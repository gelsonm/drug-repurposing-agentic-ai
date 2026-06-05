"""
CrewAI crew definition for drug repurposing pipeline.
Orchestrates all 5 specialized agents in a hierarchical process.
"""
from __future__ import annotations

import json
import time
from typing import Any

from crewai import Agent, Crew, Process, Task
from langchain_groq import ChatGroq
from loguru import logger

from src.config import get_settings
from src.models.entities import (
    Disease,
    Drug,
    EvidenceItem,
    EvidenceType,
    KGPath,
    RepurposingCandidate,
    RepurposingReport,
    ScoreBreakdown,
)
from src.scoring.composite_scorer import compute_composite_score
from src.tools import (
    analyze_trial_landscape,
    assess_safety_flag,
    build_repurposing_query,
    compute_gene_overlap_score,
    extract_safety_summary,
    extract_uniprot_ids,
    find_drug_disease_paths,
    get_drug_label,
    get_drug_mechanisms,
    get_drug_targets,
    get_drug_targets_from_kg,
    get_proteins_for_targets,
    get_top_adverse_reactions,
    ingest_articles,
    query_literature,
    search_and_fetch,
    search_drug_by_name,
    search_trials,
)
from src.tools.pubmed_client import build_cancer_repurposing_query


def _get_llm() -> ChatGroq:
    """Initialize the Groq LLM for all agents."""
    settings = get_settings()
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
    )


class DrugRepurposingCrew:
    """
    Orchestrates the 5-agent drug repurposing pipeline using CrewAI.

    The crew follows a sequential process:
    1. Molecular Mechanism Agent → gathers mechanistic evidence
    2. Literature Intelligence Agent → surfaces published evidence
    3. Clinical Signal Agent → analyzes trials & safety
    4. Scoring Agent → computes composite scores
    5. Brief Generation Agent → synthesizes the final report
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = _get_llm()

    # ------------------------------------------------------------------
    # Agent Definitions
    # ------------------------------------------------------------------

    def _molecular_agent(self) -> Agent:
        return Agent(
            role="Molecular Mechanism Analyst",
            goal=(
                "Identify molecular targets and mechanistic links between drugs "
                "and diseases using ChEMBL, UniProt, and biomedical knowledge graphs."
            ),
            backstory=(
                "You are a senior computational pharmacologist with expertise in "
                "drug-target interaction analysis, pathway biology, and knowledge graph "
                "reasoning. You rigorously evaluate mechanistic plausibility and never "
                "speculate beyond what the data supports."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
        )

    def _literature_agent(self) -> Agent:
        return Agent(
            role="Literature Intelligence Specialist",
            goal=(
                "Search PubMed and retrieve published evidence for drug-disease "
                "repurposing hypotheses using RAG over indexed abstracts."
            ),
            backstory=(
                "You are an expert in biomedical literature mining with deep knowledge "
                "of PubMed search strategies, evidence quality assessment, and scientific "
                "writing. You always cite your sources and assess the strength of evidence "
                "critically, distinguishing correlation from causation."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
        )

    def _clinical_agent(self) -> Agent:
        return Agent(
            role="Clinical & Safety Intelligence Agent",
            goal=(
                "Analyze clinical trial evidence and FDA safety data to assess "
                "the clinical feasibility and safety profile of repurposing candidates."
            ),
            backstory=(
                "You are a clinical pharmacovigilance expert with years of experience "
                "analyzing clinical trial data, FDA FAERS adverse event reports, and "
                "drug labeling. You are meticulous about safety signals and always flag "
                "potential contraindications or serious adverse events."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
        )

    def _scoring_agent(self) -> Agent:
        return Agent(
            role="Repurposing Candidate Scorer",
            goal=(
                "Synthesize evidence from all agents into composite scores and "
                "produce a ranked list of repurposing candidates."
            ),
            backstory=(
                "You are a quantitative analyst specializing in drug development "
                "decision frameworks. You combine mechanistic, literature, and clinical "
                "evidence into transparent, explainable scores with clear rationale."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2,
        )

    def _brief_agent(self) -> Agent:
        return Agent(
            role="Scientific Brief Writer",
            goal=(
                "Generate a comprehensive, boardroom-ready drug repurposing brief "
                "with ranked candidates, evidence chains, and actionable next steps."
            ),
            backstory=(
                "You are a scientific communications expert who translates complex "
                "biomedical data into clear, compelling narratives for pharma executives "
                "and R&D decision-makers. Your reports are precise, well-cited, and "
                "structured for maximum clarity."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2,
        )

    # ------------------------------------------------------------------
    # Core Pipeline (direct execution, no CrewAI task overhead)
    # ------------------------------------------------------------------

    def run(self, query_drug: str | None, query_disease: str | None, query_raw: str) -> RepurposingReport:
        """
        Execute the full drug repurposing pipeline.

        Args:
            query_drug: Drug name (e.g. "Metformin"). If None, searches broadly.
            query_disease: Disease name. If None, discovers candidates for the drug.
            query_raw: Original user query string.

        Returns:
            RepurposingReport with ranked candidates.
        """
        start_time = time.time()
        logger.info(f"Starting pipeline: drug='{query_drug}', disease='{query_disease}'")

        # Determine what to analyze
        target_pairs = self._resolve_target_pairs(query_drug, query_disease)

        candidates = []
        for drug_name, disease_name in target_pairs:
            logger.info(f"Analyzing pair: {drug_name} → {disease_name}")
            candidate = self._analyze_pair(drug_name, disease_name)
            if candidate:
                candidates.append(candidate)

        # Sort by composite score
        candidates.sort(key=lambda c: c.score.composite_score, reverse=True)

        # Generate summary via LLM
        summary = self._generate_summary(candidates, query_raw)
        next_steps = self._generate_next_steps(candidates)

        runtime = time.time() - start_time
        logger.info(f"Pipeline complete in {runtime:.1f}s. {len(candidates)} candidates found.")

        return RepurposingReport(
            query_drug=query_drug,
            query_disease=query_disease,
            query_raw=query_raw,
            candidates=candidates,
            summary=summary,
            next_steps=next_steps,
            total_runtime_seconds=round(runtime, 1),
        )

    def _resolve_target_pairs(
        self,
        query_drug: str | None,
        query_disease: str | None,
    ) -> list[tuple[str, str]]:
        """
        Determine which drug-disease pairs to analyze.

        Strategy:
        - Drug specified + no disease: analyze top 3 candidate diseases from KG
        - Disease specified + no drug: analyze known drugs for that disease
        - Both specified: analyze that specific pair
        """
        if query_drug and query_disease:
            return [(query_drug, query_disease)]

        if query_drug and not query_disease:
            # Find candidate diseases via KG
            from src.tools.kg_client import load_knowledge_graph, _find_nodes_by_name
            G = load_knowledge_graph()
            drug_nodes = _find_nodes_by_name(G, query_drug, kind_filter="Compound")
            candidate_diseases = []
            for dn in drug_nodes[:2]:
                for neighbor in G.neighbors(dn):
                    node_data = G.nodes[neighbor]
                    if node_data.get("kind") == "Disease":
                        candidate_diseases.append(node_data.get("name", str(neighbor)))
            # Deduplicate and take top 3
            seen = set()
            unique_diseases = []
            for d in candidate_diseases:
                if d not in seen:
                    seen.add(d)
                    unique_diseases.append(d)
            if unique_diseases:
                return [(query_drug, d) for d in unique_diseases[:3]]
            # Fall back to known cancer diseases
            return [(query_drug, "Colorectal cancer"), (query_drug, "Pancreatic cancer")]

        if query_disease and not query_drug:
            # Default to top approved drugs in KG
            from src.tools.kg_client import load_knowledge_graph
            G = load_knowledge_graph()
            compounds = [
                n for n, d in G.nodes(data=True) if d.get("kind") == "Compound"
            ]
            return [(c, query_disease) for c in compounds[:3]]

        # Default demo case (cancer)
        return [("Metformin", "Colorectal cancer")]

    def _analyze_pair(self, drug_name: str, disease_name: str) -> RepurposingCandidate | None:
        """Run the full 5-step analysis for a single drug-disease pair."""
        try:
            # --- Step 1: Molecular Mechanism ---
            logger.info(f"[1/5] Molecular mechanism: {drug_name} → {disease_name}")
            mol_data = self._run_molecular_analysis(drug_name, disease_name)

            # --- Step 2: Literature ---
            logger.info(f"[2/5] Literature search: {drug_name} + {disease_name}")
            lit_data = self._run_literature_analysis(drug_name, disease_name)

            # --- Step 3: Clinical Signal ---
            logger.info(f"[3/5] Clinical signal: {drug_name} + {disease_name}")
            clin_data = self._run_clinical_analysis(drug_name, disease_name)

            # --- Step 4: Scoring ---
            logger.info(f"[4/5] Scoring candidate")
            score = compute_composite_score(
                kg_paths=mol_data["kg_paths"],
                gene_overlap=mol_data["gene_overlap"],
                n_targets=mol_data["n_targets"],
                rag_passages=lit_data["passages"],
                pubmed_count=lit_data["pubmed_count"],
                trial_landscape=clin_data["trial_landscape"],
                safety_flag=clin_data["safety_flag"],
                has_chembl=mol_data["has_chembl"],
                has_uniprot=mol_data["has_uniprot"],
                has_label=clin_data["has_label"],
            )

            # --- Step 5: Build candidate object ---
            drug_obj = Drug(
                name=drug_name,
                chembl_id=mol_data.get("chembl_id"),
                targets=mol_data.get("target_names", []),
                mechanism_of_action=mol_data.get("mechanism"),
            )
            disease_obj = Disease(name=disease_name)

            # Build evidence items
            evidence_items = []
            for passage in lit_data["passages"][:8]:
                evidence_items.append(EvidenceItem(
                    evidence_type=EvidenceType.LITERATURE,
                    source="PubMed",
                    source_id=passage.get("pmid"),
                    title=passage.get("title"),
                    snippet=passage["text"][:500],
                    url=passage.get("url"),
                    year=int(passage["year"]) if passage.get("year") and str(passage["year"]).isdigit() else None,
                    relevance_score=passage.get("relevance_score", 0.5),
                ))

            # Add KG path evidence
            for path in mol_data["kg_paths"][:3]:
                evidence_items.append(EvidenceItem(
                    evidence_type=EvidenceType.KNOWLEDGE_GRAPH,
                    source="Hetionet",
                    snippet=path.get("description", "KG path"),
                    relevance_score=path.get("path_score", 0.5),
                ))

            # Build KGPath objects
            kg_path_objs = [
                KGPath(
                    nodes=p.get("nodes", []),
                    edge_types=p.get("edge_types", []),
                    path_score=p.get("path_score", 0.0),
                    description=p.get("description"),
                )
                for p in mol_data["kg_paths"]
            ]

            return RepurposingCandidate(
                drug=drug_obj,
                disease=disease_obj,
                score=score,
                kg_paths=kg_path_objs,
                evidence_items=evidence_items,
                mechanistic_rationale=mol_data.get("rationale"),
                literature_summary=lit_data.get("summary"),
                clinical_summary=clin_data.get("summary"),
                safety_summary=clin_data.get("safety_summary"),
            )

        except Exception as e:
            logger.error(f"Error analyzing pair ({drug_name}, {disease_name}): {e}")
            return None

    def _run_molecular_analysis(self, drug_name: str, disease_name: str) -> dict[str, Any]:
        """Execute molecular mechanism analysis."""
        result: dict[str, Any] = {
            "kg_paths": [], "gene_overlap": 0.0, "n_targets": 0,
            "has_chembl": False, "has_uniprot": False,
            "chembl_id": None, "target_names": [], "mechanism": None,
            "rationale": None,
        }

        # ChEMBL lookup
        molecules = search_drug_by_name(drug_name, max_results=3)
        if molecules:
            mol = molecules[0]
            result["has_chembl"] = True
            result["chembl_id"] = mol.get("molecule_chembl_id")
            result["mechanism"] = mol.get("mechanism_of_action")

            # Get targets
            if result["chembl_id"]:
                targets = get_drug_targets(result["chembl_id"])
                result["n_targets"] = len(targets)
                result["target_names"] = [t.get("pref_name", "") for t in targets[:5]]

                uniprot_ids = extract_uniprot_ids(targets)
                if uniprot_ids:
                    proteins = get_proteins_for_targets(uniprot_ids[:5])
                    result["has_uniprot"] = bool(proteins)

        # KG paths
        kg_paths = find_drug_disease_paths(drug_name, disease_name, max_paths=5)
        result["kg_paths"] = kg_paths

        # Gene overlap
        result["gene_overlap"] = compute_gene_overlap_score(drug_name, disease_name)

        # Generate rationale with LLM
        if kg_paths or result["target_names"]:
            path_desc = "; ".join(p.get("description", "") for p in kg_paths[:3])
            targets_str = ", ".join(result["target_names"][:5])
            prompt = (
                f"In 2-3 sentences, explain the molecular rationale for why {drug_name} "
                f"might be effective against {disease_name}. "
                f"Known targets: {targets_str}. "
                f"Knowledge graph paths: {path_desc}. "
                f"Be specific and mechanistically precise."
            )
            try:
                response = self.llm.invoke(prompt)
                result["rationale"] = response.content
            except Exception as e:
                logger.debug(f"LLM rationale failed: {e}")
                result["rationale"] = (
                    f"{drug_name} targets {targets_str} which are implicated in {disease_name} pathways."
                )

        return result

    def _run_literature_analysis(self, drug_name: str, disease_name: str) -> dict[str, Any]:
        """Execute literature intelligence analysis."""
        result: dict[str, Any] = {
            "passages": [], "pubmed_count": 0, "summary": None,
        }

        # Search PubMed — use cancer-optimized query if disease looks like oncology
        cancer_keywords = ["cancer", "carcinoma", "myeloma", "tumor", "glioblastoma", "leukemia", "lymphoma"]
        is_cancer = any(kw in disease_name.lower() for kw in cancer_keywords)
        if is_cancer:
            query = build_cancer_repurposing_query(drug_name, disease_name)
        else:
            query = build_repurposing_query(drug_name, disease_name)
        articles = search_and_fetch(query, max_results=15, min_year=2015)
        result["pubmed_count"] = len(articles)

        if articles:
            # Ingest into ChromaDB
            ingest_articles(articles)
            # RAG query
            rag_query = f"{drug_name} treatment for {disease_name} mechanism evidence"
            passages = query_literature(rag_query, top_k=8)
            result["passages"] = passages

            # Generate summary
            top_passages = "\n".join(p["text"][:300] for p in passages[:3])
            prompt = (
                f"Based on these PubMed abstracts, summarize in 2-3 sentences the "
                f"published evidence for {drug_name} in {disease_name}:\n\n{top_passages}"
            )
            try:
                response = self.llm.invoke(prompt)
                result["summary"] = response.content
            except Exception as e:
                logger.debug(f"LLM lit summary failed: {e}")
                result["summary"] = f"Found {len(articles)} PubMed articles on {drug_name} and {disease_name}."

        return result

    def _run_clinical_analysis(self, drug_name: str, disease_name: str) -> dict[str, Any]:
        """Execute clinical signal analysis."""
        result: dict[str, Any] = {
            "trial_landscape": {}, "safety_flag": "unknown",
            "has_label": False, "summary": None, "safety_summary": None,
        }

        # Clinical trials
        trials = search_trials(drug_name=drug_name, disease_name=disease_name, max_results=20)
        result["trial_landscape"] = analyze_trial_landscape(trials)

        # FDA drug label
        label = get_drug_label(drug_name)
        label_summary = extract_safety_summary(label)
        result["has_label"] = label_summary.get("has_label", False)
        result["safety_flag"] = assess_safety_flag(label_summary, [], disease_name)

        # Clinical summary
        trial_count = result["trial_landscape"].get("total", 0)
        active = result["trial_landscape"].get("active", 0)
        completed = result["trial_landscape"].get("completed", 0)
        result["summary"] = (
            f"Found {trial_count} clinical trials ({active} active, {completed} completed) "
            f"for {drug_name} in {disease_name}. "
            f"Trial signal: {result['trial_landscape'].get('clinical_signal', 'unknown')}."
        )
        result["safety_summary"] = (
            f"Safety flag: {result['safety_flag']}. "
            + (f"Indications: {label_summary['indications'][:200]}" if result["has_label"] else "No FDA label found.")
        )

        return result

    def _generate_summary(self, candidates: list[RepurposingCandidate], query: str) -> str:
        """Generate executive summary via LLM."""
        if not candidates:
            return "No repurposing candidates were found for the given query."

        top = candidates[:3]
        bullets = "\n".join(
            f"- {c.drug.name} for {c.disease.name}: score={c.score.composite_score:.2f} ({c.rank_label})"
            for c in top
        )
        prompt = (
            f"You are a pharmaceutical research analyst specializing in oncology drug repurposing. "
            f"Summarize the following findings in 3-4 sentences for an executive audience.\n\n"
            f"Query: {query}\n\nTop candidates:\n{bullets}\n\n"
            f"Highlight mechanistic novelty, clinical evidence strength, and why these non-oncology "
            f"drugs could be fast-tracked for cancer indications (existing safety data = reduced Phase I risk)."
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.debug(f"LLM summary failed: {e}")
            return f"Analysis identified {len(candidates)} repurposing candidates. Top: {top[0].drug.name} for {top[0].disease.name} (score: {top[0].score.composite_score:.2f})."

    def _generate_next_steps(self, candidates: list[RepurposingCandidate]) -> list[str]:
        """Generate recommended next steps based on top candidates."""
        if not candidates:
            return ["Expand search to broader drug and disease categories."]

        top = candidates[0] if candidates else None
        steps = []
        if top:
            steps.append(
                f"Validate {top.drug.name} → {top.disease.name} hypothesis with in vitro binding assays."
            )
            if top.score.active_trials > 0:
                steps.append(f"Review existing clinical trial protocols for {top.drug.name} in {top.disease.name}.")
            steps.append("Conduct systematic literature review for top 3 candidates.")
            steps.append("Engage disease experts for clinical plausibility assessment.")
            steps.append("File FDA IND application if preclinical validation succeeds.")
        return steps
