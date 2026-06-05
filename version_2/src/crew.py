"""
DrugRepurposingCrew v2 — 8-Agent Sequential Pipeline Orchestrator.

Agents run sequentially:
  1. Disease Intelligence → DiseaseProfile
  2. KG Candidate Discovery → list[CandidateDrug]
  3. Molecular Mechanism → enriched CandidateDrug (per candidate)
  4. Literature RAG → literature-enriched candidate + passages
  5. Pathway Validator → PathwayAlignment (per candidate, top N)
  6. Adversarial Critique → AdversarialReport (per candidate, top N)
  7. Scoring & Ranking → ScoredCandidate (per candidate)
  8. Report Generation → RepurposingReport

Memory: Each agent receives the accumulated context from prior agents.
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.agents.adversarial import run_adversarial_agent
from src.agents.disease_intel import run_disease_intel_agent
from src.agents.kg_candidate import run_kg_candidate_agent
from src.agents.literature_rag import run_literature_rag_agent
from src.agents.mol_mechanism import run_mol_mechanism_agent
from src.agents.pathway_validator import run_pathway_validator_agent
from src.agents.report_gen import run_report_agent
from src.agents.scorer import rank_candidates, run_scorer_agent
from src.config import get_settings
from src.schemas.models import RepurposingReport


class DrugRepurposingCrew:
    """Orchestrates the 8-agent sequential repurposing pipeline."""

    def __init__(self, top_n: int | None = None, use_llm_report: bool = True):
        settings = get_settings()
        self.top_n = top_n or settings.top_n_for_deep_analysis
        self.max_candidates = settings.max_candidates
        self.use_llm_report = use_llm_report

    def run(self, disease_name: str, progress_callback=None) -> RepurposingReport:
        """
        Run the full 8-agent pipeline for a given disease.

        Args:
            disease_name: Target disease name (e.g., "Alzheimer's disease")
            progress_callback: Optional callable(step: int, total: int, msg: str)

        Returns:
            RepurposingReport with ranked candidates and evidence
        """
        pipeline_start = time.time()
        total_steps = 8

        def _progress(step: int, msg: str):
            logger.info(f"[Pipeline] Step {step}/{total_steps}: {msg}")
            if progress_callback:
                progress_callback(step, total_steps, msg)

        # ── Agent 1: Disease Intelligence ────────────────────────────────
        _progress(1, f"Characterizing '{disease_name}' with Open Targets + DisGeNET...")
        disease_profile = run_disease_intel_agent(disease_name)

        # ── Agent 2: KG Candidate Discovery ──────────────────────────────
        _progress(2, "Discovering candidates via Hetionet KG + Open Targets...")
        raw_candidates = run_kg_candidate_agent(disease_profile, max_candidates=self.max_candidates)

        if not raw_candidates:
            logger.warning("No candidates found — returning empty report")
            return RepurposingReport(
                disease_name=disease_name,
                disease_profile=disease_profile,
                candidates=[],
                executive_summary=f"No repurposing candidates found for {disease_name}. "
                                  "Try a different disease name or check API connectivity.",
            )

        # ── Agents 3 + 4: Per-candidate deep analysis (all candidates) ───
        enriched_candidates = []
        all_passages: dict[str, list[dict]] = {}

        for i, cand in enumerate(raw_candidates):
            logger.info(f"[Pipeline] Analyzing candidate {i+1}/{len(raw_candidates)}: {cand.drug_name}")

            # Agent 3: Molecular Mechanism
            _progress(3, f"Molecular mechanism: {cand.drug_name}...")
            cand = run_mol_mechanism_agent(cand)

            # Agent 4: Literature RAG
            _progress(4, f"Literature RAG: {cand.drug_name}...")
            cand, passages = run_literature_rag_agent(
                cand, disease_name, is_cns=disease_profile.is_cns_disease
            )
            all_passages[cand.drug_name] = passages
            enriched_candidates.append(cand)

        # ── Agents 5 + 6 + 7: Deep analysis for top-N only ───────────────
        # Sort by preliminary score (KG proximity + OT score) for top-N selection
        enriched_candidates.sort(
            key=lambda c: c.network_proximity_score + c.open_targets_score * 0.3 + len(c.literature_pmids) * 0.01,
            reverse=True,
        )
        top_candidates = enriched_candidates[:self.top_n]

        scored_candidates = []

        for cand in enriched_candidates:
            drug_name = cand.drug_name
            passages = all_passages.get(drug_name, [])

            if cand in top_candidates:
                # Full deep analysis
                _progress(5, f"Pathway validation: {drug_name}...")
                pathway_alignment = run_pathway_validator_agent(cand, disease_profile, passages)

                _progress(6, f"Adversarial critique: {drug_name}...")
                adversarial_report = run_adversarial_agent(cand, disease_profile, pathway_alignment)
            else:
                # Skipped for lower-ranked candidates
                pathway_alignment = None
                adversarial_report = None

            _progress(7, f"Scoring: {drug_name}...")
            scored = run_scorer_agent(cand, pathway_alignment, adversarial_report, passages)
            scored_candidates.append(scored)

        # ── Rank all candidates ───────────────────────────────────────────
        scored_candidates = rank_candidates(scored_candidates)

        # ── Agent 8: Report Generation ────────────────────────────────────
        _progress(8, "Generating final report...")
        runtime = round(time.time() - pipeline_start, 1)

        report = run_report_agent(
            disease_name=disease_name,
            disease_profile=disease_profile,
            scored_candidates=scored_candidates,
            runtime_seconds=runtime,
            use_llm=self.use_llm_report,
        )

        logger.info(
            f"[Pipeline] Complete: {len(scored_candidates)} candidates, "
            f"runtime={runtime}s, top_score={scored_candidates[0].score.composite:.3f} "
            f"({scored_candidates[0].candidate.drug_name})"
        )
        return report
