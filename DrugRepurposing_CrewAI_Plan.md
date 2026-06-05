# Drug Repurposing with Agentic AI — Implementation Plan
## CrewAI Multi-Agent Framework | 1-Day Prep Guide

---

## PART 1: LITERATURE FOUNDATION — KEY PAPERS TO CITE & BUILD ON

### Tier 1: Must-Read (Directly Relevant)

#### 1. DrugAgent (Inoue et al., 2024) — arXiv:2408.13378
**Why it matters:** The closest prior work to what you're building. Multi-agent LLM system for drug repurposing with three agents: AI Agent (DeepPurpose DTI models), KG Agent (DrugBank, DGIdb, CTD, STITCH), Search Agent (PubMed/literature). Uses coordinator to integrate perspectives. Evaluated on kinase inhibitor dataset.

**Your novelty over it:** DrugAgent lacks (a) biological validation/hallucination checking, (b) pathway-level mechanistic reasoning, (c) a scoring system with confidence intervals, and (d) CrewAI's task delegation architecture.

**Cite as:** "Our work extends DrugAgent's three-agent architecture with explicit biological grounding, pathway validation, and confidence-aware scoring."

---

#### 2. DrugMCTS (Yang et al., Jul 2025) — arXiv:2507.07426
**Why it matters:** Most recent multi-agent drug repurposing paper. Combines RAG + multi-agent + Monte Carlo Tree Search. Five agents: retrieval, analysis, filtering, protein matching, ranking. Uses Qwen2.5-7B-Instruct. Benchmarked on DrugBank + KIBA. Achieves 55.34% recall. Validated one case study (Equol–CXCR3) with AutoDock Vina (binding score −8.4 kcal/mol).

**Your novelty over it:** DrugMCTS focuses on drug-target interaction prediction, not end-to-end repurposing with disease context. It lacks pathway mechanism validation, side-effect cross-checking, and interpretable clinical rationale generation.

**Cite as:** "While DrugMCTS achieves strong recall via MCTS-guided reasoning, our framework adds mechanistic pathway validation and adverse effect screening that are prerequisites for clinically actionable outputs."

---

#### 3. Prompt-to-Pill (Vichentijevikj et al., Aug 2025) — Bioinformatics Advances / bioRxiv:2025.08.12
**Why it matters:** Comprehensive survey of 51 LLM-based drug discovery systems + working pipeline from target ID to clinical simulation. Identifies that NO existing multi-agent system handles RWE, clinical trial design, AND scientific scoring together.

**Your novelty over it:** The gap they identify IS your opportunity: integrating mechanistic biological validation with a scoring pipeline, on open public datasets.

**Cite as:** "The Prompt-to-Pill survey identifies the absence of integrated RWE + biological validation in current MAS pipelines as a key gap — which our framework addresses."

---

#### 4. Deep Learning + GraphRAG for Drug Repurposing (2025) — bioRxiv:2025.12.08.693009
**Why it matters:** Uses DRKG (97k entities, 4.4M relationships) + TransE embeddings + LLM explanation. Shows knowledge graph embeddings can be queried by LLMs for explainable predictions.

**Your novelty over it:** You use KG embeddings as one evidence source within a multi-agent pipeline, rather than standalone prediction.

---

#### 5. LLM Hypothesis Validation for Drug Repurposing (Zunzunegui et al., Jun 2025) — bioRxiv:2025.06.13
**Why it matters:** Directly addresses LLM accuracy in validating repurposing hypotheses. Shows structured prompts + benchmark datasets raise precision significantly (p < 0.001). Provides a methodology for fact-checking repurposing outputs.

**Your novelty:** Use their validation methodology as your Validator Agent's prompt strategy.

---

### Tier 2: Contextual / Survey

#### 6. Wan et al., Applications of AI in Drug Repurposing — Advanced Science (Mar 2025)
doi: 10.1002/advs.202411325 — Comprehensive review of AI methods, datasets (OMIM, LINCS, CLUE), and modalities. Good for background framing.

#### 7. RAG-Enhanced Collaborative LLM Agents for Drug Discovery — arXiv:2502.17506 (Feb 2025)
Covers RAG architecture for multi-agent bio systems; shows domain-specific RAG outperforms general retrieval. Good technical foundation for your RAG design.

#### 8. KG-Bench: Benchmarking GNNs for Drug Repurposing — bioRxiv:2025.10.13
Uses Open Targets dataset. Shows TransformerConv as best GNN architecture. Provides benchmarking methodology you can reference for evaluation.

#### 9. AI Hallucinations in Drug Discovery — IntuitionLabs (2025)
Shows LLM hallucination rates of 15–40% on clinical tasks. Directly motivates your Validator Agent. Recommends KG-grounded RAG and self-reflection loops.

#### 10. Biological Database Mining for LLM-Driven Alzheimer's DR — bioRxiv:2024.12.04
Shows ontologies + LLMs as complementary: LLMs translate natural language, ontologies provide ground truth. Good template for how your agents should use structured databases.

---

## PART 2: WHAT'S NOVEL ABOUT YOUR PROJECT

Most existing systems do ONE of:
- KG embedding → drug-disease prediction (no LLM reasoning)
- LLM → literature hypothesis (no KG validation)
- Multi-agent DTI (no pathway mechanism validation)
- RAG retrieval (no iterative MCTS-style exploration)

**Your novel contribution: A Biologically-Grounded, Self-Validating Multi-Agent Repurposing Pipeline**

Specifically:
1. **Dual-evidence confidence scoring** — KG-based + literature-based evidence scores merged with explicit uncertainty quantification
2. **Pathway Mechanism Validator Agent** — checks that a drug's MOA aligns with the target disease pathway (not just association mining)
3. **Adversarial Critique Agent** — plays devil's advocate; cross-checks side effects, contraindications, and known drug failures for the same indication
4. **Structured hallucination guard** — every LLM claim must be anchored to a structured DB entry (DrugBank ID, PubMed PMID, or pathway ID) before passing downstream
5. **Open, reproducible pipeline** — runs entirely on public datasets (no IQVIA needed for demo)

---

## PART 3: DATASETS (NO IQVIA NEEDED)

| Dataset | What it provides | Access |
|---------|-----------------|--------|
| **DRKG** (Drug Repurposing Knowledge Graph) | 97k entities, 4.4M edges — drugs, diseases, genes, pathways, proteins. Pre-trained TransE embeddings included | Free: github.com/gnn4dr/DRKG |
| **DrugBank** (free tier / XML download) | Drug MOA, targets, pathways, interactions, approved indications | Free academic: drugbank.com |
| **Hetionet** | 29 node types, 24 edge types, integrated biological network | Free: het.io |
| **DisGeNET** | Gene-disease associations, curated + text-mined | Free API (academic) |
| **Open Targets** | Drug-disease evidence, target validation scores | Free API: platform.opentargets.org |
| **PubMed / NCBI E-utilities** | 35M+ abstracts for RAG | Free API |
| **ChEMBL** | Drug bioactivity data, clinical stage info | Free API |
| **ClinicalTrials.gov API** | Trial history per drug/disease pair | Free REST API |
| **OMIM** | Genetic disease basis | Free API (academic) |
| **STRING DB** | Protein-protein interaction network | Free API |

**Demo recommendation:** Focus on Alzheimer's Disease or a rare oncology indication (e.g., glioblastoma). These have rich public data and known repurposing success stories (e.g., metformin, rapamycin, ibuprofen for AD).

---

## PART 4: CREWAI MULTI-AGENT ARCHITECTURE

### System Overview

```
User Query: "Find repurposing candidates for [Disease X]"
         ↓
  [Orchestrator Agent] — CrewAI Manager
         ↓ delegates tasks
┌────────────────────────────────────────────────────────────┐
│  Agent 1: Disease Intelligence Agent                       │
│  Agent 2: KG Traversal & Candidate Agent                   │
│  Agent 3: Molecular Mechanism Agent                        │
│  Agent 4: Literature RAG Agent                             │
│  Agent 5: Pathway Validator Agent  ← YOUR NOVELTY         │
│  Agent 6: Adversarial Critique Agent  ← YOUR NOVELTY      │
│  Agent 7: Scoring & Ranking Agent                          │
│  Agent 8: Report Generation Agent                          │
└────────────────────────────────────────────────────────────┘
         ↓
  Structured Output: Ranked Candidates + Evidence + Confidence
```

---

### Agent Specifications

#### Agent 1: Disease Intelligence Agent
**Role:** Characterize the target disease in structured biological terms.

**Tools:**
- `DisGeNET_API` — gene-disease associations, confidence scores
- `OMIM_API` — genetic basis, phenotypic descriptions
- `Open_Targets_API` — disease-gene evidence, therapeutic areas
- `PubMed_search` — recent mechanistic papers

**Output schema (structured JSON):**
```json
{
  "disease_id": "EFO_0000249",
  "disease_name": "Alzheimer's disease",
  "key_pathways": ["amyloid cascade", "tau phosphorylation", "neuroinflammation"],
  "known_targets": [{"gene": "APP", "confidence": 0.95}, ...],
  "disease_genes": ["APOE", "PSEN1", "PSEN2", "BIN1"],
  "biological_context": "...",
  "data_sources": ["DisGeNET:v7.0", "OpenTargets:2025-03"]
}
```

**Validation rule:** Must cite at least 2 structured DB sources. Free-text claims flagged for downstream validation.

---

#### Agent 2: KG Traversal & Candidate Discovery Agent
**Role:** Query DRKG + Hetionet to find drugs with structural/network proximity to disease targets.

**Tools:**
- `DRKG_embedding_query` — cosine similarity search over pre-trained TransE embeddings
- `Hetionet_path_query` — metapath-based traversal (Drug→Gene→Disease paths)
- `DrugBank_API` — fetch drug metadata, approval status, known indications

**Method:**
1. Take disease genes from Agent 1
2. Find drugs that share targets (via DRKG edges: `DRUG-binds-GENE`, `DRUG-treats-DISEASE`)
3. Score by network proximity: shortest metapath length in Hetionet
4. Filter to approved/Phase III drugs only (using DrugBank status field)

**Output schema:**
```json
{
  "candidates": [
    {
      "drug_name": "Metformin",
      "drugbank_id": "DB00331",
      "approval_status": "approved",
      "original_indication": "Type 2 Diabetes",
      "network_proximity_score": 0.83,
      "shared_targets": ["AMPK", "mTOR"],
      "kg_evidence_paths": ["Metformin→AMPK→MTOR→AD-pathway"],
      "candidate_confidence": "high"
    }
  ]
}
```

**Validation rule:** Every candidate MUST have a valid DrugBank ID. Score only computed from structured edges, not LLM inference.

---

#### Agent 3: Molecular Mechanism Agent
**Role:** For top-N candidates, retrieve and summarize MOA at molecular level.

**Tools:**
- `DrugBank_mechanism_fetch` — official MOA text
- `ChEMBL_bioactivity_API` — binding data (Ki, IC50 for target proteins)
- `STRING_DB_API` — protein-protein interaction network for targets
- `PubChem_API` — structural data, pharmacophore

**Output:** For each candidate:
- Primary mechanism (with DrugBank citation)
- Binding affinity data for relevant targets (ChEMBL IDs)
- Protein interaction network context

**Validation rule:** Binding claims must cite a ChEMBL assay ID. No affinity values without experimental backing.

---

#### Agent 4: Literature RAG Agent
**Role:** Find published evidence for/against each candidate-disease pair.

**Architecture:** RAG over PubMed abstracts using BM25 + dense retrieval hybrid.

**Tools:**
- `PubMed_E_utilities` — bulk abstract fetch by MeSH terms + drug name + disease name
- `Vector_store_query` — semantic search over indexed abstracts (use ChromaDB or FAISS locally)
- `Citation_extractor` — pull PMIDs, publication years, study types

**Query strategy:**
```
query = f"{drug_name} {disease_name} repurposing OR treatment OR mechanism"
filter: year >= 2018, article types: clinical trial, systematic review, meta-analysis preferred
```

**Output:** Per candidate: list of PMIDs, study types, key findings (paraphrased, not copied), publication years. Distinguishes: positive evidence / negative evidence / mechanistic only.

**Validation rule:** Every literature claim tied to a PMID. Agent explicitly flags if fewer than 3 papers found (low-evidence flag).

---

#### Agent 5: Pathway Mechanism Validator Agent ⭐ (YOUR KEY NOVELTY)
**Role:** Verify that the drug's MOA mechanistically connects to the disease's known pathogenic pathway. This is the biological grounding layer that most systems lack.

**Logic:**
1. Take drug MOA (from Agent 3) + disease pathway (from Agent 1)
2. Map both to KEGG/Reactome pathway IDs
3. Check: Does the drug's target sit within or upstream of the disease pathway?
4. Score mechanistic alignment: Direct (drug targets disease pathway gene) / Indirect (drug targets upstream regulator) / Speculative (no known pathway connection)

**Tools:**
- `Reactome_API` — pathway membership queries
- `KEGG_REST_API` — pathway maps, gene-pathway associations
- `UniProt_API` — protein function, pathway memberships

**Output:**
```json
{
  "drug": "Metformin",
  "disease": "Alzheimer's Disease",
  "pathway_alignment": "INDIRECT",
  "connection": "Metformin activates AMPK → suppresses mTOR → reduces tau hyperphosphorylation (AD pathway)",
  "pathway_ids": ["KEGG:hsa04150", "Reactome:R-HSA-165159"],
  "biological_plausibility_score": 0.78,
  "validation_notes": "AMPK-mTOR-tau axis supported by 4 experimental studies (PMIDs: ...)"
}
```

**Validation rules:**
- Must map to at least one Reactome or KEGG pathway ID
- "Speculative" alignment auto-flags candidate for low priority
- Side-effect check: Cross-reference drug known ADRs (DrugBank) against disease-critical systems (e.g., CNS drugs for CNS diseases must check BBB penetration)

---

#### Agent 6: Adversarial Critique Agent ⭐ (YOUR KEY NOVELTY)
**Role:** Act as a skeptical reviewer. Find reasons the candidate SHOULDN'T work.

**Checks performed:**
1. **Previous trial failures:** Query ClinicalTrials.gov for terminated/failed trials of this drug for this indication
2. **Contraindications:** Check DrugBank contraindications against disease comorbidities
3. **Drug-drug interactions:** Flag if drug has interactions with standard-of-care for the disease
4. **Bioavailability at target:** Does the drug reach the relevant tissue/organ? (BBB for CNS, liver for metabolic, etc.)
5. **Selectivity concerns:** Is the drug too promiscuous (many off-targets) to attribute effect?

**Tools:**
- `ClinicalTrials_API` — filter by drug + disease, status = "terminated" or "withdrawn"
- `DrugBank_interactions_API`
- `DrugBank_pharmacokinetics` — Cmax, tissue distribution, BBB penetration data

**Output:**
```json
{
  "drug": "Metformin",
  "red_flags": [],
  "yellow_flags": ["No CNS-specific PK data", "mTOR effects may be dose-dependent"],
  "failed_trials": [],
  "overall_adversarial_verdict": "PROCEED — no disqualifying factors found",
  "confidence_adjustment": -0.05
}
```

**If a red flag is found (e.g., 3+ failed Phase II trials), candidate is deprioritized automatically.**

---

#### Agent 7: Scoring & Ranking Agent
**Role:** Aggregate all agent outputs into a final composite score with uncertainty.

**Scoring formula (transparent, explainable):**

```
Final Score = (
  KG_proximity_score × 0.20    [structural evidence]
  + MOA_alignment_score × 0.25  [mechanistic grounding]
  + literature_score × 0.20     [published evidence]
  + pathway_plausibility × 0.25 [biological validation]
  + adversarial_adjustment × 0.10 [risk penalty]
)

Confidence band = ±σ based on evidence count (fewer papers → wider band)
```

**Output:** Ranked table with scores, confidence intervals, and a one-line justification per candidate grounded in specific evidence.

**Validation rule:** Scores above 0.7 require evidence from at least 3 agents. Scores presented with ± uncertainty always.

---

#### Agent 8: Report Generation Agent
**Role:** Produce a structured, human-readable output that a pharma scientist could evaluate.

**Output format:**
```
# Drug Repurposing Report: [Disease] — [Date]

## Executive Summary
Top 3 candidates with composite scores

## Candidate Detail Cards (per drug)
- Drug Profile (DrugBank ID, approval, original indication)
- Evidence Summary (KG + literature + mechanism)  
- Biological Rationale (pathway alignment, cited)
- Risk Flags (adversarial findings)
- Confidence Score: X.XX ± Y.YY
- Recommended Next Step (e.g., "in vitro assay on target X")

## Methodology & Data Sources
## Limitations & Caveats
## References (all PMIDs and DB IDs)
```

**Validation rule:** Every factual claim in the report maps to a source. Final section lists all data sources with version/date.

---

## PART 5: CREWAI IMPLEMENTATION PLAN

### Tech Stack
```
crewai>=0.80.0
langchain-openai (or langchain-anthropic for Claude)
chromadb (local vector store for RAG)
requests (API calls to DrugBank, PubMed, etc.)
networkx (KG traversal on DRKG)
pandas, numpy (scoring)
python-dotenv
pydantic v2 (output validation schemas)
```

### Project Structure
```
drug_repurposing_crew/
├── main.py                  # Entry point
├── crew.py                  # CrewAI crew definition
├── agents/
│   ├── disease_intel.py
│   ├── kg_candidate.py
│   ├── mol_mechanism.py
│   ├── literature_rag.py
│   ├── pathway_validator.py  ← Novel
│   ├── adversarial.py        ← Novel
│   ├── scorer.py
│   └── report_gen.py
├── tools/
│   ├── drkg_tools.py        # DRKG embedding + graph query
│   ├── pubmed_tools.py      # E-utilities + RAG
│   ├── drugbank_tools.py    # DrugBank API wrapper
│   ├── pathway_tools.py     # KEGG + Reactome APIs
│   ├── clinical_trials.py   # ClinicalTrials.gov API
│   └── open_targets.py      # Open Targets platform API
├── data/
│   ├── drkg/                # DRKG embeddings (downloaded once)
│   └── vector_store/        # ChromaDB index
├── schemas/
│   ├── candidate.py         # Pydantic models for output validation
│   ├── disease_profile.py
│   └── report.py
├── validation/
│   ├── hallucination_guard.py  # Source-grounding check
│   └── biological_validator.py
└── tests/
    └── test_alzheimers.py   # Demo test case
```

### CrewAI Crew Configuration (crew.py sketch)
```python
from crewai import Agent, Task, Crew, Process

# Hierarchical process — Orchestrator delegates
crew = Crew(
    agents=[
        disease_intel_agent,
        kg_candidate_agent,
        mol_mechanism_agent,
        literature_rag_agent,
        pathway_validator_agent,
        adversarial_agent,
        scorer_agent,
        report_agent,
    ],
    tasks=[...],  # Sequential with dependencies
    process=Process.hierarchical,
    manager_llm=claude_opus,  # or gpt-4o
    verbose=True,
    memory=True,  # Share context across agents
)
```

### Key Implementation Decisions

**LLM choice:** Claude claude-sonnet-4-20250514 for reasoning agents (Pathway Validator, Adversarial Critique). Cheaper models (GPT-4o-mini / Claude Haiku) for retrieval/formatting agents.

**DRKG usage:** Download pre-trained TransE embeddings (~500MB). Use cosine similarity between drug entity vectors and disease-proximal gene vectors. No GPU required for inference on pre-trained embeddings.

**RAG setup:** Index last 5 years of PubMed abstracts for your demo disease (~50k–200k abstracts for AD). Use ChromaDB locally. Hybrid BM25 + semantic retrieval.

**Hallucination guard (validation/hallucination_guard.py):**
```python
def validate_claim(claim: str, source_ids: list[str]) -> bool:
    """Every factual claim must have at least one verifiable source ID."""
    if not source_ids:
        return False  # Reject unsourced claims
    # Verify IDs exist in known databases
    for src_id in source_ids:
        if not is_valid_db_id(src_id):  # Check DrugBank, PubMed, KEGG formats
            return False
    return True
```

---

## PART 6: DEMO SCENARIO (1-DAY BUILD)

### Recommended Demo Disease: Alzheimer's Disease
- Rich public data (DRKG has ~800 AD-related edges)
- Known repurposing success stories to validate against: Metformin, Rapamycin, Semaglutide
- Active research area = easy to find recent literature
- DisGeNET + OpenTargets have excellent coverage

### Demo Script
1. Input: "Find repurposing candidates for Alzheimer's Disease"
2. Pipeline runs all 8 agents sequentially
3. Output: Top 5 candidates with scores, evidence, and biological rationale
4. Highlight: Show how Metformin or Semaglutide emerges with high biological plausibility score + pathway alignment
5. Show adversarial check catching a compound with BBB penetration issues
6. Show hallucination guard rejecting an unsourced claim

### Known Ground-Truth Validation
Use these as positive controls (should score highly):
- Metformin → AD (AMPK-mTOR-tau pathway; multiple observational studies)
- Semaglutide → AD (GLP-1R pathway; ongoing Phase III NCT04777396)
- Rapamycin → AD (mTOR inhibition; multiple animal model studies)

If your pipeline surfaces these with high scores, that validates the system.

---

## PART 7: EVALUATION METRICS

| Metric | How to compute |
|--------|---------------|
| **Recall@K** | Known repurposed drugs recovered in top-K (use DRKG ground truth) |
| **Pathway alignment accuracy** | % of top-ranked candidates with mechanistic connection to disease |
| **Hallucination rate** | % of claims without valid source IDs (should be ~0%) |
| **Adversarial false positive rate** | % of known-good candidates flagged as red |
| **Score calibration** | Do higher-scored candidates have more literature support? |

---

## PART 8: WHAT MAKES THIS DEFENSIBLE IN PRESENTATION

1. **Biologically grounded, not just statistically associated** — Pathway Validator ensures MOA alignment, not just co-occurrence
2. **Self-critiquing by design** — Adversarial Agent is architecturally built in, not an afterthought
3. **Open and reproducible** — All data sources are public, all code is runnable
4. **Validated against known repurposing successes** — Metformin/Semaglutide for AD as positive controls
5. **Uncertainty-aware scoring** — Not a black-box; scores come with confidence bands
6. **Source-anchored outputs** — Every claim has a DB ID or PMID; hallucination guard rejects unsourced claims

---

## PART 9: 1-DAY BUILD TIMELINE

| Time | Task |
|------|------|
| Hour 1–2 | Download DRKG, set up project structure, install deps |
| Hour 3–4 | Build Agent 2 (KG Candidate) + Agent 1 (Disease Intel) — core pipeline |
| Hour 5–6 | Build Agent 4 (Literature RAG) + ChromaDB index for AD |
| Hour 7–8 | Build Agent 5 (Pathway Validator) + Agent 6 (Adversarial Critic) — your novelty |
| Hour 9–10 | Build Agent 7 (Scorer) + Agent 8 (Report) + CrewAI crew assembly |
| Hour 11 | Run end-to-end demo, debug, verify Metformin/Semaglutide score high |
| Hour 12 | Write README, prepare demo script, document limitations |

---

## PART 10: LIMITATIONS TO ACKNOWLEDGE (HONEST FRAMING)

- DRKG was last updated 2020 (COVID-era); newer drug-disease edges missing → supplement with Open Targets real-time API
- No wet-lab validation; all predictions are computational hypotheses
- Pathway connections are inferred, not experimentally confirmed for all candidates
- LLM reasoning steps (Agents 5, 6) still subject to errors despite grounding; human expert review required before any clinical action
- Demo uses free-tier API rate limits; production would need API keys + caching

---

*References: DrugAgent (arXiv:2408.13378) · DrugMCTS (arXiv:2507.07426) · Prompt-to-Pill (bioRxiv:2025.08.12) · GraphRAG+KG (bioRxiv:2025.12.08) · LLM Hypothesis Validation (bioRxiv:2025.06.13) · Wan et al. Advanced Science 2025 · DRKG (github.com/gnn4dr/DRKG) · KG-Bench (bioRxiv:2025.10.13)*
