# Drug Repurposing v2 — Biologically-Grounded Multi-Agent AI Pipeline

> A significant upgrade over v1: every claim is pathway-validated, adversarially critiqued, and source-anchored.

## What's New in v2

| Feature | v1 | **v2** |
|---------|-----|--------|
| Disease characterization | KG-only | Open Targets + DisGeNET |
| Pathway validation | ❌ | ✅ **KEGG + Reactome alignment** |
| Adversarial critique | ❌ | ✅ **Failed trials + DDI + BBB check** |
| Hallucination guard | Basic | ✅ **Source-anchored every claim** |
| Score uncertainty | ❌ | ✅ **±Confidence Intervals** |
| Evidence polarity | ❌ | ✅ **Positive / Negative / Mechanistic** |
| CNS disease handling | ❌ | ✅ **BBB penetration database** |
| Input | Fixed drug+disease | **Any disease name** |

## Architecture: 8-Agent Sequential Pipeline

```
Disease Input
     ↓
[Agent 1] Disease Intelligence   — Open Targets + DisGeNET + PubMed
     ↓  DiseaseProfile
[Agent 2] KG Candidate Discovery — Hetionet + Open Targets known drugs
     ↓  list[CandidateDrug]
[Agent 3] Molecular Mechanism    — ChEMBL + STRING + UniProt
     ↓  enriched candidates
[Agent 4] Literature RAG         — PubMed + BM25/ChromaDB hybrid
     ↓  passages + PMIDs
[Agent 5] Pathway Validator ⭐    — KEGG + Reactome + BBB check
     ↓  PathwayAlignment
[Agent 6] Adversarial Critic ⭐   — ClinicalTrials + failed trials + DDIs
     ↓  AdversarialReport
[Agent 7] Scorer & Ranker        — 5-component score + ±CI
     ↓  ScoredCandidate
[Agent 8] Report Generator       — GPT-4o-mini narrative + Markdown export
     ↓
RepurposingReport (JSON + Markdown)
```

## Scoring Formula

```
Composite Score = (
    KG_proximity         × 0.20   # Hetionet + Open Targets structural evidence
  + MOA_alignment        × 0.25   # Mechanism clarity and target richness
  + literature           × 0.20   # PubMed count, signal polarity, RAG relevance
  + pathway_plausibility × 0.25   # DIRECT/INDIRECT/SPECULATIVE alignment
  + adversarial_penalty  × 0.10   # Failed trials, BBB issues, red flags
)

Confidence Interval = ±σ (fewer evidence sources → wider CI)
```

## Novel Contributions

1. **Pathway Mechanism Validator (Agent 5)** — Maps drug MOA → KEGG/Reactome pathway IDs, checks overlap with disease pathways, classifies alignment as DIRECT / INDIRECT / SPECULATIVE
2. **Adversarial Critique Agent (Agent 6)** — Actively finds disqualifying factors: failed trials (ClinicalTrials.gov), BBB issues, promiscuous binding, wrong target directionality
3. **Hallucination Guard** — Every factual claim must be anchored to a ChEMBL ID, PMID, KEGG ID, Reactome ID, or NCT ID
4. **Confidence Intervals** — Score uncertainty quantified from evidence count

## Setup

### 1. Reuse existing venv

```powershell
# From drug_repurposing_v2 directory
cd C:\Users\gelso\Documents\projects\drug_repurposing_v2

# Activate v1's venv (already has all base deps)
..\drug_repurposing\.venv\Scripts\Activate.ps1

# Install new packages
pip install rank-bm25 openai langchain-openai
```

### 2. Configure environment

```powershell
Copy-Item .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the Streamlit app

```powershell
streamlit run app/streamlit_app.py
```

### 4. Run via CLI

```powershell
python main.py "Alzheimer's disease"
python main.py "Parkinson's disease" --top-n 5
python main.py "Colorectal cancer" --no-llm --output-dir results/
```

### 5. Run tests

```powershell
python -m pytest tests/test_alzheimers.py -v
```

## Data Sources

| Source | What it provides | Access |
|--------|-----------------|--------|
| **Open Targets** | Disease-gene associations, known drugs | Free, no key |
| **Hetionet** | Biomedical KG (29 node types) | Free, local file |
| **ChEMBL** | Drug targets, bioactivity, approval status | Free API |
| **PubMed** | 35M+ abstracts for RAG | Free (faster with key) |
| **KEGG** | Pathway gene membership | Free REST API |
| **Reactome** | Pathway enrichment | Free REST API |
| **STRING DB** | Protein-protein interactions | Free API |
| **ClinicalTrials.gov** | Trial history, failures | Free API v2 |
| **DisGeNET** | Gene-disease curated scores | Free key (register) |

## Demo: Alzheimer's Disease

Positive controls (known repurposing candidates):
- **Metformin** → AD (AMPK-mTOR-tau axis; INDIRECT alignment; BBB concern flagged)
- **Rapamycin** → AD (mTOR inhibition; DIRECT pathway overlap; BBB-penetrant)
- **Semaglutide** → AD (GLP-1R; NCT04777396 Phase III ongoing)

## Limitations

- All predictions are computational hypotheses — wet-lab validation required
- KEGG/Reactome pathway mapping may be incomplete for novel interactions
- BBB penetration data based on published pharmacological profiles
- LLM narrative may contain inaccuracies despite source grounding
- Scoring weights are heuristic, not calibrated against a gold-standard benchmark

## References

- DrugAgent (arXiv:2408.13378) · DrugMCTS (arXiv:2507.07426)
- Prompt-to-Pill (bioRxiv:2025.08.12)
- KG-Bench (bioRxiv:2025.10.13)
- LLM Hypothesis Validation (bioRxiv:2025.06.13)
- DRKG (github.com/gnn4dr/DRKG)
