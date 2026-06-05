# Drug Repurposing with Agentic AI — End-to-End Technical Documentation

> **Project:** IQVIA Hackathon 2026  
> **Domain:** Oncology Drug Repurposing  
> **Stack:** CrewAI · Groq (Llama 3.3 70B) · ChromaDB · NetworkX · Streamlit

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Data Sources & Knowledge Bases](#3-data-sources--knowledge-bases)
4. [Step-by-Step Pipeline Walkthrough](#4-step-by-step-pipeline-walkthrough)
   - 4.1 User Input & Query Resolution
   - 4.2 Agent 1: Molecular Mechanism Analysis
   - 4.3 Agent 2: Literature Intelligence
   - 4.4 Agent 3: Clinical Signal Analysis
   - 4.5 Agent 4: Composite Scoring
   - 4.6 Agent 5: Brief Generation & Reporting
5. [Knowledge Graph (Hetionet) Integration](#5-knowledge-graph-hetionet-integration)
6. [RAG Pipeline (ChromaDB + Sentence Transformers)](#6-rag-pipeline-chromadb--sentence-transformers)
7. [Composite Scoring Engine](#7-composite-scoring-engine)
8. [Streamlit UI](#8-streamlit-ui)
9. [CLI Interface](#9-cli-interface)
10. [Configuration & Environment](#10-configuration--environment)
11. [Data Models (Pydantic)](#11-data-models-pydantic)
12. [Setup & Installation](#12-setup--installation)
13. [Running the System](#13-running-the-system)
14. [Testing](#14-testing)
15. [Limitations](#15-limitations)
16. [Future Work](#16-future-work)

---

## 1. Problem Statement

Traditional drug development costs **$2.6 billion** and takes **10–15 years** from discovery to approval. Drug repurposing — finding new therapeutic uses for already-approved drugs — dramatically reduces risk and time-to-clinic because:

- **Safety profiles are already established** → Phase I trials can be skipped or accelerated
- **Manufacturing processes are known** → formulation costs are lower
- **Bioavailability/pharmacokinetics are characterized** → faster dose-finding

However, discovering repurposing opportunities requires synthesizing evidence across thousands of scientific papers, biomedical databases, clinical trial registries, and adverse event reports — a task that takes months of manual expert review.

**This project automates that synthesis** using a multi-agent AI pipeline that reasons across structured biomedical knowledge graphs, unstructured literature, and clinical databases to produce ranked, evidence-backed repurposing hypotheses in under 5 minutes.

**Focus Domain:** Non-oncology approved drugs repurposed for cancer indications (Metformin, Aspirin, Atorvastatin, Itraconazole, Doxycycline, Hydroxychloroquine, Rapamycin, Thalidomide).

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                               │
│              Streamlit Web App  │  CLI (python -m src.main)          │
└────────────────────────┬────────┴──────────────────────────────────┘
                         │  (drug_name, disease_name, query_raw)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DrugRepurposingCrew                               │
│                  (CrewAI Orchestrator)                               │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Agent 1    │  │   Agent 2    │  │   Agent 3    │              │
│  │  Molecular   │  │ Literature   │  │  Clinical    │              │
│  │  Mechanism   │  │ Intelligence │  │   Signal     │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                  │                        │
│  ┌──────▼───────────────────────────────────▼───────┐              │
│  │              Agent 4: Composite Scorer            │              │
│  └──────────────────────────┬───────────────────────┘              │
│                             │                                         │
│  ┌──────────────────────────▼───────────────────────┐              │
│  │          Agent 5: Brief Generation Agent          │              │
│  └──────────────────────────┬───────────────────────┘              │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                    RepurposingReport
                    (JSON / Markdown)
```

### Key Technology Choices

| Component | Technology | Reason |
|-----------|-----------|--------|
| Agent Framework | **CrewAI** | Role-based multi-agent orchestration |
| LLM Backbone | **Groq (Llama 3.3 70B)** | Ultra-fast inference, free tier available |
| Vector Database | **ChromaDB** (local) | Persistent RAG with cosine similarity |
| Embeddings | **all-MiniLM-L6-v2** | Compact, accurate sentence embeddings |
| Knowledge Graph | **NetworkX + Hetionet** | Biomedical heterogeneous network |
| Data Validation | **Pydantic v2** | Strict typed models for all entities |
| Frontend | **Streamlit** | Rapid interactive ML UI |
| Visualization | **Plotly** | Radar charts & interactive graphs |

---

## 3. Data Sources & Knowledge Bases

All data sources are **open, publicly accessible APIs** — no proprietary data is used.

| Source | What We Use | Endpoint / URL |
|--------|------------|----------------|
| **ChEMBL** | Drug targets, mechanisms of action, ChEMBL IDs, UniProt links | `ebi.ac.uk/chembl/api/data` |
| **UniProt** | Protein function, disease associations, GO terms | `rest.uniprot.org` |
| **Hetionet** | Biomedical knowledge graph (drug→gene→disease paths) | `het.io` (bundled JSON) |
| **PubMed (NCBI E-utilities)** | 35M+ scientific abstracts | `eutils.ncbi.nlm.nih.gov` |
| **ClinicalTrials.gov** | Trial counts, phase, status, termination rates | `clinicaltrials.gov/api/v2` |
| **OpenFDA (FAERS)** | Drug labels, adverse events, contraindications | `api.fda.gov` |

### Hetionet Knowledge Graph (Sample)
The bundled `data/kg/sample_kg.json` contains a cancer-focused subset:
- **Compounds:** Metformin, Aspirin, Atorvastatin, Itraconazole, Doxycycline, Hydroxychloroquine, Rapamycin, Thalidomide
- **Cancer Diseases:** Colorectal cancer, Breast cancer, Pancreatic cancer, NSCLC, Glioblastoma, Multiple myeloma, Ovarian cancer, Prostate cancer, Hepatocellular carcinoma, Basal cell carcinoma
- **Key Genes:** AMPK, MTOR, TP53, KRAS, COX2, HIF1A, EGFR, SMO, HMGCR, NFKB, BCL2, MYC
- **Edge Types:** activates, inhibits, upregulates, downregulates, promotes, suppresses, drives, associates, treats

---

## 4. Step-by-Step Pipeline Walkthrough

### 4.1 User Input & Query Resolution

**Entry points:**
- **Web UI:** User selects drug (dropdown from KG) + cancer type (dropdown from KG) → clicks "Run Analysis"
- **CLI:** `python -m src.main --drug "Metformin" --disease "Colorectal cancer"`
- **Demo mode:** Pre-fills with Metformin → Alzheimer's disease; offline cached results

**Target Pair Resolution** (`_resolve_target_pairs`):

| Input | Resolution Strategy |
|-------|-------------------|
| Drug only | KG traversal to find connected Disease nodes → top 3 candidate diseases |
| Disease only | KG compounds list → top 3 approved drugs in KG |
| Both specified | Single pair analysis |
| Neither | Default: Metformin → Colorectal cancer |

---

### 4.2 Agent 1: Molecular Mechanism Analysis

**Role:** "Molecular Mechanism Analyst"  
**Tools invoked:** ChEMBL API, UniProt API, Hetionet KG

**Steps:**

1. **ChEMBL Search** — `search_drug_by_name(drug_name, max_results=3)`
   - Returns molecule metadata: `molecule_chembl_id`, `pref_name`, `max_phase`
   - Sets `has_chembl = True`, stores `chembl_id`

2. **Target Retrieval** — `get_drug_targets(chembl_id, max_results=20)`
   - Queries `/mechanism.json` to get mechanism-of-action data
   - Fetches full target records: `target_chembl_id`, `pref_name`, `target_type`, `organism`
   - Stores `n_targets` and `target_names` (first 5)

3. **UniProt Protein Lookup** — `extract_uniprot_ids(targets)` → `get_proteins_for_targets(uniprot_ids[:5])`
   - Maps ChEMBL target components to UniProt accession IDs
   - Fetches protein function/disease annotations
   - Sets `has_uniprot = True` if proteins found

4. **KG Path Finding** — `find_drug_disease_paths(drug_name, disease_name, max_paths=5)`
   - Loads Hetionet graph into NetworkX `MultiGraph`
   - `_find_nodes_by_name()` — case-insensitive fuzzy match for drug and disease nodes
   - `nx.all_simple_paths(G, source=drug_node, target=disease_node, cutoff=4)`
   - Each path gets a score: `max(0.0, 1.0 - (path_length - 2) * 0.2)` (shorter = better)
   - Returns path as `{nodes, edge_types, path_length, path_score, description}`

5. **Gene Overlap Score** — `compute_gene_overlap_score(drug_name, disease_name)`
   - Gets drug target genes from KG: `get_drug_targets_from_kg()`
   - Gets disease-associated genes: `get_disease_genes()`
   - Computes **Jaccard similarity**: `|overlap| / |union|`

6. **LLM Rationale Generation**
   - Prompts Groq LLM with known targets + KG path descriptions
   - Generates 2–3 sentence mechanistic explanation
   - Example output: *"Metformin activates AMPK which suppresses colorectal cancer progression by inhibiting mTOR signaling and downregulating HIF1A-driven angiogenesis."*

**Output dict:**
```python
{
  "kg_paths": [...],      # list of path dicts
  "gene_overlap": 0.25,   # Jaccard score
  "n_targets": 4,
  "has_chembl": True,
  "has_uniprot": True,
  "chembl_id": "CHEMBL1431",
  "target_names": ["AMPK", "MTOR"],
  "mechanism": "AMPK activator",
  "rationale": "..."      # LLM-generated text
}
```

---

### 4.3 Agent 2: Literature Intelligence

**Role:** "Literature Intelligence Specialist"  
**Tools invoked:** PubMed E-utilities, ChromaDB RAG

**Steps:**

1. **Query Construction**
   - Cancer ontology detection: checks if disease name contains `["cancer", "carcinoma", "myeloma", "tumor", "glioblastoma", "leukemia", "lymphoma"]`
   - If cancer: uses `build_cancer_repurposing_query()` — adds oncology-specific MeSH terms (`antineoplastic`, `anticancer`, `apoptosis`, `tumor suppression`)
   - Otherwise: uses `build_repurposing_query()` — adds general repurposing filters

2. **PubMed Search** — `search_and_fetch(query, max_results=15, min_year=2015)`
   - `search_pubmed()` → NCBI ESearch → returns up to 15 PMIDs
   - `fetch_abstracts(pmids)` → NCBI EFetch (XML) → parses title, abstract, authors, journal, year, DOI, URL
   - Respects rate limits: 3 req/s without API key, 10 req/s with NCBI_API_KEY

3. **RAG Ingestion** — `ingest_articles(articles)` → ChromaDB
   - Text chunking: `_chunk_text(text, chunk_size=512, overlap=64)` — word-level sliding window
   - Document ID: MD5 hash of chunk content (deterministic, deduplication-safe)
   - Upserts chunks with metadata: `{pmid, title, year, url, journal, source}`
   - Embedding model: `all-MiniLM-L6-v2` (local, no API key required)
   - Storage: persistent ChromaDB at `./data/chroma/`

4. **RAG Query** — `query_literature(rag_query, top_k=8)`
   - Query: `"{drug_name} treatment for {disease_name} mechanism evidence"`
   - Returns cosine-similarity-ranked passages with scores
   - Score conversion: `1.0 - cosine_distance` → relevance_score ∈ [0, 1]

5. **LLM Summary Generation**
   - Prompts LLM with top 3 passage excerpts (300 chars each)
   - Generates 2–3 sentence literature evidence summary

**Output dict:**
```python
{
  "passages": [...],       # RAG-retrieved chunks with metadata
  "pubmed_count": 12,      # number of PubMed articles found
  "summary": "..."         # LLM-generated literature summary
}
```

---

### 4.4 Agent 3: Clinical Signal Analysis

**Role:** "Clinical & Safety Intelligence Agent"  
**Tools invoked:** ClinicalTrials.gov API, OpenFDA API

**Steps:**

1. **Clinical Trial Search** — `search_trials(drug_name, disease_name, max_results=20)`
   - Queries ClinicalTrials.gov v2 API
   - Returns trial count by phase and status

2. **Trial Landscape Analysis** — `analyze_trial_landscape(trials)`
   - Counts: `total`, `active`, `completed`, `terminated`, by phase (1/2/3/4)
   - Derives `clinical_signal`:
     - `"strong_evidence"`: ≥3 completed trials
     - `"emerging_evidence"`: ≥1 active + ≥1 completed
     - `"early_stage"`: any active phase 1/2
     - `"mixed_evidence"`: high termination rate (>50%)
     - `"no_trials"`: zero trials found

3. **FDA Drug Label Fetch** — `get_drug_label(drug_name)`
   - Queries OpenFDA drug label endpoint
   - Returns structured label with indications, warnings, contraindications

4. **Safety Summary Extraction** — `extract_safety_summary(label)`
   - Extracts: `has_label`, `indications` text, key warning flags

5. **Safety Flag Assessment** — `assess_safety_flag(label_summary, [], disease_name)`
   - Returns one of: `"clean"`, `"minor_concerns"`, `"major_concerns"`, `"contraindicated"`, `"unknown"`

**Output dict:**
```python
{
  "trial_landscape": {
    "total": 8, "active": 3, "completed": 4, "terminated": 1,
    "clinical_signal": "strong_evidence"
  },
  "safety_flag": "clean",
  "has_label": True,
  "summary": "Found 8 clinical trials (3 active, 4 completed)...",
  "safety_summary": "Safety flag: clean. Indications: Type 2 diabetes..."
}
```

---

### 4.5 Agent 4: Composite Scoring

**Role:** "Repurposing Candidate Scorer"  
**Module:** `src/scoring/composite_scorer.py`

The scoring engine is **LLM-free** — pure deterministic computation for reproducibility and speed.

#### Score Components

**Mechanistic Score** (weight: 35%)
```python
path_component  = 0.7 * best_path_score + 0.3 * (n_paths / 5)
overlap_component = min(1.0, gene_overlap * 3)          # Jaccard scaled up 3x
target_component  = min(1.0, log1p(n_targets) / log1p(20))

mechanistic_score = 0.50 * path_component
                  + 0.30 * overlap_component
                  + 0.20 * target_component
```

**Literature Score** (weight: 30%)
```python
avg_relevance    = mean(top 5 RAG relevance scores)
count_component  = min(1.0, log1p(pubmed_count) / log1p(100))
recent_bonus     = 0.15 if any paper ≥ 2023 else 0.10 if any ≥ 2020 else 0

literature_score = 0.60 * avg_relevance
                 + 0.30 * count_component
                 + 0.10 * recent_bonus
```

**Clinical Feasibility Score** (weight: 20%)
```python
signal_map = {
  "strong_evidence": 0.9,
  "emerging_evidence": 0.65,
  "early_stage": 0.35,
  "mixed_evidence": 0.25,
  "no_trials": 0.1,
}
safety_penalties = {
  "contraindicated": 0.0, "major_concerns": 0.4,
  "minor_concerns": 0.8, "clean": 1.0, "unknown": 0.9
}

clinical_score = signal_score * safety_multiplier
               * (0.7 if termination_rate > 0.5 else 1.0)
```

**Data Confidence Score** (weight: 15%)
```python
# 6 binary checks:
checks = [has_chembl, has_uniprot, has_kg_paths, has_label,
          pubmed_count >= 5, n_rag_passages >= 3]
confidence_score = sum(checks) / 6
```

**Composite Score**
```
composite = 0.35 × mechanistic + 0.30 × literature + 0.20 × clinical + 0.15 × confidence
```

**Rank Labels:**
- 🟢 **Strong:** composite ≥ 0.75
- 🟡 **Moderate:** 0.55 ≤ composite < 0.75
- 🔴 **Weak:** 0.35 ≤ composite < 0.55
- ⚫ **Speculative:** composite < 0.35

---

### 4.6 Agent 5: Brief Generation & Reporting

**Role:** "Scientific Brief Writer"

1. **Executive Summary** — LLM prompts with top 3 candidates ranked by score
   - Highlights mechanistic novelty, clinical evidence strength, fast-track rationale
   - Tailored for pharma executives / R&D decision-makers

2. **Next Steps Generation** — rule-based + LLM enhanced
   - If top candidate has active trials → "Review existing protocols"
   - Always: in vitro binding assay validation, systematic literature review
   - If preclinical validation succeeds → "File FDA IND application"

3. **Report Assembly** — `RepurposingReport` Pydantic model
   - Sorted by composite score (highest first)
   - Contains: all candidates, evidence items, KG paths, summaries, next steps

4. **Export Options:**
   - Markdown report via `generate_markdown_report()`
   - Downloadable via Streamlit UI button
   - Auto-saved to `reports/{drug}_{disease}_report.md` (CLI)

---

## 5. Knowledge Graph (Hetionet) Integration

Hetionet (het.io) is a heterogeneous biomedical network with 47,031 nodes and 2.2M edges in its full form.

### Node Types in Use
| Kind | Examples |
|------|---------|
| Compound | Metformin, Aspirin, Atorvastatin |
| Gene | AMPK, MTOR, TP53, KRAS, COX2 |
| Disease | Colorectal cancer, Breast cancer |

### Key Edge Types
| Edge | Meaning |
|------|---------|
| activates | Drug activates gene/protein |
| inhibits | Drug inhibits gene/protein |
| upregulates / downregulates | Expression regulation |
| promotes / drives | Gene promotes/drives disease |
| suppresses | Gene suppresses disease progression |
| treats | Direct drug-disease relationship |
| associates | Statistical gene-disease association |

### Path Scoring Logic
Paths are scored by **inverse length**:
- Direct (2-hop): Compound → Disease → 1.0
- 3-hop (via gene): Compound → Gene → Disease → 0.8
- 4-hop: Compound → Gene → Gene → Disease → 0.6
- 5-hop: → 0.4

The **"Smoking Guns"** feature (`get_shared_targets()`) identifies genes that are **both** targeted by the drug AND associated with the disease — the strongest mechanistic evidence type.

---

## 6. RAG Pipeline (ChromaDB + Sentence Transformers)

### Architecture
```
PubMed Abstracts
     │
     ▼  chunk_text(chunk_size=512, overlap=64)
Word-Level Chunks (512-word windows, 64-word overlap)
     │
     ▼  SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
Dense Vectors (384 dimensions)
     │
     ▼  ChromaDB PersistentClient (HNSW index, cosine space)
Vector Store at ./data/chroma/
     │
     ▼  collection.query(query_texts=[rag_query], n_results=8)
Ranked Passages (cosine similarity ∈ [0, 1])
```

### Deduplication
- Document IDs are MD5 hashes of chunk text content
- ChromaDB `upsert()` ignores already-existing IDs → safe for repeated runs

### ChromaDB Collection
- **Collection name:** `drug_repurposing_literature`
- **Metadata fields:** `pmid`, `title`, `year`, `url`, `journal`, `source`
- **Persistence:** Local SQLite + HNSW index in `./data/chroma/`

---

## 7. Composite Scoring Engine

See §4.5 for formula details. Additional notes:

- **Configurable weights** via `.env`: `WEIGHT_MECHANISTIC`, `WEIGHT_LITERATURE`, `WEIGHT_CLINICAL`, `WEIGHT_DATA_CONFIDENCE` (must sum to 1.0)
- **Deterministic:** same inputs always produce same score — no LLM randomness in scoring
- **Transparent:** `ScoreBreakdown` stores all 4 component scores + metadata for full auditability

---

## 8. Streamlit UI

**Launch:** `uv run streamlit run app/streamlit_app.py`  
**File:** `app/streamlit_app.py` (554 lines)

### UI Sections

| Section | Description |
|---------|-------------|
| **Sidebar** | Groq API key input, LLM model selection, Demo Mode toggle, data sources legend |
| **Header Stats** | 4 metric cards: 8 drugs, 10 cancer types, 35M+ papers, <5 min runtime |
| **Input Panel** | Drug + Disease dropdowns (populated from live KG), 4 quick-launch demo buttons |
| **Shared Target Preview** | Real-time gene overlap display before running analysis |
| **Agent Progress** | 5-step progress bar with per-agent status messages |
| **Results Table** | Ranked candidates DataFrame with Plotly progress bar score column |
| **Radar Chart** | Plotly polar chart: 4 score dimensions for top 3 candidates |
| **Candidate Cards** | Expandable cards: mechanistic rationale, shared targets, KG paths, literature, citations |
| **Next Steps** | Numbered recommended actions |
| **Download** | Markdown report download button |

### Demo Mode (Offline)
- Toggle "⚡ Demo Mode" in sidebar to disable real API calls
- Pre-cached results for Metformin, Aspirin, Atorvastatin demo cases

---

## 9. CLI Interface

```bash
# Analyze specific drug-disease pair
uv run python -m src.main --drug "Metformin" --disease "Colorectal cancer"

# Drug only (discovers top 3 candidate diseases)
uv run python -m src.main --drug "Aspirin"

# Disease only (discovers drugs from KG)
uv run python -m src.main --disease "Pancreatic cancer"

# Free-text query
uv run python -m src.main --query "non-oncology drugs for glioblastoma"

# Demo mode (Metformin → Alzheimer's)
uv run python -m src.main --demo

# Save report to file
uv run python -m src.main --drug "Aspirin" --disease "Colorectal cancer" --output report.md
```

**Output:** Rich-formatted terminal table + auto-saved Markdown report

---

## 10. Configuration & Environment

**File:** `.env` (copied from `.env.example`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Free at console.groq.com |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Also supports llama-3.1-8b-instant, gemma2-9b-it |
| `GROQ_TEMPERATURE` | `0.1` | Low for deterministic outputs |
| `GROQ_MAX_TOKENS` | `4096` | Max LLM response length |
| `NCBI_API_KEY` | *(optional)* | 10 req/s vs 3 req/s without |
| `OPENFDA_API_KEY` | *(optional)* | Higher rate limits |
| `WEIGHT_MECHANISTIC` | `0.35` | Scoring weight |
| `WEIGHT_LITERATURE` | `0.30` | Scoring weight |
| `WEIGHT_CLINICAL` | `0.20` | Scoring weight |
| `WEIGHT_DATA_CONFIDENCE` | `0.15` | Scoring weight |
| `RAG_CHUNK_SIZE` | `512` | Words per chunk |
| `RAG_TOP_K` | `8` | RAG retrieval count |

---

## 11. Data Models (Pydantic)

All entities are validated Pydantic `BaseModel` instances:

```
Drug                    Disease
├── name                ├── name
├── chembl_id           ├── mondo_id / mesh_id
├── mechanism_of_action ├── gene_associations
└── targets (UniProt)   └── is_rare

EvidenceItem            KGPath
├── evidence_type       ├── nodes
├── source (PubMed/KG)  ├── edge_types
├── snippet             ├── path_score
└── relevance_score     └── description

ScoreBreakdown                      RepurposingCandidate
├── mechanistic_score (0-1)         ├── drug
├── literature_score (0-1)          ├── disease
├── clinical_feasibility_score (0-1)├── score (ScoreBreakdown)
├── data_confidence_score (0-1)     ├── kg_paths
├── composite_score (0-1)           ├── evidence_items
├── safety_flag (enum)              ├── mechanistic_rationale
└── clinical_feasibility (enum)     ├── literature_summary
                                    └── clinical_summary

RepurposingReport
├── query_drug / query_disease
├── candidates (sorted by score)
├── summary (LLM executive summary)
├── next_steps
└── total_runtime_seconds
```

---

## 12. Setup & Installation

### Prerequisites
- Python 3.10+
- `uv` package manager (recommended) or `pip`
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Steps

```bash
# 1. Clone / navigate to project
cd drug_repurposing

# 2. Install uv (if not installed)
pip install uv

# 3. Install all dependencies (from pyproject.toml + uv.lock)
uv sync

# 4. Configure environment
cp .env.example .env
# Edit .env: set GROQ_API_KEY=gsk_...

# 5. Verify data directory (auto-created on first run)
# data/kg/sample_kg.json  ← bundled cancer KG
# data/chroma/            ← created on first RAG use
# data/cache/             ← API response cache
```

### Dependencies (key packages)
- `crewai` — multi-agent orchestration
- `langchain-groq` — Groq LLM integration
- `chromadb` — local vector database
- `sentence-transformers` — local embeddings
- `networkx` — knowledge graph traversal
- `pydantic` / `pydantic-settings` — data models
- `streamlit` — web UI
- `plotly` — radar charts
- `pandas` — data tables
- `httpx` — HTTP client with retries
- `loguru` — structured logging
- `rich` — terminal formatting

---

## 13. Running the System

### Web App (Recommended)
```bash
uv run streamlit run app/streamlit_app.py
# → Open http://localhost:8501
```

### CLI (Headless / Batch)
```bash
uv run python -m src.main --drug "Metformin" --disease "Colorectal cancer"
```

### Demo Mode (No API Key Required)
```bash
# Web
uv run streamlit run app/streamlit_app.py
# → Toggle "Demo Mode" in sidebar

# CLI
uv run python -m src.main --demo
```

---

## 14. Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_kg_client.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing
```

**Test coverage areas:**
- `tests/test_kg_client.py` — KG loading, path finding, gene overlap
- Unit tests for scoring components
- Integration tests for API clients (can be mocked)

---

## 15. Limitations

### 15.1 Data Scope Limitations
- **Partial KG:** The bundled `sample_kg.json` is a manually curated 8-drug × 10-cancer subset of the full Hetionet (47K nodes, 2.2M edges). Many real repurposing opportunities may be missed.
- **No real-time data:** ChEMBL, PubMed, and ClinicalTrials.gov are queried at run-time, but the KG is static — it does not reflect newly published mechanistic findings.
- **Cancer-only focus:** The current knowledge graph and drug list are scoped to oncology. General disease domains require KG expansion.

### 15.2 Scoring Limitations
- **Empirical weights:** The scoring weights (35/30/20/15) were chosen heuristically. No ground-truth validation dataset was used to calibrate them.
- **Jaccard gene overlap:** Jaccard similarity is low for most pairs (even biologically meaningful ones) because gene lists are incomplete. Scaled by ×3, but still underestimates true mechanistic overlap.
- **No causal direction:** The scoring does not distinguish between "drug inhibits a gene that promotes cancer" (beneficial) vs. "drug inhibits a gene that suppresses cancer" (potentially harmful). Both increase the mechanistic score.
- **Safety flag is shallow:** The safety flag is derived from FDA label text search, not from FAERS adverse event signal analysis or structured contraindication ontologies.

### 15.3 LLM Limitations
- **Hallucination risk:** Rationale and summary text generated by Llama 3.3 70B can contain plausible-sounding but factually incorrect mechanistic claims.
- **Context window:** With `max_tokens=4096`, long evidence chains may be truncated.
- **Temperature=0.1:** Near-deterministic outputs reduce creativity but may miss novel cross-domain hypotheses.

### 15.4 Infrastructure Limitations
- **Single-threaded pipeline:** Drug-disease pairs are analyzed sequentially (not in parallel). Analysis time scales linearly with number of pairs.
- **Local ChromaDB:** The vector store is local; not suitable for concurrent multi-user access without a server-mode deployment.
- **Rate limiting:** Without API keys, PubMed is limited to 3 req/s and ChEMBL to ~1 req/s, creating bottlenecks for batch analysis.
- **No caching of API responses:** Every run re-queries all APIs. A Redis or file-based cache would dramatically speed up repeat analyses.

### 15.5 Scientific Validity Limitations
- **No wet-lab validation:** All outputs are computational hypotheses requiring experimental confirmation (in vitro, in vivo, clinical).
- **Publication bias:** PubMed overrepresents positive findings; negative trial results may be underweighted.
- **No patient stratification:** Repurposing candidates are assessed at the population level, not for specific patient subgroups or molecular subtypes (e.g., KRAS-mutant vs. KRAS-wild-type).

---

## 16. Future Work

### 16.1 Near-Term (1–3 months)
- **Full Hetionet integration:** Load the complete 47K-node, 2.2M-edge Hetionet graph instead of the sample subset. This requires ~4GB RAM and proper graph indexing.
- **Parallel agent execution:** Run molecular, literature, and clinical agents concurrently (async) instead of sequentially — estimated 3× speedup.
- **API response caching:** Implement Redis or SQLite caching for ChEMBL, PubMed, and ClinicalTrials.gov responses. Reduce repeat analysis time from minutes to seconds.
- **Score calibration:** Collect labeled benchmark pairs (confirmed repurposing successes like Metformin→CRC) to calibrate scoring weights via logistic regression.

### 16.2 Medium-Term (3–6 months)
- **GNN-based mechanistic scoring:** Replace the heuristic path-scoring with a Graph Neural Network (e.g., trained on Hetionet) to learn drug-disease link prediction scores — similar to TxGNN (Nature Medicine 2024).
- **FAERS adverse event integration:** Parse FDA FAERS (Adverse Event Reporting System) to detect safety signals specific to cancer populations, not just label text.
- **Multi-modal evidence fusion:** Incorporate genomics (TCGA expression data), proteomics (Human Protein Atlas), and structural data (AlphaFold pLDDT) as additional scoring dimensions.
- **Patient subgroup stratification:** Cross-reference with TCGA mutation data to identify which molecular subtypes of a cancer are most likely to respond.
- **Uncertainty quantification:** Report confidence intervals on scores, not just point estimates. Bootstrap sampling over evidence subsets.

### 16.3 Long-Term (6–12 months)
- **Multi-disease expansion:** Extend beyond oncology to neurodegeneration (Alzheimer's, Parkinson's), metabolic diseases, and rare diseases.
- **Feedback loop from clinicians:** Build a human-in-the-loop interface where domain experts can annotate hypotheses as plausible/implausible, feeding back into score recalibration.
- **Automated literature monitoring:** Schedule weekly PubMed sweeps to detect new publications for existing candidates and automatically update scores.
- **Integration with drug design tools:** Connect to generative chemistry tools (e.g., diffusion models) to propose structural analogs of repurposing candidates with improved selectivity.
- **Regulatory pathway assistant:** Integrate FDA 505(b)(2) guidance to automatically draft repurposing regulatory strategy documents.
- **Enterprise deployment:** Productionize with FastAPI backend, PostgreSQL for persistence, Kubernetes for scaling, and role-based access control for pharma teams.

---

## Appendix A: Sample KG Drug-Gene-Disease Paths

| Drug | Gene | Cancer | Relationship |
|------|------|--------|-------------|
| Metformin | AMPK | Colorectal cancer | activates AMPK → AMPK suppresses CRC |
| Metformin | MTOR | Colorectal cancer | inhibits MTOR → MTOR promotes CRC |
| Aspirin | COX2 | Colorectal cancer | inhibits COX2 → COX2 promotes CRC |
| Aspirin | TP53 | Colorectal cancer | upregulates TP53 → TP53 suppresses CRC |
| Atorvastatin | KRAS | Pancreatic cancer | inhibits KRAS → KRAS drives Pancreatic |
| Itraconazole | SMO | Basal cell carcinoma | inhibits SMO → SMO drives BCC |
| Itraconazole | MTOR | NSCLC | inhibits MTOR → MTOR promotes NSCLC |

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **Drug Repurposing** | Finding new therapeutic uses for existing approved drugs |
| **KG** | Knowledge Graph — a network of entities and their relationships |
| **RAG** | Retrieval-Augmented Generation — LLM + vector search for grounded answers |
| **Hetionet** | A heterogeneous biomedical network integrating 29 databases |
| **FAERS** | FDA Adverse Event Reporting System |
| **Jaccard Similarity** | |intersection| / |union| — set overlap metric |
| **pLDDT** | Predicted Local Distance Difference Test — AlphaFold confidence score |
| **IND** | Investigational New Drug Application (FDA) |
| **MeSH** | Medical Subject Headings — NLM's controlled vocabulary |
| **ChEMBL** | EMBL-EBI bioactivity database for drug-like molecules |
| **CrewAI** | Python framework for role-based multi-agent AI collaboration |
| **Groq** | LLM inference provider with ultra-low latency (LPU hardware) |
