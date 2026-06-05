# Drug Repurposing with Agentic AI

A multi-agent AI system that autonomously discovers drug repurposing candidates by reasoning across biomedical knowledge graphs, scientific literature, and clinical databases.

## Architecture

```
User Query → Orchestrator → [Molecular Agent | Literature Agent | Clinical Agent] → Scoring Agent → Brief Agent → Report
```

**5 Specialized Agents:**
1. **Molecular Mechanism Agent** — ChEMBL, UniProt, Hetionet KG traversal  
2. **Literature Intelligence Agent** — PubMed RAG + Europe PMC full-text  
3. **Clinical Signal Agent** — ClinicalTrials.gov + OpenFDA FAERS  
4. **Scoring & Ranking Agent** — Composite multi-factor scoring  
5. **Brief Generation Agent** — Citation-backed markdown reports  

**Stack:** CrewAI · Groq (Llama 3.1 70B) · ChromaDB · NetworkX · Streamlit

## Setup

### 1. Install uv (if not already installed)
```bash
pip install uv
```

### 2. Install dependencies
```bash
uv sync
```

### 3. Configure API keys
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a free Groq API key at: https://console.groq.com

### 4. Run the Streamlit demo
```bash
uv run streamlit run app/streamlit_app.py
```

### 5. Run CLI (headless)
```bash
uv run python -m src.main --drug "Metformin" --disease "Alzheimer's disease"
```

## Demo Mode (Offline / No API Keys)
```bash
uv run streamlit run app/streamlit_app.py -- --demo
```
Uses pre-cached results for 3 demo cases. No API calls needed.

## Running Tests
```bash
uv run pytest tests/ -v
```

## Project Structure
```
src/
├── agents/       # Agent definitions
├── tools/        # API client wrappers (framework-agnostic)
├── data/         # KG loader + static data
├── crews/        # CrewAI orchestration
├── models/       # Pydantic data models
├── scoring/      # Composite scoring engine
└── reporting/    # Report templates
app/              # Streamlit UI
tests/            # Unit + integration tests
```

## Public Data Sources
| Source | Data | URL |
|--------|------|-----|
| ChEMBL | Drug targets, bioactivity | https://www.ebi.ac.uk/chembl/ |
| UniProt | Protein function, disease | https://www.uniprot.org/ |
| Hetionet | Biomedical knowledge graph | https://het.io/ |
| PubMed | 35M+ scientific abstracts | https://pubmed.ncbi.nlm.nih.gov/ |
| ClinicalTrials.gov | Trial evidence | https://clinicaltrials.gov/ |
| OpenFDA | Adverse events, drug labels | https://open.fda.gov/ |

## Research Background
- DrugAgent (ICLR 2025) — Multi-agent drug-target interaction framework
- TxGNN (Nature Medicine 2024) — Zero-shot GNN for drug repurposing
- Keramida et al. (MDPI Medicines 2025) — AI-driven repurposing review
