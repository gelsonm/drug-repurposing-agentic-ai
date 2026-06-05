"""Quick smoke test of the KG and scoring pipeline."""
import sys
sys.path.insert(0, ".")

print("=== Testing KG Client ===")
from src.tools.kg_client import _create_demo_graph, find_drug_disease_paths, compute_gene_overlap_score

G = _create_demo_graph()
print(f"KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

paths = find_drug_disease_paths("Metformin", "Alzheimer's disease")
print(f"Paths found (Metformin -> Alzheimer's): {len(paths)}")
for p in paths:
    desc = p['description'].replace('\u2192', '->').encode('ascii', 'replace').decode()
    print(f"  {desc} (score={p['path_score']})")


overlap = compute_gene_overlap_score("Metformin", "Alzheimer's disease")
print(f"Gene overlap score: {overlap}")

print("\n=== Testing Scoring Engine ===")
from src.scoring.composite_scorer import compute_composite_score

landscape = {"clinical_signal": "emerging_evidence", "total": 5, "active": 2, "completed": 2, "terminated": 1}
score = compute_composite_score(
    kg_paths=paths,
    gene_overlap=overlap,
    n_targets=5,
    rag_passages=[{"relevance_score": 0.7, "year": "2023", "text": "Metformin activates AMPK..."}],
    pubmed_count=20,
    trial_landscape=landscape,
    safety_flag="clean",
    has_chembl=True,
    has_uniprot=True,
    has_label=True,
)
print(f"Composite Score: {score.composite_score:.3f}")
print(f"  Mechanistic: {score.mechanistic_score:.3f}")
print(f"  Literature:  {score.literature_score:.3f}")
print(f"  Clinical:    {score.clinical_feasibility_score:.3f}")
print(f"  Confidence:  {score.data_confidence_score:.3f}")
print(f"  Safety: {score.safety_flag.value}, Feasibility: {score.clinical_feasibility.value}")

print("\n=== Testing PubMed Client (live) ===")
from src.tools.pubmed_client import search_pubmed
pmids = search_pubmed("Metformin Alzheimer disease", max_results=3)
print(f"PubMed results: {len(pmids)} PMIDs: {pmids}")

print("\n=== All smoke tests PASSED ===")
