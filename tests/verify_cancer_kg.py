"""Verify the cancer-focused KG loads correctly."""
import sys
sys.path.insert(0, ".")

import src.tools.kg_client as kg
kg._graph = None  # Reset singleton

G = kg.load_knowledge_graph()
print(f"KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

cancers = kg.get_cancer_types()
print(f"\nCancer types ({len(cancers)}):")
for c in cancers:
    print(f"  - {c}")

drugs = kg.get_repurposable_drugs()
print(f"\nRepurposable drugs ({len(drugs)}): {drugs}")

print("\n--- Shared Gene Targets (Smoking Guns) ---")
pairs = [
    ("Metformin", "Colorectal cancer"),
    ("Aspirin", "Colorectal cancer"),
    ("Atorvastatin", "Pancreatic cancer"),
    ("Itraconazole", "Non-small cell lung cancer"),
    ("Metformin", "Breast cancer"),
    ("Rapamycin", "Glioblastoma"),
]
for drug, cancer in pairs:
    shared = kg.get_shared_targets(drug, cancer)
    genes = [f"{t['gene']}({t['drug_relationship']})" for t in shared]
    paths = kg.find_drug_disease_paths(drug, cancer, max_paths=2)
    overlap = kg.compute_gene_overlap_score(drug, cancer)
    print(f"\n{drug} -> {cancer}:")
    print(f"  Shared targets: {genes}")
    print(f"  KG paths: {len(paths)}, Gene overlap: {overlap}")
    if paths:
        desc = paths[0]['description'].encode('ascii', 'replace').decode()
        print(f"  Best path: {desc}")
