"""Extend the sample KG to include Alzheimer's, Parkinson's, and T2D nodes + edges"""
import json
from pathlib import Path

kg_path = Path("data/kg/sample_kg.json")
with open(kg_path, encoding="utf-8") as f:
    kg = json.load(f)

# New nodes to add
new_nodes = [
    # Compounds (all have published repurposing evidence for AD/PD/T2D)
    {"identifier": "Semaglutide",      "name": "Semaglutide",      "kind": "Compound"},
    {"identifier": "Liraglutide",      "name": "Liraglutide",      "kind": "Compound"},
    {"identifier": "Berberine",        "name": "Berberine",        "kind": "Compound"},
    {"identifier": "Resveratrol",      "name": "Resveratrol",      "kind": "Compound"},
    {"identifier": "Nilotinib",        "name": "Nilotinib",        "kind": "Compound"},
    {"identifier": "Exenatide",        "name": "Exenatide",        "kind": "Compound"},
    {"identifier": "Lithium",          "name": "Lithium",          "kind": "Compound"},
    {"identifier": "Valproic_acid",    "name": "Valproic acid",    "kind": "Compound"},

    # Genes relevant to AD/PD/T2D
    {"identifier": "MAPT",   "name": "MAPT",   "kind": "Gene"},   # tau
    {"identifier": "APP",    "name": "APP",     "kind": "Gene"},   # amyloid precursor
    {"identifier": "BACE1",  "name": "BACE1",   "kind": "Gene"},   # beta secretase
    {"identifier": "APOE",   "name": "APOE",    "kind": "Gene"},   # AD risk gene
    {"identifier": "SNCA",   "name": "SNCA",    "kind": "Gene"},   # alpha-synuclein
    {"identifier": "LRRK2",  "name": "LRRK2",   "kind": "Gene"},   # PD kinase
    {"identifier": "GLP1R",  "name": "GLP1R",   "kind": "Gene"},   # GLP-1 receptor
    {"identifier": "INSR",   "name": "INSR",    "kind": "Gene"},   # insulin receptor
    {"identifier": "GSK3B",  "name": "GSK3B",   "kind": "Gene"},   # tau kinase
    {"identifier": "PRKN",   "name": "PRKN",    "kind": "Gene"},   # parkin
    {"identifier": "PINK1",  "name": "PINK1",   "kind": "Gene"},   # mitophagy
    {"identifier": "SIRT1",  "name": "SIRT1",   "kind": "Gene"},   # longevity deacetylase
    {"identifier": "NLRP3",  "name": "NLRP3",   "kind": "Gene"},   # inflammasome
    {"identifier": "NRF2",   "name": "NRF2",    "kind": "Gene"},   # antioxidant TF

    # Disease nodes
    {"identifier": "Alzheimers_disease",  "name": "Alzheimer's disease", "kind": "Disease"},
    {"identifier": "Parkinsons_disease",  "name": "Parkinson's disease", "kind": "Disease"},
    {"identifier": "Type_2_diabetes",     "name": "Type 2 diabetes",     "kind": "Disease"},
    {"identifier": "ALS",                 "name": "Amyotrophic lateral sclerosis", "kind": "Disease"},
    {"identifier": "Epilepsy",            "name": "Epilepsy",             "kind": "Disease"},

    # Pathways
    {"identifier": "mTOR_signaling",  "name": "mTOR signaling pathway",   "kind": "Pathway"},
    {"identifier": "Neuroinflammation","name": "Neuroinflammation",        "kind": "Pathway"},
    {"identifier": "Mitophagy",       "name": "Mitophagy pathway",         "kind": "Pathway"},
]

# New edges
new_edges = [
    # Metformin -> AD via AMPK-mTOR-tau axis
    {"source_id": "Metformin",   "target_id": "GSK3B",  "kind": "inhibits"},
    {"source_id": "Metformin",   "target_id": "MAPT",   "kind": "reduces_phosphorylation"},
    {"source_id": "Metformin",   "target_id": "NLRP3",  "kind": "inhibits"},
    {"source_id": "Metformin",   "target_id": "NRF2",   "kind": "activates"},
    {"source_id": "Metformin",   "target_id": "INSR",   "kind": "sensitizes"},

    # Rapamycin -> AD via mTOR
    {"source_id": "Rapamycin",   "target_id": "MAPT",   "kind": "reduces_phosphorylation"},
    {"source_id": "Rapamycin",   "target_id": "APP",    "kind": "reduces"},
    {"source_id": "Rapamycin",   "target_id": "NLRP3",  "kind": "inhibits"},

    # Semaglutide -> AD/PD via GLP-1R
    {"source_id": "Semaglutide", "target_id": "GLP1R",  "kind": "activates"},
    {"source_id": "Semaglutide", "target_id": "INSR",   "kind": "sensitizes"},
    {"source_id": "Semaglutide", "target_id": "NLRP3",  "kind": "inhibits"},
    {"source_id": "Semaglutide", "target_id": "NRF2",   "kind": "activates"},

    # Liraglutide -> AD/PD
    {"source_id": "Liraglutide", "target_id": "GLP1R",  "kind": "activates"},
    {"source_id": "Liraglutide", "target_id": "SNCA",   "kind": "reduces"},

    # Exenatide -> PD (clinical trial)
    {"source_id": "Exenatide",   "target_id": "GLP1R",  "kind": "activates"},
    {"source_id": "Exenatide",   "target_id": "SNCA",   "kind": "reduces"},
    {"source_id": "Exenatide",   "target_id": "PINK1",  "kind": "upregulates"},

    # Nilotinib -> PD/AD (autophagy)
    {"source_id": "Nilotinib",   "target_id": "LRRK2",  "kind": "inhibits"},
    {"source_id": "Nilotinib",   "target_id": "SNCA",   "kind": "reduces"},
    {"source_id": "Nilotinib",   "target_id": "PRKN",   "kind": "upregulates"},

    # Atorvastatin -> AD (neuroprotection)
    {"source_id": "Atorvastatin","target_id": "APOE",   "kind": "modulates"},
    {"source_id": "Atorvastatin","target_id": "NLRP3",  "kind": "inhibits"},
    {"source_id": "Atorvastatin","target_id": "NRF2",   "kind": "activates"},

    # Aspirin -> AD (anti-inflammatory)
    {"source_id": "Aspirin",     "target_id": "NLRP3",  "kind": "inhibits"},
    {"source_id": "Aspirin",     "target_id": "NRF2",   "kind": "activates"},

    # Resveratrol -> AD (SIRT1/mTOR)
    {"source_id": "Resveratrol", "target_id": "SIRT1",  "kind": "activates"},
    {"source_id": "Resveratrol", "target_id": "MTOR",   "kind": "inhibits"},
    {"source_id": "Resveratrol", "target_id": "NLRP3",  "kind": "inhibits"},
    {"source_id": "Resveratrol", "target_id": "NRF2",   "kind": "activates"},

    # Berberine -> T2D/AD
    {"source_id": "Berberine",   "target_id": "AMPK",   "kind": "activates"},
    {"source_id": "Berberine",   "target_id": "INSR",   "kind": "sensitizes"},
    {"source_id": "Berberine",   "target_id": "NLRP3",  "kind": "inhibits"},
    {"source_id": "Berberine",   "target_id": "GSK3B",  "kind": "inhibits"},

    # Lithium -> AD (GSK3B/tau)
    {"source_id": "Lithium",     "target_id": "GSK3B",  "kind": "inhibits"},
    {"source_id": "Lithium",     "target_id": "MAPT",   "kind": "reduces_phosphorylation"},

    # Hydroxychloroquine -> AD (autophagy/inflammation)
    {"source_id": "Hydroxychloroquine", "target_id": "NLRP3", "kind": "inhibits"},
    {"source_id": "Hydroxychloroquine", "target_id": "APP",   "kind": "reduces"},

    # Valproic acid -> AD/epilepsy
    {"source_id": "Valproic_acid", "target_id": "GSK3B", "kind": "inhibits"},
    {"source_id": "Valproic_acid", "target_id": "MAPT",  "kind": "reduces_phosphorylation"},

    # Gene-Disease edges (Alzheimer's)
    {"source_id": "MAPT",   "target_id": "Alzheimers_disease", "kind": "drives"},
    {"source_id": "APP",    "target_id": "Alzheimers_disease", "kind": "drives"},
    {"source_id": "BACE1",  "target_id": "Alzheimers_disease", "kind": "drives"},
    {"source_id": "APOE",   "target_id": "Alzheimers_disease", "kind": "risk_factor"},
    {"source_id": "GSK3B",  "target_id": "Alzheimers_disease", "kind": "drives"},
    {"source_id": "NLRP3",  "target_id": "Alzheimers_disease", "kind": "promotes"},
    {"source_id": "MTOR",   "target_id": "Alzheimers_disease", "kind": "promotes"},
    {"source_id": "GLP1R",  "target_id": "Alzheimers_disease", "kind": "protects"},
    {"source_id": "NRF2",   "target_id": "Alzheimers_disease", "kind": "protects"},
    {"source_id": "SIRT1",  "target_id": "Alzheimers_disease", "kind": "protects"},
    {"source_id": "INSR",   "target_id": "Alzheimers_disease", "kind": "associates"},
    {"source_id": "AMPK",   "target_id": "Alzheimers_disease", "kind": "protects"},

    # Gene-Disease edges (Parkinson's)
    {"source_id": "SNCA",   "target_id": "Parkinsons_disease", "kind": "drives"},
    {"source_id": "LRRK2",  "target_id": "Parkinsons_disease", "kind": "drives"},
    {"source_id": "PRKN",   "target_id": "Parkinsons_disease", "kind": "suppresses"},
    {"source_id": "PINK1",  "target_id": "Parkinsons_disease", "kind": "suppresses"},
    {"source_id": "GLP1R",  "target_id": "Parkinsons_disease", "kind": "protects"},
    {"source_id": "NLRP3",  "target_id": "Parkinsons_disease", "kind": "promotes"},

    # Gene-Disease edges (T2D)
    {"source_id": "INSR",   "target_id": "Type_2_diabetes", "kind": "associates"},
    {"source_id": "GLP1R",  "target_id": "Type_2_diabetes", "kind": "treats"},
    {"source_id": "AMPK",   "target_id": "Type_2_diabetes", "kind": "protects"},
    {"source_id": "MTOR",   "target_id": "Type_2_diabetes", "kind": "promotes"},
    {"source_id": "NLRP3",  "target_id": "Type_2_diabetes", "kind": "promotes"},

    # Pathway links
    {"source_id": "MTOR",   "target_id": "mTOR_signaling",    "kind": "member_of"},
    {"source_id": "AMPK",   "target_id": "mTOR_signaling",    "kind": "suppresses"},
    {"source_id": "NLRP3",  "target_id": "Neuroinflammation", "kind": "drives"},
    {"source_id": "PINK1",  "target_id": "Mitophagy",         "kind": "activates"},
    {"source_id": "PRKN",   "target_id": "Mitophagy",         "kind": "activates"},
    {"source_id": "mTOR_signaling",  "target_id": "Alzheimers_disease", "kind": "promotes"},
    {"source_id": "Neuroinflammation","target_id": "Alzheimers_disease", "kind": "drives"},
    {"source_id": "Neuroinflammation","target_id": "Parkinsons_disease", "kind": "drives"},
    {"source_id": "Mitophagy",       "target_id": "Parkinsons_disease", "kind": "protects"},

    # Epilepsy
    {"source_id": "GSK3B",      "target_id": "Epilepsy",  "kind": "associates"},
    {"source_id": "Valproic_acid","target_id": "Epilepsy", "kind": "treats"},
    {"source_id": "Lithium",    "target_id": "Epilepsy",   "kind": "associates"},
]

# Add new nodes (avoid duplicates)
existing_ids = {n["identifier"] for n in kg["nodes"]}
for n in new_nodes:
    if n["identifier"] not in existing_ids:
        kg["nodes"].append(n)
        existing_ids.add(n["identifier"])

# Add new edges (avoid exact duplicates)
existing_edges = {(e["source_id"], e["target_id"], e["kind"]) for e in kg["edges"]}
for e in new_edges:
    key = (e["source_id"], e["target_id"], e["kind"])
    if key not in existing_edges:
        kg["edges"].append(e)
        existing_edges.add(key)

kg["_description"] = "Multi-disease knowledge graph: oncology + neurodegenerative (AD, PD) + metabolic (T2D). Curated from Hetionet v1.0, DrugBank, and published repurposing literature."
kg["_version"] = "2.1-multi-disease"

with open(kg_path, "w", encoding="utf-8") as f:
    json.dump(kg, f, indent=2, ensure_ascii=False)

print(f"KG updated: {len(kg['nodes'])} nodes, {len(kg['edges'])} edges")
diseases = [n["name"] for n in kg["nodes"] if n["kind"] == "Disease"]
compounds = [n["name"] for n in kg["nodes"] if n["kind"] == "Compound"]
print(f"Diseases: {diseases}")
print(f"Compounds: {compounds}")
