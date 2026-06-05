"""
Knowledge Graph client using NetworkX over Hetionet data.
Hetionet: https://het.io/ — MIT License

Node types: Gene, Compound, Disease, Anatomy, Pathway, etc.
Edge types: treats, binds, upregulates, downregulates, associates, etc.

For the demo we load a lightweight subset bundled in data/kg/sample_kg.json.
In production, load the full Hetionet JSON from het.io.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from loguru import logger

# Singleton graph instance
_graph: nx.MultiGraph | None = None


def load_knowledge_graph(kg_path: str | Path | None = None) -> nx.MultiGraph:
    """
    Load Hetionet into a NetworkX MultiGraph.

    Falls back to bundled sample_kg.json if full KG is not found.
    Graph is cached as a module-level singleton.
    """
    global _graph
    if _graph is not None:
        return _graph

    if kg_path is None:
        # Try full Hetionet first, fall back to sample
        kg_path = Path("data/kg/hetionet-v1.0.json")
        if not kg_path.exists():
            kg_path = Path("data/kg/sample_kg.json")

    kg_path = Path(kg_path)
    if not kg_path.exists():
        logger.warning(f"KG file not found at {kg_path}. Creating minimal demo graph.")
        _graph = _create_demo_graph()
        return _graph

    logger.info(f"Loading knowledge graph from {kg_path}")
    with open(kg_path, encoding="utf-8") as f:
        data = json.load(f)

    G = nx.MultiGraph()

    # Load nodes
    for node in data.get("nodes", []):
        G.add_node(
            node["identifier"],
            name=node.get("name", str(node["identifier"])),
            kind=node.get("kind", "Unknown"),
        )

    # Load edges
    for edge in data.get("edges", []):
        G.add_edge(
            edge["source_id"],
            edge["target_id"],
            kind=edge.get("kind", "relates_to"),
        )

    logger.info(f"KG loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    _graph = G
    return _graph


def find_drug_disease_paths(
    drug_name: str,
    disease_name: str,
    max_paths: int = 5,
    max_path_length: int = 4,
) -> list[dict[str, Any]]:
    """
    Find paths in the knowledge graph between a drug and a disease.

    Returns list of paths, each as:
    {
        "nodes": [...],
        "edge_types": [...],
        "path_length": int,
        "path_score": float,
        "description": str,
    }
    """
    G = load_knowledge_graph()

    # Find matching node IDs (case-insensitive)
    drug_nodes = _find_nodes_by_name(G, drug_name, kind_filter="Compound")
    disease_nodes = _find_nodes_by_name(G, disease_name, kind_filter="Disease")

    if not drug_nodes:
        logger.warning(f"KG: no compound nodes found for '{drug_name}'")
        return []
    if not disease_nodes:
        logger.warning(f"KG: no disease nodes found for '{disease_name}'")
        return []

    paths = []
    for drug_node in drug_nodes[:2]:
        for disease_node in disease_nodes[:2]:
            try:
                # Simple shortest paths (NetworkX)
                simple_paths = list(
                    nx.all_simple_paths(
                        G, source=drug_node, target=disease_node, cutoff=max_path_length
                    )
                )
                for path in simple_paths[:max_paths]:
                    edge_types = _get_path_edge_types(G, path)
                    path_names = [G.nodes[n].get("name", str(n)) for n in path]
                    description = _describe_path(path_names, edge_types)
                    # Score: shorter paths are better
                    score = max(0.0, 1.0 - (len(path) - 2) * 0.2)
                    paths.append({
                        "nodes": path_names,
                        "edge_types": edge_types,
                        "path_length": len(path),
                        "path_score": round(score, 3),
                        "description": description,
                    })
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue

    # Sort by score, deduplicate
    paths = sorted(paths, key=lambda p: p["path_score"], reverse=True)
    return paths[:max_paths]


def get_drug_targets_from_kg(drug_name: str) -> list[dict[str, Any]]:
    """Get all gene/protein targets for a drug from the KG."""
    G = load_knowledge_graph()
    drug_nodes = _find_nodes_by_name(G, drug_name, kind_filter="Compound")

    targets = []
    for drug_node in drug_nodes[:2]:
        for neighbor in G.neighbors(drug_node):
            node_data = G.nodes[neighbor]
            if node_data.get("kind") == "Gene":
                edge_data = list(G.get_edge_data(drug_node, neighbor).values())
                edge_type = edge_data[0].get("kind", "") if edge_data else ""
                targets.append({
                    "gene_name": node_data.get("name", str(neighbor)),
                    "node_id": neighbor,
                    "relationship": edge_type,
                })
    return targets


def get_disease_genes(disease_name: str) -> list[dict[str, Any]]:
    """Get genes associated with a disease from the KG."""
    G = load_knowledge_graph()
    disease_nodes = _find_nodes_by_name(G, disease_name, kind_filter="Disease")

    genes = []
    for disease_node in disease_nodes[:2]:
        for neighbor in G.neighbors(disease_node):
            node_data = G.nodes[neighbor]
            if node_data.get("kind") == "Gene":
                edge_data = list(G.get_edge_data(disease_node, neighbor).values())
                edge_type = edge_data[0].get("kind", "") if edge_data else ""
                genes.append({
                    "gene_name": node_data.get("name", str(neighbor)),
                    "node_id": neighbor,
                    "relationship": edge_type,
                })
    return genes


def compute_gene_overlap_score(drug_name: str, disease_name: str) -> float:
    """
    Compute a gene overlap score between drug targets and disease genes.
    Returns a score from 0.0 to 1.0.
    """
    drug_targets = {t["gene_name"] for t in get_drug_targets_from_kg(drug_name)}
    disease_genes = {g["gene_name"] for g in get_disease_genes(disease_name)}

    if not drug_targets or not disease_genes:
        return 0.0

    overlap = drug_targets & disease_genes
    jaccard = len(overlap) / len(drug_targets | disease_genes)
    return round(jaccard, 3)


def get_cancer_types() -> list[str]:
    """Return all cancer disease nodes in the current KG."""
    G = load_knowledge_graph()
    cancer_keywords = [
        "cancer", "carcinoma", "myeloma", "glioblastoma", "leukemia",
        "lymphoma", "sarcoma", "melanoma", "tumor",
    ]
    cancers = []
    for _, data in G.nodes(data=True):
        if data.get("kind") == "Disease":
            name = data.get("name", "").lower()
            if any(kw in name for kw in cancer_keywords):
                cancers.append(data["name"])
    return sorted(cancers)


def get_repurposable_drugs() -> list[str]:
    """Return all drug compound nodes in the current KG."""
    G = load_knowledge_graph()
    return sorted(
        data.get("name", str(nid))
        for nid, data in G.nodes(data=True)
        if data.get("kind") == "Compound"
    )


def get_shared_targets(drug_name: str, disease_name: str) -> list[dict[str, Any]]:
    """
    Return genes that are both targeted by the drug AND associated with the disease.
    These are the mechanistic 'smoking guns' for repurposing.
    """
    drug_targets = {t["gene_name"]: t for t in get_drug_targets_from_kg(drug_name)}
    disease_genes = {g["gene_name"]: g for g in get_disease_genes(disease_name)}
    shared = set(drug_targets.keys()) & set(disease_genes.keys())
    return [
        {
            "gene": g,
            "drug_relationship": drug_targets[g]["relationship"],
            "disease_relationship": disease_genes[g]["relationship"],
        }
        for g in sorted(shared)
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_nodes_by_name(
    G: nx.MultiGraph,
    name: str,
    kind_filter: str | None = None,
) -> list[Any]:
    """Find node IDs whose name matches (case-insensitive)."""
    name_lower = name.lower()
    matches = []
    for node_id, data in G.nodes(data=True):
        node_name = data.get("name", str(node_id)).lower()
        if name_lower in node_name or node_name in name_lower:
            if kind_filter is None or data.get("kind") == kind_filter:
                matches.append(node_id)
    return matches


def _get_path_edge_types(G: nx.MultiGraph, path: list) -> list[str]:
    """Get edge type labels for each step in a path."""
    edge_types = []
    for i in range(len(path) - 1):
        edges = G.get_edge_data(path[i], path[i + 1])
        if edges:
            kinds = [v.get("kind", "relates_to") for v in edges.values()]
            edge_types.append(kinds[0])
        else:
            edge_types.append("connects")
    return edge_types


def _describe_path(node_names: list[str], edge_types: list[str]) -> str:
    """Generate a human-readable path description."""
    parts = [node_names[0]]
    for i, edge in enumerate(edge_types):
        parts.append(f"→[{edge}]→")
        if i + 1 < len(node_names):
            parts.append(node_names[i + 1])
    return " ".join(parts)


def _create_demo_graph() -> nx.MultiGraph:
    """
    Fallback in-memory demo graph (cancer-focused) used when no KG file is found.
    Mirrors the cancer subset in data/kg/sample_kg.json.
    """
    # Try loading from the bundled JSON first
    bundled = Path("data/kg/sample_kg.json")
    if bundled.exists():
        with open(bundled, encoding="utf-8") as f:
            data = json.load(f)
        G = nx.MultiGraph()
        for node in data.get("nodes", []):
            G.add_node(node["identifier"], name=node.get("name", node["identifier"]), kind=node.get("kind", "Unknown"))
        for edge in data.get("edges", []):
            G.add_edge(edge["source_id"], edge["target_id"], kind=edge.get("kind", "relates_to"))
        logger.info(f"Demo KG created from bundled JSON: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    # Pure in-memory fallback (cancer subset)
    G = nx.MultiGraph()
    cancer_nodes = [
        ("Metformin", "Compound"), ("Aspirin", "Compound"),
        ("Atorvastatin", "Compound"), ("Itraconazole", "Compound"),
        ("AMPK", "Gene"), ("MTOR", "Gene"), ("TP53", "Gene"),
        ("KRAS", "Gene"), ("COX2", "Gene"), ("HIF1A", "Gene"),
        ("EGFR", "Gene"), ("SMO", "Gene"), ("HMGCR", "Gene"),
        ("NFKB", "Gene"), ("BCL2", "Gene"), ("MYC", "Gene"),
        ("Colorectal cancer", "Disease"), ("Breast cancer", "Disease"),
        ("Pancreatic cancer", "Disease"), ("Non-small cell lung cancer", "Disease"),
        ("Basal cell carcinoma", "Disease"),
    ]
    for name, kind in cancer_nodes:
        G.add_node(name, name=name, kind=kind)
    cancer_edges = [
        ("Metformin", "AMPK", "activates"), ("Metformin", "MTOR", "inhibits"),
        ("Metformin", "HIF1A", "downregulates"), ("Metformin", "NFKB", "inhibits"),
        ("Aspirin", "COX2", "inhibits"), ("Aspirin", "NFKB", "inhibits"),
        ("Aspirin", "TP53", "upregulates"), ("Aspirin", "BCL2", "downregulates"),
        ("Atorvastatin", "HMGCR", "inhibits"), ("Atorvastatin", "KRAS", "inhibits"),
        ("Atorvastatin", "MYC", "downregulates"),
        ("Itraconazole", "SMO", "inhibits"), ("Itraconazole", "MTOR", "inhibits"),
        ("AMPK", "Colorectal cancer", "suppresses"), ("MTOR", "Colorectal cancer", "promotes"),
        ("COX2", "Colorectal cancer", "promotes"), ("TP53", "Colorectal cancer", "suppresses"),
        ("KRAS", "Colorectal cancer", "drives"), ("NFKB", "Colorectal cancer", "promotes"),
        ("MTOR", "Breast cancer", "promotes"), ("HIF1A", "Breast cancer", "promotes"),
        ("MYC", "Breast cancer", "drives"),
        ("KRAS", "Pancreatic cancer", "drives"), ("MTOR", "Pancreatic cancer", "promotes"),
        ("HIF1A", "Pancreatic cancer", "promotes"), ("NFKB", "Pancreatic cancer", "promotes"),
        ("EGFR", "Non-small cell lung cancer", "drives"), ("KRAS", "Non-small cell lung cancer", "drives"),
        ("MTOR", "Non-small cell lung cancer", "promotes"),
        ("SMO", "Basal cell carcinoma", "drives"), ("TP53", "Basal cell carcinoma", "suppresses"),
    ]
    for src, tgt, kind in cancer_edges:
        G.add_edge(src, tgt, kind=kind)
    logger.info(f"Demo KG created (in-memory): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G
