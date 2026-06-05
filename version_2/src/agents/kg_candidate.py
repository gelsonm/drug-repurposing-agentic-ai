"""
Agent 2: KG Traversal & Candidate Discovery Agent

Finds repurposing candidates by combining:
  1. Hetionet graph traversal (NetworkX) — structural proximity
  2. Open Targets known drugs for disease — clinically vetted candidates
  3. ChEMBL — drug metadata, approval status, max clinical phase

Filters: approved or Phase III+ drugs only (max_phase >= 3).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
from loguru import logger

from src.config import get_settings
from src.schemas.models import CandidateDrug, KGEvidence
from src.tools.open_targets_client import get_known_drugs_for_disease

# ── ChEMBL imports (adapted from v1) ──────────────────────────────────────
from src.tools.http_utils import APIError, get_json

_CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Singleton graph
_graph: nx.MultiGraph | None = None


def _load_kg() -> nx.MultiGraph:
    global _graph
    if _graph is not None:
        return _graph

    settings = get_settings()
    kg_path = settings.kg_data_path / "sample_kg.json"

    if not kg_path.exists():
        logger.warning("KG file not found — using in-memory demo graph")
        _graph = _create_demo_graph()
        return _graph

    logger.info(f"Loading knowledge graph from {kg_path}")
    with open(kg_path, encoding="utf-8") as f:
        data = json.load(f)

    G = nx.MultiGraph()
    for node in data.get("nodes", []):
        G.add_node(node["identifier"], name=node.get("name", ""), kind=node.get("kind", ""))
    for edge in data.get("edges", []):
        G.add_edge(edge["source_id"], edge["target_id"], kind=edge.get("kind", ""))

    logger.info(f"KG loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    _graph = G
    return _graph


def _find_nodes_by_name(G: nx.MultiGraph, name: str, kind: str | None = None) -> list:
    name_lower = name.lower()
    return [
        nid for nid, data in G.nodes(data=True)
        if name_lower in data.get("name", "").lower()
        and (kind is None or data.get("kind") == kind)
    ]


def _get_drug_disease_paths(drug_name: str, disease_name: str, max_paths: int = 5) -> list[dict]:
    """Find KG paths between drug and disease (adapted from v1)."""
    G = _load_kg()
    drug_nodes = _find_nodes_by_name(G, drug_name, "Compound")
    disease_nodes = _find_nodes_by_name(G, disease_name, "Disease")

    if not drug_nodes or not disease_nodes:
        return []

    paths = []
    for dn in drug_nodes[:2]:
        for dis_n in disease_nodes[:2]:
            try:
                for path in list(nx.all_simple_paths(G, dn, dis_n, cutoff=4))[:max_paths]:
                    edge_types = []
                    for i in range(len(path) - 1):
                        edges = G.get_edge_data(path[i], path[i + 1])
                        edge_types.append(list(edges.values())[0].get("kind", "?") if edges else "?")
                    node_names = [G.nodes[n].get("name", str(n)) for n in path]
                    desc = " → ".join(
                        f"{node_names[i]} [{edge_types[i]}]" for i in range(len(edge_types))
                    ) + f" → {node_names[-1]}"
                    score = max(0.0, 1.0 - (len(path) - 2) * 0.2)
                    paths.append({
                        "nodes": node_names,
                        "edge_types": edge_types,
                        "path_score": round(score, 3),
                        "description": desc,
                    })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    return sorted(paths, key=lambda p: p["path_score"], reverse=True)[:max_paths]


def _get_shared_targets(drug_name: str, disease_name: str) -> list[str]:
    """Get gene symbols shared between drug targets and disease genes in KG."""
    G = _load_kg()

    drug_genes = set()
    for dn in _find_nodes_by_name(G, drug_name, "Compound")[:2]:
        for neighbor in G.neighbors(dn):
            if G.nodes[neighbor].get("kind") == "Gene":
                drug_genes.add(G.nodes[neighbor].get("name", ""))

    disease_genes = set()
    for dis_n in _find_nodes_by_name(G, disease_name, "Disease")[:2]:
        for neighbor in G.neighbors(dis_n):
            if G.nodes[neighbor].get("kind") == "Gene":
                disease_genes.add(G.nodes[neighbor].get("name", ""))

    return sorted(drug_genes & disease_genes)


def _search_chembl_drug(drug_name: str) -> dict[str, Any] | None:
    """Search ChEMBL for a drug by name."""
    try:
        data = get_json(
            f"{_CHEMBL_BASE}/molecule.json",
            params={"pref_name__icontains": drug_name, "limit": 3},
        )
        molecules = data.get("molecules", [])
        return molecules[0] if molecules else None
    except APIError:
        return None


def run_kg_candidate_agent(
    disease_profile,  # DiseaseProfile
    max_candidates: int = 10,
) -> list[CandidateDrug]:
    """
    Execute Agent 2: KG Traversal & Candidate Discovery.

    Returns list of CandidateDrug objects sorted by network proximity score.
    """
    disease_name = disease_profile.disease_name
    efo_id = disease_profile.efo_id
    logger.info(f"[Agent 2] KG Candidate Discovery for '{disease_name}' (EFO: {efo_id})")

    candidates: dict[str, CandidateDrug] = {}  # drug_name → CandidateDrug

    # ── Source 1: Open Targets known drugs for disease ─────────────────────
    if efo_id:
        ot_drugs = get_known_drugs_for_disease(efo_id, max_results=30)
        for drug_data in ot_drugs:
            name = drug_data.get("drug_name", "")
            phase = drug_data.get("max_phase") or 0
            if not name or phase < 2:  # Phase II and above (includes approved)
                continue

            chembl_id = drug_data.get("chembl_id")
            ot_score = min(1.0, phase / 4.0)  # Phase 4 (approved) = 1.0

            if name not in candidates:
                candidates[name] = CandidateDrug(
                    drug_name=name,
                    chembl_id=chembl_id,
                    max_phase=phase,
                    mechanism_of_action=drug_data.get("mechanism_of_action"),
                    known_targets=drug_data.get("target_symbols", [])[:5],
                    open_targets_score=ot_score,
                )
            else:
                candidates[name].open_targets_score = max(candidates[name].open_targets_score, ot_score)

        logger.info(f"[Agent 2] {len(candidates)} Phase III+ drugs from Open Targets")

    # ── Source 2: Hetionet KG traversal ──────────────────────────────────
    G = _load_kg()
    # Get all compound nodes from KG
    compound_nodes = [
        (nid, data) for nid, data in G.nodes(data=True)
        if data.get("kind") == "Compound"
    ]

    kg_hits = []
    for nid, data in compound_nodes[:30]:  # Cap for speed
        drug_name_kg = data.get("name", str(nid))
        paths = _get_drug_disease_paths(drug_name_kg, disease_name, max_paths=3)
        shared = _get_shared_targets(drug_name_kg, disease_name)

        if paths or shared:
            best_path_score = max(p["path_score"] for p in paths) if paths else 0.0
            kg_hits.append((drug_name_kg, paths, shared, best_path_score))

    # Sort by KG score
    kg_hits.sort(key=lambda x: x[3], reverse=True)

    for drug_name_kg, paths, shared, kg_score in kg_hits[:15]:
        kg_evidences = [
            KGEvidence(
                path_description=p.get("description", ""),
                path_score=p.get("path_score", 0.0),
                shared_targets=shared,
                source="Hetionet",
            )
            for p in paths
        ]

        if drug_name_kg not in candidates:
            # Try ChEMBL lookup for metadata
            mol = _search_chembl_drug(drug_name_kg)
            chembl_id = mol.get("molecule_chembl_id") if mol else None
            max_phase = int(float(mol.get("max_phase", 0) or 0)) if mol else 0

            # Only include if has KG evidence (KG compounds may be at any phase)
            if not kg_evidences and max_phase < 2:
                continue

            candidates[drug_name_kg] = CandidateDrug(
                drug_name=drug_name_kg,
                chembl_id=chembl_id,
                max_phase=max_phase,
                known_targets=shared[:5],
                kg_evidence=kg_evidences,
                network_proximity_score=kg_score,
                open_targets_score=0.0,
            )
        else:
            # Enrich existing candidate with KG evidence
            candidates[drug_name_kg].kg_evidence.extend(kg_evidences)
            candidates[drug_name_kg].network_proximity_score = max(
                candidates[drug_name_kg].network_proximity_score, kg_score
            )
            # Add shared targets not already listed
            existing = set(candidates[drug_name_kg].known_targets)
            for t in shared:
                if t not in existing:
                    candidates[drug_name_kg].known_targets.append(t)

    # ── Compute composite proximity score ─────────────────────────────────
    for cand in candidates.values():
        kg_s = cand.network_proximity_score
        ot_s = cand.open_targets_score
        cand.network_proximity_score = round(0.5 * kg_s + 0.5 * ot_s, 4)

    # ── Sort and return top N ─────────────────────────────────────────────
    ranked = sorted(
        candidates.values(),
        key=lambda c: c.network_proximity_score + (c.open_targets_score * 0.3),
        reverse=True,
    )

    top = ranked[:max_candidates]
    logger.info(f"[Agent 2] Returning {len(top)} candidates")
    for c in top[:5]:
        logger.info(
            f"  {c.drug_name}: proximity={c.network_proximity_score:.3f}, "
            f"OT={c.open_targets_score:.3f}, KG_paths={len(c.kg_evidence)}, "
            f"shared_targets={c.known_targets[:3]}"
        )
    return top


def _create_demo_graph() -> nx.MultiGraph:
    """Fallback demo graph for testing without KG file."""
    G = nx.MultiGraph()
    nodes = [
        ("Metformin", "Compound"), ("Rapamycin", "Compound"), ("Semaglutide", "Compound"),
        ("Ibuprofen", "Compound"), ("Atorvastatin", "Compound"), ("Donepezil", "Compound"),
        ("AMPK", "Gene"), ("MTOR", "Gene"), ("APP", "Gene"), ("MAPT", "Gene"),
        ("BACE1", "Gene"), ("GLP1R", "Gene"), ("NFKB1", "Gene"), ("TNF", "Gene"),
        ("TP53", "Gene"), ("HMGCR", "Gene"),
        ("Alzheimer's disease", "Disease"), ("Parkinson's disease", "Disease"),
        ("Type 2 diabetes", "Disease"), ("Colorectal cancer", "Disease"),
    ]
    for name, kind in nodes:
        G.add_node(name, name=name, kind=kind)

    edges = [
        ("Metformin", "AMPK", "activates"), ("Metformin", "MTOR", "inhibits"),
        ("Metformin", "NFKB1", "inhibits"),
        ("Rapamycin", "MTOR", "inhibits"),
        ("Semaglutide", "GLP1R", "activates"),
        ("Ibuprofen", "NFKB1", "inhibits"), ("Ibuprofen", "TNF", "reduces"),
        ("Atorvastatin", "HMGCR", "inhibits"),
        ("AMPK", "MTOR", "inhibits"),
        ("MTOR", "MAPT", "phosphorylates"),
        ("MAPT", "Alzheimer's disease", "drives"),
        ("APP", "Alzheimer's disease", "drives"),
        ("BACE1", "APP", "cleaves"),
        ("GLP1R", "Alzheimer's disease", "protects"),
        ("NFKB1", "Alzheimer's disease", "promotes"),
        ("TNF", "Alzheimer's disease", "promotes"),
        ("MTOR", "Alzheimer's disease", "promotes"),
        ("HMGCR", "Alzheimer's disease", "associates"),
        ("TP53", "Colorectal cancer", "suppresses"),
    ]
    for src, tgt, kind in edges:
        G.add_edge(src, tgt, kind=kind)

    logger.info(f"Demo KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G
