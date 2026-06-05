"""
STRING Database API client.
Documentation: https://string-db.org/help/api/

Used by Agent 3 (Molecular Mechanism) to:
- Get protein-protein interaction network context for drug targets
- Find functional enrichment for a set of drug targets
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.tools.http_utils import APIError, get_json

_BASE = "https://string-db.org/api"
_SPECIES = 9606  # Homo sapiens
_RATE_DELAY = 0.5


def get_interaction_partners(
    gene_symbols: list[str],
    min_score: int = 700,
    max_partners: int = 20,
) -> list[dict[str, Any]]:
    """
    Get protein-protein interaction partners for a list of gene symbols.

    Args:
        gene_symbols: List of human gene symbols
        min_score: Minimum STRING interaction score (0-1000). 700 = high confidence.
        max_partners: Max interaction partners to return

    Returns:
        List of {protein_a, protein_b, score, interaction_types}
    """
    if not gene_symbols:
        return []

    logger.info(f"STRING: finding interaction partners for {gene_symbols[:5]}")
    time.sleep(_RATE_DELAY)

    try:
        data = get_json(
            f"{_BASE}/json/interaction_partners",
            params={
                "identifiers": "%0d".join(gene_symbols[:10]),
                "species": _SPECIES,
                "required_score": min_score,
                "limit": max_partners,
                "network_type": "functional",
            },
        )

        interactions = []
        for item in (data if isinstance(data, list) else []):
            interactions.append({
                "protein_a": item.get("preferredName_A", ""),
                "protein_b": item.get("preferredName_B", ""),
                "score": item.get("score", 0),
                "nscore": item.get("nscore", 0),
                "escore": item.get("escore", 0),
                "dscore": item.get("dscore", 0),
            })

        logger.info(f"STRING: {len(interactions)} interactions found")
        return interactions

    except APIError as e:
        logger.warning(f"STRING interaction lookup failed: {e}")
        return []


def get_functional_enrichment(gene_symbols: list[str]) -> list[dict[str, Any]]:
    """
    Get functional enrichment (GO, KEGG, Reactome) for a set of proteins.

    Returns:
        List of {category, term_id, term_name, p_value, genes}
    """
    if not gene_symbols:
        return []

    logger.info(f"STRING: functional enrichment for {len(gene_symbols)} genes")
    time.sleep(_RATE_DELAY)

    try:
        data = get_json(
            f"{_BASE}/json/enrichment",
            params={
                "identifiers": "%0d".join(gene_symbols[:20]),
                "species": _SPECIES,
            },
        )

        enrichment = []
        for item in (data if isinstance(data, list) else [])[:20]:
            p = float(item.get("fdr", 1.0))
            if p > 0.05:
                continue
            enrichment.append({
                "category": item.get("category", ""),
                "term_id": item.get("term", ""),
                "term_name": item.get("description", ""),
                "p_value": round(p, 6),
                "genes": item.get("inputGenes", "").split(",") if item.get("inputGenes") else [],
            })

        logger.info(f"STRING: {len(enrichment)} enriched terms")
        return enrichment

    except APIError as e:
        logger.warning(f"STRING enrichment failed: {e}")
        return []


def get_network_hub_score(gene_symbols: list[str]) -> dict[str, float]:
    """
    Estimate hub connectivity for each gene in the network.
    Higher degree = more connected = more likely to be biologically relevant.

    Returns dict of {gene_symbol: normalized_degree}
    """
    interactions = get_interaction_partners(gene_symbols, min_score=400, max_partners=50)
    degrees: dict[str, int] = {}
    for inter in interactions:
        for key in ["protein_a", "protein_b"]:
            gene = inter.get(key, "")
            if gene:
                degrees[gene] = degrees.get(gene, 0) + 1

    if not degrees:
        return {}

    max_deg = max(degrees.values())
    return {g: round(d / max_deg, 3) for g, d in degrees.items()}
