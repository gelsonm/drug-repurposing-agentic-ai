"""
Reactome Pathway Analysis API client.
Documentation: https://reactome.org/ContentService/

Used by Agent 5 (Pathway Validator) to:
- Map gene lists → Reactome pathway enrichment
- Get pathway hierarchy and description
- Find overlapping pathways between drug targets and disease pathways
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.tools.http_utils import APIError, get_json, post_json

_ANALYSIS_BASE = "https://reactome.org/AnalysisService"
_CONTENT_BASE = "https://reactome.org/ContentService"
_RATE_DELAY = 0.4


def get_pathways_for_genes(
    gene_symbols: list[str],
    species: str = "Homo sapiens",
    p_value: float = 0.05,
    max_results: int = 15,
) -> list[dict[str, Any]]:
    """
    Submit a gene list for pathway enrichment analysis via Reactome.

    Args:
        gene_symbols: List of HGNC gene symbols (e.g. ["MTOR", "AMPK", "APP"])
        species: Species name (default "Homo sapiens")
        p_value: Max p-value threshold
        max_results: Max pathways to return

    Returns:
        List of {pathway_id, pathway_name, p_value, found_entities, entities_count}
    """
    if not gene_symbols:
        return []

    logger.info(f"Reactome: enrichment analysis for {len(gene_symbols)} genes")
    time.sleep(_RATE_DELAY)

    # Reactome Analysis Service: POST newfile/form
    gene_list = "\n".join(gene_symbols)

    try:
        import httpx
        with httpx.Client(timeout=30) as client:
            r = client.post(
                f"{_ANALYSIS_BASE}/identifiers/",
                content=gene_list,
                headers={"Content-Type": "text/plain"},
                params={"species": species, "pageSize": max_results, "page": 1},
            )
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning(f"Reactome analysis submission failed: {e}")
        return []

    pathways_data = data.get("pathways", [])
    results = []
    for pw in pathways_data[:max_results]:
        entities = pw.get("entities", {})
        p = pw.get("entities", {}).get("pValue", 1.0)
        if p > p_value:
            continue
        results.append({
            "pathway_id": pw.get("stId", ""),
            "pathway_name": pw.get("name", ""),
            "p_value": round(p, 6),
            "found_entities": entities.get("found", 0),
            "total_entities": entities.get("total", 0),
            "species": pw.get("species", {}).get("name", ""),
        })

    logger.info(f"Reactome: {len(results)} significant pathways found")
    return results


def get_pathway_details(reactome_id: str) -> dict[str, Any]:
    """
    Get details for a specific Reactome pathway.

    Returns: {id, displayName, summation, species, isInferred}
    """
    logger.info(f"Reactome: fetching pathway details for {reactome_id}")
    time.sleep(_RATE_DELAY)

    try:
        data = get_json(f"{_CONTENT_BASE}/data/query/{reactome_id}")
        return {
            "pathway_id": data.get("stId", reactome_id),
            "name": data.get("displayName", ""),
            "summation": data.get("summation", [{}])[0].get("text", "") if data.get("summation") else "",
            "species": data.get("speciesName", ""),
        }
    except APIError as e:
        logger.warning(f"Reactome pathway details failed for {reactome_id}: {e}")
        return {}


def find_pathway_by_name(query: str) -> list[dict[str, Any]]:
    """
    Search Reactome for pathways matching a keyword.

    Returns list of {pathway_id, name, species}.
    """
    logger.info(f"Reactome: searching pathways for '{query}'")
    time.sleep(_RATE_DELAY)

    try:
        data = get_json(
            f"{_CONTENT_BASE}/search/query",
            params={"q": query, "species": "Homo sapiens", "types": "Pathway", "rows": 10},
        )
        results = data.get("results", [])
        pathways = []
        for group in results:
            for entry in group.get("entries", []):
                pathways.append({
                    "pathway_id": entry.get("stId", ""),
                    "name": entry.get("name", ""),
                    "species": entry.get("species", ""),
                })
        logger.info(f"Reactome: {len(pathways)} pathways found for '{query}'")
        return pathways[:10]
    except APIError as e:
        logger.warning(f"Reactome search failed for '{query}': {e}")
        return []


def get_genes_in_pathway(reactome_id: str) -> list[str]:
    """
    Get all gene/protein participants in a Reactome pathway.

    Returns list of gene symbols (may include UniProt IDs — best effort).
    """
    time.sleep(_RATE_DELAY)
    try:
        data = get_json(f"{_CONTENT_BASE}/data/participants/{reactome_id}/participatingGenes")
        genes = []
        if isinstance(data, list):
            for entry in data:
                symbol = entry.get("displayName") or entry.get("geneName") or ""
                if symbol and len(symbol) <= 20:
                    genes.append(symbol)
        return genes[:50]
    except APIError:
        return []


# ── Curated disease-pathway map for reliable fallback ─────────────────────

DISEASE_REACTOME_MAP: dict[str, list[str]] = {
    "alzheimer": ["R-HSA-112314", "R-HSA-165159", "R-HSA-9612973", "R-HSA-5205685"],
    "alzheimer's disease": ["R-HSA-112314", "R-HSA-165159", "R-HSA-9612973"],
    "parkinson": ["R-HSA-9619665", "R-HSA-9612973", "R-HSA-112315"],
    "parkinson's disease": ["R-HSA-9619665", "R-HSA-9612973"],
    "colorectal cancer": ["R-HSA-5633007", "R-HSA-4791275", "R-HSA-73857"],
    "breast cancer": ["R-HSA-5633007", "R-HSA-9634638"],
    "diabetes": ["R-HSA-422085", "R-HSA-392499", "R-HSA-165159"],
    "type 2 diabetes": ["R-HSA-422085", "R-HSA-392499"],
}


def get_disease_reactome_pathways(disease_name: str) -> list[str]:
    """Return curated Reactome IDs for a disease (fast fallback)."""
    key = disease_name.lower()
    for k, pathways in DISEASE_REACTOME_MAP.items():
        if k in key or key in k:
            return pathways
    return []
