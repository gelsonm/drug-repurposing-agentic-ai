"""
DisGeNET REST API client.
Documentation: https://www.disgenet.com/api/

DisGeNET provides curated gene-disease associations with:
- Association score (GDA score)
- Evidence Index (EI): proportion of evidence supporting association
- Disease Specificity Index (DSI)
- Gene Pleiotropy Index (GPI)

Requires an API key (free academic registration at disgenet.com).
Falls back gracefully if no key is set.
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.config import get_settings
from src.tools.http_utils import APIError, get_json

_BASE_URL = "https://www.disgenet.com/api"
_RATE_DELAY = 0.5  # seconds between requests (free tier)


def _auth_header() -> dict[str, str]:
    """Return authorization header if API key is configured."""
    key = get_settings().disgenet_api_key
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def get_disease_gene_associations(
    disease_name: str,
    max_results: int = 20,
    min_score: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Get gene-disease associations from DisGeNET for a disease name.

    Args:
        disease_name: Disease name to search (e.g. "Alzheimer's disease")
        max_results: Max associations to return
        min_score: Minimum GDA score to include

    Returns:
        List of dicts with: gene_symbol, gene_id, score, ei, dsi, disease_name
    """
    settings = get_settings()
    if not settings.disgenet_api_key:
        logger.info("DisGeNET: no API key configured — skipping (using Open Targets instead)")
        return []

    logger.info(f"DisGeNET: searching gene associations for '{disease_name}'")
    time.sleep(_RATE_DELAY)

    try:
        # Step 1: Search for disease
        search_data = get_json(
            f"{_BASE_URL}/disease/search",
            params={"q": disease_name, "format": "json"},
            headers=_auth_header(),
        )
        if not search_data:
            return []

        # Get top disease match
        diseases = search_data if isinstance(search_data, list) else search_data.get("results", [])
        if not diseases:
            logger.warning(f"DisGeNET: no disease found for '{disease_name}'")
            return []

        disease_id = diseases[0].get("diseaseId") or diseases[0].get("disease_id", "")
        if not disease_id:
            return []

        logger.info(f"DisGeNET: disease ID = {disease_id}")
        time.sleep(_RATE_DELAY)

        # Step 2: Get gene associations
        assoc_data = get_json(
            f"{_BASE_URL}/gda/disease/{disease_id}",
            params={"format": "json", "limit": max_results},
            headers=_auth_header(),
        )

        if not assoc_data:
            return []

        associations = assoc_data if isinstance(assoc_data, list) else assoc_data.get("results", [])

        results = []
        for assoc in associations:
            score = float(assoc.get("score", assoc.get("gdaScore", 0.0)))
            if score < min_score:
                continue
            results.append({
                "gene_symbol": assoc.get("gene_symbol", assoc.get("geneSymbol", "")),
                "gene_id": str(assoc.get("ncbi_id", assoc.get("geneNcbiId", ""))),
                "score": round(score, 4),
                "ei": float(assoc.get("ei", assoc.get("EI", 0.0))),
                "dsi": float(assoc.get("dsi", assoc.get("DSI", 0.5))),
                "disease_name": assoc.get("disease_name", disease_name),
                "source": "DisGeNET",
            })

        logger.info(f"DisGeNET: {len(results)} gene associations for '{disease_name}'")
        return results

    except APIError as e:
        logger.warning(f"DisGeNET request failed: {e}")
        return []
    except Exception as e:
        logger.warning(f"DisGeNET unexpected error: {e}")
        return []
