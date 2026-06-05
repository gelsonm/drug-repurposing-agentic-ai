"""
ChEMBL REST API client.
Docs: https://www.ebi.ac.uk/chembl/api/data/docs
No authentication required. Rate limit: ~1 req/s.
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.tools.http_utils import APIError, get_json

_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
_RATE_LIMIT_DELAY = 0.5  # seconds between requests


def search_drug_by_name(drug_name: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search ChEMBL for a drug by name.

    Returns list of molecule metadata dicts with fields:
    molecule_chembl_id, pref_name, max_phase, molecule_type, etc.
    """
    logger.info(f"ChEMBL: searching for drug '{drug_name}'")
    try:
        data = get_json(
            f"{_BASE_URL}/molecule.json",
            params={
                "pref_name__icontains": drug_name,
                "limit": max_results,
                "format": "json",
            },
        )
        molecules = data.get("molecules", [])
        logger.info(f"ChEMBL: found {len(molecules)} molecules for '{drug_name}'")
        return molecules
    except APIError as e:
        logger.warning(f"ChEMBL search failed for '{drug_name}': {e}")
        return []


def get_drug_by_chembl_id(chembl_id: str) -> dict[str, Any] | None:
    """Fetch full molecule record by ChEMBL ID (e.g. 'CHEMBL1431')."""
    logger.info(f"ChEMBL: fetching molecule {chembl_id}")
    time.sleep(_RATE_LIMIT_DELAY)
    try:
        return get_json(f"{_BASE_URL}/molecule/{chembl_id}.json")
    except APIError as e:
        logger.warning(f"ChEMBL: failed to fetch {chembl_id}: {e}")
        return None


def get_drug_targets(chembl_id: str, max_results: int = 20) -> list[dict[str, Any]]:
    """
    Get all protein targets for a drug via its ChEMBL ID.

    Returns list of target dicts with: target_chembl_id, pref_name,
    target_type, organism, target_components (UniProt accessions).
    """
    logger.info(f"ChEMBL: fetching targets for {chembl_id}")
    time.sleep(_RATE_LIMIT_DELAY)
    try:
        # Get mechanism of action data
        moa_data = get_json(
            f"{_BASE_URL}/mechanism.json",
            params={
                "molecule_chembl_id": chembl_id,
                "limit": max_results,
                "format": "json",
            },
        )
        mechanisms = moa_data.get("mechanisms", [])

        # Get unique target IDs
        target_ids = list({m["target_chembl_id"] for m in mechanisms if m.get("target_chembl_id")})
        targets = []
        for tid in target_ids[:10]:  # cap at 10 targets
            time.sleep(_RATE_LIMIT_DELAY)
            target = get_json(f"{_BASE_URL}/target/{tid}.json")
            if target:
                targets.append(target)

        logger.info(f"ChEMBL: found {len(targets)} targets for {chembl_id}")
        return targets
    except APIError as e:
        logger.warning(f"ChEMBL: failed to get targets for {chembl_id}: {e}")
        return []


def get_drug_mechanisms(chembl_id: str) -> list[dict[str, Any]]:
    """Get mechanism of action data for a drug."""
    time.sleep(_RATE_LIMIT_DELAY)
    try:
        data = get_json(
            f"{_BASE_URL}/mechanism.json",
            params={"molecule_chembl_id": chembl_id, "limit": 20, "format": "json"},
        )
        return data.get("mechanisms", [])
    except APIError as e:
        logger.warning(f"ChEMBL mechanisms failed for {chembl_id}: {e}")
        return []


def get_bioactivities_for_target(
    target_chembl_id: str,
    activity_type: str = "IC50",
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Get bioactivity data (e.g. IC50 values) for a given target."""
    time.sleep(_RATE_LIMIT_DELAY)
    try:
        data = get_json(
            f"{_BASE_URL}/activity.json",
            params={
                "target_chembl_id": target_chembl_id,
                "standard_type": activity_type,
                "limit": max_results,
                "format": "json",
            },
        )
        return data.get("activities", [])
    except APIError as e:
        logger.warning(f"ChEMBL bioactivities failed for {target_chembl_id}: {e}")
        return []


def extract_uniprot_ids(targets: list[dict[str, Any]]) -> list[str]:
    """Extract UniProt accession IDs from ChEMBL target records."""
    uniprot_ids = []
    for target in targets:
        components = target.get("target_components", [])
        for comp in components:
            for xref in comp.get("target_component_xrefs", []):
                if xref.get("xref_src_db") == "UniProt":
                    uniprot_ids.append(xref["xref_id"])
    return list(set(uniprot_ids))
