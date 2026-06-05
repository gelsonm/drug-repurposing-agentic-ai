"""
OpenFDA API client for adverse event (FAERS) and drug label data.
Docs: https://open.fda.gov/apis/
No key: 240 req/min | With key: 1000 req/min
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.config import get_settings
from src.tools.http_utils import APIError, get_json

_BASE_URL = "https://api.fda.gov"
_RATE_LIMIT_DELAY = 0.26  # ~240 req/min without key


def _api_key_param() -> dict[str, str]:
    settings = get_settings()
    return {"api_key": settings.openfda_api_key} if settings.openfda_api_key else {}


def search_adverse_events(
    drug_name: str,
    disease_name: str | None = None,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """
    Search FDA Adverse Event Reporting System (FAERS) for a drug.

    Returns list of adverse event reports relevant to the drug.
    """
    logger.info(f"OpenFDA FAERS: searching for drug '{drug_name}'")
    time.sleep(_RATE_LIMIT_DELAY)

    query = f'patient.drug.medicinalproduct:"{drug_name}"'
    if disease_name:
        query += f' AND patient.reaction.reactionmeddrapt:"{disease_name}"'

    try:
        data = get_json(
            f"{_BASE_URL}/drug/event.json",
            params={
                **_api_key_param(),
                "search": query,
                "limit": max_results,
            },
        )
        results = data.get("results", [])
        logger.info(f"OpenFDA FAERS: found {len(results)} adverse event reports")
        return results
    except APIError as e:
        logger.warning(f"OpenFDA FAERS failed for '{drug_name}': {e}")
        return []


def get_drug_label(drug_name: str) -> dict[str, Any] | None:
    """
    Fetch FDA drug label (prescribing information) for a drug.

    Returns structured label with: indications, warnings, contraindications,
    adverse reactions, clinical pharmacology.
    """
    logger.info(f"OpenFDA label: fetching label for '{drug_name}'")
    time.sleep(_RATE_LIMIT_DELAY)

    try:
        data = get_json(
            f"{_BASE_URL}/drug/label.json",
            params={
                **_api_key_param(),
                "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
                "limit": 1,
            },
        )
        results = data.get("results", [])
        return results[0] if results else None
    except APIError as e:
        logger.warning(f"OpenFDA label failed for '{drug_name}': {e}")
        return None


def get_top_adverse_reactions(drug_name: str, max_reactions: int = 20) -> list[dict[str, Any]]:
    """
    Get the most commonly reported adverse reactions for a drug (count aggregation).

    Returns list of dicts: {reaction: str, count: int}
    """
    logger.info(f"OpenFDA: getting top adverse reactions for '{drug_name}'")
    time.sleep(_RATE_LIMIT_DELAY)

    try:
        data = get_json(
            f"{_BASE_URL}/drug/event.json",
            params={
                **_api_key_param(),
                "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": max_reactions,
            },
        )
        return data.get("results", [])
    except APIError as e:
        logger.warning(f"OpenFDA adverse reactions failed for '{drug_name}': {e}")
        return []


def extract_safety_summary(label: dict[str, Any] | None) -> dict[str, Any]:
    """
    Extract key safety information from an FDA drug label.

    Returns dict with: indications, warnings, contraindications, adverse_reactions.
    """
    if not label:
        return {
            "indications": [],
            "warnings": [],
            "contraindications": [],
            "adverse_reactions": [],
            "has_label": False,
        }

    def _get_section(key: str) -> str:
        val = label.get(key, [])
        if isinstance(val, list) and val:
            return val[0][:1000]  # truncate long sections
        return str(val)[:1000] if val else ""

    return {
        "indications": _get_section("indications_and_usage"),
        "warnings": _get_section("warnings"),
        "contraindications": _get_section("contraindications"),
        "adverse_reactions": _get_section("adverse_reactions"),
        "boxed_warning": _get_section("boxed_warning"),
        "has_label": True,
    }


def assess_safety_flag(
    label_summary: dict[str, Any],
    adverse_reactions: list[dict[str, Any]],
    target_disease: str,
) -> str:
    """
    Assess safety flag for a potential repurposing candidate.

    Returns: "clean" | "minor_concerns" | "major_concerns" | "contraindicated"
    """
    if not label_summary.get("has_label"):
        return "unknown"

    # Check for explicit contraindications
    contraindications = label_summary.get("contraindications", "").lower()
    boxed_warning = label_summary.get("boxed_warning", "").lower()

    disease_lower = target_disease.lower()

    if disease_lower in contraindications:
        return "contraindicated"

    if boxed_warning and len(boxed_warning) > 50:
        return "major_concerns"

    warnings = label_summary.get("warnings", "").lower()
    if any(word in warnings for word in ["serious", "fatal", "death", "severe"]):
        return "minor_concerns"

    return "clean"
