"""
ClinicalTrials.gov API v2 client.
Docs: https://clinicaltrials.gov/data-api/api
No authentication required.
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.tools.http_utils import APIError, get_json

_BASE_URL = "https://clinicaltrials.gov/api/v2"
_RATE_LIMIT_DELAY = 0.5


class TrialStatus:
    """Constants for clinical trial status values."""
    RECRUITING = "RECRUITING"
    ACTIVE = "ACTIVE_NOT_RECRUITING"
    COMPLETED = "COMPLETED"
    TERMINATED = "TERMINATED"
    WITHDRAWN = "WITHDRAWN"
    NOT_YET = "NOT_YET_RECRUITING"


def search_trials(
    drug_name: str | None = None,
    disease_name: str | None = None,
    status: str | None = None,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """
    Search ClinicalTrials.gov for trials involving a drug and/or disease.

    Args:
        drug_name: Drug/intervention name to search for.
        disease_name: Condition/disease to search for.
        status: Filter by trial status (use TrialStatus constants).
        max_results: Maximum number of results to return.

    Returns:
        List of trial summary dicts.
    """
    # Build query
    query_parts = []
    if drug_name:
        query_parts.append(f'AREA[InterventionName]"{drug_name}"')
    if disease_name:
        query_parts.append(f'AREA[Condition]"{disease_name}"')

    query = " AND ".join(query_parts) if query_parts else drug_name or disease_name or ""

    logger.info(f"ClinicalTrials: searching drug='{drug_name}', disease='{disease_name}'")
    time.sleep(_RATE_LIMIT_DELAY)

    params: dict[str, Any] = {
        "query.cond": disease_name or "",
        "query.intr": drug_name or "",
        "pageSize": min(max_results, 100),
        "format": "json",
        "fields": (
            "NCTId,BriefTitle,OverallStatus,Phase,StartDate,"
            "CompletionDate,Condition,InterventionName,"
            "EnrollmentCount,StudyType,BriefSummary"
        ),
    }
    if status:
        params["filter.overallStatus"] = status

    # Remove empty params
    params = {k: v for k, v in params.items() if v}

    try:
        data = get_json(f"{_BASE_URL}/studies", params=params)
        studies = data.get("studies", [])
        logger.info(f"ClinicalTrials: found {len(studies)} studies")
        return [_flatten_study(s) for s in studies]
    except APIError as e:
        logger.warning(f"ClinicalTrials search failed: {e}")
        return []


def get_trial_by_nct_id(nct_id: str) -> dict[str, Any] | None:
    """Fetch full details for a specific trial by NCT ID."""
    logger.info(f"ClinicalTrials: fetching {nct_id}")
    time.sleep(_RATE_LIMIT_DELAY)
    try:
        data = get_json(f"{_BASE_URL}/studies/{nct_id}", params={"format": "json"})
        return _flatten_study(data)
    except APIError as e:
        logger.warning(f"ClinicalTrials: failed to fetch {nct_id}: {e}")
        return None


def _flatten_study(study: dict[str, Any]) -> dict[str, Any]:
    """Flatten the nested ClinicalTrials API v2 response to a flat dict."""
    try:
        proto = study.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        desc_mod = proto.get("descriptionModule", {})
        design_mod = proto.get("designModule", {})
        arms_mod = proto.get("armsInterventionsModule", {})
        cond_mod = proto.get("conditionsModule", {})

        interventions = [
            i.get("interventionName", "")
            for i in arms_mod.get("interventions", [])
            if i.get("interventionType") in ("DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT")
        ]

        return {
            "nct_id": id_mod.get("nctId", ""),
            "title": id_mod.get("briefTitle", ""),
            "status": status_mod.get("overallStatus", ""),
            "phase": design_mod.get("phases", []),
            "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
            "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
            "conditions": cond_mod.get("conditions", []),
            "interventions": interventions,
            "enrollment": design_mod.get("enrollmentInfo", {}).get("count", 0),
            "study_type": design_mod.get("studyType", ""),
            "summary": desc_mod.get("briefSummary", ""),
            "url": f"https://clinicaltrials.gov/study/{id_mod.get('nctId', '')}",
        }
    except Exception as e:
        logger.debug(f"Error flattening study: {e}")
        return study


def analyze_trial_landscape(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarize the trial landscape for a drug-disease pair.

    Returns:
        dict with counts by status, phases, and clinical feasibility signal.
    """
    if not trials:
        return {
            "total": 0,
            "active": 0,
            "completed": 0,
            "terminated": 0,
            "phases": {},
            "clinical_signal": "no_trials",
        }

    status_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}

    for trial in trials:
        status = trial.get("status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

        phases = trial.get("phase", [])
        for phase in phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

    completed = status_counts.get("COMPLETED", 0)
    active = status_counts.get("RECRUITING", 0) + status_counts.get("ACTIVE_NOT_RECRUITING", 0)
    terminated = status_counts.get("TERMINATED", 0)

    # Infer clinical signal strength
    if completed >= 2:
        signal = "strong_evidence"
    elif active >= 1 or completed >= 1:
        signal = "emerging_evidence"
    elif terminated > active:
        signal = "mixed_evidence"
    else:
        signal = "early_stage"

    return {
        "total": len(trials),
        "active": active,
        "completed": completed,
        "terminated": terminated,
        "phases": phase_counts,
        "clinical_signal": signal,
    }
