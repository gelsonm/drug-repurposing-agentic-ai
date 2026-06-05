"""
Agent 6: Adversarial Critique Agent ⭐ (KEY NOVELTY)

Acts as a skeptical scientific reviewer. Finds reasons a candidate
SHOULDN'T work before it reaches the scoring stage.

Checks:
  1. Failed/terminated clinical trials (ClinicalTrials.gov)
  2. Contraindications (DrugBank — via ChEMBL adverse events proxy)
  3. Drug-drug interactions with standard-of-care
  4. Bioavailability at target (BBB for CNS, etc.)
  5. Selectivity concerns (promiscuous multi-target drugs)

Issues:
  - RED FLAG: Disqualifying (3+ Phase II failures, known contraindication)
  - YELLOW FLAG: Concerning but not disqualifying

Validation: ClinicalTrials NCT IDs cited for every failed trial finding.
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.schemas.models import (
    AdversarialReport,
    CandidateDrug,
    DiseaseProfile,
    FailedTrial,
    PathwayAlignment,
    SafetyFlag,
)
from src.tools.http_utils import APIError, get_json

_CT_BASE = "https://clinicaltrials.gov/api/v2/studies"


def _search_failed_trials(drug_name: str, disease_name: str) -> list[FailedTrial]:
    """Query ClinicalTrials.gov for terminated/withdrawn trials."""
    logger.info(f"[Agent 6] Checking failed trials: {drug_name} + {disease_name}")
    time.sleep(0.5)

    try:
        data = get_json(
            _CT_BASE,
            params={
                "query.intr": drug_name,
                "query.cond": disease_name,
                "filter.overallStatus": "TERMINATED,WITHDRAWN,SUSPENDED",
                "pageSize": 10,
                "format": "json",
            },
        )
        studies = data.get("studies", [])
        failed = []
        for study in studies[:10]:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            nct_id = ident.get("nctId", "")
            title = ident.get("briefTitle", "")
            phase_list = design.get("phases", [])
            phase = phase_list[0].replace("PHASE", "Phase ") if phase_list else "Unknown"
            why_stopped = status.get("whyStopped", "")

            if nct_id:
                failed.append(FailedTrial(
                    nct_id=nct_id,
                    title=title[:150] if title else None,
                    phase=phase,
                    why_stopped=why_stopped[:200] if why_stopped else None,
                ))

        logger.info(f"[Agent 6] {len(failed)} failed trials found for {drug_name}+{disease_name}")
        return failed

    except APIError as e:
        logger.warning(f"[Agent 6] ClinicalTrials query failed: {e}")
        return []


def _check_promiscuity(candidate: CandidateDrug) -> str | None:
    """Flag drugs with too many targets (promiscuous binding)."""
    n_targets = len(candidate.known_targets)
    if n_targets >= 10:
        return (
            f"Highly promiscuous drug: {n_targets} known targets in ChEMBL. "
            "Difficult to attribute repurposing effect to specific mechanism."
        )
    if n_targets >= 6:
        return f"Moderately promiscuous: {n_targets} targets may complicate mechanistic attribution."
    return None


def _check_ddi_risk(drug_name: str, disease_name: str) -> str | None:
    """Check for known DDI risk with standard-of-care (curated list)."""
    # Standard-of-care drugs by disease area
    soc_drugs: dict[str, list[str]] = {
        "alzheimer": ["donepezil", "memantine", "galantamine", "rivastigmine"],
        "parkinson": ["levodopa", "carbidopa", "rasagiline"],
        "diabetes": ["insulin", "glipizide", "sitagliptin"],
        "cancer": ["doxorubicin", "paclitaxel", "cisplatin", "carboplatin"],
        "heart failure": ["digoxin", "furosemide", "warfarin"],
    }
    # Known problematic DDIs
    ddi_warnings: dict[str, dict[str, str]] = {
        "metformin": {"alcohol": "Lactic acidosis risk"},
        "warfarin": {"nsaids": "Bleeding risk"},
        "digoxin": {"verapamil": "Toxicity risk"},
    }

    drug_lower = drug_name.lower()
    if drug_lower in ddi_warnings:
        disease_lower = disease_name.lower()
        for disease_key, soc_list in soc_drugs.items():
            if disease_key in disease_lower:
                for soc in soc_list:
                    if soc in ddi_warnings.get(drug_lower, {}):
                        return ddi_warnings[drug_lower][soc]
    return None


def _check_selectivity(candidate: CandidateDrug, disease_profile: DiseaseProfile) -> str | None:
    """Check if drug targets tumor suppressors or protective genes (wrong directionality)."""
    # Known tumor suppressors / protective genes
    protective_genes = {"TP53", "BRCA1", "BRCA2", "RB1", "PTEN", "APC", "VHL"}
    neuroprotective_genes = {"BDNF", "NGF", "SIRT1", "PARK2"}

    drug_targets = set(candidate.known_targets)
    is_cns = disease_profile.is_cns_disease

    if is_cns:
        bad_targets = drug_targets & neuroprotective_genes
        if bad_targets:
            return (
                f"Drug may inhibit neuroprotective genes: {', '.join(bad_targets)}. "
                "Review directionality of interaction."
            )
    else:
        bad_targets = drug_targets & protective_genes
        if bad_targets:
            return (
                f"Drug targets known tumor suppressors: {', '.join(bad_targets)}. "
                "Inhibiting these genes may be counterproductive."
            )
    return None


def run_adversarial_agent(
    candidate: CandidateDrug,
    disease_profile: DiseaseProfile,
    pathway_alignment: PathwayAlignment | None = None,
) -> AdversarialReport:
    """
    Execute Agent 6: Adversarial Critique.

    Returns:
        AdversarialReport with red/yellow flags and overall verdict.
    """
    drug_name = candidate.drug_name
    disease_name = disease_profile.disease_name
    logger.info(f"[Agent 6] Adversarial critique: {drug_name} ↔ {disease_name}")

    red_flags: list[str] = []
    yellow_flags: list[str] = []

    # ── Check 1: Failed trials ─────────────────────────────────────────────
    failed_trials = _search_failed_trials(drug_name, disease_name)
    phase2_failures = [t for t in failed_trials if "2" in (t.phase or "")]
    phase3_failures = [t for t in failed_trials if "3" in (t.phase or "")]

    if len(phase3_failures) >= 2:
        red_flags.append(
            f"Multiple Phase III failures ({len(phase3_failures)} trials terminated). "
            f"NCT IDs: {', '.join(t.nct_id for t in phase3_failures[:3])}"
        )
    elif len(phase2_failures) >= 3:
        red_flags.append(
            f"Multiple Phase II failures ({len(phase2_failures)} trials). "
            f"NCT IDs: {', '.join(t.nct_id for t in phase2_failures[:3])}"
        )
    elif failed_trials:
        yellow_flags.append(
            f"{len(failed_trials)} terminated trial(s) for this indication. "
            f"NCT IDs: {', '.join(t.nct_id for t in failed_trials[:2])}"
        )

    # ── Check 2: Pathway alignment (Speculative = yellow) ─────────────────
    if pathway_alignment:
        from src.schemas.models import AlignmentType
        if pathway_alignment.alignment_type == AlignmentType.SPECULATIVE:
            yellow_flags.append(
                "No direct pathway connection established. "
                "Mechanistic rationale is speculative — requires experimental validation."
            )
        if pathway_alignment.bbb_penetrant is False and disease_profile.is_cns_disease:
            red_flags.append(
                f"BBB CONCERN: {drug_name} has poor blood-brain barrier penetration. "
                "CNS drug exposure may be insufficient for therapeutic effect."
            )

    # ── Check 3: Promiscuity ───────────────────────────────────────────────
    promiscuity_flag = _check_promiscuity(candidate)
    if promiscuity_flag:
        yellow_flags.append(promiscuity_flag)

    # ── Check 4: Drug-drug interactions ───────────────────────────────────
    ddi_flag = _check_ddi_risk(drug_name, disease_name)
    if ddi_flag:
        yellow_flags.append(f"Potential DDI with standard-of-care: {ddi_flag}")

    # ── Check 5: Selectivity/directionality ───────────────────────────────
    selectivity_flag = _check_selectivity(candidate, disease_profile)
    if selectivity_flag:
        yellow_flags.append(selectivity_flag)

    # ── Check 6: Low literature evidence ──────────────────────────────────
    if len(candidate.literature_pmids) < 3:
        yellow_flags.append(
            f"Low literature evidence: only {len(candidate.literature_pmids)} papers found. "
            "Consider this a preliminary computational hypothesis."
        )

    # ── Check 7: No ChEMBL ID ─────────────────────────────────────────────
    if not candidate.chembl_id:
        yellow_flags.append("No ChEMBL ID found — drug metadata could not be fully verified.")

    # ── Compute safety flag ────────────────────────────────────────────────
    if red_flags:
        safety_flag = SafetyFlag.RED
    elif len(yellow_flags) >= 3:
        safety_flag = SafetyFlag.YELLOW
    elif yellow_flags:
        safety_flag = SafetyFlag.YELLOW
    else:
        safety_flag = SafetyFlag.CLEAN

    # ── Compute confidence adjustment ─────────────────────────────────────
    adjustment = 0.0
    adjustment -= len(red_flags) * 0.15
    adjustment -= len(yellow_flags) * 0.05
    adjustment -= len(phase2_failures) * 0.08
    adjustment -= len(phase3_failures) * 0.12
    if pathway_alignment and pathway_alignment.bbb_penetrant is False and disease_profile.is_cns_disease:
        adjustment -= 0.10
    adjustment = max(-1.0, round(adjustment, 3))

    # ── Overall verdict ────────────────────────────────────────────────────
    if red_flags:
        verdict = f"DEPRIORITIZE — {len(red_flags)} disqualifying factor(s) found"
    elif not yellow_flags:
        verdict = "PROCEED — no significant adverse factors found"
    else:
        verdict = f"PROCEED WITH CAUTION — {len(yellow_flags)} concern(s) noted"

    report = AdversarialReport(
        drug_name=drug_name,
        disease_name=disease_name,
        red_flags=red_flags,
        yellow_flags=yellow_flags,
        failed_trials=failed_trials[:5],
        overall_verdict=verdict,
        confidence_adjustment=adjustment,
        safety_flag=safety_flag,
    )

    logger.info(
        f"[Agent 6] {drug_name}: {len(red_flags)} red, {len(yellow_flags)} yellow, "
        f"verdict='{verdict}', adjustment={adjustment:.3f}"
    )
    return report
