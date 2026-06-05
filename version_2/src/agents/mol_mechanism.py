"""
Agent 3: Molecular Mechanism Agent

For each candidate drug, retrieves detailed MOA at molecular level:
  - ChEMBL: binding targets, Ki/IC50 values, assay IDs
  - STRING DB: protein-protein interaction network context
  - UniProt: protein function annotations

Validation: Binding claims must cite a ChEMBL assay ID.
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.schemas.models import CandidateDrug
from src.tools.http_utils import APIError, get_json
from src.tools.string_db_client import get_functional_enrichment, get_interaction_partners

_CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
_UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"


def _get_chembl_targets(chembl_id: str) -> list[dict[str, Any]]:
    """Get mechanism-of-action targets from ChEMBL."""
    try:
        data = get_json(
            f"{_CHEMBL_BASE}/mechanism.json",
            params={"molecule_chembl_id": chembl_id, "limit": 20},
        )
        mechanisms = data.get("mechanisms", [])
        targets = []
        for m in mechanisms:
            tid = m.get("target_chembl_id")
            if not tid:
                continue
            time.sleep(0.3)
            try:
                t_data = get_json(f"{_CHEMBL_BASE}/target/{tid}.json")
                components = t_data.get("target_components", [])
                uniprot_ids = [
                    xref.get("xref_id", "")
                    for comp in components
                    for xref in comp.get("target_component_xrefs", [])
                    if xref.get("xref_src_db") == "UniProt"
                ]
                targets.append({
                    "target_chembl_id": tid,
                    "pref_name": t_data.get("pref_name", ""),
                    "target_type": t_data.get("target_type", ""),
                    "mechanism": m.get("mechanism_of_action", ""),
                    "action_type": m.get("action_type", ""),
                    "uniprot_ids": uniprot_ids,
                })
            except APIError:
                continue
        return targets
    except APIError as e:
        logger.warning(f"ChEMBL targets failed for {chembl_id}: {e}")
        return []


def _get_chembl_bioactivities(target_chembl_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Get top IC50/Ki bioactivities for a target (up to limit)."""
    try:
        data = get_json(
            f"{_CHEMBL_BASE}/activity.json",
            params={
                "target_chembl_id": target_chembl_id,
                "standard_type__in": "IC50,Ki,Kd",
                "limit": limit,
                "assay_type": "B",  # Binding assays only
            },
        )
        activities = data.get("activities", [])
        results = []
        for act in activities:
            val = act.get("standard_value")
            unit = act.get("standard_units", "nM")
            assay_id = act.get("assay_chembl_id", "")
            if val:
                results.append({
                    "assay_id": assay_id,
                    "type": act.get("standard_type", ""),
                    "value": val,
                    "units": unit,
                    "pchembl": act.get("pchembl_value"),
                })
        return results
    except APIError:
        return []


def _get_uniprot_function(uniprot_id: str) -> str:
    """Get protein function text from UniProt."""
    try:
        data = get_json(
            f"{_UNIPROT_BASE}/{uniprot_id}",
            params={"fields": "function_cc", "format": "json"},
        )
        comments = data.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    return texts[0].get("value", "")[:300]
    except APIError:
        pass
    return ""


def run_mol_mechanism_agent(candidate: CandidateDrug) -> CandidateDrug:
    """
    Execute Agent 3: Molecular Mechanism Analysis.
    Enriches the candidate with detailed MOA and binding data.

    Returns:
        Enriched CandidateDrug with mechanism_of_action and targets populated.
    """
    drug_name = candidate.drug_name
    chembl_id = candidate.chembl_id

    logger.info(f"[Agent 3] Molecular Mechanism: {drug_name} (ChEMBL: {chembl_id})")

    if not chembl_id:
        # Try to find ChEMBL ID by drug name
        try:
            data = get_json(
                f"{_CHEMBL_BASE}/molecule.json",
                params={"pref_name__icontains": drug_name, "limit": 3},
            )
            molecules = data.get("molecules", [])
            if molecules:
                chembl_id = molecules[0].get("molecule_chembl_id")
                candidate.chembl_id = chembl_id
                candidate.max_phase = int(molecules[0].get("max_phase", 0) or 0)
                logger.info(f"[Agent 3] Found ChEMBL ID: {chembl_id}")
        except APIError:
            pass

    if not chembl_id:
        logger.warning(f"[Agent 3] No ChEMBL ID for '{drug_name}' — skipping deep mechanism")
        return candidate

    # ── Step 1: ChEMBL targets ─────────────────────────────────────────────
    chembl_targets = _get_chembl_targets(chembl_id)

    # ── Step 2: Bioactivity data for top 2 targets ─────────────────────────
    binding_citations = []  # ChEMBL assay IDs for hallucination guard
    moa_pieces = []

    for tgt in chembl_targets[:4]:
        target_name = tgt.get("pref_name", "")
        mechanism = tgt.get("mechanism", "")
        action = tgt.get("action_type", "")
        if target_name:
            moa_pieces.append(f"{action} {target_name}")
            candidate.known_targets.append(target_name)

        # Get binding data
        tid = tgt.get("target_chembl_id", "")
        if tid:
            acts = _get_chembl_bioactivities(tid, limit=3)
            for act in acts:
                assay_id = act.get("assay_id", "")
                if assay_id:
                    binding_citations.append(assay_id)
                    val = act.get("value", "")
                    atype = act.get("type", "")
                    units = act.get("units", "nM")
                    if val:
                        moa_pieces.append(f"  [{atype}={val}{units} assay:{assay_id}]")

        # UniProt function
        uniprot_ids = tgt.get("uniprot_ids", [])
        for uid in uniprot_ids[:1]:
            func = _get_uniprot_function(uid)
            if func:
                logger.debug(f"[Agent 3] UniProt {uid}: {func[:80]}")

    # ── Step 3: STRING PPI context ─────────────────────────────────────────
    unique_targets = list(dict.fromkeys(candidate.known_targets))[:8]
    ppi_interactions = []
    if unique_targets:
        ppi_interactions = get_interaction_partners(unique_targets, min_score=700, max_partners=10)

    ppi_note = ""
    if ppi_interactions:
        top_partners = list({i.get("protein_b", "") for i in ppi_interactions[:5] if i.get("protein_b")})
        ppi_note = f" Interacts with: {', '.join(top_partners[:4])}."

    # ── Step 4: Build MOA string ───────────────────────────────────────────
    if moa_pieces:
        candidate.mechanism_of_action = (
            f"{drug_name} acts via: " + "; ".join(p for p in moa_pieces if "=" not in p)[:200] + ppi_note
        )
    elif not candidate.mechanism_of_action:
        candidate.mechanism_of_action = f"{drug_name}: mechanism not found in ChEMBL."

    # Deduplicate targets
    candidate.known_targets = list(dict.fromkeys(candidate.known_targets))[:10]

    logger.info(
        f"[Agent 3] {drug_name}: {len(candidate.known_targets)} targets, "
        f"{len(binding_citations)} binding citations, {len(ppi_interactions)} PPI edges"
    )
    return candidate
