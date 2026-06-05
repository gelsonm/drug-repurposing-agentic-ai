"""
Agent 5: Pathway Mechanism Validator Agent ⭐ (KEY NOVELTY)

Verifies that the drug's MOA mechanistically connects to the disease's
known pathogenic pathway. This is the biological grounding layer.

Logic:
  1. Take drug targets (from Agent 3) + disease pathways (from Agent 1)
  2. Map drug targets to KEGG/Reactome pathway IDs
  3. Check overlap with disease pathways
  4. Score alignment: DIRECT / INDIRECT / SPECULATIVE
  5. For CNS diseases: check BBB penetration from known database

Validation:
  - Must map to at least one Reactome or KEGG pathway ID
  - SPECULATIVE alignment auto-flags candidate for low priority
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.schemas.models import AlignmentType, CandidateDrug, DiseaseProfile, PathwayAlignment
from src.tools.kegg_client import (
    get_disease_pathways_curated,
    get_pathways_for_gene,
    search_pathways_by_keyword,
)
from src.tools.reactome_client import (
    get_disease_reactome_pathways,
    get_pathways_for_genes,
)

# BBB penetration database (known from published pharmacology)
# True = crosses BBB, False = does NOT cross BBB, None = uncertain
_BBB_DATABASE: dict[str, bool | None] = {
    "metformin": False,         # Does NOT cross BBB well (polar, low lipophilicity)
    "rapamycin": True,          # Crosses BBB (lipophilic)
    "semaglutide": False,       # Peptide — poor BBB penetration (but some CNS effect via vagal nerve)
    "ibuprofen": True,          # Crosses BBB
    "atorvastatin": False,      # Hydrophilic statins — poor BBB
    "simvastatin": True,        # Lipophilic statin — crosses BBB
    "aspirin": True,            # Crosses BBB
    "dexamethasone": True,      # Crosses BBB
    "donepezil": True,          # Designed for CNS (AD drug)
    "memantine": True,          # CNS drug
    "hydroxychloroquine": False,# Poor CNS penetration
    "tamoxifen": True,          # Crosses BBB
    "valproic acid": True,      # Crosses BBB
    "lithium": True,            # Crosses BBB
    "pioglitazone": True,       # Can cross BBB
    "liraglutide": False,       # GLP-1 agonist — poor BBB
    "exenatide": False,         # GLP-1 agonist — poor BBB
}


def _check_bbb(drug_name: str) -> bool | None:
    return _BBB_DATABASE.get(drug_name.lower())


def _compute_pathway_overlap_score(
    drug_pathway_ids: set[str],
    disease_pathway_ids: set[str],
    drug_target_set: set[str],
    disease_gene_set: set[str],
) -> tuple[AlignmentType, float]:
    """
    Compute alignment type and plausibility score.

    Returns: (AlignmentType, score 0-1)
    """
    # Direct: drug targets exactly in disease pathway
    pathway_overlap = drug_pathway_ids & disease_pathway_ids
    target_overlap = drug_target_set & disease_gene_set

    if pathway_overlap and len(pathway_overlap) >= 1:
        # Strong direct connection via shared pathway
        overlap_ratio = len(pathway_overlap) / max(1, len(disease_pathway_ids))
        score = 0.7 + 0.3 * min(1.0, overlap_ratio * 3)
        return AlignmentType.DIRECT, round(score, 4)

    if target_overlap:
        # Indirect: drug targets disease genes but not necessarily same pathway
        target_ratio = len(target_overlap) / max(1, len(disease_gene_set))
        score = 0.45 + 0.25 * min(1.0, target_ratio * 5)
        return AlignmentType.INDIRECT, round(score, 4)

    if drug_pathway_ids:
        # Speculative: drug has pathways but no overlap with disease
        return AlignmentType.SPECULATIVE, 0.15

    return AlignmentType.SPECULATIVE, 0.05


def run_pathway_validator_agent(
    candidate: CandidateDrug,
    disease_profile: DiseaseProfile,
    rag_passages: list[dict[str, Any]] | None = None,
) -> PathwayAlignment:
    """
    Execute Agent 5: Pathway Mechanism Validator.

    Args:
        candidate: Drug candidate from Agents 2-4
        disease_profile: Disease profile from Agent 1
        rag_passages: Literature passages from Agent 4 (for evidence notes)

    Returns:
        PathwayAlignment with mechanistic connection assessment
    """
    drug_name = candidate.drug_name
    disease_name = disease_profile.disease_name
    logger.info(f"[Agent 5] Pathway Validator: {drug_name} ↔ {disease_name}")

    # ── Step 1: Get disease pathway IDs ─────────────────────────────────
    disease_kegg = set(get_disease_pathways_curated(disease_name))
    disease_reactome = set(get_disease_reactome_pathways(disease_name))

    # ── Step 2: Map drug targets to pathways ─────────────────────────────
    drug_kegg_pathways: set[str] = set()
    drug_reactome_pathways: set[str] = set()

    for target in candidate.known_targets[:6]:
        time.sleep(0.2)
        gene_kegg = get_pathways_for_gene(target)
        for pw in gene_kegg:
            drug_kegg_pathways.add(pw["pathway_id"])

    # Reactome enrichment for drug target set
    if candidate.known_targets:
        reactome_results = get_pathways_for_genes(
            candidate.known_targets[:8],
            p_value=0.1,
            max_results=10,
        )
        for r in reactome_results:
            drug_reactome_pathways.add(r.get("pathway_id", ""))

    # Fallback: search by drug mechanism keywords
    if not drug_kegg_pathways and candidate.mechanism_of_action:
        moa = candidate.mechanism_of_action
        for keyword in [drug_name, "mTOR", "AMPK"]:
            if keyword.upper() in moa.upper():
                found = search_pathways_by_keyword(keyword)
                for pw in found:
                    drug_kegg_pathways.add(pw["pathway_id"])
                break

    logger.info(
        f"[Agent 5] Drug pathways: KEGG={len(drug_kegg_pathways)}, "
        f"Reactome={len(drug_reactome_pathways)}"
    )
    logger.info(
        f"[Agent 5] Disease pathways: KEGG={len(disease_kegg)}, "
        f"Reactome={len(disease_reactome)}"
    )

    # ── Step 3: Compute overlap and alignment ─────────────────────────────
    drug_target_set = set(candidate.known_targets)
    disease_gene_set = set(disease_profile.disease_genes)

    all_drug_pathways = drug_kegg_pathways | drug_reactome_pathways
    all_disease_pathways = disease_kegg | disease_reactome

    alignment_type, plausibility_score = _compute_pathway_overlap_score(
        drug_pathway_ids=all_drug_pathways,
        disease_pathway_ids=all_disease_pathways,
        drug_target_set=drug_target_set,
        disease_gene_set=disease_gene_set,
    )

    # ── Step 4: Build connection description ─────────────────────────────
    shared_targets = sorted(drug_target_set & disease_gene_set)
    shared_pathways = sorted(all_drug_pathways & all_disease_pathways)

    connection = _build_connection_description(
        drug_name, disease_name, candidate.mechanism_of_action or "",
        shared_targets, shared_pathways, alignment_type
    )

    # ── Step 5: BBB check for CNS diseases ───────────────────────────────
    bbb_penetrant = None
    tissue_note = None
    if disease_profile.is_cns_disease:
        bbb_penetrant = _check_bbb(drug_name)
        if bbb_penetrant is False:
            tissue_note = (
                f"⚠️ BBB concern: {drug_name} has limited blood-brain barrier penetration, "
                "which may reduce CNS drug exposure despite systemic activity."
            )
            plausibility_score = max(0.0, plausibility_score - 0.15)
        elif bbb_penetrant is True:
            tissue_note = f"✅ {drug_name} is known to cross the blood-brain barrier."
            plausibility_score = min(1.0, plausibility_score + 0.05)
        else:
            tissue_note = f"❓ BBB penetration data for {drug_name} is uncertain."

    # ── Step 6: Evidence notes from literature ────────────────────────────
    validation_notes = []
    if rag_passages:
        supporting_pmids = [
            p["pmid"] for p in rag_passages[:3]
            if p.get("pmid") and p.get("evidence_type") in ("positive", "mechanistic")
        ]
        if supporting_pmids:
            validation_notes.append(f"Literature support: PMIDs {', '.join(supporting_pmids)}")

    if alignment_type == AlignmentType.SPECULATIVE:
        validation_notes.append("⚠️ No direct pathway connection found — speculative hypothesis.")
    elif alignment_type == AlignmentType.INDIRECT:
        validation_notes.append("Indirect mechanistic link via shared upstream/downstream regulators.")

    full_notes = " | ".join(validation_notes) if validation_notes else None
    if tissue_note:
        full_notes = f"{full_notes} | {tissue_note}" if full_notes else tissue_note

    alignment = PathwayAlignment(
        drug_name=drug_name,
        disease_name=disease_name,
        alignment_type=alignment_type,
        connection_description=connection,
        kegg_pathway_ids=sorted(list(drug_kegg_pathways & disease_kegg))[:5],
        reactome_pathway_ids=sorted(list(drug_reactome_pathways & disease_reactome))[:5],
        biological_plausibility_score=round(plausibility_score, 4),
        validation_notes=full_notes,
        bbb_penetrant=bbb_penetrant,
        tissue_distribution_note=tissue_note,
        evidence_pmids=[p.get("pmid", "") for p in (rag_passages or [])[:5] if p.get("pmid")],
    )

    logger.info(
        f"[Agent 5] {drug_name}: alignment={alignment_type.value}, "
        f"plausibility={plausibility_score:.3f}, BBB={bbb_penetrant}"
    )
    return alignment


def _build_connection_description(
    drug: str, disease: str, moa: str,
    shared_targets: list[str], shared_pathways: list[str],
    alignment: AlignmentType,
) -> str:
    """Build human-readable mechanistic connection description."""
    parts = []

    if shared_targets:
        target_str = ", ".join(shared_targets[:3])
        parts.append(f"{drug} modulates {target_str}, which are implicated in {disease}")

    if shared_pathways:
        parts.append(f"Shared pathway IDs: {', '.join(shared_pathways[:3])}")

    if moa and not parts:
        parts.append(f"{drug} mechanism: {moa[:120]}")

    if alignment == AlignmentType.SPECULATIVE:
        parts.append(
            f"Note: No direct pathway overlap found between {drug}'s targets "
            f"and {disease}'s pathogenic pathways. This is a speculative hypothesis."
        )

    return " | ".join(parts) if parts else f"{drug} ↔ {disease}: connection under investigation."
