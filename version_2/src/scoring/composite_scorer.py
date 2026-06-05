"""
Composite scoring engine for drug repurposing v2.
5-component formula with confidence intervals.

Formula:
  composite = (
    kg_proximity        × 0.20  [KG structural evidence]
    + moa_alignment     × 0.25  [mechanistic grounding]
    + literature        × 0.20  [published evidence]
    + pathway_plausibility × 0.25  [biological validation]
    + adversarial_adjustment × 0.10  [risk penalty, ≤ 0]
  )

Confidence interval:
  CI = ±σ based on evidence count — fewer sources → wider CI
"""
from __future__ import annotations

import math
from typing import Any

from loguru import logger

from src.config import get_settings
from src.schemas.models import (
    AdversarialReport,
    AlignmentType,
    PathwayAlignment,
    SafetyFlag,
    ScoreBreakdown,
)


def score_kg_proximity(
    kg_paths: list[dict[str, Any]],
    open_targets_score: float,
    shared_targets: list[str],
) -> float:
    """
    Score KG structural evidence.

    Components:
    - Best KG path score (from Hetionet traversal)
    - Open Targets drug-disease association score
    - Shared target count (gene overlap)

    Returns float in [0, 1].
    """
    # KG path component
    if kg_paths:
        best_path = max(p.get("path_score", 0.0) for p in kg_paths)
        n_paths_factor = min(1.0, len(kg_paths) / 5)
        path_component = 0.7 * best_path + 0.3 * n_paths_factor
    else:
        path_component = 0.0

    # Open Targets score (already 0–1)
    ot_component = min(1.0, open_targets_score)

    # Shared target bonus
    shared_bonus = min(0.5, len(shared_targets) * 0.1)

    score = 0.45 * path_component + 0.40 * ot_component + 0.15 * shared_bonus
    return round(min(1.0, score), 4)


def score_moa_alignment(mechanism: str | None, n_targets: int) -> float:
    """
    Score mechanistic evidence (MOA clarity and target count).

    Returns float in [0, 1].
    """
    if not mechanism:
        return 0.1

    moa_length = len(mechanism.split())
    # More detailed MOA = better mechanistic understanding
    detail_score = min(1.0, moa_length / 30)

    # Target count: more known targets = better characterized
    target_score = min(1.0, math.log1p(n_targets) / math.log1p(10))

    return round(0.6 * detail_score + 0.4 * target_score, 4)


def score_literature(
    pubmed_count: int,
    positive_count: int,
    negative_count: int,
    avg_relevance: float = 0.5,
) -> float:
    """
    Score literature evidence quality.

    Returns float in [0, 1].
    """
    if pubmed_count == 0:
        return 0.0

    # Volume component (log-scaled)
    volume_component = min(1.0, math.log1p(pubmed_count) / math.log1p(50))

    # Signal balance: penalize if many negatives
    if positive_count + negative_count > 0:
        signal_ratio = positive_count / (positive_count + negative_count)
    else:
        signal_ratio = 0.5  # neutral if no explicit classification

    # Relevance component
    relevance_component = min(1.0, avg_relevance)

    score = (
        0.35 * volume_component
        + 0.35 * signal_ratio
        + 0.30 * relevance_component
    )
    return round(min(1.0, score), 4)


def score_pathway_plausibility(alignment: PathwayAlignment | None) -> float:
    """
    Score biological plausibility from pathway validation.

    Returns float in [0, 1].
    """
    if alignment is None:
        return 0.1

    base_scores = {
        AlignmentType.DIRECT: 0.90,
        AlignmentType.INDIRECT: 0.65,
        AlignmentType.SPECULATIVE: 0.15,
    }
    base = base_scores.get(alignment.alignment_type, 0.1)

    # Bonus for having pathway IDs
    pathway_bonus = 0.0
    if alignment.kegg_pathway_ids or alignment.reactome_pathway_ids:
        n_pathways = len(alignment.kegg_pathway_ids) + len(alignment.reactome_pathway_ids)
        pathway_bonus = min(0.1, n_pathways * 0.02)

    # BBB penetration bonus for CNS diseases
    bbb_bonus = 0.0
    if alignment.bbb_penetrant is True:
        bbb_bonus = 0.05
    elif alignment.bbb_penetrant is False:
        bbb_bonus = -0.15  # Penalize if BBB impermeable for CNS disease

    # Use agent's plausibility score as weight
    agent_score = alignment.biological_plausibility_score

    # Blend formula and agent scores
    score = 0.6 * base + 0.3 * agent_score + 0.1 * min(1.0, base + pathway_bonus + bbb_bonus)
    return round(min(1.0, max(0.0, score)), 4)


def score_adversarial(report: AdversarialReport | None) -> float:
    """
    Compute adversarial adjustment (always ≤ 0, penalty).

    Returns float in [-1, 0].
    """
    if report is None:
        return 0.0

    adjustment = 0.0

    # Red flags: severe penalty
    adjustment -= len(report.red_flags) * 0.15

    # Yellow flags: mild penalty
    adjustment -= len(report.yellow_flags) * 0.05

    # Failed trials: compound penalty
    adjustment -= len(report.failed_trials) * 0.10

    # Safety flag
    flag_penalties = {
        SafetyFlag.CLEAN: 0.0,
        SafetyFlag.YELLOW: -0.05,
        SafetyFlag.RED: -0.25,
        SafetyFlag.UNKNOWN: -0.02,
    }
    adjustment += flag_penalties.get(report.safety_flag, -0.02)

    # Cap from agent's own adjustment
    adjustment = min(adjustment, report.confidence_adjustment)

    return round(max(-1.0, adjustment), 4)


def compute_confidence_interval(
    pubmed_count: int,
    n_kg_paths: int,
    has_pathway: bool,
    has_adversarial: bool,
) -> float:
    """
    Compute ±CI based on evidence completeness.
    Fewer evidence sources → wider CI.

    Returns CI value (±this many points around composite score).
    """
    evidence_points = 0
    if pubmed_count >= 5:
        evidence_points += 2
    elif pubmed_count >= 1:
        evidence_points += 1
    if n_kg_paths >= 2:
        evidence_points += 2
    elif n_kg_paths >= 1:
        evidence_points += 1
    if has_pathway:
        evidence_points += 2
    if has_adversarial:
        evidence_points += 1

    # More evidence = smaller CI
    max_points = 7
    coverage = evidence_points / max_points
    ci = 0.25 * (1.0 - coverage) + 0.05  # CI ranges from 0.05 (full evidence) to 0.30 (no evidence)
    return round(ci, 3)


def compute_composite_score(
    kg_paths: list[dict[str, Any]],
    open_targets_score: float,
    shared_targets: list[str],
    mechanism: str | None,
    n_targets: int,
    pubmed_count: int,
    positive_lit_count: int,
    negative_lit_count: int,
    avg_relevance: float,
    pathway_alignment: PathwayAlignment | None,
    adversarial_report: AdversarialReport | None,
) -> ScoreBreakdown:
    """
    Compute the full composite ScoreBreakdown.
    This is the main entry point called by Agent 7.
    """
    settings = get_settings()

    kg = score_kg_proximity(kg_paths, open_targets_score, shared_targets)
    moa = score_moa_alignment(mechanism, n_targets)
    lit = score_literature(pubmed_count, positive_lit_count, negative_lit_count, avg_relevance)
    path = score_pathway_plausibility(pathway_alignment)
    adv = score_adversarial(adversarial_report)

    composite = (
        settings.weight_kg_proximity * kg
        + settings.weight_moa_alignment * moa
        + settings.weight_literature * lit
        + settings.weight_pathway_plausibility * path
        + settings.weight_adversarial_adjustment * adv
    )
    composite = round(min(1.0, max(0.0, composite)), 4)

    ci = compute_confidence_interval(
        pubmed_count=pubmed_count,
        n_kg_paths=len(kg_paths),
        has_pathway=pathway_alignment is not None,
        has_adversarial=adversarial_report is not None,
    )

    logger.info(
        f"Score: kg={kg:.2f} moa={moa:.2f} lit={lit:.2f} "
        f"path={path:.2f} adv={adv:.2f} → composite={composite:.3f} ±{ci:.3f}"
    )

    return ScoreBreakdown(
        kg_proximity=kg,
        moa_alignment=moa,
        literature=lit,
        pathway_plausibility=path,
        adversarial_adjustment=adv,
        composite=composite,
        confidence_interval=ci,
    )
