"""
Composite scoring engine for drug repurposing candidates.
This module is standalone — no CrewAI or LLM dependencies.
"""
from __future__ import annotations

import math
from typing import Any

from loguru import logger

from src.config import get_settings
from src.models.entities import (
    ClinicalFeasibility,
    SafetyFlag,
    ScoreBreakdown,
)


def score_mechanistic(
    kg_paths: list[dict[str, Any]],
    gene_overlap: float,
    n_targets: int,
) -> float:
    """
    Score mechanistic evidence for a drug-disease pair.

    Components:
    - KG path quality: higher score for more/shorter paths
    - Gene overlap between drug targets and disease genes
    - Number of known drug targets (data richness)

    Returns float in [0, 1].
    """
    # KG path component
    if kg_paths:
        path_scores = [p.get("path_score", 0.0) for p in kg_paths]
        best_path_score = max(path_scores)
        n_paths = min(len(kg_paths), 5)
        path_component = 0.7 * best_path_score + 0.3 * (n_paths / 5)
    else:
        path_component = 0.0

    # Gene overlap component
    overlap_component = min(1.0, gene_overlap * 3)  # scale up since Jaccard tends to be small

    # Target richness component
    target_component = min(1.0, math.log1p(n_targets) / math.log1p(20))

    # Weighted combination
    mech_score = (
        0.50 * path_component
        + 0.30 * overlap_component
        + 0.20 * target_component
    )

    return round(min(1.0, mech_score), 4)


def score_literature(
    rag_passages: list[dict[str, Any]],
    pubmed_count: int,
) -> float:
    """
    Score literature evidence for a drug-disease pair.

    Components:
    - RAG retrieval quality (average relevance of top passages)
    - Number of PubMed articles found
    - Recency bonus (passages from recent years)

    Returns float in [0, 1].
    """
    # RAG relevance component
    if rag_passages:
        top_scores = sorted(
            [p.get("relevance_score", 0.0) for p in rag_passages],
            reverse=True,
        )[:5]
        avg_relevance = sum(top_scores) / len(top_scores)
    else:
        avg_relevance = 0.0

    # PubMed hit count component (log scale)
    count_component = min(1.0, math.log1p(pubmed_count) / math.log1p(100))

    # Recency bonus: check if any passage is from 2020+
    recent_bonus = 0.0
    if rag_passages:
        years = []
        for p in rag_passages:
            try:
                y = int(p.get("year", 0))
                years.append(y)
            except (ValueError, TypeError):
                pass
        if years and max(years) >= 2020:
            recent_bonus = 0.1
        if years and max(years) >= 2023:
            recent_bonus = 0.15

    lit_score = (
        0.60 * avg_relevance
        + 0.30 * count_component
        + 0.10 * min(1.0, recent_bonus)
    )

    return round(min(1.0, lit_score), 4)


def score_clinical(
    trial_landscape: dict[str, Any],
    safety_flag: str,
) -> tuple[float, ClinicalFeasibility]:
    """
    Score clinical evidence and feasibility.

    Returns:
        Tuple of (clinical_score float [0,1], ClinicalFeasibility enum)
    """
    signal = trial_landscape.get("clinical_signal", "no_trials")
    total = trial_landscape.get("total", 0)
    completed = trial_landscape.get("completed", 0)
    active = trial_landscape.get("active", 0)
    terminated = trial_landscape.get("terminated", 0)

    # Base score from trial signal
    signal_scores = {
        "strong_evidence": 0.9,
        "emerging_evidence": 0.65,
        "early_stage": 0.35,
        "mixed_evidence": 0.25,
        "no_trials": 0.1,
    }
    base_score = signal_scores.get(signal, 0.1)

    # Penalty for high termination rate
    if total > 0 and terminated / total > 0.5:
        base_score *= 0.7

    # Safety penalty
    safety_penalties = {
        "contraindicated": 0.0,
        "major_concerns": 0.4,
        "minor_concerns": 0.8,
        "clean": 1.0,
        "unknown": 0.9,
    }
    safety_multiplier = safety_penalties.get(safety_flag, 0.9)
    clinical_score = round(base_score * safety_multiplier, 4)

    # Map to ClinicalFeasibility enum
    if clinical_score >= 0.7:
        feasibility = ClinicalFeasibility.HIGH
    elif clinical_score >= 0.4:
        feasibility = ClinicalFeasibility.MEDIUM
    elif clinical_score > 0:
        feasibility = ClinicalFeasibility.LOW
    else:
        feasibility = ClinicalFeasibility.UNKNOWN

    return clinical_score, feasibility


def score_data_confidence(
    has_chembl: bool,
    has_uniprot: bool,
    has_kg_paths: bool,
    has_label: bool,
    pubmed_count: int,
    n_rag_passages: int,
) -> float:
    """
    Score data completeness/confidence (how much data we actually found).

    Returns float in [0, 1].
    """
    checks = [
        has_chembl,
        has_uniprot,
        has_kg_paths,
        has_label,
        pubmed_count >= 5,
        n_rag_passages >= 3,
    ]
    completeness = sum(1 for c in checks if c) / len(checks)
    return round(completeness, 4)


def compute_composite_score(
    kg_paths: list[dict[str, Any]],
    gene_overlap: float,
    n_targets: int,
    rag_passages: list[dict[str, Any]],
    pubmed_count: int,
    trial_landscape: dict[str, Any],
    safety_flag: str,
    has_chembl: bool,
    has_uniprot: bool,
    has_label: bool,
) -> ScoreBreakdown:
    """
    Compute the full composite ScoreBreakdown for a drug-disease candidate.

    This is the main entry point for the Scoring Agent.
    """
    settings = get_settings()

    # Individual component scores
    mech_score = score_mechanistic(kg_paths, gene_overlap, n_targets)
    lit_score = score_literature(rag_passages, pubmed_count)
    clin_score, feasibility = score_clinical(trial_landscape, safety_flag)
    conf_score = score_data_confidence(
        has_chembl=has_chembl,
        has_uniprot=has_uniprot,
        has_kg_paths=bool(kg_paths),
        has_label=has_label,
        pubmed_count=pubmed_count,
        n_rag_passages=len(rag_passages),
    )

    # Parse safety flag
    try:
        safety_enum = SafetyFlag(safety_flag)
    except ValueError:
        safety_enum = SafetyFlag.UNKNOWN

    breakdown = ScoreBreakdown(
        mechanistic_score=mech_score,
        literature_score=lit_score,
        clinical_feasibility_score=clin_score,
        data_confidence_score=conf_score,
        kg_paths_found=len(kg_paths),
        pubmed_hits=pubmed_count,
        active_trials=trial_landscape.get("active", 0),
        safety_flag=safety_enum,
        clinical_feasibility=feasibility,
    )

    # Compute weighted composite
    breakdown.compute_composite(
        w_mech=settings.weight_mechanistic,
        w_lit=settings.weight_literature,
        w_clin=settings.weight_clinical,
        w_conf=settings.weight_data_confidence,
    )

    logger.info(
        f"Composite score: {breakdown.composite_score:.3f} "
        f"(mech={mech_score:.2f}, lit={lit_score:.2f}, "
        f"clin={clin_score:.2f}, conf={conf_score:.2f})"
    )

    return breakdown
