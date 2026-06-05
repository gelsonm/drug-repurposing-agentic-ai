"""
Agent 7: Scoring & Ranking Agent

Aggregates all evidence from Agents 1-6 into a final composite score
with confidence intervals. Uses the deterministic composite_scorer module.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from src.schemas.models import (
    AdversarialReport,
    CandidateDrug,
    PathwayAlignment,
    ScoreBreakdown,
    ScoredCandidate,
)
from src.scoring.composite_scorer import compute_composite_score
from src.validation.hallucination_guard import validate_candidate_sources


def run_scorer_agent(
    candidate: CandidateDrug,
    pathway_alignment: PathwayAlignment | None,
    adversarial_report: AdversarialReport | None,
    rag_passages: list[dict[str, Any]] | None = None,
) -> ScoredCandidate:
    """
    Execute Agent 7: Score & Rank a single candidate.

    Returns:
        ScoredCandidate with composite score and justification.
    """
    drug_name = candidate.drug_name
    logger.info(f"[Agent 7] Scoring: {drug_name}")

    # Gather avg relevance from RAG passages
    avg_relevance = 0.5
    if rag_passages:
        scores = [p.get("relevance_score", 0.5) for p in rag_passages if p.get("relevance_score")]
        avg_relevance = sum(scores) / len(scores) if scores else 0.5

    # Compute composite score
    score = compute_composite_score(
        kg_paths=[e.dict() for e in candidate.kg_evidence],
        open_targets_score=candidate.open_targets_score,
        shared_targets=candidate.known_targets[:5],
        mechanism=candidate.mechanism_of_action,
        n_targets=len(candidate.known_targets),
        pubmed_count=len(candidate.literature_pmids),
        positive_lit_count=candidate.literature_positive_count,
        negative_lit_count=candidate.literature_negative_count,
        avg_relevance=avg_relevance,
        pathway_alignment=pathway_alignment,
        adversarial_report=adversarial_report,
    )

    # Build all source IDs for hallucination guard
    source_ids = []
    if candidate.chembl_id:
        source_ids.append(candidate.chembl_id)
    if candidate.drugbank_id:
        source_ids.append(candidate.drugbank_id)
    source_ids.extend(candidate.literature_pmids[:5])
    if pathway_alignment:
        source_ids.extend(pathway_alignment.kegg_pathway_ids[:3])
        source_ids.extend(pathway_alignment.reactome_pathway_ids[:3])
    if adversarial_report:
        source_ids.extend([t.nct_id for t in adversarial_report.failed_trials[:3]])

    source_ids = [s for s in source_ids if s]

    # Validate hallucination guard
    guard_result = validate_candidate_sources(drug_name, source_ids)
    if not guard_result["pass"]:
        logger.warning(f"[Agent 7] HALLUCINATION GUARD FAILED for {drug_name}: no valid source IDs")
        score.composite = min(score.composite, 0.3)  # Cap score for unsourced candidates

    # Build justification
    justification = _build_justification(candidate, pathway_alignment, adversarial_report, score)

    # Recommended next step
    next_step = _recommend_next_step(score, adversarial_report, pathway_alignment)

    scored = ScoredCandidate(
        candidate=candidate,
        pathway_alignment=pathway_alignment,
        adversarial_report=adversarial_report,
        score=score,
        one_line_justification=justification,
        recommended_next_step=next_step,
        all_source_ids=list(dict.fromkeys(source_ids)),
    )

    logger.info(
        f"[Agent 7] {drug_name}: composite={score.composite:.3f} ±{score.confidence_interval:.3f} "
        f"({score.rank_label})"
    )
    return scored


def rank_candidates(scored: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """Sort scored candidates by composite score descending."""
    return sorted(scored, key=lambda s: s.score.composite, reverse=True)


def _build_justification(
    candidate: CandidateDrug,
    pathway: PathwayAlignment | None,
    adversarial: AdversarialReport | None,
    score: ScoreBreakdown,
) -> str:
    """Build a one-line evidence-based justification."""
    parts = []

    if candidate.known_targets:
        parts.append(f"targets {', '.join(candidate.known_targets[:2])}")

    if pathway:
        parts.append(f"{pathway.alignment_type.value} pathway alignment")

    if candidate.literature_pmids:
        n = len(candidate.literature_pmids)
        pos = candidate.literature_positive_count
        parts.append(f"{n} papers ({pos} positive)")

    if adversarial and adversarial.safety_flag.value == "clean":
        parts.append("no safety flags")
    elif adversarial and adversarial.red_flags:
        parts.append(f"{len(adversarial.red_flags)} red flag(s)")

    body = "; ".join(parts) if parts else "limited evidence"
    return f"{candidate.drug_name} — {body} — score {score.composite:.2f} ±{score.confidence_interval:.2f}"


def _recommend_next_step(
    score: ScoreBreakdown,
    adversarial: AdversarialReport | None,
    pathway: PathwayAlignment | None,
) -> str:
    """Recommend the most appropriate next step based on evidence."""
    if adversarial and adversarial.safety_flag.value == "red":
        return "Review red flags before proceeding. Consult clinical pharmacologist."

    if score.composite >= 0.70:
        return (
            "High priority: Design in vitro validation assay targeting shared pathway genes. "
            "Consider retrospective cohort analysis using EHR data."
        )
    elif score.composite >= 0.50:
        return (
            "Moderate priority: Run phenotypic screen in relevant disease cell model. "
            "Perform systematic literature review on mechanism."
        )
    elif score.composite >= 0.35:
        return (
            "Weak evidence: Conduct computational docking study against key disease targets. "
            "Gather more primary literature before experimental investment."
        )
    else:
        return (
            "Low priority: Computational hypothesis only. "
            "Requires substantial mechanistic evidence before experimental consideration."
        )
