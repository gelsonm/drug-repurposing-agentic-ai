"""
Agent 8: Report Generation Agent

Uses GPT-4o-mini (primary) or Groq llama-3.3-70b (fallback) to synthesize
all evidence into a structured, human-readable repurposing report.

LLM selection order:
  1. OpenAI  — if OPENAI_API_KEY is set
  2. Groq    — if GROQ_API_KEY is set (free tier, no credit card)
  3. No LLM  — deterministic fallback summary used instead

Output format:
  - Executive Summary (top 3 candidates)
  - Per-candidate evidence cards
  - Methodology notes
  - Limitations
  - All references (PMIDs + DB IDs)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from src.config import get_settings
from src.schemas.models import DiseaseProfile, RepurposingReport, ScoredCandidate
from src.tools.llm_client import Provider, get_llm_client, get_provider_display_name, llm_chat


def _candidate_to_context(sc: ScoredCandidate) -> str:
    """Serialize a scored candidate to a concise LLM-ready context string."""
    c = sc.candidate
    score = sc.score
    pa = sc.pathway_alignment
    adv = sc.adversarial_report

    lines = [
        f"Drug: {c.drug_name} (ChEMBL: {c.chembl_id or 'N/A'}, Phase: {c.max_phase})",
        f"Original indication: {c.original_indication or 'Unknown'}",
        f"Mechanism: {c.mechanism_of_action or 'Not characterized'}",
        f"Known targets: {', '.join(c.known_targets[:5]) or 'None'}",
        f"KG proximity score: {c.network_proximity_score:.3f}",
        f"Open Targets score: {c.open_targets_score:.3f}",
        f"Literature: {len(c.literature_pmids)} papers (+{c.literature_positive_count}/-{c.literature_negative_count})",
        f"PMIDs: {', '.join(c.literature_pmids[:5])}",
    ]
    if pa:
        lines += [
            f"Pathway alignment: {pa.alignment_type.value.upper()} (score: {pa.biological_plausibility_score:.3f})",
            f"Mechanistic connection: {pa.connection_description or 'None'}",
            f"KEGG pathways: {', '.join(pa.kegg_pathway_ids[:3])}",
            f"Reactome pathways: {', '.join(pa.reactome_pathway_ids[:3])}",
            f"BBB penetrant: {pa.bbb_penetrant}",
            f"Pathway notes: {pa.validation_notes or 'None'}",
        ]
    if adv:
        lines += [
            f"Adversarial verdict: {adv.overall_verdict}",
            f"Red flags: {'; '.join(adv.red_flags) or 'None'}",
            f"Yellow flags: {'; '.join(adv.yellow_flags[:3]) or 'None'}",
            f"Failed trials: {len(adv.failed_trials)}",
            f"Safety flag: {adv.safety_flag.value}",
        ]
    lines += [
        f"Composite score: {score.composite:.3f} ±{score.confidence_interval:.3f} ({score.rank_label})",
        f"Score breakdown — KG:{score.kg_proximity:.2f} MOA:{score.moa_alignment:.2f} "
        f"Lit:{score.literature:.2f} Pathway:{score.pathway_plausibility:.2f} Adv:{score.adversarial_adjustment:.2f}",
        f"Next step: {sc.recommended_next_step}",
    ]
    return "\n".join(lines)


def _generate_executive_summary(
    disease_name: str,
    top_candidates: list[ScoredCandidate],
    client: Any,
    provider: Provider,
    model: str,
) -> str:
    """Use the available LLM to write an executive summary."""
    candidates_context = "\n\n---\n".join(
        f"#{i+1} {_candidate_to_context(sc)}" for i, sc in enumerate(top_candidates[:3])
    )

    prompt = f"""You are a senior computational drug repurposing scientist. 
Write a concise executive summary (3–5 paragraphs) for a drug repurposing analysis of {disease_name}.

Top candidates are:
{candidates_context}

Requirements:
- Cite specific drug names, scores, and mechanistic evidence
- Reference pathway connections (e.g., KEGG:hsa04150) when available
- Cite PMIDs when available
- Mention confidence intervals
- Flag any adversarial concerns
- Be scientifically precise but accessible to a pharmaceutical scientist

Write only the summary, no headers."""

    try:
        return llm_chat(client, provider, [{"role": "user", "content": prompt}], model=model, max_tokens=800, temperature=0.2)
    except Exception as e:
        logger.warning(f"[Agent 8] LLM executive summary failed: {e}")
        return _fallback_summary(disease_name, top_candidates)


def _generate_candidate_rationale(
    sc: ScoredCandidate,
    disease_name: str,
    client: Any,
    provider: Provider,
    model: str,
) -> str:
    """Use the available LLM to write a biological rationale for one candidate."""
    context = _candidate_to_context(sc)
    prompt = f"""You are a computational pharmacologist. 
Write a detailed biological rationale (2–3 paragraphs) for why {sc.candidate.drug_name} 
might be repurposed for {disease_name}.

Evidence available:
{context}

Requirements:
- Focus on mechanistic pathway connections
- Cite KEGG/Reactome pathway IDs when mentioned
- Cite PMIDs for literature claims
- Acknowledge pathway alignment type (direct/indirect/speculative)
- Mention confidence score and what it means
- Be honest about limitations

Write only the rationale, no headers."""

    try:
        return llm_chat(client, provider, [{"role": "user", "content": prompt}], model=model, max_tokens=500, temperature=0.2)
    except Exception as e:
        logger.warning(f"[Agent 8] LLM rationale failed for {sc.candidate.drug_name}: {e}")
        return sc.one_line_justification or "Rationale not available."


def _fallback_summary(disease_name: str, candidates: list[ScoredCandidate]) -> str:
    """Fallback summary without LLM."""
    top = candidates[:3]
    lines = [f"Drug repurposing analysis for {disease_name} identified {len(candidates)} candidates."]
    for i, sc in enumerate(top):
        lines.append(
            f"#{i+1}: {sc.candidate.drug_name} (score: {sc.score.composite:.2f} ±{sc.score.confidence_interval:.2f}, "
            f"{sc.score.rank_label}). {sc.one_line_justification}"
        )
    return " ".join(lines)


def run_report_agent(
    disease_name: str,
    disease_profile: DiseaseProfile | None,
    scored_candidates: list[ScoredCandidate],
    runtime_seconds: float | None = None,
    use_llm: bool = True,
) -> RepurposingReport:
    """
    Execute Agent 8: Report Generation.

    Args:
        disease_name: Target disease
        disease_profile: From Agent 1
        scored_candidates: From Agent 7 (pre-sorted)
        runtime_seconds: Total pipeline runtime
        use_llm: Whether to use GPT-4o-mini for narrative generation

    Returns:
        Complete RepurposingReport
    """
    settings = get_settings()
    logger.info(f"[Agent 8] Generating report for '{disease_name}' ({len(scored_candidates)} candidates)")

    client = None
    provider: Provider = "openai"
    model = settings.openai_model

    if use_llm:
        try:
            client, provider = get_llm_client()
            model = settings.openai_model if provider == "openai" else settings.groq_model
            logger.info(f"[Agent 8] Using LLM provider: {get_provider_display_name(provider)}")
        except RuntimeError as e:
            logger.warning(f"[Agent 8] No LLM available ({e}) — using deterministic fallback")
            client = None

    top_candidates = scored_candidates[:5]

    # Executive summary
    if client:
        executive_summary = _generate_executive_summary(disease_name, top_candidates, client, provider, model)
    else:
        executive_summary = _fallback_summary(disease_name, top_candidates)

    # Generate per-candidate rationales
    for sc in top_candidates:
        if client and len(sc.one_line_justification or "") < 100:
            sc.one_line_justification = _generate_candidate_rationale(sc, disease_name, client, provider, model)

    # Collect all references
    all_refs = set()
    for sc in scored_candidates:
        all_refs.update(sc.all_source_ids)

    report = RepurposingReport(
        disease_name=disease_name,
        disease_profile=disease_profile,
        candidates=scored_candidates,
        executive_summary=executive_summary,
        methodology_notes=(
            "Analysis performed using Drug Repurposing v2 pipeline: "
            "8-agent sequential system with Open Targets, Hetionet KG, "
            "PubMed RAG, KEGG/Reactome pathway validation, and adversarial critique. "
            f"LLM: {get_provider_display_name(provider) if client else 'None (deterministic fallback)'}. "
            "Scoring: weighted composite (KG×0.20 + MOA×0.25 + "
            "Literature×0.20 + Pathway×0.25 + Adversarial×0.10)."
        ),
        limitations=[
            "All predictions are computational hypotheses requiring experimental validation",
            "KEGG/Reactome pathway mapping may be incomplete for novel drug-target interactions",
            "DisGeNET/Open Targets associations reflect available evidence as of database snapshot",
            "LLM-generated narrative (Agent 8) may contain inaccuracies despite source grounding",
            "BBB penetration data is from published pharmacological profiles — may not apply to all doses",
            "Scoring weights are heuristic, not calibrated against a benchmark dataset",
            "ClinicalTrials.gov query may miss trials with non-standard disease terminology",
        ],
        all_references=sorted(list(all_refs)),
        total_runtime_seconds=runtime_seconds,
        pipeline_version="2.0.0",
    )

    logger.info(f"[Agent 8] Report generated: {len(scored_candidates)} candidates, {len(all_refs)} references")
    return report


def report_to_markdown(report: RepurposingReport) -> str:
    """Convert RepurposingReport to a Markdown string for export."""
    lines = [
        f"# Drug Repurposing Report: {report.disease_name}",
        f"*Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')} | Pipeline v{report.pipeline_version}*",
        "",
        "## Executive Summary",
        report.executive_summary or "N/A",
        "",
        "---",
        "",
        "## Candidate Detail Cards",
        "",
    ]

    for i, sc in enumerate(report.top_candidates):
        c = sc.candidate
        score = sc.score
        pa = sc.pathway_alignment
        adv = sc.adversarial_report

        lines += [
            f"### #{i+1} {c.drug_name} — Score: {score.composite:.3f} ±{score.confidence_interval:.3f} ({score.rank_label})",
            "",
            f"**Drug Profile:** {c.drug_name} (ChEMBL: `{c.chembl_id or 'N/A'}`, Phase: {c.max_phase})",
            f"**Original Indication:** {c.original_indication or 'Unknown'}",
            f"**Mechanism of Action:** {c.mechanism_of_action or 'Not characterized'}",
            f"**Known Targets:** {', '.join(c.known_targets[:5]) or 'N/A'}",
            "",
            "**Score Breakdown:**",
            f"| Component | Score | Weight |",
            f"|-----------|-------|--------|",
            f"| KG Proximity | {score.kg_proximity:.3f} | 20% |",
            f"| MOA Alignment | {score.moa_alignment:.3f} | 25% |",
            f"| Literature | {score.literature:.3f} | 20% |",
            f"| Pathway Plausibility | {score.pathway_plausibility:.3f} | 25% |",
            f"| Adversarial Adjustment | {score.adversarial_adjustment:.3f} | 10% |",
            f"| **Composite** | **{score.composite:.3f} ±{score.confidence_interval:.3f}** | — |",
            "",
        ]

        if pa:
            lines += [
                f"**Pathway Alignment:** {pa.alignment_type.value.upper()} (plausibility: {pa.biological_plausibility_score:.3f})",
                f"**Mechanistic Connection:** {pa.connection_description or 'N/A'}",
                f"**KEGG Pathways:** {', '.join(f'`{p}`' for p in pa.kegg_pathway_ids[:3]) or 'None'}",
                f"**Reactome Pathways:** {', '.join(f'`{p}`' for p in pa.reactome_pathway_ids[:3]) or 'None'}",
                f"**BBB Penetrant:** {'✅ Yes' if pa.bbb_penetrant else ('❌ No' if pa.bbb_penetrant is False else '❓ Unknown')}",
                "",
            ]

        if adv:
            safety_emoji = {"clean": "✅", "yellow": "⚠️", "red": "🔴", "unknown": "❓"}[adv.safety_flag.value]
            lines += [
                f"**Adversarial Verdict:** {adv.overall_verdict} {safety_emoji}",
            ]
            if adv.red_flags:
                lines.append("**🔴 Red Flags:**")
                for rf in adv.red_flags:
                    lines.append(f"  - {rf}")
            if adv.yellow_flags:
                lines.append("**⚠️ Yellow Flags:**")
                for yf in adv.yellow_flags[:3]:
                    lines.append(f"  - {yf}")
            if adv.failed_trials:
                lines.append(f"**Failed Trials:** {', '.join(t.nct_id for t in adv.failed_trials[:3])}")
            lines.append("")

        lines += [
            f"**Literature:** {len(c.literature_pmids)} papers (+{c.literature_positive_count}/-{c.literature_negative_count})",
            f"**PMIDs:** {', '.join(c.literature_pmids[:5]) or 'None'}",
            "",
            f"**Biological Rationale:**",
            sc.one_line_justification or "N/A",
            "",
            f"**Recommended Next Step:** {sc.recommended_next_step or 'N/A'}",
            "",
            "---",
            "",
        ]

    lines += [
        "## Methodology & Data Sources",
        report.methodology_notes or "",
        "",
        "## Limitations & Caveats",
    ]
    for lim in report.limitations:
        lines.append(f"- {lim}")

    lines += [
        "",
        "## All References",
        "",
    ]
    for ref in sorted(report.all_references):
        lines.append(f"- `{ref}`")

    return "\n".join(lines)
