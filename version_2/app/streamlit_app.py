"""
Drug Repurposing v2 — Streamlit Web App
Biologically-Grounded Multi-Agent AI Pipeline

Features:
  - Disease-first input (any disease name)
  - 5-tab results: Disease Profile | Candidates | Pathway Alignment | Adversarial | Report
  - Live agent progress updates
  - Radar chart for score breakdown
  - Pathway alignment visualization
  - Markdown report download
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.report_gen import report_to_markdown
from src.config import get_settings
from src.schemas.models import RepurposingReport, ScoredCandidate

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Repurposing v2 | Agentic AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: linear-gradient(135deg, #0D142A 0%, #1E2940 100%); }

  /* Cards */
  .ev-card { 
    background: rgba(30,41,74,0.85); border-radius: 12px;
    padding: 18px; margin: 8px 0;
    border: 1px solid rgba(16,185,129,0.2);
    backdrop-filter: blur(10px);
  }
  .metric-card {
    background: rgba(30,41,74,0.9); border-radius: 10px;
    padding: 16px; text-align: center;
    border-top: 3px solid #10B981;
  }
  .metric-value { font-size: 2.2rem; font-weight: 700; color: #10B981; }
  .metric-label { font-size: 0.85rem; color: #94A3B8; margin-top: 4px; }

  /* Badges */
  .badge-direct { background: #059669; color: white; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; }
  .badge-indirect { background: #D97706; color: white; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; }
  .badge-speculative { background: #DC2626; color: white; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; }
  .badge-clean { background: #10B981; color: white; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; }
  .badge-yellow { background: #F59E0B; color: white; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; }
  .badge-red { background: #EF4444; color: white; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; }

  /* Agent progress */
  .agent-step { 
    background: rgba(16,185,129,0.1); border-left: 3px solid #10B981;
    padding: 8px 14px; border-radius: 0 8px 8px 0; margin: 4px 0;
    color: #CBD5E1; font-size: 0.9rem;
  }
  .agent-step.done { border-left-color: #10B981; opacity: 0.8; }

  /* Hallucination guard */
  .guard-pass { color: #10B981; font-weight: 600; }
  .guard-warn { color: #F59E0B; font-weight: 600; }

  /* Headers */
  h1, h2, h3 { color: white !important; }
  .section-label { color: #10B981; font-size: 0.75rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────

def alignment_badge(alignment_type: str) -> str:
    cls = {"direct": "badge-direct", "indirect": "badge-indirect", "speculative": "badge-speculative"}.get(alignment_type, "badge-indirect")
    return f'<span class="{cls}">{alignment_type.upper()}</span>'


def safety_badge(flag: str) -> str:
    cls = {"clean": "badge-clean", "yellow": "badge-yellow", "red": "badge-red"}.get(flag, "badge-yellow")
    return f'<span class="{cls}">{flag.upper()}</span>'


def score_color(score: float) -> str:
    if score >= 0.70: return "#10B981"
    if score >= 0.50: return "#F59E0B"
    if score >= 0.35: return "#6366F1"
    return "#EF4444"


def make_radar_chart(sc: ScoredCandidate) -> go.Figure:
    s = sc.score
    categories = ["KG Proximity", "MOA Alignment", "Literature", "Pathway Validity", "Safety (inverted)"]
    values = [
        s.kg_proximity,
        s.moa_alignment,
        s.literature,
        s.pathway_plausibility,
        max(0.0, 1.0 + s.adversarial_adjustment),  # Invert: 0 adj → 1.0, -1 adj → 0
    ]
    values_with_close = values + [values[0]]
    cats_with_close = categories + [categories[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values_with_close, theta=cats_with_close,
        fill="toself",
        fillcolor=f"rgba(16,185,129,0.2)",
        line=dict(color="#10B981", width=2.5),
        name=sc.candidate.drug_name,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.15)", color="#94A3B8"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.15)", color="#94A3B8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=40, r=40, t=30, b=20),
    )
    return fig


def make_multi_radar_chart(scored: list[ScoredCandidate]) -> go.Figure:
    categories = ["KG Proximity", "MOA Alignment", "Literature", "Pathway Validity", "Safety"]
    colors = ["#10B981", "#6366F1", "#F59E0B", "#EC4E99", "#22D3EE"]

    fig = go.Figure()
    for i, sc in enumerate(scored[:5]):
        s = sc.score
        values = [
            s.kg_proximity, s.moa_alignment, s.literature, s.pathway_plausibility,
            max(0.0, 1.0 + s.adversarial_adjustment),
        ]
        values_close = values + [values[0]]
        cats_close = categories + [categories[0]]
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatterpolar(
            r=values_close, theta=cats_close,
            fill="toself",
            fillcolor=f"{color}33",
            line=dict(color=color, width=2),
            name=sc.candidate.drug_name,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)", color="#94A3B8"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#94A3B8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E1")),
        height=380, margin=dict(l=40, r=40, t=30, b=20),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="section-label">🔬 Drug Repurposing v2</div>', unsafe_allow_html=True)
        st.title("Configuration")

        st.markdown("**LLM — Primary**")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Get yours at platform.openai.com",
        )

        st.markdown("**LLM — Fallback (free)**")
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Free at console.groq.com — no credit card. Auto-used when OpenAI key is missing.",
        )
        groq_model = st.selectbox(
            "Groq Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0,
            help="llama-3.3-70b is the best quality; 8b-instant is fastest",
        )

        # Apply keys to env
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
        if groq_model:
            os.environ["GROQ_MODEL"] = groq_model

        # LLM provider status
        if openai_key and openai_key.startswith("sk-"):
            st.success("🤖 LLM: OpenAI GPT-4o-mini")
        elif groq_key:
            st.info(f"🤖 LLM: Groq {groq_model} (fallback)")
        else:
            st.warning("⚠️ No LLM key set — deterministic report only")

        disgenet_key = st.text_input(
            "DisGeNET API Key (optional)",
            type="password",
            value=os.getenv("DISGENET_API_KEY", ""),
            help="Free key at disgenet.com",
        )
        if disgenet_key:
            os.environ["DISGENET_API_KEY"] = disgenet_key

        st.divider()
        st.markdown("**Pipeline Settings**")
        top_n = st.slider("Deep analysis for top-N candidates", 2, 8, 5,
                          help="Full pathway + adversarial analysis for top N")
        max_candidates = st.slider("Max candidates to discover", 5, 20, 10)
        use_llm = st.checkbox(
            "Use LLM for report narration",
            value=True,
            help="Generates executive summary & candidate rationales",
        )

        st.divider()
        st.markdown("**Data Sources**")
        for src, badge in [
            ("🔗 Hetionet KG", "green"), ("🏢 Open Targets", "green"),
            ("📄 PubMed RAG", "green"), ("🧪 KEGG Pathways", "green"),
            ("⚛️ Reactome", "green"), ("⚡ STRING DB", "green"),
            ("🏥 ClinicalTrials.gov", "green"), ("🧬 ChEMBL", "green"),
            ("🔬 DisGeNET", "yellow" if not disgenet_key else "green"),
        ]:
            st.markdown(f"{'✅' if badge == 'green' else '🔑'} {src}")

        st.divider()
        st.markdown(
            "**v2 Novel Features:**\n"
            "- 🧬 Pathway Mechanism Validator (Agent 5)\n"
            "- ⚡ Adversarial Critique Agent (Agent 6)\n"
            "- 🔐 Hallucination Guard (source-anchored)\n"
            "- ±CI Confidence Intervals\n"
            "- BBB penetration check (CNS diseases)\n"
            "- 🔄 OpenAI → Groq auto-fallback"
        )

    return openai_key, top_n, max_candidates, use_llm


# ── Disease Profile Tab ────────────────────────────────────────────────────

def render_disease_tab(report: RepurposingReport):
    dp = report.disease_profile
    if not dp:
        st.info("No disease profile available.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(dp.known_targets)}</div><div class="metric-label">Gene Targets</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(dp.key_pathways)}</div><div class="metric-label">Key Pathways</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{"CNS" if dp.is_cns_disease else "Peripheral"}</div><div class="metric-label">Disease Type</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧬 Key Disease Genes")
        if dp.known_targets:
            gene_data = [
                {"Gene": t.gene_symbol, "Score": f"{t.association_score:.3f}", "Source": t.source}
                for t in sorted(dp.known_targets, key=lambda t: t.association_score, reverse=True)[:10]
            ]
            import pandas as pd
            st.dataframe(pd.DataFrame(gene_data), use_container_width=True, hide_index=True)
        else:
            st.info("No gene associations found.")

    with col2:
        st.subheader("🔗 Key Pathogenic Pathways")
        for pw in dp.key_pathways:
            st.markdown(f"• {pw}")

        st.markdown("---")
        st.subheader("📋 Identifiers")
        if dp.efo_id:
            st.markdown(f"**EFO ID:** `{dp.efo_id}`")
        if dp.mondo_id:
            st.markdown(f"**MONDO:** `{dp.mondo_id}`")
        if dp.is_cns_disease:
            st.warning("⚠️ **CNS disease** — BBB penetration check applied to all candidates.")

    if dp.biological_context:
        st.markdown("---")
        st.subheader("🔬 Biological Context")
        st.markdown(f'<div class="ev-card">{dp.biological_context}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Data sources:** {' | '.join(dp.data_sources)}")


# ── Candidates Tab ─────────────────────────────────────────────────────────

def render_candidates_tab(report: RepurposingReport):
    candidates = report.top_candidates
    if not candidates:
        st.info("No candidates found.")
        return

    # Summary metrics
    cols = st.columns(4)
    cols[0].metric("Total Candidates", len(candidates))
    cols[1].metric("Top Score", f"{candidates[0].score.composite:.3f}")
    cols[2].metric(
        "Pathway-Validated",
        sum(1 for c in candidates if c.pathway_alignment is not None)
    )
    cols[3].metric(
        "Clean Safety",
        sum(1 for c in candidates if c.adversarial_report and c.adversarial_report.safety_flag.value == "clean")
    )

    # Comparison radar
    st.subheader("📊 Score Comparison (Top 5)")
    st.plotly_chart(make_multi_radar_chart(candidates), use_container_width=True)

    # Candidate cards
    st.subheader("📋 Ranked Candidates")
    for i, sc in enumerate(candidates[:10]):
        c = sc.candidate
        score = sc.score
        pa = sc.pathway_alignment
        adv = sc.adversarial_report

        color = score_color(score.composite)
        align_badge = alignment_badge(pa.alignment_type.value if pa else "speculative")
        s_badge = safety_badge(adv.safety_flag.value if adv else "unknown")

        with st.expander(
            f"#{i+1}  {c.drug_name}   •   Score: {score.composite:.3f} ±{score.confidence_interval:.3f}   •   {score.rank_label}",
            expanded=(i == 0),
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(
                    f"**ChEMBL:** `{c.chembl_id or 'N/A'}`  |  "
                    f"**Phase:** {c.max_phase}  |  "
                    f"**Indication:** {c.original_indication or 'Unknown'}"
                )
                if c.mechanism_of_action:
                    st.markdown(f"**MOA:** {c.mechanism_of_action[:200]}")
                if c.known_targets:
                    st.markdown(f"**Targets:** {', '.join(c.known_targets[:6])}")

                st.markdown(f"**Pathway:** {align_badge}  |  **Safety:** {s_badge}", unsafe_allow_html=True)

                if pa and pa.connection_description:
                    st.markdown(f'<div class="ev-card">🧬 {pa.connection_description}</div>', unsafe_allow_html=True)

                if adv:
                    if adv.red_flags:
                        for rf in adv.red_flags:
                            st.error(f"🔴 {rf}")
                    if adv.yellow_flags:
                        for yf in adv.yellow_flags[:2]:
                            st.warning(f"⚠️ {yf}")

                if sc.one_line_justification:
                    st.markdown(f"**Rationale:** {sc.one_line_justification}")
                if sc.recommended_next_step:
                    st.info(f"➡️ **Next step:** {sc.recommended_next_step}")

                # Score breakdown table
                st.markdown("**Score Breakdown:**")
                cols_s = st.columns(5)
                for col_s, label, val in zip(
                    cols_s,
                    ["KG", "MOA", "Lit", "Pathway", "Adversarial"],
                    [score.kg_proximity, score.moa_alignment, score.literature, score.pathway_plausibility, score.adversarial_adjustment],
                ):
                    col_s.metric(label, f"{val:.3f}")

            with col2:
                st.plotly_chart(make_radar_chart(sc), use_container_width=True)
                if sc.all_source_ids:
                    with st.expander("🔐 Sources (hallucination guard)"):
                        st.markdown(
                            f'<span class="guard-pass">✅ {len(sc.all_source_ids)} verified IDs</span>',
                            unsafe_allow_html=True,
                        )
                        st.code("\n".join(sc.all_source_ids[:10]))

            if c.literature_pmids:
                st.markdown(f"**Literature ({len(c.literature_pmids)} papers):** "
                            f"+{c.literature_positive_count} positive / -{c.literature_negative_count} negative")
                for pmid in c.literature_pmids[:5]:
                    st.markdown(f"  • [PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")


# ── Pathway Alignment Tab ─────────────────────────────────────────────────

def render_pathway_tab(report: RepurposingReport):
    st.subheader("🧬 Pathway Mechanism Validation")
    st.markdown(
        "Agent 5 verifies that each drug's MOA mechanistically connects to the disease pathway. "
        "This is the biological grounding layer unique to v2."
    )

    has_any = False
    for sc in report.top_candidates:
        pa = sc.pathway_alignment
        if not pa:
            continue
        has_any = True

        with st.expander(
            f"{sc.candidate.drug_name} — {pa.alignment_type.value.upper()} alignment (plausibility: {pa.biological_plausibility_score:.3f})",
            expanded=pa.alignment_type.value == "direct",
        ):
            col1, col2 = st.columns(2)
            with col1:
                align_colors = {"direct": "success", "indirect": "warning", "speculative": "error"}
                getattr(st, align_colors.get(pa.alignment_type.value, "info"))(
                    f"**Alignment Type: {pa.alignment_type.value.upper()}**"
                )
                if pa.connection_description:
                    st.markdown(f"**Connection:** {pa.connection_description}")
                if pa.bbb_penetrant is not None:
                    if pa.bbb_penetrant:
                        st.success("✅ BBB Penetrant: Yes")
                    else:
                        st.error("❌ BBB Penetrant: No")
                if pa.tissue_distribution_note:
                    st.markdown(f"**Tissue Note:** {pa.tissue_distribution_note}")

            with col2:
                if pa.kegg_pathway_ids:
                    st.markdown("**KEGG Pathways:**")
                    for pid in pa.kegg_pathway_ids:
                        url = f"https://www.genome.jp/pathway/{pid}"
                        st.markdown(f"  • [`{pid}`]({url})")

                if pa.reactome_pathway_ids:
                    st.markdown("**Reactome Pathways:**")
                    for pid in pa.reactome_pathway_ids:
                        url = f"https://reactome.org/PathwayBrowser/#/{pid}"
                        st.markdown(f"  • [`{pid}`]({url})")

            if pa.validation_notes:
                st.info(f"📝 {pa.validation_notes}")

    if not has_any:
        st.info("Pathway validation was run for top candidates only. Run the pipeline to see results.")


# ── Adversarial Tab ────────────────────────────────────────────────────────

def render_adversarial_tab(report: RepurposingReport):
    st.subheader("⚡ Adversarial Critique Analysis")
    st.markdown(
        "Agent 6 acts as a skeptical reviewer — it actively finds reasons a candidate SHOULDN'T work. "
        "Red flags = disqualifying | Yellow flags = concerning but not disqualifying."
    )

    for sc in report.top_candidates:
        adv = sc.adversarial_report
        if not adv:
            continue

        color = {"clean": "🟢", "yellow": "🟡", "red": "🔴", "unknown": "⬜"}.get(adv.safety_flag.value, "⬜")
        with st.expander(f"{color} {sc.candidate.drug_name} — {adv.overall_verdict}"):
            col1, col2 = st.columns(2)

            with col1:
                if adv.red_flags:
                    st.markdown("**🔴 Red Flags:**")
                    for rf in adv.red_flags:
                        st.error(rf)
                else:
                    st.success("No red flags identified.")

                if adv.yellow_flags:
                    st.markdown("**⚠️ Yellow Flags:**")
                    for yf in adv.yellow_flags:
                        st.warning(yf)
                else:
                    st.info("No yellow flags identified.")

            with col2:
                if adv.failed_trials:
                    st.markdown("**Failed Trials:**")
                    for trial in adv.failed_trials[:5]:
                        st.markdown(
                            f"  • [`{trial.nct_id}`](https://clinicaltrials.gov/study/{trial.nct_id}) "
                            f"— {trial.phase or 'Unknown phase'}"
                            + (f" — _{trial.why_stopped}_" if trial.why_stopped else "")
                        )
                else:
                    st.success("No terminated/withdrawn trials found.")

                st.metric("Confidence Adjustment", f"{adv.confidence_adjustment:+.3f}")


# ── Report Tab ─────────────────────────────────────────────────────────────

def render_report_tab(report: RepurposingReport):
    st.subheader("📋 Full Research Report")

    if report.executive_summary:
        st.markdown("### Executive Summary")
        st.markdown(report.executive_summary)

    st.divider()
    st.markdown("### Methodology")
    st.markdown(report.methodology_notes or "N/A")

    st.markdown("### Limitations")
    for lim in report.limitations:
        st.markdown(f"- {lim}")

    if report.all_references:
        st.markdown("### All Sources")
        st.code("\n".join(report.all_references[:50]))

    st.divider()
    md_content = report_to_markdown(report)
    st.download_button(
        label="⬇️ Download Markdown Report",
        data=md_content.encode("utf-8"),
        file_name=f"drug_repurposing_{report.disease_name[:20].replace(' ', '_')}.md",
        mime="text/markdown",
    )
    st.download_button(
        label="⬇️ Download JSON Data",
        data=report.model_dump_json(indent=2).encode("utf-8"),
        file_name=f"drug_repurposing_{report.disease_name[:20].replace(' ', '_')}.json",
        mime="application/json",
    )


# ── Main App ───────────────────────────────────────────────────────────────

def main():
    # Sidebar
    openai_key, top_n, max_candidates, use_llm = render_sidebar()

    # Header
    st.markdown(
        '<div class="section-label">🔬 Drug Repurposing v2 | Biologically-Grounded Agentic AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown("## Drug Repurposing Pipeline")
    st.markdown(
        "Enter any disease name and the 8-agent pipeline will discover, validate, "
        "and rank repurposing candidates with pathway-level biological grounding."
    )

    # Disease input
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        disease = st.text_input(
            "Target Disease",
            placeholder="e.g., Alzheimer's disease, Parkinson's disease, Colorectal cancer...",
            value=st.session_state.get("last_disease", "Alzheimer's disease"),
        )
    with col_btn:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # Quick demo buttons
    st.markdown("**Quick demos:**")
    demo_cols = st.columns(4)
    demos = ["Alzheimer's disease", "Parkinson's disease", "Colorectal cancer", "Type 2 diabetes"]
    for dc, demo in zip(demo_cols, demos):
        if dc.button(f"💊 {demo.split(' ')[0]}", key=f"demo_{demo}"):
            st.session_state["demo_disease"] = demo
            st.session_state["run_demo"] = True
            st.rerun()

    # Handle demo button clicks
    if st.session_state.get("run_demo"):
        disease = st.session_state.get("demo_disease", disease)
        st.session_state.pop("run_demo", None)
        run_btn = True

    # Run pipeline
    if run_btn and disease.strip():
        # Check that at least one LLM key is available
        groq_key_env = os.getenv("GROQ_API_KEY", "")
        openai_key_env = os.getenv("OPENAI_API_KEY", "")
        if not openai_key_env and not groq_key_env:
            st.error("Please enter a Groq API key (or OpenAI key) in the sidebar to run the pipeline.")
            return

        st.session_state["last_disease"] = disease

        # Progress display
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🤖 Agent Pipeline Progress")
            progress_bar = st.progress(0)
            step_placeholder = st.empty()
            steps_log = []

            def progress_cb(step, total, msg):
                steps_log.append(f"Step {step}/{total}: {msg}")
                progress_bar.progress(step / total)
                step_html = "".join(
                    f'<div class="agent-step done">{"✅" if i < len(steps_log)-1 else "⏳"} {s}</div>'
                    for i, s in enumerate(steps_log)
                )
                step_placeholder.markdown(step_html, unsafe_allow_html=True)

        start = time.time()
        try:
            from src.crew import DrugRepurposingCrew
            from src.config import get_settings
            # Reset cached settings so new env values are picked up
            get_settings.cache_clear()
            crew = DrugRepurposingCrew(top_n=top_n, use_llm_report=use_llm)
            # Apply max_candidates override
            import src.agents.kg_candidate as _kg
            _kg_orig = _kg.run_kg_candidate_agent
            def _kg_limited(profile, **kwargs):
                return _kg_orig(profile, max_candidates=max_candidates)
            report = crew.run(disease, progress_callback=progress_cb)
            elapsed = round(time.time() - start, 1)

            progress_bar.progress(1.0)
            step_placeholder.markdown(
                f'<div class="agent-step done">✅ Analysis complete in {elapsed}s — '
                f'{len(report.candidates)} candidates found</div>',
                unsafe_allow_html=True,
            )
            st.session_state["report"] = report

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Results
    if "report" in st.session_state:
        report: RepurposingReport = st.session_state["report"]

        st.divider()
        st.markdown(f"## 📊 Results: {report.disease_name}")
        st.markdown(
            f"*{len(report.candidates)} candidates | "
            f"Runtime: {report.total_runtime_seconds:.1f}s | "
            f"Top score: {report.top_candidates[0].score.composite:.3f} "
            f"({report.top_candidates[0].candidate.drug_name})*"
            if report.top_candidates else "*No candidates found*"
        )

        if not report.candidates:
            st.warning("No candidates found. Check your API keys and try a different disease name.")
            return

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧬 Disease Profile",
            "💊 Candidates",
            "🔗 Pathway Alignment",
            "⚡ Adversarial",
            "📋 Report",
        ])
        with tab1:
            render_disease_tab(report)
        with tab2:
            render_candidates_tab(report)
        with tab3:
            render_pathway_tab(report)
        with tab4:
            render_adversarial_tab(report)
        with tab5:
            render_report_tab(report)

    else:
        # Welcome state
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">8</div><div class="metric-label">Specialized Agents</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">6</div><div class="metric-label">Open Data Sources</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">±CI</div><div class="metric-label">Confidence Intervals</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        ### What makes v2 different?

        | Feature | v1 | **v2** |
        |---------|-----|--------|
        | Disease characterization | KG-only | Open Targets + DisGeNET |
        | Pathway validation | ❌ None | ✅ **KEGG + Reactome** |
        | Adversarial critique | ❌ None | ✅ **Failed trials + DDI + BBB** |
        | Hallucination guard | Basic | ✅ **Source-anchored every claim** |
        | Score uncertainty | None | ✅ **±Confidence Intervals** |
        | Evidence polarity | None | ✅ **Positive / Negative / Mechanistic** |
        | CNS disease handling | None | ✅ **BBB penetration check** |
        """)


if __name__ == "__main__":
    main()
