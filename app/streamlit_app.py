"""
Drug Repurposing with Agentic AI — Streamlit Demo App

A beautiful, interactive frontend for the multi-agent drug repurposing pipeline.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.crews.repurposing_crew import DrugRepurposingCrew
from src.models.entities import RepurposingReport
from src.reporting.markdown_reporter import generate_markdown_report

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drug Repurposing · Agentic AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f8f9fb;
    color: #1e293b;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] * { color: #334155 !important; }

/* Main content */
[data-testid="stAppViewContainer"] > .main { background-color: #f8f9fb; }
[data-testid="block-container"] { padding-top: 2rem; max-width: 1100px; }

/* Headings */
h1 { color: #0f172a !important; font-weight: 600 !important; letter-spacing: -0.02em; }
h2 { color: #1e293b !important; font-weight: 600 !important; }
h3 { color: #334155 !important; font-weight: 500 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 600; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem; }
[data-testid="stMetricDelta"]  { color: #64748b !important; }

/* Buttons */
.stButton > button {
    background: #0f766e;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    transition: background 0.18s;
}
.stButton > button:hover { background: #0d6460; }

/* Agent step log */
.agent-step {
    background: #f1f5f9;
    border-left: 3px solid #0f766e;
    border-radius: 0 6px 6px 0;
    padding: 0.45rem 1rem;
    margin: 0.25rem 0;
    font-size: 0.875rem;
    color: #334155;
}

/* Score badges */
.score-strong   { color: #059669; font-weight: 600; font-size: 1rem; }
.score-moderate { color: #d97706; font-weight: 600; font-size: 1rem; }
.score-weak     { color: #dc2626; font-weight: 600; font-size: 1rem; }

/* Citation card */
.citation {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #0f766e;
    border-radius: 6px;
    padding: 0.55rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.84rem;
    color: #334155;
}
.citation a { color: #0f766e; text-decoration: none; }
.citation a:hover { text-decoration: underline; }

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
}

/* Divider */
hr { border-color: #e2e8f0 !important; }

/* Inputs */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div {
    border-color: #cbd5e1 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}

/* Progress bar */
[data-testid="stProgressBar"] > div { background-color: #0f766e !important; }

/* DataFrame */
[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna-helix.png", width=60)
    st.title("🔬 Drug Repurposing")
    st.caption("Agentic AI Pipeline · Hackathon Demo")

    st.divider()

    # API Key input
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a free key at https://console.groq.com",
        placeholder="gsk_...",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()

    # Model selection
    model = st.selectbox(
        "LLM Model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
        index=0,
    )
    os.environ["GROQ_MODEL"] = model

    st.divider()

    # Demo mode toggle
    demo_mode = st.toggle("⚡ Demo Mode (Offline Cache)", value=False)
    st.caption("Demo mode uses pre-cached results. No API calls needed.")

    st.divider()

    st.markdown("""
    **Data Sources:**
    - 🧬 ChEMBL (drug targets)
    - 🔗 Hetionet (knowledge graph)
    - 📚 PubMed (35M+ papers)
    - 🏥 ClinicalTrials.gov
    - 💊 OpenFDA (FAERS)
    """)

# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("🔬 Drug Repurposing with Agentic AI")
    st.markdown(
        "**Multi-agent AI system** that autonomously discovers repurposing candidates for **oncology** "
        "by reasoning across biomedical knowledge graphs, scientific literature, and clinical databases."
    )
    st.caption("Domain: Cancer Drug Repurposing · Drugs: Metformin, Aspirin, Atorvastatin, Itraconazole, Doxycycline, Rapamycin")

# ---------------------------------------------------------------------------
# Quick Stats Row
# ---------------------------------------------------------------------------
s1, s2, s3, s4 = st.columns(4)
s1.metric("💊 Repurposable Drugs", "8", "in cancer KG")
s2.metric("🦠 Cancer Types", "10", "in knowledge graph")
s3.metric("📄 PubMed Papers", "35M+", "Live search")
s4.metric("⏱️ Analysis Time", "<5 min", "vs 3-6 months")

st.divider()

# ---------------------------------------------------------------------------
# Input Section — Cancer Domain
# ---------------------------------------------------------------------------
st.subheader("🎯 Select Drug & Cancer Type")

# Load cancer options dynamically from KG
try:
    import src.tools.kg_client as _kg
    _kg._graph = None  # Reset singleton so the new JSON loads
    _cancer_types = _kg.get_cancer_types()
    _drug_list = _kg.get_repurposable_drugs()
except Exception:
    _cancer_types = [
        "Colorectal cancer", "Breast cancer", "Pancreatic cancer",
        "Non-small cell lung cancer", "Glioblastoma", "Multiple myeloma",
        "Ovarian cancer", "Prostate cancer", "Hepatocellular carcinoma",
        "Basal cell carcinoma",
    ]
    _drug_list = ["Metformin", "Aspirin", "Atorvastatin", "Itraconazole",
                  "Doxycycline", "Hydroxychloroquine", "Rapamycin", "Thalidomide"]

col1, col2 = st.columns(2)
with col1:
    drug_name = st.selectbox(
        "💊 Drug (Non-oncology / Approved)",
        options=_drug_list,
        index=0,
        help="Approved drugs being investigated for cancer repurposing",
    )
with col2:
    disease_name = st.selectbox(
        "🦠 Cancer Type",
        options=_cancer_types,
        index=0,
        help="Target cancer for repurposing analysis",
    )

# Quick-start demo buttons
st.markdown("**⚡ Top Repurposing Cases (click to auto-fill):**")
demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)

if demo_col1.button("💊 Metformin\n→ Colorectal", use_container_width=True):
    st.session_state["demo_drug"] = "Metformin"
    st.session_state["demo_disease"] = "Colorectal cancer"
    st.session_state["trigger_run"] = True

if demo_col2.button("💊 Aspirin\n→ Colorectal", use_container_width=True):
    st.session_state["demo_drug"] = "Aspirin"
    st.session_state["demo_disease"] = "Colorectal cancer"
    st.session_state["trigger_run"] = True

if demo_col3.button("💊 Atorvastatin\n→ Pancreatic", use_container_width=True):
    st.session_state["demo_drug"] = "Atorvastatin"
    st.session_state["demo_disease"] = "Pancreatic cancer"
    st.session_state["trigger_run"] = True

if demo_col4.button("💊 Itraconazole\n→ Lung Cancer", use_container_width=True):
    st.session_state["demo_drug"] = "Itraconazole"
    st.session_state["demo_disease"] = "Non-small cell lung cancer"
    st.session_state["trigger_run"] = True

# Apply demo values from buttons
if "demo_drug" in st.session_state:
    drug_name = st.session_state.pop("demo_drug", drug_name)
if "demo_disease" in st.session_state:
    disease_name = st.session_state.pop("demo_disease", disease_name)

# Show shared gene targets between selected drug and cancer (instant preview)
if drug_name and disease_name:
    try:
        shared = _kg.get_shared_targets(drug_name, disease_name)
        if shared:
            genes_str = ", ".join(f"**{t['gene']}** ({t['drug_relationship']} by drug; {t['disease_relationship']} in cancer)" for t in shared[:5])
            st.success(f"🧬 **Shared targets detected:** {genes_str}")
        else:
            paths = _kg.find_drug_disease_paths(drug_name, disease_name, max_paths=1)
            if paths:
                st.info(f"🔗 **Indirect KG path found:** {paths[0].get('description', 'multi-hop connection')}")
    except Exception:
        pass

# Run button
st.divider()
run_btn = st.button(
    "🚀 Run Cancer Repurposing Analysis",
    type="primary",
    use_container_width=True,
)

trigger_run = st.session_state.pop("trigger_run", False) or run_btn

# ---------------------------------------------------------------------------
# Run Pipeline
# ---------------------------------------------------------------------------
if trigger_run and (drug_name or disease_name):
    if not os.environ.get("GROQ_API_KEY") and not demo_mode:
        st.error("⚠️ Please enter your Groq API key in the sidebar. Get one free at https://console.groq.com")
        st.stop()

    st.divider()
    st.subheader("🤖 Multi-Agent Pipeline Running...")

    # Agent progress display
    progress_container = st.container()
    with progress_container:
        prog = st.progress(0, text="Initializing pipeline...")

        agent_steps = [
            ("🧬 Molecular Mechanism Agent", "Querying ChEMBL, UniProt, and Hetionet KG...", 20),
            ("📚 Literature Intelligence Agent", "Searching PubMed and indexing abstracts...", 40),
            ("🏥 Clinical Signal Agent", "Analyzing ClinicalTrials.gov and FDA FAERS...", 60),
            ("📊 Scoring & Ranking Agent", "Computing composite scores...", 80),
            ("📝 Brief Generation Agent", "Synthesizing final report...", 100),
        ]

        step_placeholders = []
        for step_name, step_desc, _ in agent_steps:
            ph = st.empty()
            step_placeholders.append(ph)

        # Run the pipeline
        try:
            query_raw = f"Find repurposing candidates: {drug_name or ''} for {disease_name or 'new indications'}"

            # Simulate step-by-step progress for UI
            report_placeholder = st.empty()

            with st.spinner(""):
                # Update progress as pipeline runs
                for i, (step_name, step_desc, pct) in enumerate(agent_steps):
                    step_placeholders[i].markdown(
                        f'<div class="agent-step">✅ <b>{step_name}</b> — {step_desc}</div>',
                        unsafe_allow_html=True,
                    )
                    prog.progress(pct, text=f"{step_name}: {step_desc}")

                    if i == 0:  # Actually run pipeline on first step
                        crew = DrugRepurposingCrew()
                        report: RepurposingReport = crew.run(
                            query_drug=drug_name or None,
                            query_disease=disease_name or None,
                            query_raw=query_raw,
                        )
                        # Store in session state
                        st.session_state["last_report"] = report

                    time.sleep(0.3)

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            st.stop()

    st.success(f"✅ Analysis complete! Found {len(report.candidates)} candidates in {report.total_runtime_seconds}s")

# ---------------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------------
if "last_report" in st.session_state:
    report: RepurposingReport = st.session_state["last_report"]

    st.divider()
    st.subheader("📊 Results")

    # Executive summary
    if report.summary:
        with st.expander("📋 Executive Summary", expanded=True):
            st.markdown(report.summary)

    # Candidates table
    if report.candidates:
        st.subheader(f"🏆 Top {min(5, len(report.candidates))} Repurposing Candidates")

        # Build DataFrame for table
        rows = []
        for i, c in enumerate(report.top_candidates[:8], 1):
            score = c.score.composite_score
            badge = "🟢" if score >= 0.75 else "🟡" if score >= 0.55 else "🔴"
            rows.append({
                "Rank": i,
                "Drug": c.drug.name,
                "Disease": c.disease.name,
                f"Score": round(score, 3),
                "Strength": c.rank_label,
                "Mechanistic": round(c.score.mechanistic_score, 3),
                "Literature": round(c.score.literature_score, 3),
                "Clinical": round(c.score.clinical_feasibility_score, 3),
                "PubMed Hits": c.score.pubmed_hits,
                "Active Trials": c.score.active_trials,
                "Safety": c.score.safety_flag.value,
            })

        df = pd.DataFrame(rows).set_index("Rank")
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Composite Score", min_value=0, max_value=1, format="%.3f"
                ),
            },
        )

        # Radar chart for top 3 candidates
        st.subheader("📈 Score Breakdown — Top Candidates")
        categories = ["Mechanistic", "Literature", "Clinical", "Data Confidence"]
        fig = go.Figure()
        colors = ["#6366f1", "#22d3ee", "#34d399", "#f59e0b"]

        for i, c in enumerate(report.top_candidates[:3]):
            values = [
                c.score.mechanistic_score,
                c.score.literature_score,
                c.score.clinical_feasibility_score,
                c.score.data_confidence_score,
            ]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=f"{c.drug.name} → {c.disease.name}",
                line_color=colors[i % len(colors)],
                opacity=0.7,
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#94a3b8")),
                angularaxis=dict(tickfont=dict(color="#94a3b8")),
                bgcolor="rgba(15,15,35,0.8)",
            ),
            showlegend=True,
            paper_bgcolor="rgba(15,15,35,0)",
            plot_bgcolor="rgba(15,15,35,0)",
            font=dict(color="#e2e8f0"),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed candidate cards
        st.subheader("🔍 Candidate Deep-Dive")
        for c in report.top_candidates[:5]:
            score = c.score.composite_score
            badge = "🟢 Strong" if score >= 0.75 else "🟡 Moderate" if score >= 0.55 else "🔴 Weak"
            with st.expander(
                f"{badge} | {c.drug.name} → {c.disease.name} | Score: {score:.3f}",
                expanded=(score >= 0.7),
            ):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**🧬 Mechanistic Rationale**")
                    st.markdown(c.mechanistic_rationale or "_No mechanistic data found._")

                    # Shared gene targets (smoking guns)
                    try:
                        shared = _kg.get_shared_targets(c.drug.name, c.disease.name)
                        if shared:
                            st.markdown("**🎯 Shared Gene Targets (Smoking Guns)**")
                            for t in shared[:5]:
                                rel_drug = t['drug_relationship']
                                rel_dis = t['disease_relationship']
                                icon = "🔴" if rel_dis in ("promotes", "drives") and rel_drug in ("inhibits", "downregulates") else "🟡"
                                st.markdown(f"{icon} `{t['gene']}` — drug *{rel_drug}* it; cancer *{rel_dis}* it")
                    except Exception:
                        pass

                    if c.kg_paths:
                        st.markdown("**🔗 Knowledge Graph Paths**")
                        for path in c.kg_paths[:3]:
                            st.code(path.description or "", language=None)

                with col_b:
                    st.markdown("**📚 Literature Evidence**")
                    st.markdown(c.literature_summary or "_No literature data found._")

                    st.markdown("**🏥 Clinical Evidence**")
                    st.markdown(c.clinical_summary or "_No clinical data found._")

                # Citations
                if c.top_citations:
                    st.markdown("**📎 Top Citations**")
                    for ev in c.top_citations[:5]:
                        # EvidenceItem uses source_id (not pmid)
                        ev_id = ev.source_id or ""
                        ev_title = ev.title or ev_id or "View source"
                        ev_url = ev.url or f"https://pubmed.ncbi.nlm.nih.gov/{ev_id}/" if ev_id else "#"
                        ev_year = f" ({ev.year})" if ev.year else ""
                        if ev_id or ev.url:
                            st.markdown(
                                f'<div class="citation">📄 '
                                f'<a href="{ev_url}" target="_blank">{ev_title}</a>'
                                f'{ev_year} — <i>{ev.source}</i> '
                                f'[relevance: {ev.relevance_score:.2f}]</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            # KG or non-PubMed evidence — show as plain text
                            st.markdown(
                                f'<div class="citation">🔗 {ev_title}{ev_year} — <i>{ev.source}</i> '
                                f'[relevance: {ev.relevance_score:.2f}]</div>',
                                unsafe_allow_html=True,
                            )

        # Next steps
        if report.next_steps:
            st.subheader("➡️ Recommended Next Steps")
            for i, step in enumerate(report.next_steps, 1):
                st.markdown(f"{i}. {step}")

        # Download report
        st.divider()
        st.subheader("💾 Export Report")
        md_report = generate_markdown_report(report)
        st.download_button(
            label="📥 Download Markdown Report",
            data=md_report,
            file_name=f"repurposing_{(report.query_drug or 'query').lower().replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem;">
🔬 Drug Repurposing Agentic AI v1.0 · IQVIA Hackathon 2026 <br>
Data: ChEMBL · UniProt · Hetionet · PubMed · ClinicalTrials.gov · OpenFDA <br>
Framework: CrewAI · Groq · ChromaDB · NetworkX
</div>
""", unsafe_allow_html=True)
