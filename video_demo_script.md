# Video Demo Script: Drug Repurposing with Agentic AI
## IQVIA Hackathon 2026

> **Total duration:** ~8–10 minutes  
> **Format:** Screen recording + voiceover  
> **Resolution:** 1920×1080, 60fps recommended  
> **Screen:** Show Streamlit app fullscreen, switch to VS Code/terminal for architecture slides

---

## 🎬 Scene 1: Hook & Problem Statement (0:00 – 0:45)

**[SCREEN: Show a slide or clean full-screen text overlay]**  
*Display text: "Drug Development: $2.6 Billion. 12 Years. 90% Failure Rate."*

**[SAY:]**
> "Developing a new drug from scratch costs over two and a half billion dollars and takes more than a decade. And even then, nine out of ten candidates fail in clinical trials. But what if we could take drugs that are already safe — drugs that millions of people take every day — and discover that they work against completely different diseases? That's drug repurposing. And instead of taking months of expert review, we've built an AI system that does it in under five minutes."

---

## 🎬 Scene 2: Project Introduction (0:45 – 1:30)

**[SCREEN: Show the Streamlit app home page, not yet running — just the header and sidebar visible]**

**[SAY:]**
> "This is the Drug Repurposing Agentic AI — a system we built for the IQVIA 2026 Hackathon. It's a multi-agent AI pipeline that autonomously discovers cancer drug repurposing candidates by reasoning across biomedical knowledge graphs, scientific literature, and clinical trial databases."

**[SCREEN: Slowly scroll down to show the four metric cards]**

**[SAY:]**
> "We're focusing on a particularly impactful use case: taking approved non-oncology drugs — things like Metformin for diabetes, or Aspirin for pain — and evaluating their potential against ten different cancer types. We have eight candidate drugs, we search over 35 million PubMed papers in real time, and we complete the full analysis in under five minutes — compared to three to six months of manual expert work."

---

## 🎬 Scene 3: Architecture Walkthrough (1:30 – 2:30)

**[SCREEN: Show the README.md architecture diagram, or a clean architecture slide]**

**[SAY:]**
> "Before we run the demo, let me quickly walk you through how this works. We have five specialized AI agents, each with a distinct role, orchestrated using the CrewAI framework with Groq's Llama 3.3 70B model for lightning-fast inference."

**[SCREEN: Highlight each agent as you name it]**

**[SAY:]**
> "Agent one is the **Molecular Mechanism Analyst**. It queries ChEMBL to find the drug's protein targets, then queries UniProt for protein-disease links, and traverses a biomedical knowledge graph called Hetionet to find paths between the drug and the cancer."

> "Agent two is the **Literature Intelligence Specialist**. It searches PubMed with an optimized query — up to 15 papers — then ingests those abstracts into a local vector database using sentence-transformer embeddings, and retrieves the most relevant passages using cosine similarity search. This is RAG — Retrieval-Augmented Generation."

> "Agent three is the **Clinical Signal Agent**. It hits ClinicalTrials.gov to see how many trials are running, at what phase, and whether they're completing or being terminated. It also checks the FDA drug label for safety warnings."

> "Agent four is a **deterministic scoring engine** — no LLM randomness — that combines all the evidence into a transparent composite score with four weighted components."

> "And agent five, the **Brief Generation Agent**, synthesizes everything into a boardroom-ready summary with ranked candidates and recommended next steps."

---

## 🎬 Scene 4: Live Demo — Setting Up (2:30 – 3:00)

**[SCREEN: Show the sidebar of the Streamlit app]**

**[SAY:]**
> "Let's jump into the live demo. In the sidebar, you can see I've already entered my Groq API key — you can get one for free at console.groq.com — and I've selected the Llama 3.3 70B model. The data sources are shown here: ChEMBL for drug targets, Hetionet for the knowledge graph, PubMed for the literature, ClinicalTrials.gov for clinical evidence, and OpenFDA for safety data."

---

## 🎬 Scene 5: Live Demo — Selecting Drug & Disease (3:00 – 3:30)

**[SCREEN: Click on the Drug dropdown — show it populating from KG]**

**[SAY:]**
> "The drug and cancer dropdowns are populated dynamically from the knowledge graph. I'm going to select **Metformin** — the world's most prescribed diabetes drug — and let's pair it with **Colorectal cancer**."

**[SCREEN: After selecting both, pause and point to the green banner that appears]**

**[SAY:]**
> "Notice what happened instantly — even before running the analysis. The system has already detected shared gene targets between Metformin and colorectal cancer directly from the knowledge graph. It's showing us AMPK and MTOR — genes that Metformin is known to modulate and that are directly implicated in colorectal cancer progression. These are what we call 'smoking guns' — the mechanistic fingerprints of a repurposing hypothesis."

---

## 🎬 Scene 6: Live Demo — Running the Pipeline (3:30 – 5:00)

**[SCREEN: Click the "🚀 Run Cancer Repurposing Analysis" button]**

**[SAY:]**
> "Now let's run the full analysis. I'm clicking the Run button."

**[SCREEN: Show the agent progress bar updating step by step]**

**[SAY as each step appears:]**
> "You can see the five agents activating in sequence. First, the **Molecular Mechanism Agent** is hitting ChEMBL and traversing the knowledge graph..."

> "Now the **Literature Intelligence Agent** is searching PubMed and indexing abstracts into the vector database..."

> "The **Clinical Signal Agent** is now querying ClinicalTrials.gov and checking the FDA label..."

> "The **Scoring Agent** is computing the composite scores across all four dimensions..."

> "And finally the **Brief Generation Agent** is synthesizing the executive summary..."

**[SCREEN: Show "✅ Analysis complete!" message]**

**[SAY:]**
> "Done — the full analysis completed in about two minutes. Let's look at the results."

---

## 🎬 Scene 7: Results — Candidates Table (5:00 – 5:45)

**[SCREEN: Scroll to the results table with the Plotly progress bar column]**

**[SAY:]**
> "Here's our ranked candidate. We're analyzing Metformin against Colorectal cancer, and the system has assigned it a **composite score** — shown here as a progress bar. The score is broken down into four dimensions: mechanistic, literature, clinical feasibility, and data confidence."

**[SCREEN: Pause on the table row]**

**[SAY:]**
> "We can see the PubMed hit count — how many published papers discuss Metformin in the context of colorectal cancer — and the number of active clinical trials. The safety flag is shown in the last column. Metformin has a clean safety profile since it's been used in hundreds of millions of patients for decades."

---

## 🎬 Scene 8: Results — Radar Chart (5:45 – 6:15)

**[SCREEN: Scroll down to the Plotly radar chart]**

**[SAY:]**
> "This radar chart gives us an instant visual breakdown of the score across all four dimensions. A candidate that scores high across all four quadrants — mechanistic evidence, literature support, clinical feasibility, and data confidence — would fill the entire polygon. We can compare multiple candidates side-by-side here."

---

## 🎬 Scene 9: Results — Candidate Deep Dive (6:15 – 7:00)

**[SCREEN: Click to expand the candidate card for Metformin → Colorectal cancer]**

**[SAY:]**
> "Now let's deep-dive into the evidence for this candidate."

**[SCREEN: Highlight the Mechanistic Rationale section]**

**[SAY:]**
> "On the left, the Molecular Agent's conclusion: a precise mechanistic rationale generated by the LLM based on the actual data it gathered. It explains *why* Metformin might work — specifically mentioning AMPK activation and mTOR inhibition as key mechanisms."

**[SCREEN: Scroll to the Shared Gene Targets section within the card]**

**[SAY:]**
> "Below that, the knowledge graph smoking guns: genes that are both targeted by Metformin AND implicated in colorectal cancer. You can see the directionality — Metformin inhibits MTOR, and MTOR promotes colorectal cancer. That's a perfect mechanistic chain."

**[SCREEN: Scroll right to Literature Evidence]**

**[SAY:]**
> "On the right, the literature summary — a synthesis of what the RAG-retrieved PubMed papers say. And below that, the clinical summary showing how many trials are running and what signal they give."

**[SCREEN: Scroll to the citations section]**

**[SAY:]**
> "And at the bottom, clickable PubMed citations with relevance scores — so a researcher can immediately drill into the source papers."

---

## 🎬 Scene 10: Next Steps & Export (7:00 – 7:20)

**[SCREEN: Scroll to the "Recommended Next Steps" section]**

**[SAY:]**
> "The system concludes with concrete recommended next steps — from in vitro binding assay validation, to reviewing existing clinical protocols, to the eventual FDA IND application if preclinical validation succeeds."

**[SCREEN: Click the "Download Markdown Report" button]**

**[SAY:]**
> "And the full analysis can be exported as a structured Markdown report — ready to share with a research team or attach to a project brief."

---

## 🎬 Scene 11: Quick Demo — Atorvastatin → Pancreatic Cancer (7:20 – 7:50)

**[SCREEN: Click the demo button "💊 Atorvastatin → Pancreatic"]**

**[SAY:]**
> "Let me quickly show you a second candidate — Atorvastatin, a cholesterol drug, against Pancreatic cancer. Atorvastatin inhibits KRAS — one of the most commonly mutated driver genes in pancreatic cancer — making it a mechanistically compelling hypothesis. You can see the KG path: Atorvastatin inhibits HMGCR and KRAS, and KRAS drives pancreatic cancer."

---

## 🎬 Scene 12: Limitations (7:50 – 8:20)

**[SCREEN: Switch to a clean slide or text overlay]**

**[SAY:]**
> "Now, we want to be transparent about the limitations. First, the knowledge graph is a curated cancer subset — not the full 47,000-node Hetionet, so we may miss some connections. Second, the scoring weights are heuristic — not calibrated against a validated gold-standard benchmark dataset. Third — and this is critical — all outputs are **computational hypotheses**. They are not medical advice. Every candidate would need rigorous wet-lab validation before any clinical consideration."

> "The system also can't currently detect causal directionality in all path types — so it needs domain expert review to filter out cases where a drug might interfere with a tumor-suppressing gene."

---

## 🎬 Scene 13: Future Work & Closing (8:20 – 9:00)

**[SCREEN: Return to the Streamlit app or show a roadmap slide]**

**[SAY:]**
> "Looking ahead, we have an exciting roadmap. In the near term, we'll integrate the full Hetionet graph, add parallel agent execution for 3× speedup, and calibrate scoring weights against confirmed repurposing successes like Metformin in colorectal cancer."

> "In the medium term, we want to replace the heuristic path scoring with a Graph Neural Network trained on Hetionet — similar to the TxGNN paper published in Nature Medicine last year — and incorporate genomic and proteomic data for patient subgroup stratification."

> "Long term, the vision is a continuously updated repurposing intelligence platform: monitoring new literature weekly, integrating with FDA regulatory pathways, and eventually providing actionable recommendations to R&D teams across multiple disease areas."

**[SCREEN: Show the footer of the Streamlit app or a final title slide]**

**[SAY:]**
> "What we've built in this hackathon is a proof of concept that demonstrates what's possible when you combine the reasoning power of large language models with structured biomedical knowledge and multi-agent orchestration. Drug repurposing could be the fastest path to new cancer treatments — and AI is the key to unlocking it at scale."

> "Thank you."

---

## 🎬 End Card (9:00 – 9:15)

**[SCREEN: Static slide with project details]**

```
Drug Repurposing with Agentic AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 IQVIA Hackathon 2026

Stack: CrewAI · Groq · ChromaDB · Hetionet · Streamlit

Data: ChEMBL · UniProt · PubMed · ClinicalTrials.gov · OpenFDA

GitHub: [your repo URL]
```

---

## 📋 Recording Checklist

Before recording:
- [ ] Groq API key set in `.env` and Streamlit sidebar
- [ ] ChromaDB collection cleared for a "fresh" demo (`clear_collection()`)
- [ ] Browser zoom at 110% for readability
- [ ] Terminal font size ≥ 16pt if showing CLI
- [ ] Close notifications/Slack/email
- [ ] Confirm internet connectivity for live API calls
- [ ] Have the 4 demo drug-disease pairs pre-noted to avoid typing mistakes

Recommended screen layout:
- **Main screen:** Streamlit app at 1920×1080 (fullscreen Chrome)
- **Transition slides:** Black background with white text for architecture diagram

---

## ⏱️ Timing Summary

| Scene | Section | Duration |
|-------|---------|----------|
| 1 | Hook & Problem Statement | 0:45 |
| 2 | Project Introduction | 0:45 |
| 3 | Architecture Walkthrough | 1:00 |
| 4 | Demo Setup | 0:30 |
| 5 | Drug & Disease Selection | 0:30 |
| 6 | Pipeline Execution | 1:30 |
| 7 | Candidates Table | 0:45 |
| 8 | Radar Chart | 0:30 |
| 9 | Deep Dive | 0:45 |
| 10 | Next Steps & Export | 0:20 |
| 11 | Second Candidate | 0:30 |
| 12 | Limitations | 0:30 |
| 13 | Future Work & Close | 0:40 |
| — | End Card | 0:15 |
| **Total** | | **~8:30** |
