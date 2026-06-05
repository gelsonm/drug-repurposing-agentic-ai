import sys, os
sys.stdout.reconfigure(encoding='utf-8')

print("=== Agent 1: Disease Intel ===")
from src.agents.disease_intel import run_disease_intel_agent
profile = run_disease_intel_agent("Alzheimer's disease")
print(f"  targets: {len(profile.known_targets)}, EFO: {profile.efo_id}, CNS: {profile.is_cns_disease}")
print(f"  pathways: {profile.key_pathways[:2]}")
print(f"  sources: {profile.data_sources}")
print("  PASS")

print()
print("=== Agent 2: KG Candidates ===")
from src.agents.kg_candidate import run_kg_candidate_agent
candidates = run_kg_candidate_agent(profile, max_candidates=5)
print(f"  Found {len(candidates)} candidates")
for c in candidates[:3]:
    print(f"  - {c.drug_name}: prox={c.network_proximity_score:.3f} OT={c.open_targets_score:.3f}")
print("  PASS")

print()
print("=== Agent 3: Mol Mechanism (first candidate) ===")
from src.agents.mol_mechanism import run_mol_mechanism_agent
c = candidates[0]
c = run_mol_mechanism_agent(c)
print(f"  {c.drug_name}: ChEMBL={c.chembl_id}, targets={c.known_targets[:3]}")
print(f"  MOA: {(c.mechanism_of_action or '')[:80]}")
print("  PASS")

print()
print("=== Agent 4: Literature RAG (first candidate) ===")
from src.agents.literature_rag import run_literature_rag_agent
c, passages = run_literature_rag_agent(c, "Alzheimer's disease", is_cns=True, max_papers=10)
print(f"  PMIDs: {len(c.literature_pmids)}, positive: {c.literature_positive_count}, passages: {len(passages)}")
print("  PASS")

print()
print("=== Agent 5: Pathway Validator ===")
from src.agents.pathway_validator import run_pathway_validator_agent
alignment = run_pathway_validator_agent(c, profile, passages)
print(f"  {c.drug_name}: alignment={alignment.alignment_type.value}, plausibility={alignment.biological_plausibility_score:.3f}")
print(f"  BBB: {alignment.bbb_penetrant}")
print("  PASS")

print()
print("=== Agent 6: Adversarial ===")
from src.agents.adversarial import run_adversarial_agent
adv = run_adversarial_agent(c, profile, alignment)
print(f"  verdict: {adv.overall_verdict}")
print(f"  red: {adv.red_flags}, yellow_count: {len(adv.yellow_flags)}, failed_trials: {len(adv.failed_trials)}")
print("  PASS")

print()
print("=== Agent 7: Scorer ===")
from src.agents.scorer import run_scorer_agent
scored = run_scorer_agent(c, alignment, adv, passages)
print(f"  {c.drug_name}: score={scored.score.composite:.3f} +/-{scored.score.confidence_interval:.3f} ({scored.score.rank_label})")
print("  PASS")

print()
print("=== Agent 8: Report (Groq) ===")
from src.agents.report_gen import run_report_agent, report_to_markdown
report = run_report_agent("Alzheimer's disease", profile, [scored], use_llm=True)
print(f"  Executive summary length: {len(report.executive_summary or '')} chars")
print(f"  Summary snippet: {(report.executive_summary or 'N/A')[:120]}")
print("  PASS")

print()
print("=== ALL 8 AGENTS PASSED ===")
