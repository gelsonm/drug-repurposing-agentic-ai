"""
CLI entry point for Drug Repurposing v2.

Usage:
  python main.py "Alzheimer's disease"
  python main.py "Alzheimer's disease" --top-n 5 --no-llm
  python main.py "Parkinson's disease" --output-dir results/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

from src.config import get_settings
from src.crew import DrugRepurposingCrew
from src.agents.report_gen import report_to_markdown

console = Console()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Drug Repurposing v2 — Biologically-Grounded Multi-Agent AI Pipeline"
    )
    parser.add_argument("disease", help="Target disease name (e.g., \"Alzheimer's disease\")")
    parser.add_argument("--top-n", type=int, default=5, help="How many candidates to run full deep analysis")
    parser.add_argument("--max-candidates", type=int, default=10, help="Max candidates to discover")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM report generation")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for reports")
    args = parser.parse_args()

    settings = get_settings()
    disease = args.disease
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Header
    console.print(Panel.fit(
        f"[bold cyan]Drug Repurposing v2 — Biologically-Grounded Multi-Agent Pipeline[/bold cyan]\n"
        f"Disease: [bold white]{disease}[/bold white]\n"
        f"LLM: {settings.openai_model}  |  Top-N: {args.top_n}  |  Max candidates: {args.max_candidates}",
        border_style="cyan",
    ))

    # Progress tracking
    steps_done = []
    def progress_cb(step, total, msg):
        steps_done.append(msg)
        console.print(f"  [bold cyan][{step}/{total}][/bold cyan] {msg}")

    # Run pipeline
    console.print("\n[bold]Starting pipeline...[/bold]\n")

    try:
        crew = DrugRepurposingCrew(top_n=args.top_n, use_llm_report=not args.no_llm)
        report = crew.run(disease, progress_callback=progress_cb)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        console.print(f"[bold red]Pipeline error:[/bold red] {e}")
        sys.exit(1)

    # ── Display Results ────────────────────────────────────────────────────
    console.print(f"\n[bold green]✅ Analysis complete in {report.total_runtime_seconds:.1f}s[/bold green]\n")

    if report.executive_summary:
        console.print(Panel(report.executive_summary, title="Executive Summary", border_style="green"))

    # Candidates table
    table = Table(title="Ranked Repurposing Candidates", box=box.ROUNDED)
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Drug", style="bold white", min_width=18)
    table.add_column("Score ± CI", style="bold yellow", width=12)
    table.add_column("Rank", width=10)
    table.add_column("Pathway", width=12)
    table.add_column("Safety", width=10)
    table.add_column("Papers", width=8)

    for i, sc in enumerate(report.top_candidates[:10]):
        pa = sc.pathway_alignment
        adv = sc.adversarial_report
        score = sc.score

        alignment_emoji = {
            "direct": "🟢 Direct",
            "indirect": "🟡 Indirect",
            "speculative": "🔴 Spec.",
        }.get(pa.alignment_type.value if pa else "", "⬜ N/A")

        safety_emoji = {
            "clean": "✅ Clean",
            "yellow": "⚠️ Yellow",
            "red": "🔴 Red",
            "unknown": "❓ Unk.",
        }.get(adv.safety_flag.value if adv else "unknown", "❓")

        table.add_row(
            str(i + 1),
            sc.candidate.drug_name,
            f"{score.composite:.3f} ±{score.confidence_interval:.3f}",
            score.rank_label,
            alignment_emoji,
            safety_emoji,
            str(len(sc.candidate.literature_pmids)),
        )

    console.print(table)

    # ── Save outputs ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    disease_slug = disease.lower().replace(" ", "_").replace("'", "")[:30]

    # Markdown report
    md_path = output_dir / f"report_{disease_slug}_{timestamp}.md"
    md_content = report_to_markdown(report)
    md_path.write_text(md_content, encoding="utf-8")
    console.print(f"\n📄 Markdown report: [link file://{md_path}]{md_path}[/link file://{md_path}]")

    # JSON dump
    json_path = output_dir / f"report_{disease_slug}_{timestamp}.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"📦 JSON export: {json_path}")

    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
