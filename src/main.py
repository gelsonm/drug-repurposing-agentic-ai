"""Main CLI entry point for the drug repurposing pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.config import get_settings
from src.crews.repurposing_crew import DrugRepurposingCrew
from src.reporting.markdown_reporter import generate_markdown_report, save_report

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drug Repurposing Agentic AI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find repurposing candidates for Metformin
  python -m src.main --drug "Metformin"

  # Analyze specific drug-disease pair
  python -m src.main --drug "Metformin" --disease "Alzheimer's disease"

  # Run demo with pre-cached results
  python -m src.main --demo

  # Save report to file
  python -m src.main --drug "Aspirin" --disease "Colorectal cancer" --output report.md
        """,
    )
    parser.add_argument("--drug", type=str, help="Drug name to analyze")
    parser.add_argument("--disease", type=str, help="Target disease name")
    parser.add_argument("--query", type=str, help="Free-text query")
    parser.add_argument("--output", type=str, help="Output file path for report (.md)")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    return parser.parse_args()


def print_results_table(report) -> None:
    """Print a rich table of ranked candidates."""
    table = Table(title="🔬 Ranked Repurposing Candidates", show_header=True)
    table.add_column("Rank", style="bold cyan", width=6)
    table.add_column("Drug", style="bold green")
    table.add_column("Disease", style="bold yellow")
    table.add_column("Score", justify="right")
    table.add_column("Strength", style="bold")
    table.add_column("PubMed", justify="right")
    table.add_column("Trials", justify="right")
    table.add_column("Safety")

    for i, c in enumerate(report.top_candidates, 1):
        score = c.score.composite_score
        color = "green" if score >= 0.75 else "yellow" if score >= 0.55 else "red"
        table.add_row(
            str(i),
            c.drug.name,
            c.disease.name,
            f"[{color}]{score:.3f}[/{color}]",
            c.rank_label,
            str(c.score.pubmed_hits),
            str(c.score.active_trials),
            c.score.safety_flag.value,
        )

    console.print(table)


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()

    console.print(
        Panel.fit(
            "[bold blue]Drug Repurposing with Agentic AI[/bold blue]\n"
            "[dim]Powered by CrewAI · Groq · ChromaDB · Hetionet[/dim]",
            border_style="blue",
        )
    )

    settings = get_settings()

    # Demo mode
    if args.demo:
        args.drug = "Metformin"
        args.disease = "Alzheimer's disease"
        console.print("[yellow]Running in DEMO mode: Metformin → Alzheimer's disease[/yellow]")

    if not settings.groq_api_key:
        console.print("[red]ERROR: GROQ_API_KEY not set. Copy .env.example to .env and add your key.[/red]")
        sys.exit(1)

    # Build query
    query_raw = args.query or f"{args.drug or ''} repurposing for {args.disease or 'new indications'}".strip()

    console.print(f"\n[bold]Query:[/bold] {query_raw}")
    console.print("[dim]Starting multi-agent analysis...[/dim]\n")

    # Run pipeline
    crew = DrugRepurposingCrew()
    report = crew.run(
        query_drug=args.drug,
        query_disease=args.disease,
        query_raw=query_raw,
    )

    # Print results
    print_results_table(report)

    if report.summary:
        console.print(f"\n[bold]Executive Summary:[/bold]\n{report.summary}")

    if report.next_steps:
        console.print("\n[bold]Recommended Next Steps:[/bold]")
        for i, step in enumerate(report.next_steps, 1):
            console.print(f"  {i}. {step}")

    # Save report
    if args.output:
        output_path = save_report(report, args.output)
        console.print(f"\n[green]✓ Report saved to: {output_path}[/green]")
    else:
        # Auto-save to reports/
        drug_slug = (args.drug or "unknown").lower().replace(" ", "_")
        disease_slug = (args.disease or "general").lower().replace(" ", "_")
        output_path = save_report(report, f"reports/{drug_slug}_{disease_slug}_report.md")
        console.print(f"\n[green]✓ Report saved to: {output_path}[/green]")

    console.print(f"\n[dim]Runtime: {report.total_runtime_seconds}s[/dim]")


if __name__ == "__main__":
    main()
