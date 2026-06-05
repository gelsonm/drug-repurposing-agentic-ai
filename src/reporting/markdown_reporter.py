"""Markdown report generator using Jinja2."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.models.entities import RepurposingReport

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_markdown_report(report: RepurposingReport) -> str:
    """
    Render a RepurposingReport to a Markdown string using the Jinja2 template.

    Args:
        report: The completed RepurposingReport object.

    Returns:
        Rendered markdown string.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("brief_template.md.j2")
    return template.render(report=report)


def save_report(report: RepurposingReport, output_path: str | Path) -> Path:
    """Save the rendered report to a file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = generate_markdown_report(report)
    output_path.write_text(content, encoding="utf-8")
    return output_path
