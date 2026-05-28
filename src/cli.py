"""Typer CLI entry point for the Annex IV generator.

Provides commands to ingest artifacts, run extraction, generate gap reports,
and produce PDF/JSON output. Supports --model flag to select between
gpt-4o-mini (default) and gpt-4o for harder extraction.

See CLAUDE.md: "typer for the CLI"
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import sys
from pathlib import Path

import typer

from annex_iv.schema import AnnexIVDocument
from extraction.pipeline import generate_annex_iv
from gap_analysis.analyzer import analyze
from ingestion.huggingface import HuggingFaceModelCardLoader
from output.pdf import generate_pdf

app = typer.Typer(no_args_is_help=True)


def _err(msg: str) -> None:
    typer.echo(msg, err=True)


def _derive_system_name(source: str, is_local: bool) -> str:
    if is_local:
        return Path(source).stem
    return source.replace("/", "__")


@app.command()
def generate(
    source: str = typer.Argument(..., help="HuggingFace model ID or local model card path"),
    output: Path = typer.Option(Path("./output"), "--output", "-o", help="Output directory"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="OpenAI model name"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable INFO-level logging"),
    no_gap_report: bool = typer.Option(False, "--no-gap-report", help="Skip gap report"),
    pdf: bool = typer.Option(True, "--pdf/--no-pdf", help="Generate PDF output"),
) -> None:
    """Generate Annex IV technical documentation from a model card."""
    if not os.environ.get("OPENAI_API_KEY"):
        _err("Error: OPENAI_API_KEY environment variable is not set. Set it with `export OPENAI_API_KEY=...` and re-run.")
        raise typer.Exit(code=1)

    if verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")

    is_local = Path(source).exists()

    loader = HuggingFaceModelCardLoader()
    source_uri = f"file://{Path(source).resolve()}" if is_local else f"huggingface://{source}"
    _err(f"Loading artifact from {source_uri}...")

    try:
        artifact = loader.load(source)
    except Exception as exc:
        exc_type = type(exc).__name__
        if "RepositoryNotFoundError" in exc_type or "404" in str(exc) or "not found" in str(exc).lower():
            _err(f"Error: model '{source}' not found on HuggingFace Hub or as a local file.")
            raise typer.Exit(code=1)
        _err(f"Error: failed to fetch from HuggingFace ({exc_type}). Check your connection and try again.")
        raise typer.Exit(code=1)

    document = generate_annex_iv(artifact, model=model)

    if not verbose:
        _print_item_progress(document)

    gap_report = analyze(document)

    system_name = _derive_system_name(source, is_local)

    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _err(f"Error: cannot write to {output}: {exc.strerror}.")
        raise typer.Exit(code=1)

    annex_path = output / f"{system_name}.annex_iv.json"
    gap_path = output / f"{system_name}.gap_report.json"
    summary_path = output / f"{system_name}.summary.txt"
    pdf_path = output / f"{system_name}.pdf"

    try:
        annex_path.write_text(document.model_dump_json(indent=2))
        if not no_gap_report:
            gap_path.write_text(gap_report.model_dump_json(indent=2))
        summary_path.write_text(gap_report.summary)
        if pdf:
            generate_pdf(document, gap_report, pdf_path)
    except OSError as exc:
        _err(f"Error: cannot write to {output}: {exc.strerror}.")
        raise typer.Exit(code=1)

    typer.echo(f"\n{gap_report.summary}")
    typer.echo(f"Gaps: {len(gap_report.gaps)}")
    typer.echo(f"\nOutput files:")
    typer.echo(f"  {annex_path}")
    if not no_gap_report:
        typer.echo(f"  {gap_path}")
    typer.echo(f"  {summary_path}")
    if pdf:
        typer.echo(f"  {pdf_path}")


def _print_item_progress(document: AnnexIVDocument) -> None:
    _ITEM_ATTRS = [
        (1, "item_1_general"),
        (2, "item_2_development"),
        (3, "item_3_monitoring_control"),
        (4, "item_4_performance_metrics"),
        (5, "item_5_risk_management"),
        (6, "item_6_lifecycle_changes"),
        (7, "item_7_harmonised_standards"),
        (8, "item_8_declaration_of_conformity"),
        (9, "item_9_post_market_monitoring"),
    ]
    for num, attr in _ITEM_ATTRS:
        item = getattr(document, attr)
        filled = 0
        unavailable = 0
        for field_name in type(item).model_fields:
            f = getattr(item, field_name)
            if f.status == "filled":
                filled += 1
            else:
                unavailable += 1
        _err(f"Item {num}/9: {filled} filled, {unavailable} unavailable")


@app.command()
def report(
    path: Path = typer.Argument(..., help="Path to a .annex_iv.json file"),
    pdf: bool = typer.Option(False, "--pdf", help="Generate PDF output"),
) -> None:
    """Re-run gap analysis on a previously generated AnnexIVDocument JSON."""
    if not path.exists():
        _err(f"Error: file '{path}' does not exist.")
        raise typer.Exit(code=1)

    try:
        raw = path.read_text()
        document = AnnexIVDocument.model_validate_json(raw)
    except Exception:
        _err(f"Error: '{path}' is not a valid AnnexIVDocument JSON file.")
        raise typer.Exit(code=1)

    gap_report = analyze(document)

    gap_path = path.with_suffix("").with_suffix(".gap_report.json")
    if path.name.endswith(".annex_iv.json"):
        stem = path.name.removesuffix(".annex_iv.json")
        gap_path = path.parent / f"{stem}.gap_report.json"

    gap_path.write_text(gap_report.model_dump_json(indent=2))
    typer.echo(gap_report.summary)
    typer.echo(f"Gap report written to {gap_path}")

    if pdf:
        if path.name.endswith(".annex_iv.json"):
            stem = path.name.removesuffix(".annex_iv.json")
        else:
            stem = path.stem
        pdf_path = path.parent / f"{stem}.pdf"
        generate_pdf(document, gap_report, pdf_path)
        typer.echo(f"PDF written to {pdf_path}")


@app.command()
def benchmark(
    models_yaml: Path = typer.Option(Path("benchmark/models.yaml"), "--models", help="Path to models YAML"),
    output_dir: Path = typer.Option(Path("benchmark/results"), "--output", "-o", help="Output directory"),
    force: bool = typer.Option(False, "--force", help="Re-run even if cached"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="OpenAI model name"),
) -> None:
    """Run the 10-model benchmark and write aggregate results."""
    if not os.environ.get("OPENAI_API_KEY"):
        _err("Error: OPENAI_API_KEY environment variable is not set. Set it with `export OPENAI_API_KEY=...` and re-run.")
        raise typer.Exit(code=1)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")

    from benchmark.run_benchmark import run_benchmark as _run

    _run(models_yaml, output_dir, openai_model=model, force=force)


@app.command()
def version() -> None:
    """Print the package version."""
    try:
        v = importlib.metadata.version("ai-act-compliance-copilot")
    except importlib.metadata.PackageNotFoundError:
        v = "unknown"
    typer.echo(v)
