"""Command line interface for the evidence pack.

    evidence-pack build -q questionnaire.yaml [-d README.md ...] [--llm] -o out/
    evidence-pack demo -o out/
    evidence-pack template > questionnaire.yaml

Plain argparse so it runs with no extra dependencies.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

from evidence_pack.export import write_all
from evidence_pack.pipeline import run
from evidence_pack.questionnaire import QuestionnaireError, template
from evidence_pack.report import GPAI_TEXT, ROLE_TEXT, TIER_TEXT

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_QUESTIONNAIRE = REPO_ROOT / "examples" / "brightloop" / "questionnaire.yaml"


def _date(text: str) -> dt.date:
    return dt.date.fromisoformat(text)


def _build(questionnaire: Path | None, docs: list[Path], out: Path, use_llm: bool,
           model: str, as_of: dt.date | None) -> int:
    if use_llm and not os.environ.get("OPENAI_API_KEY"):
        print("Error: --llm needs OPENAI_API_KEY. Without --llm the questionnaire path runs offline.",
              file=sys.stderr)
        return 1
    if questionnaire is None and not (use_llm and docs):
        print("Error: give a questionnaire (-q), or documents (-d) together with --llm.", file=sys.stderr)
        return 1
    try:
        pack, log = run(questionnaire, docs, use_llm=use_llm, model=model, as_of=as_of)
    except (QuestionnaireError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    c = pack.classification
    if pack.label:
        print(pack.label)
    print(f"{pack.system_name} ({pack.company_name})")
    print(f"  Role:         {ROLE_TEXT[c.role]}")
    print(f"  Risk tier:    {TIER_TEXT[c.risk_tier]}")
    print(f"  Article 50:   {', '.join(t.article for t in c.art50_triggers) or 'none'}")
    print(f"  GPAI:         {GPAI_TEXT[c.gpai_use]}")
    print(f"  AI literacy:  {'applies' if c.ai_literacy_applies else 'does not apply'}")
    print(f"  Obligations:  {len(pack.obligations)}   Gaps: {len(pack.gaps)}")
    if log:
        known = sum(1 for e in log if e.outcome == "known")
        dropped = sum(1 for e in log if e.outcome in {"quote_not_found", "not_grounded"})
        print(f"  Extraction:   {known} facts grounded, {dropped} candidates rejected")
    paths = write_all(pack, out)
    print("\nOutput files:")
    for p in paths.values():
        print(f"  {p}")
    print("\nNot legal advice. Re-check dates against the Official Journal.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidence-pack", description="EU AI Act evidence pack generator")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Build an evidence pack from a questionnaire and/or documents")
    b.add_argument("-q", "--questionnaire", type=Path, help="Questionnaire YAML/JSON file")
    b.add_argument("-d", "--doc", type=Path, action="append", default=[],
                   help="Product document (.md, .txt, .pdf); repeatable")
    b.add_argument("--llm", action="store_true",
                   help="Extract facts from documents with the two-pass LLM pipeline (needs OPENAI_API_KEY)")
    b.add_argument("-m", "--model", default="gpt-4o-mini")
    b.add_argument("--as-of", type=_date, default=None, help="Assess deadlines as of this date (YYYY-MM-DD)")
    b.add_argument("-o", "--output", type=Path, default=Path("output/evidence_pack"))

    d = sub.add_parser("demo", help="Run the fictional example company offline")
    d.add_argument("-o", "--output", type=Path, default=Path("output/brightloop"))
    d.add_argument("--as-of", type=_date, default=None)

    sub.add_parser("template", help="Print a blank questionnaire")

    args = parser.parse_args(argv)
    if args.command == "template":
        sys.stdout.write(template())
        return 0
    if args.command == "demo":
        return _build(DEMO_QUESTIONNAIRE, [], args.output, False, "gpt-4o-mini", args.as_of)
    return _build(args.questionnaire, args.doc, args.output, args.llm, args.model, args.as_of)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
