"""Evaluate the rule-based classifier on the fictional systems in eval/systems/.

Runs offline through the questionnaire path (no LLM). Prints accuracy per
field and lists every mismatch.

    python eval/run_eval.py [--systems eval/systems] [--fail-under 1.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_pack.classify import classify  # noqa: E402
from evidence_pack.questionnaire import facts_from_answers  # noqa: E402

FIELDS = ["role", "risk_tier", "art50_triggers", "gpai_use", "ai_literacy"]


def predict(answers: dict) -> dict:
    c = classify(facts_from_answers(answers))
    return {
        "role": c.role,
        "risk_tier": c.risk_tier,
        "art50_triggers": sorted(c.trigger_ids),
        "gpai_use": c.gpai_use,
        "ai_literacy": c.ai_literacy_applies,
    }


def evaluate(systems_dir: Path) -> tuple[dict[str, float], list[str], int]:
    files = sorted(systems_dir.glob("*.yaml"))
    if not files:
        raise SystemExit(f"no eval systems in {systems_dir}")
    correct = {f: 0 for f in FIELDS}
    mismatches: list[str] = []
    for path in files:
        case = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = dict(case["expected"])
        expected["art50_triggers"] = sorted(expected["art50_triggers"])
        got = predict(case["answers"])
        for f in FIELDS:
            if got[f] == expected[f]:
                correct[f] += 1
            else:
                mismatches.append(f"{path.name}: {f} expected {expected[f]!r}, got {got[f]!r}")
    n = len(files)
    return {f: correct[f] / n for f in FIELDS}, mismatches, n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", type=Path, default=ROOT / "eval" / "systems")
    parser.add_argument("--fail-under", type=float, default=0.0,
                        help="Exit 1 if any field's accuracy is below this value")
    args = parser.parse_args(argv)

    acc, mismatches, n = evaluate(args.systems)
    print(f"Evaluated {n} fictional systems (questionnaire path, no LLM)\n")
    print(f"{'field':<16} accuracy")
    for f in FIELDS:
        print(f"{f:<16} {acc[f]:.0%} ({round(acc[f] * n)}/{n})")
    all_fields = sum(acc.values()) / len(acc)
    print(f"{'mean':<16} {all_fields:.0%}")
    if mismatches:
        print("\nMismatches:")
        for m in mismatches:
            print(f"  {m}")
    return 1 if min(acc.values()) < args.fail_under else 0


if __name__ == "__main__":
    raise SystemExit(main())
