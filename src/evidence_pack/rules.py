"""Load the versioned rules and deadlines files and evaluate rule conditions.

The condition grammar is documented at the top of data/rules.yaml. Evaluation
is pure and deterministic. Unknown values never satisfy a condition, and the
names of unknown values a condition looked at are collected so the caller can
report them as gaps.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

DATA_DIR = Path(__file__).resolve().parent / "data"
RULES_PATH = DATA_DIR / "rules.yaml"
DEADLINES_PATH = DATA_DIR / "deadlines.yaml"

_OPERATORS = {"eq", "nonempty", "intersects"}


class RuleError(ValueError):
    """The rules file contains a condition this evaluator does not understand."""


@lru_cache(maxsize=4)
def load_rules(path: Path = RULES_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        rules = yaml.safe_load(fh)
    if not isinstance(rules, dict) or "version" not in rules:
        raise RuleError(f"{path} has no version")
    return rules


@lru_cache(maxsize=4)
def load_deadlines(path: Path = DEADLINES_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    for key, entry in data["deadlines"].items():
        if not isinstance(entry.get("date"), dt.date):
            raise RuleError(f"deadline {key!r} has no ISO date")
    return data


def catalogue_ids(name: str) -> list[str]:
    return [entry["id"] for entry in load_rules()[name]]


def catalogue_entry(name: str, entry_id: str) -> dict[str, Any]:
    for entry in load_rules()[name]:
        if entry["id"] == entry_id:
            return entry
    raise KeyError(f"{entry_id!r} not in catalogue {name!r}")


def deadline(key: str) -> dict[str, Any]:
    entries = load_deadlines()["deadlines"]
    if key not in entries:
        raise KeyError(f"unknown deadline key {key!r}")
    return {"key": key, **entries[key]}


def evaluate(cond: Mapping[str, Any], values: Mapping[str, Any], unknowns: set[str] | None = None) -> bool:
    """Evaluate one condition against *values* (facts and derived values).

    ``None`` in *values* means unknown. Every unknown field that the
    condition inspects is added to *unknowns*.
    """
    if unknowns is None:
        unknowns = set()
    if not isinstance(cond, Mapping) or not cond:
        raise RuleError(f"condition must be a non-empty mapping, got {cond!r}")

    if "always" in cond:
        return bool(cond["always"])
    if "all" in cond or "any" in cond:
        # Only report unknowns that could change the outcome: if one branch
        # settles the result on known facts alone (a known False inside
        # `all`, a known True inside `any`), the other branches' unknowns
        # do not matter.
        is_all = "all" in cond
        branches = cond["all"] if is_all else cond["any"]
        outcomes = []
        for c in branches:
            branch_unknowns: set[str] = set()
            outcomes.append((evaluate(c, values, branch_unknowns), branch_unknowns))
        settling = False if is_all else True
        if any(res is settling and not unk for res, unk in outcomes):
            return settling
        for _res, unk in outcomes:
            unknowns |= unk
        results = [res for res, _unk in outcomes]
        return all(results) if is_all else any(results)
    if "not" in cond:
        inner_unknowns: set[str] = set()
        result = evaluate(cond["not"], values, inner_unknowns)
        unknowns |= inner_unknowns
        return not result
    if "field" in cond:
        name = cond["field"]
        ops = set(cond) - {"field"}
        if len(ops) != 1 or not ops <= _OPERATORS:
            raise RuleError(f"field condition needs exactly one of {_OPERATORS}: {cond!r}")
        if name not in values:
            raise RuleError(f"condition refers to unknown field name {name!r}")
        value = values[name]
        if value is None:
            unknowns.add(name)
            return False
        if "eq" in cond:
            return value == cond["eq"]
        if "nonempty" in cond:
            return bool(value) == bool(cond["nonempty"])
        if "intersects" in cond:
            if not isinstance(value, list):
                raise RuleError(f"intersects needs a list field, {name!r} is {type(value).__name__}")
            return bool(set(value) & set(cond["intersects"]))
    raise RuleError(f"unrecognised condition {cond!r}")
