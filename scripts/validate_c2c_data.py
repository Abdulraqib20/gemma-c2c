#!/usr/bin/env python3
"""Validate C2C JSONL datasets and produce QC report."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

INTENTS = {"remind", "schedule", "log", "notify"}
PRIORITIES = {"H", "M", "L"}
DOMAIN_HINTS = {
    "business": {
        "invoice",
        "client",
        "meeting",
        "crm",
        "budget",
        "contract",
        "team",
        "roadmap",
        "expense",
        "sales",
        "kickoff",
        "procurement",
        "sprint",
        "hiring",
    },
    "personal": {
        "groceries",
        "gym",
        "dentist",
        "mom",
        "dad",
        "landlord",
        "cat",
        "dry cleaning",
        "electricity",
        "medication",
        "weekend",
        "roommate",
        "car service",
    },
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\.,;:!?]+", "", text)
    return text


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def parse_label(label_yaml: str, max_tasks: int) -> Tuple[bool, str, dict | None]:
    try:
        obj = yaml.safe_load(label_yaml)
    except yaml.YAMLError as exc:
        return False, f"YAML parse error: {exc}", None

    if not isinstance(obj, dict):
        return False, "label must parse into a mapping", None

    if set(obj.keys()) != {"is_act", "intent", "tasks"}:
        return False, "top-level keys must be exactly is_act,intent,tasks", None

    if obj["is_act"] not in (0, 1):
        return False, "is_act must be 0 or 1", None

    if obj["intent"] not in INTENTS:
        return False, "intent must be one of remind/schedule/log/notify", None

    if not isinstance(obj["tasks"], list):
        return False, "tasks must be a list", None

    tasks = obj["tasks"]
    if obj["is_act"] == 0 and tasks:
        return False, "is_act=0 requires empty tasks", None

    if obj["is_act"] == 1 and not (1 <= len(tasks) <= max_tasks):
        return False, f"is_act=1 requires tasks length between 1 and {max_tasks}", None

    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            return False, f"task[{idx}] must be a mapping", None
        if set(task.keys()) != {"act", "who", "due", "pri"}:
            return False, f"task[{idx}] keys must be exactly act/who/due/pri", None
        if not all(isinstance(task[k], str) and task[k].strip() for k in ("act", "who", "due")):
            return False, f"task[{idx}] act/who/due must be non-empty strings", None
        if task["pri"] not in PRIORITIES:
            return False, f"task[{idx}] pri must be H/M/L", None

        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", task["due"].strip()):
            return False, f"task[{idx}] due must be natural-language, not strict ISO date", None

    return True, "ok", obj


def infer_domain_from_text_and_tasks(text: str, tasks: list[dict]) -> str:
    haystack = (text + " " + " ".join(task["act"] for task in tasks)).lower()

    scores = {
        domain: sum(1 for token in hints if token in haystack)
        for domain, hints in DOMAIN_HINTS.items()
    }

    if scores["business"] > scores["personal"]:
        return "business"
    if scores["personal"] > scores["business"]:
        return "personal"
    return "unknown"


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error in {path}:{line_no}: {exc}") from exc
    return rows


def ratio(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def check_close(actual: float, expected: float, tol: float) -> bool:
    return abs(actual - expected) <= tol


def validate_rows(rows: Iterable[dict], split_name: str, max_tasks: int) -> Tuple[List[str], List[dict]]:
    issues: List[str] = []
    parsed: List[dict] = []

    for idx, row in enumerate(rows, start=1):
        if set(row.keys()) != {"text", "label"}:
            issues.append(f"{split_name}[{idx}]: row keys must be exactly text,label")
            continue

        if not isinstance(row["text"], str) or not row["text"].strip():
            issues.append(f"{split_name}[{idx}]: text must be non-empty string")
            continue

        if not isinstance(row["label"], str) or not row["label"].strip():
            issues.append(f"{split_name}[{idx}]: label must be non-empty YAML string")
            continue

        ok, msg, obj = parse_label(row["label"], max_tasks=max_tasks)
        if not ok or obj is None:
            issues.append(f"{split_name}[{idx}]: {msg}")
            continue

        parsed.append(
            {
                "text": row["text"],
                "label": row["label"],
                "obj": obj,
                "norm": normalize_text(row["text"]),
                "tokens": token_set(row["text"]),
            }
        )

    return issues, parsed


def exact_duplicate_count(rows: List[dict]) -> int:
    seen: set[str] = set()
    dups = 0
    for row in rows:
        if row["norm"] in seen:
            dups += 1
        else:
            seen.add(row["norm"])
    return dups


def near_duplicates_across(train: List[dict], test: List[dict], threshold: float) -> int:
    hits = 0
    test_tokens = [row["tokens"] for row in test]
    for row in train:
        tok = row["tokens"]
        if any(jaccard(tok, other) >= threshold for other in test_tokens):
            hits += 1
    return hits


def summarize(rows: List[dict]) -> dict:
    is_act_counts = Counter()
    intent_counts = Counter()
    domain_counts = Counter()
    tasks_per_record: List[int] = []

    for row in rows:
        obj = row["obj"]
        is_act_counts[obj["is_act"]] += 1
        intent_counts[obj["intent"]] += 1
        tasks_per_record.append(len(obj["tasks"]))
        domain_counts[infer_domain_from_text_and_tasks(row["text"], obj["tasks"])] += 1

    avg_tasks = sum(tasks_per_record) / max(1, len(tasks_per_record))
    return {
        "total": len(rows),
        "is_act": {str(k): v for k, v in sorted(is_act_counts.items())},
        "intent": dict(intent_counts),
        "domain_estimate": dict(domain_counts),
        "avg_tasks_per_record": round(avg_tasks, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate C2C dataset JSONL files")
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/c2c_qc.json"))
    parser.add_argument("--expected-train", type=int, default=800)
    parser.add_argument("--expected-test", type=int, default=200)
    parser.add_argument("--expected-non-actionable-ratio", type=float, default=0.30)
    parser.add_argument("--expected-business-ratio", type=float, default=0.50)
    parser.add_argument("--ratio-tolerance", type=float, default=0.02)
    parser.add_argument("--max-tasks", type=int, default=3)
    parser.add_argument("--near-dup-threshold", type=float, default=0.92)
    parser.add_argument(
        "--enforce-domain-ratio-check",
        action="store_true",
        help="Fail validation when estimated business ratio is out of tolerance",
    )
    parser.add_argument("--strict", action="store_true", help="Return non-zero when checks fail")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    train_rows_raw = read_jsonl(args.train)
    test_rows_raw = read_jsonl(args.test)

    issues_train, train_rows = validate_rows(train_rows_raw, "train", args.max_tasks)
    issues_test, test_rows = validate_rows(test_rows_raw, "test", args.max_tasks)

    issues: List[str] = []
    issues.extend(issues_train)
    issues.extend(issues_test)

    if len(train_rows_raw) != args.expected_train:
        issues.append(f"train size mismatch: got {len(train_rows_raw)} expected {args.expected_train}")
    if len(test_rows_raw) != args.expected_test:
        issues.append(f"test size mismatch: got {len(test_rows_raw)} expected {args.expected_test}")

    train_exact_dups = exact_duplicate_count(train_rows)
    test_exact_dups = exact_duplicate_count(test_rows)
    cross_exact_dups = len({row["norm"] for row in train_rows} & {row["norm"] for row in test_rows})
    cross_near_dups = near_duplicates_across(train_rows, test_rows, threshold=args.near_dup_threshold)

    if train_exact_dups:
        issues.append(f"train has {train_exact_dups} exact duplicate texts")
    if test_exact_dups:
        issues.append(f"test has {test_exact_dups} exact duplicate texts")
    if cross_exact_dups:
        issues.append(f"train/test share {cross_exact_dups} exact duplicate texts")
    if cross_near_dups:
        issues.append(f"train/test have {cross_near_dups} near-duplicate pairs above threshold")

    all_rows = train_rows + test_rows
    non_actionable = sum(1 for row in all_rows if row["obj"]["is_act"] == 0)
    actual_non_actionable_ratio = ratio(non_actionable, len(all_rows))

    domain_counter = Counter(
        infer_domain_from_text_and_tasks(row["text"], row["obj"]["tasks"])
        for row in all_rows
    )
    business_plus_personal = domain_counter["business"] + domain_counter["personal"]
    actual_business_ratio = ratio(domain_counter["business"], business_plus_personal)

    if not check_close(actual_non_actionable_ratio, args.expected_non_actionable_ratio, args.ratio_tolerance):
        issues.append(
            "non-actionable ratio out of tolerance: "
            f"actual={actual_non_actionable_ratio:.3f} "
            f"expected={args.expected_non_actionable_ratio:.3f} tol={args.ratio_tolerance:.3f}"
        )

    if (
        args.enforce_domain_ratio_check
        and business_plus_personal > 0
        and not check_close(actual_business_ratio, args.expected_business_ratio, args.ratio_tolerance)
    ):
        issues.append(
            "business ratio out of tolerance (estimated): "
            f"actual={actual_business_ratio:.3f} "
            f"expected={args.expected_business_ratio:.3f} tol={args.ratio_tolerance:.3f}"
        )

    report = {
        "config": {
            "expected_train": args.expected_train,
            "expected_test": args.expected_test,
            "expected_non_actionable_ratio": args.expected_non_actionable_ratio,
            "expected_business_ratio": args.expected_business_ratio,
            "ratio_tolerance": args.ratio_tolerance,
            "max_tasks": args.max_tasks,
            "near_dup_threshold": args.near_dup_threshold,
            "enforce_domain_ratio_check": args.enforce_domain_ratio_check,
        },
        "summary": {
            "train": summarize(train_rows),
            "test": summarize(test_rows),
            "all": summarize(all_rows),
            "duplicates": {
                "train_exact": train_exact_dups,
                "test_exact": test_exact_dups,
                "cross_exact": cross_exact_dups,
                "cross_near": cross_near_dups,
            },
            "actual_non_actionable_ratio": round(actual_non_actionable_ratio, 3),
            "actual_business_ratio_estimated": round(actual_business_ratio, 3),
        },
        "issues": issues,
        "pass": len(issues) == 0,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote report to {args.report}")
    print(f"Validation pass: {report['pass']}")
    if issues:
        for issue in issues:
            print(f"- {issue}")

    if args.strict and issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
