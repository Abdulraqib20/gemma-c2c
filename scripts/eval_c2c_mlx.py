#!/usr/bin/env python3
"""Evaluate local MLX C2C model on JSONL test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from mlx_lm import load

from c2c_mlx_core import run_once

ALLOWED_INTENTS = {"remind", "schedule", "log", "notify"}
ALLOWED_PRI = {"H", "M", "L"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error in {path}:{line_no}: {exc}"
                ) from exc
    return rows


def valid_schema(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"is_act", "intent", "tasks"}:
        return False
    if obj["is_act"] not in (0, 1):
        return False
    if obj["intent"] not in ALLOWED_INTENTS:
        return False
    if not isinstance(obj["tasks"], list):
        return False
    if obj["is_act"] == 0 and obj["tasks"]:
        return False
    for task in obj["tasks"]:
        if not isinstance(task, dict):
            return False
        if set(task.keys()) != {"act", "who", "due", "pri"}:
            return False
        if not all(
            isinstance(task[k], str) and task[k].strip() for k in ("act", "who", "due")
        ):
            return False
        if task["pri"] not in ALLOWED_PRI:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local MLX C2C model")
    parser.add_argument(
        "--model", type=str, default="mlx_models/c2c-gemma4-e4b-it-4bit"
    )
    parser.add_argument("--data", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--no-repair-schema",
        action="store_true",
        help="Disable post-repair to measure raw cleaned generations.",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/mlx_eval_summary.json")
    )
    parser.add_argument(
        "--predictions", type=Path, default=Path("reports/mlx_eval_predictions.jsonl")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.data)
    if args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        raise ValueError("No rows to evaluate")

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    parsed_ok = 0
    schema_ok = 0
    exact_ok = 0
    is_act_ok = 0
    intent_ok = 0
    task_count_ok = 0

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as pred_f:
        for idx, row in enumerate(rows, start=1):
            text = str(row.get("text", ""))
            gold_raw = str(row.get("label", ""))
            pred_raw = run_once(
                model,
                tokenizer,
                text,
                max_tokens=args.max_tokens,
                temp=args.temp,
                top_p=args.top_p,
                verbose=False,
                repair_schema=not args.no_repair_schema,
            )

            gold_obj = None
            pred_obj = None
            try:
                gold_obj = yaml.safe_load(gold_raw)
            except yaml.YAMLError:
                gold_obj = None
            try:
                pred_obj = yaml.safe_load(pred_raw)
                parsed_ok += 1
            except yaml.YAMLError:
                pred_obj = None

            pred_schema_ok = valid_schema(pred_obj)
            if pred_schema_ok:
                schema_ok += 1

            if isinstance(gold_obj, dict) and isinstance(pred_obj, dict):
                if pred_obj.get("is_act") == gold_obj.get("is_act"):
                    is_act_ok += 1
                if pred_obj.get("intent") == gold_obj.get("intent"):
                    intent_ok += 1
                if len(pred_obj.get("tasks", [])) == len(gold_obj.get("tasks", [])):
                    task_count_ok += 1
                if pred_obj == gold_obj:
                    exact_ok += 1

            record = {
                "index": idx,
                "text": text,
                "gold": gold_obj,
                "pred_text": pred_raw,
                "pred": pred_obj,
                "pred_parse_ok": pred_obj is not None,
                "pred_schema_ok": pred_schema_ok,
            }
            pred_f.write(json.dumps(record, ensure_ascii=True) + "\n")

            if idx % 25 == 0:
                print(f"Processed {idx}/{len(rows)}")

    n = len(rows)
    summary = {
        "model": args.model,
        "data": str(args.data),
        "count": n,
        "settings": {
            "max_tokens": args.max_tokens,
            "temp": args.temp,
            "top_p": args.top_p,
            "repair_schema": not args.no_repair_schema,
        },
        "metrics": {
            "parse_rate": round(parsed_ok / n, 4),
            "schema_rate": round(schema_ok / n, 4),
            "is_act_accuracy": round(is_act_ok / n, 4),
            "intent_accuracy": round(intent_ok / n, 4),
            "task_count_accuracy": round(task_count_ok / n, 4),
            "exact_match": round(exact_ok / n, 4),
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nEvaluation complete")
    print(json.dumps(summary, indent=2))
    print(f"Predictions: {args.predictions}")
    print(f"Summary: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
