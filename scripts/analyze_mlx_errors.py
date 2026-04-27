#!/usr/bin/env python3
"""
Error Analysis Script for C2C MLX Evaluation

Parses mlx_eval_predictions.jsonl and buckets errors into categories:
- INTENT_WRONG: gold vs pred intent mismatch
- WHO_IS_TIME: who field contains time-like values
- ACT_CONTAINS_FOR_WHO: act field absorbed "for <who>" pattern
- PRIORITY_OVERCALL_H: gold M/L but pred H
- TASK_COUNT_DIFF: different number of tasks

Outputs:
- reports/error_analysis.json (machine-readable summary)
- Console output with top-10 worst examples per bucket
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any

# Paths
SCRIPT_DIR = Path(__file__).parent
REPORTS_DIR = SCRIPT_DIR.parent / "reports"
PREDICTIONS_FILE = REPORTS_DIR / "mlx_eval_predictions.jsonl"
OUTPUT_FILE = REPORTS_DIR / "error_analysis.json"

# Time-like patterns for WHO_IS_TIME detection
TIME_PATTERNS = [
    r"\b(asap|today|tonight|tomorrow|morning|afternoon|evening|weekend)\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(eod|eow|end of (day|week|month))\b",
    r"\b(next|this|before|after|in \d+)\b",
    r"\b(urgent|high prio|low prio|super important)\b",
]
TIME_REGEX = re.compile("|".join(TIME_PATTERNS), re.IGNORECASE)

# Pattern for detecting "for <who>" absorbed into act
FOR_WHO_PATTERN = re.compile(r"\bfor\s+\w+\s*$", re.IGNORECASE)


def load_predictions(path: Path) -> list[dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def analyze_intent_error(gold: dict, pred: dict) -> dict | None:
    """Check for intent mismatch."""
    gold_intent = gold.get("intent", "")
    pred_intent = pred.get("intent", "")
    if gold_intent != pred_intent:
        return {
            "type": "INTENT_WRONG",
            "gold_intent": gold_intent,
            "pred_intent": pred_intent,
        }
    return None


def analyze_who_is_time(pred: dict) -> list[dict]:
    """Check if any task's who field contains time-like values."""
    errors = []
    for i, task in enumerate(pred.get("tasks", [])):
        who = task.get("who", "")
        if TIME_REGEX.search(who):
            errors.append({
                "type": "WHO_IS_TIME",
                "task_index": i,
                "who_value": who,
            })
    return errors


def analyze_act_contains_for_who(gold: dict, pred: dict) -> list[dict]:
    """Check if act field absorbed 'for <who>' pattern."""
    errors = []
    gold_tasks = gold.get("tasks", [])
    pred_tasks = pred.get("tasks", [])
    
    for i, pred_task in enumerate(pred_tasks):
        pred_act = pred_task.get("act", "")
        # Check if pred act ends with "for <something>"
        if FOR_WHO_PATTERN.search(pred_act):
            # Compare with corresponding gold task if exists
            if i < len(gold_tasks):
                gold_act = gold_tasks[i].get("act", "")
                gold_who = gold_tasks[i].get("who", "")
                # If gold act is shorter and gold who matches absorbed part
                if len(gold_act) < len(pred_act):
                    errors.append({
                        "type": "ACT_CONTAINS_FOR_WHO",
                        "task_index": i,
                        "gold_act": gold_act,
                        "pred_act": pred_act,
                        "gold_who": gold_who,
                        "pred_who": pred_task.get("who", ""),
                    })
    return errors


def analyze_priority_overcall(gold: dict, pred: dict) -> list[dict]:
    """Check if model overcalled H when gold was M or L."""
    errors = []
    gold_tasks = gold.get("tasks", [])
    pred_tasks = pred.get("tasks", [])
    
    for i, (g, p) in enumerate(zip(gold_tasks, pred_tasks)):
        gold_pri = g.get("pri", "M")
        pred_pri = p.get("pri", "M")
        if gold_pri in ("M", "L") and pred_pri == "H":
            errors.append({
                "type": "PRIORITY_OVERCALL_H",
                "task_index": i,
                "gold_pri": gold_pri,
                "pred_pri": pred_pri,
            })
    return errors


def analyze_task_count(gold: dict, pred: dict) -> dict | None:
    """Check for task count mismatch."""
    gold_count = len(gold.get("tasks", []))
    pred_count = len(pred.get("tasks", []))
    if gold_count != pred_count:
        return {
            "type": "TASK_COUNT_DIFF",
            "gold_count": gold_count,
            "pred_count": pred_count,
        }
    return None


def analyze_single_prediction(record: dict) -> list[dict]:
    """Analyze a single prediction record for all error types."""
    errors = []
    gold = record.get("gold", {})
    pred = record.get("pred", {})
    index = record.get("index", -1)
    text = record.get("text", "")
    
    base_info = {"index": index, "text": text[:100] + "..." if len(text) > 100 else text}
    
    # Skip non-actionable items for task-level analysis
    is_actionable = gold.get("is_act", 0) == 1
    
    # Intent error (always check)
    if is_actionable:
        intent_err = analyze_intent_error(gold, pred)
        if intent_err:
            errors.append({**base_info, **intent_err})
    
        # WHO_IS_TIME
        who_time_errs = analyze_who_is_time(pred)
        for err in who_time_errs:
            errors.append({**base_info, **err})
        
        # ACT_CONTAINS_FOR_WHO
        act_who_errs = analyze_act_contains_for_who(gold, pred)
        for err in act_who_errs:
            errors.append({**base_info, **err})
        
        # PRIORITY_OVERCALL_H
        pri_errs = analyze_priority_overcall(gold, pred)
        for err in pri_errs:
            errors.append({**base_info, **err})
        
        # TASK_COUNT_DIFF
        task_count_err = analyze_task_count(gold, pred)
        if task_count_err:
            errors.append({**base_info, **task_count_err})
    
    return errors


def bucket_errors(all_errors: list[dict]) -> dict[str, list[dict]]:
    """Group errors by type."""
    buckets = defaultdict(list)
    for err in all_errors:
        buckets[err["type"]].append(err)
    return dict(buckets)


def compute_summary(predictions: list[dict], buckets: dict) -> dict:
    """Compute summary statistics."""
    total = len(predictions)
    actionable = sum(1 for p in predictions if p.get("gold", {}).get("is_act", 0) == 1)
    
    # Count unique indices per error type
    unique_by_type = {}
    for err_type, errs in buckets.items():
        unique_indices = set(e["index"] for e in errs)
        unique_by_type[err_type] = len(unique_indices)
    
    return {
        "total_samples": total,
        "actionable_samples": actionable,
        "error_counts": {k: len(v) for k, v in buckets.items()},
        "unique_samples_with_error": unique_by_type,
        "error_rate_by_type": {
            k: round(unique_by_type.get(k, 0) / actionable * 100, 2) if actionable > 0 else 0
            for k in buckets.keys()
        }
    }


def print_top_examples(buckets: dict, n: int = 10):
    """Print top N examples per error bucket."""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS REPORT")
    print("=" * 80)
    
    for err_type in ["INTENT_WRONG", "ACT_CONTAINS_FOR_WHO", "WHO_IS_TIME", 
                     "PRIORITY_OVERCALL_H", "TASK_COUNT_DIFF"]:
        errors = buckets.get(err_type, [])
        print(f"\n{'─' * 40}")
        print(f"[{err_type}] — {len(errors)} occurrences")
        print(f"{'─' * 40}")
        
        if not errors:
            print("  (none)")
            continue
        
        for i, err in enumerate(errors[:n]):
            print(f"\n  Example {i+1} (index={err['index']}):")
            print(f"    Text: {err['text']}")
            
            if err_type == "INTENT_WRONG":
                print(f"    Gold: {err['gold_intent']} → Pred: {err['pred_intent']}")
            elif err_type == "WHO_IS_TIME":
                print(f"    Task {err['task_index']}: who='{err['who_value']}'")
            elif err_type == "ACT_CONTAINS_FOR_WHO":
                print(f"    Task {err['task_index']}:")
                print(f"      Gold act: '{err['gold_act']}' | who: '{err['gold_who']}'")
                print(f"      Pred act: '{err['pred_act']}' | who: '{err['pred_who']}'")
            elif err_type == "PRIORITY_OVERCALL_H":
                print(f"    Task {err['task_index']}: gold={err['gold_pri']} → pred={err['pred_pri']}")
            elif err_type == "TASK_COUNT_DIFF":
                print(f"    Gold: {err['gold_count']} tasks → Pred: {err['pred_count']} tasks")


def main():
    print(f"Loading predictions from: {PREDICTIONS_FILE}")
    predictions = load_predictions(PREDICTIONS_FILE)
    print(f"Loaded {len(predictions)} samples")
    
    # Analyze all predictions
    all_errors = []
    for record in predictions:
        errors = analyze_single_prediction(record)
        all_errors.extend(errors)
    
    print(f"Found {len(all_errors)} total error instances")
    
    # Bucket errors
    buckets = bucket_errors(all_errors)
    
    # Compute summary
    summary = compute_summary(predictions, buckets)
    
    # Build output
    output = {
        "summary": summary,
        "buckets": {k: v[:20] for k, v in buckets.items()},  # Top 20 per bucket
    }
    
    # Save to JSON
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved analysis to: {OUTPUT_FILE}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Actionable samples: {summary['actionable_samples']}")
    print("\nError counts (unique samples affected):")
    for err_type, count in summary["unique_samples_with_error"].items():
        rate = summary["error_rate_by_type"].get(err_type, 0)
        print(f"  {err_type}: {count} samples ({rate}%)")
    
    # Print top examples
    print_top_examples(buckets, n=10)
    
    return output


if __name__ == "__main__":
    main()
