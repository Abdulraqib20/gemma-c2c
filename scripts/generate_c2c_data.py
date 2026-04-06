#!/usr/bin/env python3
"""Generate synthetic C2C dataset with strict YAML labels and split controls.

Outputs JSONL with fields:
- text: messy user message
- label: YAML string matching the C2C schema
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml

INTENTS: Tuple[str, ...] = ("remind", "schedule", "log", "notify")
PRIORITIES: Tuple[str, ...] = ("H", "M", "L")
DOMAINS: Tuple[str, ...] = ("business", "personal")

BUSINESS_TASK_BANK: Tuple[Dict[str, Sequence[str]], ...] = (
    {
        "acts": (
            "send the revised invoice to Acme",
            "follow up on the unpaid invoice",
            "prepare the Q2 budget summary",
            "review the contract redlines",
            "share sprint status with the team",
            "confirm the client kickoff agenda",
            "book a room for the roadmap review",
            "submit the expense report",
            "check procurement on laptop order",
            "update the CRM notes",
            "draft the hiring panel schedule",
            "finalize the sales deck",
        ),
        "who": ("me", "ops", "finance", "Aisha", "Ken", "product", "sales", "legal"),
        "due": (
            "today",
            "tomorrow",
            "Friday EOD",
            "next Monday morning",
            "this afternoon",
            "before standup",
            "after lunch",
            "in two days",
            "end of week",
        ),
    },
)

PERSONAL_TASK_BANK: Tuple[Dict[str, Sequence[str]], ...] = (
    {
        "acts": (
            "buy groceries for the week",
            "schedule a dentist appointment",
            "call mom about Sunday plans",
            "pay the electricity bill",
            "refill my gym membership",
            "pick up dry cleaning",
            "book the car service",
            "track my weight update",
            "set up a reminder for medication",
            "message my landlord about the leak",
            "organize the study desk",
            "order cat food",
        ),
        "who": ("me", "myself", "brother", "roommate", "Dad", "Mom", "trainer"),
        "due": (
            "tonight",
            "tomorrow morning",
            "this weekend",
            "Friday",
            "before dinner",
            "after work",
            "next week",
            "Saturday afternoon",
            "asap",
        ),
    },
)

NON_ACTIONABLE_TEXTS: Dict[str, Tuple[str, ...]] = {
    "business": (
        "brain is mush today, meetings everywhere and honestly just venting, no asks right now",
        "quick thought dump: the week felt chaotic but i just needed to write this out, nothing to do",
        "i keep overthinking the roadmap presentation but this is only a note to self, no action",
        "random update, inbox is noisy and i am tired, not asking for anything",
        "sharing this so i remember the vibe from today, not a task list",
    ),
    "personal": (
        "just journaling here, slept late and felt weird all day, no action needed",
        "tiny rant: weather changed and my mood is off, not asking to do anything",
        "writing this to clear my head, no tasks from this message",
        "just a brain dump about life admin stress, nothing actionable",
        "note to self mood check, no request and no reminder needed",
    ),
}

FILLERS: Tuple[str, ...] = (
    "uh",
    "like",
    "anyway",
    "kinda",
    "pls",
    "thx",
    "btw",
    "real quick",
    "if possible",
)

OPENERS: Dict[str, Tuple[str, ...]] = {
    "business": (
        "hey team, messy note",
        "quick ops brain dump",
        "sorry this is all over the place",
        "random work thought",
        "dropping this before i forget",
    ),
    "personal": (
        "ok random life note",
        "brain dump incoming",
        "this is messy sorry",
        "quick personal reminder cloud",
        "dumping thoughts before i sleep",
    ),
}


@dataclass
class Sample:
    text: str
    label: str
    domain: str
    is_act: int


def canonical_yaml(obj: dict) -> str:
    dumped = yaml.safe_dump(obj, sort_keys=False, allow_unicode=False)
    return dumped.strip()


def validate_label_yaml(label_yaml: str, max_tasks: int) -> Tuple[bool, str, dict | None]:
    try:
        obj = yaml.safe_load(label_yaml)
    except yaml.YAMLError as exc:
        return False, f"YAML parse error: {exc}", None

    if not isinstance(obj, dict):
        return False, "Label must parse to mapping", None

    expected_top = {"is_act", "intent", "tasks"}
    if set(obj.keys()) != expected_top:
        return False, "Top-level keys must be exactly is_act,intent,tasks", None

    if obj["is_act"] not in (0, 1):
        return False, "is_act must be 0 or 1", None

    if obj["intent"] not in INTENTS:
        return False, "intent must be remind|schedule|log|notify", None

    tasks = obj["tasks"]
    if not isinstance(tasks, list):
        return False, "tasks must be a list", None

    if obj["is_act"] == 0 and tasks:
        return False, "is_act=0 requires empty tasks", None

    if obj["is_act"] == 1 and not (1 <= len(tasks) <= max_tasks):
        return False, f"is_act=1 requires 1..{max_tasks} tasks", None

    for task in tasks:
        if not isinstance(task, dict):
            return False, "each task must be a mapping", None
        if set(task.keys()) != {"act", "who", "due", "pri"}:
            return False, "task keys must be exactly act,who,due,pri", None
        if not all(isinstance(task[k], str) and task[k].strip() for k in ("act", "who", "due")):
            return False, "task act/who/due must be non-empty strings", None
        if task["pri"] not in PRIORITIES:
            return False, "task pri must be H|M|L", None

        # Keep due in natural-language style by rejecting strict ISO date.
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", task["due"].strip()):
            return False, "due must be natural-language, not strict ISO date", None

    return True, "ok", obj


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


def typo_word(word: str, rnd: random.Random) -> str:
    if len(word) < 4 or rnd.random() > 0.08:
        return word

    mode = rnd.choice(("drop", "swap", "double"))
    if mode == "drop":
        i = rnd.randrange(1, len(word) - 1)
        return word[:i] + word[i + 1 :]
    if mode == "swap" and len(word) >= 5:
        i = rnd.randrange(1, len(word) - 2)
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    i = rnd.randrange(1, len(word) - 1)
    return word[:i] + word[i] + word[i:]


def messify(text: str, rnd: random.Random) -> str:
    words = text.split(" ")
    out = [typo_word(w, rnd) for w in words]
    result = " ".join(out)

    if rnd.random() < 0.4:
        result += rnd.choice((" ...", " lol", "", "  "))
    if rnd.random() < 0.35:
        result = result.replace(" and ", rnd.choice((" + ", " & ", " annd ")), 1)
    return re.sub(r"\s+", " ", result).strip()


def pick_task(domain: str, rnd: random.Random) -> Dict[str, str]:
    bank = BUSINESS_TASK_BANK[0] if domain == "business" else PERSONAL_TASK_BANK[0]
    return {
        "act": rnd.choice(tuple(bank["acts"])),
        "who": rnd.choice(tuple(bank["who"])),
        "due": rnd.choice(tuple(bank["due"])),
        "pri": rnd.choice(PRIORITIES),
    }


def infer_intent(tasks: Sequence[Dict[str, str]], rnd: random.Random) -> str:
    if not tasks:
        return rnd.choice(INTENTS)

    acts = " ".join(task["act"].lower() for task in tasks)
    if any(k in acts for k in ("schedule", "book", "appointment", "meeting", "kickoff")):
        return "schedule"
    if any(k in acts for k in ("log", "track", "update", "record")):
        return "log"
    if any(k in acts for k in ("message", "notify", "share", "call")):
        return "notify"
    return "remind"


def render_actionable_text(
    domain: str,
    tasks: Sequence[Dict[str, str]],
    intent: str,
    rnd: random.Random,
) -> str:
    opener = rnd.choice(OPENERS[domain])
    chunks: List[str] = [opener]

    for idx, task in enumerate(tasks, start=1):
        connectors = (
            "also",
            "and",
            "plus",
            "one more",
            "dont let me miss",
            "need this too",
        )
        prefix = rnd.choice(connectors) if idx > 1 else rnd.choice(("", "please", "can you", "remind me to"))

        line = f"{prefix} {task['act']} for {task['who']} by {task['due']}"
        if task["pri"] == "H":
            line += rnd.choice((" high prio", " urgent", " super important"))
        elif task["pri"] == "L":
            line += rnd.choice((" low prio", " not urgent", " whenever"))

        chunks.append(line.strip())

    # Add distracting side chatter to make text messy and multi-topic.
    if rnd.random() < 0.7:
        chunks.append(
            rnd.choice(
                (
                    "i also spilled coffee on my notes so this is chaotic",
                    "ignore typos i am walking while typing",
                    "calendar is a mess this week",
                    "my brain is tab-overloaded right now",
                    "also i forgot where i saved the file",
                )
            )
        )

    if rnd.random() < 0.5:
        chunks.append(rnd.choice(FILLERS))

    text = " ; ".join(chunks)
    text = messify(text, rnd)

    if intent == "notify" and rnd.random() < 0.4:
        text += " and maybe ping me once done"
    return text


def render_non_actionable_text(domain: str, rnd: random.Random) -> str:
    base = rnd.choice(NON_ACTIONABLE_TEXTS[domain])
    extras = (
        "anyway just wanted to vent",
        "no action from this btw",
        "dont convert this into tasks",
        "just context, nothing to track",
        "sharing for memory only",
    )

    text = base
    if rnd.random() < 0.55:
        text = f"{text} ; {rnd.choice(extras)}"
    return messify(text, rnd)


def make_sample(domain: str, is_act: int, max_tasks: int, rnd: random.Random) -> Sample:
    if is_act:
        task_count = rnd.randint(1, max_tasks)
        tasks = [pick_task(domain, rnd) for _ in range(task_count)]
    else:
        tasks = []

    intent = infer_intent(tasks, rnd)
    label_obj = {"is_act": int(is_act), "intent": intent, "tasks": tasks}
    label = canonical_yaml(label_obj)

    text = render_actionable_text(domain, tasks, intent, rnd) if is_act else render_non_actionable_text(domain, rnd)
    return Sample(text=text, label=label, domain=domain, is_act=int(is_act))


def compute_bucket_targets(total: int, non_actionable_ratio: float, business_ratio: float) -> Dict[Tuple[str, int], int]:
    business_total = round(total * business_ratio)
    personal_total = total - business_total

    non_total = round(total * non_actionable_ratio)
    business_non = round(non_total * business_ratio)
    personal_non = non_total - business_non

    business_act = business_total - business_non
    personal_act = personal_total - personal_non

    return {
        ("business", 1): business_act,
        ("business", 0): business_non,
        ("personal", 1): personal_act,
        ("personal", 0): personal_non,
    }


def generate_dataset(
    total: int,
    non_actionable_ratio: float,
    business_ratio: float,
    max_tasks: int,
    seed: int,
    near_dup_threshold: float,
) -> List[Sample]:
    rnd = random.Random(seed)
    targets = compute_bucket_targets(total, non_actionable_ratio, business_ratio)
    remaining = dict(targets)

    samples: List[Sample] = []
    seen_norm: set[str] = set()
    token_cache: List[set[str]] = []

    max_attempts = total * 120
    attempts = 0

    while sum(remaining.values()) > 0 and attempts < max_attempts:
        attempts += 1

        buckets = [key for key, left in remaining.items() if left > 0]
        weights = [remaining[key] for key in buckets]
        domain, is_act = rnd.choices(buckets, weights=weights, k=1)[0]

        sample = make_sample(domain=domain, is_act=is_act, max_tasks=max_tasks, rnd=rnd)
        ok, _, obj = validate_label_yaml(sample.label, max_tasks=max_tasks)
        if not ok or obj is None:
            continue

        norm = normalize_text(sample.text)
        if norm in seen_norm:
            continue

        tok = token_set(sample.text)
        if any(jaccard(tok, other) >= near_dup_threshold for other in token_cache):
            continue

        seen_norm.add(norm)
        token_cache.append(tok)
        samples.append(sample)
        remaining[(domain, is_act)] -= 1

    if sum(remaining.values()) > 0:
        raise RuntimeError(
            "Could not satisfy generation quotas. "
            f"remaining={remaining}, generated={len(samples)}, attempts={attempts}"
        )

    return samples


def allocate_test_counts(bucket_sizes: Dict[Tuple[str, int], int], test_total: int) -> Dict[Tuple[str, int], int]:
    total = sum(bucket_sizes.values())
    raw = {k: (v * test_total) / total for k, v in bucket_sizes.items()}

    base = {k: int(raw[k]) for k in raw}
    remainder = test_total - sum(base.values())

    if remainder > 0:
        ranked = sorted(raw.keys(), key=lambda k: (raw[k] - base[k]), reverse=True)
        for key in ranked[:remainder]:
            base[key] += 1

    return base


def split_dataset(samples: Sequence[Sample], train_size: int, test_size: int, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rnd = random.Random(seed)

    by_bucket: Dict[Tuple[str, int], List[Sample]] = {
        ("business", 1): [],
        ("business", 0): [],
        ("personal", 1): [],
        ("personal", 0): [],
    }

    for sample in samples:
        by_bucket[(sample.domain, sample.is_act)].append(sample)

    for bucket in by_bucket.values():
        rnd.shuffle(bucket)

    bucket_sizes = {k: len(v) for k, v in by_bucket.items()}
    bucket_test_counts = allocate_test_counts(bucket_sizes, test_size)

    train: List[Sample] = []
    test: List[Sample] = []

    for key, bucket in by_bucket.items():
        n_test = bucket_test_counts[key]
        test.extend(bucket[:n_test])
        train.extend(bucket[n_test:])

    rnd.shuffle(train)
    rnd.shuffle(test)

    if len(train) != train_size or len(test) != test_size:
        raise RuntimeError(f"Unexpected split sizes train={len(train)} test={len(test)}")

    return train, test


def write_jsonl(path: Path, samples: Iterable[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            row = {"text": sample.text, "label": sample.label}
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize(samples: Sequence[Sample]) -> dict:
    intent_counter: Counter[str] = Counter()
    act_counter: Counter[int] = Counter()
    domain_counter: Counter[str] = Counter()
    task_counts: List[int] = []

    for sample in samples:
        _, _, obj = validate_label_yaml(sample.label, max_tasks=10)
        if not obj:
            continue
        intent_counter[obj["intent"]] += 1
        act_counter[obj["is_act"]] += 1
        domain_counter[sample.domain] += 1
        task_counts.append(len(obj["tasks"]))

    avg_tasks = sum(task_counts) / max(1, len(task_counts))
    return {
        "total": len(samples),
        "intent_distribution": dict(intent_counter),
        "is_act_distribution": {str(k): v for k, v in sorted(act_counter.items())},
        "domain_distribution": dict(domain_counter),
        "avg_tasks_per_record": round(avg_tasks, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate C2C train/test JSONL dataset")
    parser.add_argument("--train", type=int, default=800)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--non-actionable-ratio", type=float, default=0.30)
    parser.add_argument("--business-ratio", type=float, default=0.50)
    parser.add_argument("--max-tasks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-dup-threshold", type=float, default=0.92)
    parser.add_argument("--out-train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--out-test", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/generation_summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    total = args.train + args.test
    samples = generate_dataset(
        total=total,
        non_actionable_ratio=args.non_actionable_ratio,
        business_ratio=args.business_ratio,
        max_tasks=args.max_tasks,
        seed=args.seed,
        near_dup_threshold=args.near_dup_threshold,
    )

    train_samples, test_samples = split_dataset(samples, args.train, args.test, seed=args.seed + 13)

    write_jsonl(args.out_train, train_samples)
    write_jsonl(args.out_test, test_samples)

    summary = {
        "config": {
            "train": args.train,
            "test": args.test,
            "non_actionable_ratio": args.non_actionable_ratio,
            "business_ratio": args.business_ratio,
            "max_tasks": args.max_tasks,
            "seed": args.seed,
            "near_dup_threshold": args.near_dup_threshold,
        },
        "all": summarize(samples),
        "train": summarize(train_samples),
        "test": summarize(test_samples),
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(train_samples)} train examples to {args.out_train}")
    print(f"Wrote {len(test_samples)} test examples to {args.out_test}")
    print(f"Wrote generation summary to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
