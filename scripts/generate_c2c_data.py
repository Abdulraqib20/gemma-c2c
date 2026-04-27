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
NON_ACTIONABLE_INTENT = "log"
GENERIC_WHO_CHOICES: Tuple[str, ...] = ("me", "Aisha", "Ken", "Alex", "Mina", "legal", "finance", "ops")
POLITE_FILLERS: Tuple[str, ...] = ("pls", "please", "thanks", "if possible", "thx")

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

INTENT_PHRASES = {
    "remind": (
        "remind me to",
        "can you remind me to",
        "dont let me forget to",
        "make sure i",
        "remember to",
    ),
    "schedule": (
        "schedule",
        "book",
        "set up",
        "put on my calendar",
        "move",
    ),
    "log": (
        "log this",
        "note down",
        "track",
        "record",
        "capture",
    ),
    "notify": (
        "ping",
        "let",
        "tell",
        "message",
        "notify",
    ),
}

PRIORITY_CUES: Dict[str, Tuple[str, ...]] = {
    "H": ("urgent", "high prio", "super important", "asap", "critical"),
    "M": ("", "normal priority", "standard", "medium prio"),
    "L": ("not urgent", "whenever", "low prio", "no rush", "backlog"),
}

INTENT_ACTION_BANKS: Dict[str, Dict[str, Tuple[Dict[str, str], ...]]] = {
    "business": {
        "remind": (
            {"act": "send the revised invoice to Acme", "who": "me", "due": "tomorrow noon"},
            {"act": "follow up on the unpaid invoice", "who": "me", "due": "Friday EOD"},
            {"act": "review the contract redlines", "who": "me", "due": "this afternoon"},
            {"act": "submit the expense report", "who": "me", "due": "end of week"},
        ),
        "schedule": (
            {"act": "set up the deliverables meeting", "who": "Alex", "due": "Wednesday close of business"},
            {"act": "book the roadmap review", "who": "ops", "due": "next Monday morning"},
            {"act": "move standup to 9:30", "who": "me", "due": "tomorrow"},
            {"act": "schedule the hiring panel", "who": "Aisha", "due": "Friday afternoon"},
        ),
        "log": (
            {"act": "log the CRM notes from the client call", "who": "me", "due": "after lunch"},
            {"act": "record the Q2 budget delta", "who": "finance", "due": "today"},
            {"act": "track the procurement status on the laptop order", "who": "ops", "due": "end of week"},
            {"act": "note down the sprint blockers", "who": "me", "due": "before standup"},
        ),
        "notify": (
            {"act": "send the signed contract", "who": "legal", "due": "tomorrow 5pm sharp"},
            {"act": "ping the team about moving standup", "who": "team", "due": "tomorrow"},
            {"act": "share sprint status", "who": "product", "due": "today"},
            {"act": "tell finance about the late invoice", "who": "finance", "due": "this afternoon"},
        ),
    },
    "personal": {
        "remind": (
            {"act": "call the dentist", "who": "me", "due": "next Tuesday"},
            {"act": "pay the electricity bill", "who": "me", "due": "tomorrow evening"},
            {"act": "refill my gym membership", "who": "me", "due": "this weekend"},
            {"act": "pick up the dry cleaning", "who": "me", "due": "after work"},
        ),
        "schedule": (
            {"act": "book a haircut", "who": "me", "due": "Saturday morning"},
            {"act": "schedule the car service", "who": "me", "due": "next week"},
            {"act": "book the dentist appointment", "who": "me", "due": "Friday afternoon"},
            {"act": "set up the study block", "who": "me", "due": "tonight"},
        ),
        "log": (
            {"act": "track my weight update", "who": "me", "due": "Sunday night"},
            {"act": "note down the medication refill count", "who": "me", "due": "tonight"},
            {"act": "record the grocery spend", "who": "me", "due": "after the Saturday shop"},
            {"act": "capture the landlord leak notes", "who": "me", "due": "after work"},
        ),
        "notify": (
            {"act": "text the landlord about the leak", "who": "landlord", "due": "asap"},
            {"act": "message Mom about Sunday plans", "who": "Mom", "due": "Friday"},
            {"act": "tell my roommate about the rent transfer", "who": "roommate", "due": "tonight"},
            {"act": "ping my trainer about the new gym time", "who": "trainer", "due": "tomorrow morning"},
        ),
    },
}

TARGETED_CASES: Tuple[Dict[str, object], ...] = (
    {
        "domain": "personal",
        "intent": "remind",
        "tasks": [{"act": "call the dentist", "who": "me", "due": "next Tuesday", "pri": "M"}],
        "texts": (
            "remind me call dentist pls sometime next week idk tuesday maybe",
            "can you remind me to call the dentist next tuesday pls mom keeps asking",
        ),
    },
    {
        "domain": "business",
        "intent": "notify",
        "tasks": [{"act": "send the signed contract", "who": "legal", "due": "tomorrow 5pm sharp", "pri": "H"}],
        "texts": (
            "tell legal about the signed contract by tomorrow 5pm sharp",
            "please tell legal about the signed contract by tomorrow 5pm sharp",
        ),
    },
    {
        "domain": "business",
        "intent": "schedule",
        "tasks": [{"act": "set up the meeting", "who": "boss", "due": "9pm ET", "pri": "H"}],
        "texts": (
            "lol yesterday my boss asked me to set up a meeting by 9pm ET for the next deliverables chat",
            "set up a meeting for boss by 9pm ET re the next business deliverables",
        ),
    },
    {
        "domain": "personal",
        "intent": "log",
        "tasks": [
            {
                "act": "add bananas eggs coffee and that weird cheese Marc likes to the grocery list",
                "who": "me",
                "due": "Saturday shop",
                "pri": "M",
            }
        ],
        "texts": (
            "yo add bananas eggs coffee and that weird cheese Marc likes to the list for saturday shop thanks",
            "note down bananas eggs coffee and the weird cheese Marc likes for the saturday shop list",
        ),
    },
)


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


def canonicalize_who(value: str) -> str:
    clean = re.sub(r"\s+", " ", value.strip())
    lowered = clean.lower()
    alias = {
        "myself": "me",
        "i": "me",
        "my": "me",
        "dad": "Dad",
        "mom": "Mom",
    }
    return alias.get(lowered, clean)


def choose_priority(rnd: random.Random) -> str:
    return rnd.choices(PRIORITIES, weights=(0.2, 0.6, 0.2), k=1)[0]


def pick_intent(is_act: int, rnd: random.Random) -> str:
    if not is_act:
        return NON_ACTIONABLE_INTENT
    return rnd.choices(INTENTS, weights=(0.3, 0.25, 0.2, 0.25), k=1)[0]


def pick_task(domain: str, intent: str, rnd: random.Random) -> Dict[str, str]:
    base = dict(rnd.choice(INTENT_ACTION_BANKS[domain][intent]))
    base["who"] = canonicalize_who(base["who"])
    base["pri"] = choose_priority(rnd)
    return base


def sample_unique_tasks(
    domain: str,
    intent: str,
    task_count: int,
    rnd: random.Random,
) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    attempts = 0
    max_attempts = max(12, task_count * 10)

    while len(tasks) < task_count and attempts < max_attempts:
        attempts += 1
        task = pick_task(domain, intent, rnd)
        key = (task["act"], task["who"], task["due"])
        if key in seen:
            continue
        seen.add(key)
        tasks.append(task)

    if len(tasks) < task_count:
        raise RuntimeError(
            f"Could not sample {task_count} unique tasks for domain={domain} intent={intent}"
        )
    return tasks


def maybe_make_targeted_sample(domain: str, is_act: int, rnd: random.Random) -> Sample | None:
    if not is_act or rnd.random() >= 0.28:
        return None
    candidates = [case for case in TARGETED_CASES if case["domain"] == domain]
    case = rnd.choice(candidates)
    tasks = [dict(task) for task in case["tasks"]]  # shallow copy
    label_obj = {"is_act": 1, "intent": case["intent"], "tasks": tasks}
    label = canonical_yaml(label_obj)
    text = messify(rnd.choice(case["texts"]), rnd)
    return Sample(text=text, label=label, domain=domain, is_act=1)


def render_actionable_text(
    domain: str,
    tasks: Sequence[Dict[str, str]],
    intent: str,
    rnd: random.Random,
) -> str:
    opener = rnd.choice(OPENERS[domain])
    chunks: List[str] = [opener]
    intent_phrase = rnd.choice(INTENT_PHRASES[intent])

    for idx, task in enumerate(tasks, start=1):
        connectors = (
            "also",
            "and",
            "plus",
            "one more",
            "need this too",
            "adding one more",
        )
        if idx == 1:
            prefix = intent_phrase
        else:
            prefix = rnd.choice(connectors)

        if intent == "notify":
            if prefix in ("let", "tell"):
                line = f"{prefix} {task['who']} know about {task['act']} by {task['due']}"
            elif prefix == "message":
                line = f"message {task['who']} about {task['act']} by {task['due']}"
            else:
                line = f"{prefix} {task['who']} about {task['act']} by {task['due']}"
        elif intent == "schedule":
            variants = (
                f"{prefix} {task['act']} for {task['who']} by {task['due']}",
                f"{prefix} {task['act']} by {task['due']} for {task['who']}",
                f"{prefix} {task['act']} and keep {task['who']} on it by {task['due']}",
            )
            line = rnd.choice(variants)
        elif intent == "log":
            variants = (
                f"{prefix} {task['act']} for {task['who']} by {task['due']}",
                f"{prefix} {task['act']} before {task['due']} for {task['who']}",
            )
            line = rnd.choice(variants)
        else:
            variants = (
                f"{prefix} {task['act']} by {task['due']}",
                f"{prefix} {task['act']} for {task['who']} by {task['due']}",
                f"{prefix} {task['act']} by {task['due']} for {task['who']}",
            )
            line = rnd.choice(variants)

        if rnd.random() < 0.35:
            line += f" {rnd.choice(POLITE_FILLERS)}"

        cue = rnd.choice(PRIORITY_CUES[task["pri"]])
        if cue:
            line += f" {cue}"

        chunks.append(line.strip())

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
    targeted = maybe_make_targeted_sample(domain=domain, is_act=is_act, rnd=rnd)
    if targeted is not None:
        return targeted

    intent = pick_intent(is_act, rnd)
    if is_act:
        task_count = rnd.randint(1, max_tasks)
        tasks = sample_unique_tasks(domain, intent, task_count, rnd)
    else:
        tasks = []
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
