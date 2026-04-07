#!/usr/bin/env python3
"""Shared MLX inference helpers for the C2C project."""

from __future__ import annotations

import re
from typing import Any

import yaml
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

C2C_INSTRUCTION = """You are a structured extractor for the Chaos-to-Clarity (C2C) task.

Output rules (mandatory):
- Respond with YAML only. No markdown fences, no prose, no explanations.
- Top-level keys must be exactly: is_act, intent, tasks.
- is_act must be 0 or 1.
- intent must be one of: remind, schedule, log, notify.
- tasks must be a list.
- If is_act is 0, tasks must be [] exactly.
- If is_act is 1, extract every actionable task from the message.
- Each task must contain exactly: act, who, due, pri.
- pri must be one of H, M, L.
- Keep `who` as person/owner/target, not a time phrase.
- Keep `due` as natural-language time phrase.

The user message after the '---' separator is messy human text to extract from."""

ALLOWED_INTENTS = {"remind", "schedule", "log", "notify"}
ALLOWED_PRI = {"H", "M", "L"}


def c2c_user_content(raw_user_text: str) -> str:
    return f"{C2C_INSTRUCTION}\n\n---\n\n{raw_user_text.strip()}"


def clean_yaml_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("yaml"):
            text = text[4:].lstrip()
        idx = text.rfind("```")
        if idx != -1:
            text = text[:idx].strip()

    stop_tokens = ["<turn|>", "<|eot_id|>", "<|end_of_text|>", "<eos>"]
    cut = len(text)
    for tok in stop_tokens:
        i = text.find(tok)
        if i != -1:
            cut = min(cut, i)
    text = text[:cut].strip()

    key_match = re.search(r"(?m)^\s*is_act\s*:\s*", text)
    if key_match:
        text = text[key_match.start() :].strip()

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _sanitize_obj(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None

    is_act = obj.get("is_act", 0)
    if isinstance(is_act, str) and is_act.strip().isdigit():
        is_act = int(is_act.strip())
    is_act = 1 if is_act == 1 else 0

    intent = str(obj.get("intent", "remind")).strip().lower()
    if intent not in ALLOWED_INTENTS:
        intent = "remind"

    raw_tasks = obj.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raw_tasks = []

    tasks: list[dict[str, str]] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            continue
        act = str(item.get("act", "")).strip()
        who = str(item.get("who", "")).strip()
        due = str(item.get("due", "")).strip()
        pri = str(item.get("pri", "M")).strip().upper()
        if not act or not who or not due:
            continue
        if pri not in ALLOWED_PRI:
            pri = "M"
        tasks.append({"act": act, "who": who, "due": due, "pri": pri})

    if is_act == 0:
        tasks = []
    elif not tasks:
        is_act = 0

    return {"is_act": is_act, "intent": intent, "tasks": tasks}


def postprocess_yaml(raw: str, repair_schema: bool = True) -> str:
    cleaned = clean_yaml_text(raw)
    if not repair_schema:
        return cleaned
    try:
        obj = yaml.safe_load(cleaned)
    except yaml.YAMLError:
        return cleaned
    fixed = _sanitize_obj(obj)
    if fixed is None:
        return cleaned
    return yaml.safe_dump(fixed, sort_keys=False, allow_unicode=False).strip()


def run_once(
    model,
    tokenizer,
    user_text: str,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 1.0,
    verbose: bool = False,
    repair_schema: bool = True,
) -> str:
    messages = [{"role": "user", "content": c2c_user_content(user_text)}]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=False,
    )

    top_p_kw = top_p if 0 < top_p < 1.0 else 0.0
    raw = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temp, top_p=top_p_kw),
        verbose=verbose,
    )
    return postprocess_yaml(raw, repair_schema=repair_schema)
