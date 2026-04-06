#!/usr/bin/env python3
"""Local M1 C2C demo using mlx-lm.

Run after converting your fused model to MLX format.

Examples:
  python scripts/demo_c2c_mlx.py --model mlx_models/c2c-gemma4-e4b-it-4bit
  python scripts/demo_c2c_mlx.py --model mlx_models/c2c-gemma4-e4b-it-4bit --prompt "remind me to pay rent tomorrow"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


C2C_INSTRUCTION = """You are a structured extractor for the Chaos-to-Clarity (C2C) task.

Output rules (mandatory):
- Respond with YAML only. No markdown fences, no prose, no bullet options, no explanations.
- Keys: is_act (0 or 1), intent (remind|schedule|log|notify), tasks (list).
- Each task must have: act, who, due, pri (H|M|L).
- If is_act is 0, tasks must be an empty list.

The user message after the '---' separator is messy human text to extract from."""


def c2c_user_content(raw_user_text: str) -> str:
    return f"{C2C_INSTRUCTION}\n\n---\n\n{raw_user_text.strip()}"


def clean_yaml(raw: str) -> str:
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

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local C2C demo with mlx-lm")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx_models/c2c-gemma4-e4b-it-4bit",
        help="Path or HF repo id for MLX model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Messy user text. If empty, enters interactive mode.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--verbose", action="store_true", help="Print mlx-lm timing output."
    )
    return parser.parse_args()


def run_once(
    model,
    tokenizer,
    user_text: str,
    max_tokens: int,
    temp: float,
    top_p: float,
    verbose: bool,
) -> str:
    messages = [{"role": "user", "content": c2c_user_content(user_text)}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False
    )
    raw = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temp,
        top_p=top_p,
        verbose=verbose,
    )
    return clean_yaml(raw)


def main() -> int:
    args = parse_args()

    from mlx_lm import generate, load

    if args.model and not args.model.startswith(
        ("mlx-community/", "http://", "https://")
    ):
        _ = Path(args.model)

    print(f"Loading MLX model: {args.model}")
    model, tokenizer = load(args.model)

    if args.prompt.strip():
        out = run_once(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temp=args.temp,
            top_p=args.top_p,
            verbose=args.verbose,
        )
        print("\n--- C2C YAML ---\n")
        print(out)
        return 0

    print("Interactive mode. Enter a blank line to exit.\n")
    while True:
        user_text = input("Messy text > ").strip()
        if not user_text:
            break
        out = run_once(
            model,
            tokenizer,
            user_text,
            max_tokens=args.max_tokens,
            temp=args.temp,
            top_p=args.top_p,
            verbose=False,
        )
        print("\n--- C2C YAML ---")
        print(out)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
