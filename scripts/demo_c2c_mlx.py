#!/usr/bin/env python3
"""Local M1 C2C demo using mlx-lm.

Run after converting your fused model to MLX format.

Examples:
  python scripts/demo_c2c_mlx.py --model mlx_models/c2c-gemma4-e4b-it-4bit
  python scripts/demo_c2c_mlx.py --model mlx_models/c2c-gemma4-e4b-it-4bit --prompt "remind me to pay rent tomorrow"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from c2c_mlx_core import run_once


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
        "--no-repair-schema",
        action="store_true",
        help="Disable YAML/schema post-repair; return raw cleaned model output.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print mlx-lm timing output."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

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
            repair_schema=not args.no_repair_schema,
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
            repair_schema=not args.no_repair_schema,
        )
        print("\n--- C2C YAML ---")
        print(out)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
