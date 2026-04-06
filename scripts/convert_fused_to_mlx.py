#!/usr/bin/env python3
"""Convert fused Hugging Face model into MLX format for Apple Silicon.

This script wraps `mlx_lm.convert` and defaults to 4-bit quantization.

Example:
  python scripts/convert_fused_to_mlx.py \
    --hf-path fused/c2c_gemma4_e4b_it_fused \
    --mlx-path mlx_models/c2c-gemma4-e4b-it-4bit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert fused model to MLX format")
    parser.add_argument(
        "--hf-path",
        type=str,
        default="fused/c2c_gemma4_e4b_it_fused",
        help="Local fused model folder or Hugging Face repo id.",
    )
    parser.add_argument(
        "--mlx-path",
        type=Path,
        default=Path("mlx_models/c2c-gemma4-e4b-it-4bit"),
        help="Destination folder for MLX model.",
    )
    parser.add_argument("--q-bits", type=int, default=4, help="Quantization bits.")
    parser.add_argument(
        "--q-group-size", type=int, default=64, help="Quantization group size."
    )
    parser.add_argument(
        "--q-mode",
        type=str,
        default="affine",
        choices=("affine", "mxfp4", "nvfp4", "mxfp8"),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="Conversion dtype for non-quantized params.",
    )
    parser.add_argument(
        "--upload-repo",
        type=str,
        default="",
        help="Optional HF repo id to upload converted MLX model.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing --mlx-path before converting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mlx_path = args.mlx_path.resolve()

    if mlx_path.exists():
        if not args.force:
            print(
                f"Destination already exists: {mlx_path}\n"
                "Use --force to remove it first.",
                file=sys.stderr,
            )
            return 2
        import shutil

        shutil.rmtree(mlx_path)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.convert",
        "--hf-path",
        args.hf_path,
        "--mlx-path",
        str(mlx_path),
        "--quantize",
        "--q-bits",
        str(args.q_bits),
        "--q-group-size",
        str(args.q_group_size),
        "--q-mode",
        args.q_mode,
        "--dtype",
        args.dtype,
    ]

    if args.upload_repo.strip():
        cmd.extend(["--upload-repo", args.upload_repo.strip()])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    print("Running:")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        return proc.returncode

    print("\nMLX conversion complete.")
    print(f"Output: {mlx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
