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
import json
import shutil
import subprocess
import sys
import tempfile
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


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_hf_path_for_mlx(hf_path: str) -> tuple[str, Path | None]:
    source = Path(hf_path)
    if not source.exists() or not source.is_dir():
        return hf_path, None

    index_path = source / "model.safetensors.index.json"
    config_path = source / "config.json"
    if not index_path.exists() or not config_path.exists():
        return hf_path, None

    index = _load_json(index_path)
    weight_map = index.get("weight_map", {})
    has_lm_head = any("lm_head.weight" in key for key in weight_map)
    has_embed_tokens = "model.language_model.embed_tokens.weight" in weight_map
    if has_lm_head or not has_embed_tokens:
        return hf_path, None

    config = _load_json(config_path)
    text_config = config.get("text_config")
    updated = False

    if config.get("tie_word_embeddings") is False:
        config["tie_word_embeddings"] = True
        updated = True
    if isinstance(text_config, dict) and text_config.get("tie_word_embeddings") is False:
        text_config["tie_word_embeddings"] = True
        updated = True

    if not updated:
        return hf_path, None

    temp_dir = Path(tempfile.mkdtemp(prefix="mlx_convert_hf_fix_"))
    staged = temp_dir / source.name
    shutil.copytree(source, staged, symlinks=True)
    (staged / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Patched config for MLX conversion: checkpoint has tied embeddings "
        "but no standalone lm_head; using tie_word_embeddings=true."
    )
    return str(staged), temp_dir


def main() -> int:
    args = parse_args()
    mlx_path = args.mlx_path.resolve()
    prepared_hf_path, temp_dir = _prepare_hf_path_for_mlx(args.hf_path)

    try:
        if mlx_path.exists():
            if not args.force:
                print(
                    f"Destination already exists: {mlx_path}\n"
                    "Use --force to remove it first.",
                    file=sys.stderr,
                )
                return 2
            shutil.rmtree(mlx_path)

        cmd = [
            sys.executable,
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            prepared_hf_path,
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
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
