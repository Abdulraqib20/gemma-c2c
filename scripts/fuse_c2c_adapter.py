#!/usr/bin/env python3
"""Fuse the trained C2C PEFT adapter into a standalone Hugging Face model.

This does not retrain anything. It reuses your existing LoRA weights from:
- local folder (e.g. c2c_output/adapter), or
- Hub repo (e.g. raqibcodes/c2c-checkpoints)

Typical use on Kaggle/Colab GPU:
  python scripts/fuse_c2c_adapter.py \
    --hub-repo raqibcodes/c2c-checkpoints \
    --output-dir fused/c2c_gemma4_e4b_it_fused

Then optionally upload fused model:
  python scripts/fuse_c2c_adapter.py \
    --hub-repo raqibcodes/c2c-checkpoints \
    --output-dir fused/c2c_gemma4_e4b_it_fused \
    --push-repo raqibcodes/c2c-gemma4-e4b-it-fused
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_dtype(value: str):
    import torch

    value = value.strip().lower()
    if value == "auto":
        return None
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse C2C PEFT adapter into full model"
    )
    parser.add_argument("--base-model", type=str, default="google/gemma-4-E4B-it")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Local adapter directory (e.g. c2c_output/adapter). If omitted, uses --hub-repo.",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default="raqibcodes/c2c-checkpoints",
        help="Hub model repo that stores adapter weights.",
    )
    parser.add_argument(
        "--hub-subfolder",
        type=str,
        default="",
        help='Optional adapter subfolder in Hub repo (e.g. "last-checkpoint").',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fused/c2c_gemma4_e4b_it_fused"),
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Transformers device_map, e.g. "auto", "cpu", "cuda:0".',
    )
    parser.add_argument("--max-shard-size", type=str, default="5GB")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Optional HF token. If omitted, uses cached auth from `hf auth login`.",
    )
    parser.add_argument(
        "--push-repo",
        type=str,
        default="",
        help="Optional Hub repo id to upload fused model to.",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create pushed repo as private."
    )
    return parser.parse_args()


def infer_base_model(base_model_arg: str, adapter_dir: Path | None) -> str:
    if adapter_dir is None:
        return base_model_arg
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return base_model_arg
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return base_model_arg
    inferred = payload.get("base_model_name_or_path")
    if isinstance(inferred, str) and inferred.strip():
        return inferred.strip()
    return base_model_arg


def load_tokenizer(
    *,
    base_model: str,
    adapter_dir: Path | None,
    hub_repo: str,
    hub_subfolder: str | None,
    trust_remote_code: bool,
    hf_token: str,
):
    from transformers import AutoTokenizer

    tok_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if hf_token:
        tok_kwargs["token"] = hf_token

    if adapter_dir is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), **tok_kwargs)
            return tokenizer
        except Exception:
            pass

    try:
        if hub_subfolder:
            tokenizer = AutoTokenizer.from_pretrained(
                hub_repo, subfolder=hub_subfolder, **tok_kwargs
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(hub_repo, **tok_kwargs)
        return tokenizer
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, **tok_kwargs)
        return tokenizer


def push_folder_to_hub(
    folder: Path, repo_id: str, private: bool, hf_token: str
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token or None)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(folder), repo_id=repo_id, repo_type="model"
    )
    print(f"Uploaded fused model to https://huggingface.co/{repo_id}")
    print(f"Commit: {commit}")


def main() -> int:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter_dir = args.adapter.resolve() if args.adapter is not None else None
    if adapter_dir is not None and not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    hub_subfolder = args.hub_subfolder.strip() or None
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = infer_base_model(args.base_model, adapter_dir)
    dtype = parse_dtype(args.dtype)

    print(f"Base model: {base_model}")
    if adapter_dir is not None:
        print(f"Adapter source: local path {adapter_dir}")
    else:
        source = args.hub_repo + (f"/{hub_subfolder}" if hub_subfolder else "")
        print(f"Adapter source: hub {source}")
    print(f"Output directory: {output_dir}")

    tokenizer = load_tokenizer(
        base_model=base_model,
        adapter_dir=adapter_dir,
        hub_repo=args.hub_repo,
        hub_subfolder=hub_subfolder,
        trust_remote_code=args.trust_remote_code,
        hf_token=args.hf_token.strip(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.hf_token.strip():
        model_kwargs["token"] = args.hf_token.strip()

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    peft_kwargs: dict[str, Any] = {}
    if args.hf_token.strip():
        peft_kwargs["token"] = args.hf_token.strip()

    print("Loading adapter and merging...")
    if adapter_dir is not None:
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir), **peft_kwargs)
    elif hub_subfolder:
        peft_model = PeftModel.from_pretrained(
            base, args.hub_repo, subfolder=hub_subfolder, **peft_kwargs
        )
    else:
        peft_model = PeftModel.from_pretrained(base, args.hub_repo, **peft_kwargs)

    fused_model = peft_model.merge_and_unload()

    print("Saving fused model + tokenizer...")
    fused_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "base_model": base_model,
        "adapter_local": str(adapter_dir) if adapter_dir is not None else None,
        "adapter_hub_repo": None if adapter_dir is not None else args.hub_repo,
        "adapter_hub_subfolder": None if adapter_dir is not None else hub_subfolder,
        "dtype": args.dtype,
        "device_map": args.device_map,
    }
    (output_dir / "c2c_fuse_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Fuse complete.")
    print(f"Saved fused model to: {output_dir}")

    push_repo = args.push_repo.strip()
    if push_repo:
        print(f"Uploading to Hub repo: {push_repo}")
        push_folder_to_hub(
            output_dir, push_repo, private=args.private, hf_token=args.hf_token.strip()
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
