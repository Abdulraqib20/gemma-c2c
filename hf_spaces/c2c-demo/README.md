---
title: C2C Chaos-to-Clarity
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Messy text → structured YAML (Gemma 4 E4B-it + LoRA)
---

# Chaos-to-Clarity (C2C) demo

Gradio UI for the C2C extractor: paste messy human text, get **YAML** with `is_act`, `intent`, and `tasks`.

## Run this on Hugging Face Spaces

1. **Create a new Space** (Gradio SDK). You can duplicate this folder into a Space repo, or connect a subfolder if your monorepo supports it.
2. **Hardware → GPU** (e.g. **T4**). This app uses **4-bit bitsandbytes** and needs **CUDA**; CPU-only Spaces will not work.
3. **Gemma access**: Accept the license for [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) on Hugging Face (same account as the Space).
4. **Private adapter repo**: In the Space **Settings → Secrets**, add `HF_TOKEN` with a read token that can access `raqibcodes/c2c-checkpoints` (and the base model if gated). The app calls `huggingface_hub.login` when `HF_TOKEN` is set.

## Optional environment variables

| Variable | Default |
|----------|---------|
| `C2C_BASE_MODEL` | `google/gemma-4-E4B-it` |
| `C2C_ADAPTER_REPO` | `raqibcodes/c2c-checkpoints` |
| `C2C_ADAPTER_SUBFOLDER` | _(empty = repo root adapter)_ or e.g. `last-checkpoint` |

## Project

Developed in the [gemma-project](https://huggingface.co/raqibcodes) C2C weekend track; training uses TRL QLoRA + Hub checkpoints.
