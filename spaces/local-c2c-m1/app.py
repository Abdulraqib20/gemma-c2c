#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import yaml
from mlx_lm import load

import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from c2c_mlx_core import run_once

DEFAULT_MODEL = str(ROOT / "mlx_models" / "c2c-gemma4-e4b-it-4bit")
# Override without exposing in the UI: C2C_MLX_MODEL=/path/to/mlx_model
MODEL_PATH = os.environ.get("C2C_MLX_MODEL", DEFAULT_MODEL).strip()

STATE = {
    "model_path": None,
    "model": None,
    "tokenizer": None,
}


def get_model_and_tokenizer(model_path: str):
    if STATE["model"] is None or STATE["model_path"] != model_path:
        model, tokenizer = load(model_path)
        STATE["model"] = model
        STATE["tokenizer"] = tokenizer
        STATE["model_path"] = model_path
    return STATE["model"], STATE["tokenizer"]


def infer(
    messy_text: str,
    max_tokens: int,
    temp: float,
    top_p: float,
    repair_schema: bool,
):
    text = (messy_text or "").strip()
    if not text:
        return "Please enter a message.", "", ""

    model, tokenizer = get_model_and_tokenizer(MODEL_PATH)
    yaml_text = run_once(
        model,
        tokenizer,
        text,
        max_tokens=max_tokens,
        temp=temp,
        top_p=top_p,
        verbose=False,
        repair_schema=repair_schema,
    )

    parsed = ""
    status = "YAML generated."
    try:
        obj = yaml.safe_load(yaml_text)
        parsed = yaml.safe_dump(obj, sort_keys=False, allow_unicode=False).strip()
    except yaml.YAMLError as exc:
        status = f"YAML parse failed: {exc}"

    return status, yaml_text, parsed


with gr.Blocks(title="C2C M1 Local Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# C2C Gemma 4 Local Demo (MLX on Apple Silicon)")
    gr.Markdown(
        "Paste messy text and get strict C2C YAML extraction. "
        "This app loads your local MLX model once and reuses it."
    )

    with gr.Row():
        messy_text = gr.Textbox(
            label="Messy input text",
            lines=6,
            placeholder="e.g. hey remind me to send invoice to Sarah tomorrow noon and add milk",
        )

    with gr.Row():
        max_tokens = gr.Slider(
            label="Max new tokens", minimum=64, maximum=512, value=220, step=1
        )
        temp = gr.Slider(
            label="Temperature", minimum=0.0, maximum=1.0, value=0.0, step=0.05
        )
        top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=1.0, step=0.05)

    repair_schema = gr.Checkbox(label="Repair schema after generation", value=True)

    run_btn = gr.Button("Extract YAML", variant="primary")

    status = gr.Textbox(label="Status", interactive=False)
    yaml_out = gr.Code(label="YAML output", language="yaml")
    parsed_out = gr.Code(label="Parsed object (normalized YAML)", language="yaml")

    run_btn.click(
        fn=infer,
        inputs=[messy_text, max_tokens, temp, top_p, repair_schema],
        outputs=[status, yaml_out, parsed_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
