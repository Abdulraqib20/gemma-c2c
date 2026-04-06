"""
Gradio Space: Chaos-to-Clarity (C2C) — messy text → YAML via Gemma 4 E4B + LoRA from the Hub.

Env (optional):
  C2C_BASE_MODEL         default google/gemma-4-E4B-it
  C2C_ADAPTER_REPO       default raqibcodes/c2c-checkpoints
  C2C_ADAPTER_SUBFOLDER  e.g. last-checkpoint (empty = repo root)
  HF_TOKEN               Space secret if adapter or base is private / gated
"""

from __future__ import annotations

import os
import threading

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

MODEL_NAME = os.environ.get("C2C_BASE_MODEL", "google/gemma-4-E4B-it").strip()
HUB_REPO = os.environ.get("C2C_ADAPTER_REPO", "raqibcodes/c2c-checkpoints").strip()
_sub = os.environ.get("C2C_ADAPTER_SUBFOLDER", "").strip()
HUB_SUBFOLDER = _sub or None

C2C_INSTRUCTION = """You are a structured extractor for the Chaos-to-Clarity (C2C) task.

Output rules (mandatory):
- Respond with YAML only. No markdown fences, no prose, no bullet options, no explanations.
- Keys: is_act (0 or 1), intent (remind|schedule|log|notify), tasks (list).
- Each task must have: act, who, due, pri (H|M|L).
- If is_act is 0, tasks must be an empty list.

The user message after the '---' separator is messy human text to extract from."""

_tokenizer = None
_model = None
_load_lock = threading.Lock()


def c2c_user_content(raw_user_text: str) -> str:
    return f"{C2C_INSTRUCTION}\n\n---\n\n{raw_user_text.strip()}"


def strip_thinking(raw: str) -> str:
    raw = raw.strip()
    if "<|channel|>thought" in raw:
        end = raw.rfind("<|channel|>")
        if end != -1:
            raw = raw[end + len("<|channel|>") :].strip()
    if raw.startswith("```"):
        raw = raw[3:]
        if raw.lower().startswith("yaml"):
            raw = raw[4:].lstrip()
        fence = raw.rfind("```")
        if fence != -1:
            raw = raw[:fence].strip()
    return raw


def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. In Space Settings → Hardware, enable a GPU (e.g. T4)."
        )

    _tok_kw = {}
    if HUB_SUBFOLDER:
        _tok_kw["subfolder"] = HUB_SUBFOLDER
    tokenizer = AutoTokenizer.from_pretrained(HUB_REPO, **_tok_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if HUB_SUBFOLDER:
        model = PeftModel.from_pretrained(base, HUB_REPO, subfolder=HUB_SUBFOLDER)
    else:
        model = PeftModel.from_pretrained(base, HUB_REPO)
    model.eval()
    _tokenizer, _model = tokenizer, model


@torch.inference_mode()
def extract(messy_text: str, max_new_tokens: int) -> str:
    text = (messy_text or "").strip()
    if not text:
        return "Paste some messy text first."

    with _load_lock:
        _load_model()

    messages = [{"role": "user", "content": c2c_user_content(text)}]
    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    out = _model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=_tokenizer.pad_token_id,
    )
    gen = _tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
    )
    return strip_thinking(gen)


with gr.Blocks(title="C2C Extractor") as demo:
    gr.Markdown(
        "### Chaos-to-Clarity (C2C)\n"
        "Fine-tuned **Gemma 4 E4B-it** + LoRA: messy message → **YAML** tasks.\n\n"
        "_First click may take a minute while the model loads._"
    )
    messy = gr.Textbox(
        label="Messy message",
        lines=5,
        placeholder="e.g. hey can u remind me to email sarah about the invoice by friday and grab oat milk",
    )
    max_tok = gr.Slider(64, 512, value=256, step=32, label="Max new tokens")
    out = gr.Textbox(label="C2C YAML", lines=14)
    go = gr.Button("Extract", variant="primary")
    go.click(fn=extract, inputs=[messy, max_tok], outputs=out)
