# C2C Local M1 Gradio Demo

Run this app locally on your Mac (Apple Silicon) after MLX conversion.

## Install

```bash
conda run -n ai pip install -r spaces/local-c2c-m1/requirements.txt
```

## Launch

```bash
conda run -n ai python spaces/local-c2c-m1/app.py
```

Then open `http://127.0.0.1:7860` in your browser.

## Notes

- Default model path is `mlx_models/c2c-gemma4-e4b-it-4bit` (repo root). Override with env `C2C_MLX_MODEL` if needed.
- The UI shows one **C2C YAML** panel: normalized via PyYAML when parse succeeds; on parse failure, raw model text is shown and **Status** explains the error.
- Optional schema-repair runs inside `run_once` when the checkbox is enabled.
