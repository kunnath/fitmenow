# CatVTON Workspace Guide

This workspace contains the CatVTON project source code in the [CatVTON](CatVTON) folder. The app is a Gradio-based virtual try-on demo that loads a try-on pipeline, downloads the model checkpoint if needed, and runs locally on your GPU.

## Project structure

- [CatVTON/app.py](CatVTON/app.py) — main Gradio app for the standard CatVTON pipeline
- [CatVTON/app_flux.py](CatVTON/app_flux.py) — FLUX-based variant
- [CatVTON/inference.py](CatVTON/inference.py) — batch inference workflow
- [CatVTON/requirements.txt](CatVTON/requirements.txt) — Python dependencies
- [CatVTON/README.md](CatVTON/README.md) — upstream project documentation and training/inference notes

## Requirements

- Python 3.9+
- NVIDIA GPU recommended for normal use
- CUDA-enabled PyTorch build
- Git and Hugging Face access for model downloads

## Quick start

From the workspace root:

```bash
cd /Users/kunnath/projects/CatVTON
python3 -m venv .venv
source .venv/bin/activate
pip install -r CatVTON/requirements.txt
```

Then start the app from the project directory:

```bash
cd CatVTON
python app.py \
  --output_dir="resource/demo/output" \
  --mixed_precision="bf16" \
  --allow_tf32
```

If you are using a non-Ampere or CPU-only setup, use `--mixed_precision="no"` instead of `bf16`.

## Run the FLUX version

If you want to launch the FLUX demo instead of the default app:

```bash
cd CatVTON
python app_flux.py \
  --output_dir="resource/demo/output" \
  --mixed_precision="bf16" \
  --allow_tf32
```

## Common issues

### Running from the wrong directory

The app is not at the workspace root. It is under `CatVTON/`, so this will fail:

```bash
python app.py
```

Use:

```bash
cd CatVTON
python app.py
```

### Dependency problems

If the environment is missing libraries, reinstall dependencies:

```bash
source .venv/bin/activate
pip install -r CatVTON/requirements.txt
```

### GPU / precision issues

- Use `bf16` on modern NVIDIA GPUs
- Use `fp16` if your GPU supports it but not `bf16`
- Use `no` for CPU-only or older GPUs

## Inference and evaluation

The project also supports dataset inference and metric evaluation. See the upstream docs in [CatVTON/README.md](CatVTON/README.md) for the full commands for:

- data preparation
- VITON-HD / DressCode inference
- metric calculation
- ComfyUI workflow usage

## Notes

- The first run may download model weights from Hugging Face, which can take time.
- The generated output is saved under `CatVTON/resource/demo/output` by default.
- If the terminal prints a local Gradio URL, open it in the browser to use the app.

## Quick summary

```bash
cd /Users/kunnath/projects/CatVTON
source .venv/bin/activate
cd CatVTON
python app.py --output_dir="resource/demo/output" --mixed_precision="bf16" --allow_tf32
```
