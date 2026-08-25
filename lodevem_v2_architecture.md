# Lodevem v2.0 Architecture Roadmap: Universal Benchmarking

This document outlines the implementation plan for transitioning `lodevem` from a lightweight PyTorch-only edge simulator into a universal ML benchmarking framework, **without** bloating the core package size.

## 1. The Bloat Solution: Optional Plugins (Extras)

Currently, `lodevem` forces users to download PyTorch (~800MB) just to install the package. To solve this, we will strip the core dependencies down to the absolute minimum (`psutil`, `pyyaml`, `rich`, `click`). 

Everything else will be modularized into **Optional Dependencies (PyPI Extras)**.

Users will install only what they need:
* `pip install lodevem` (Core only, ~5MB. Can benchmark ONNX if onnxruntime is installed)
* `pip install lodevem[pytorch]` (Installs `torch`, `torchvision`, `nn-meter`)
* `pip install lodevem[hf]` (Installs `transformers`, `accelerate`)
* `pip install lodevem[sklearn]` (Installs `scikit-learn`, `hummingbird-ml`)
* `pip install lodevem[all]` (The kitchen sink)

If a user tries to load a `.pt` file without the `[pytorch]` extra installed, `lodevem` will gracefully fail fast and say: *"To benchmark PyTorch models, please run: pip install lodevem[pytorch]"*.

## 2. Universal Loader Bridge

We will replace the hardcoded `torch.load()` with a `LoaderBridge` class that detects file extensions and delegates to the correct backend:

### A. ONNX (`.onnx`) - *New Core Standard*
Since ONNX is the industry standard for edge deployment, `lodevem` will treat ONNX as a first-class citizen. `onnxruntime` is very lightweight compared to PyTorch.
* **Loader:** `onnx.load()`
* **Execution:** `onnxruntime.InferenceSession()`

### B. Hugging Face (`.safetensors` / `config.json`)
* **Loader:** `transformers.AutoModel.from_pretrained()` (if `[hf]` is installed).
* **Execution:** Converted to PyTorch graph in-memory.

### C. Scikit-Learn (`.joblib` / `.pkl`)
* **Loader:** `joblib.load()`
* **Execution:** Passed through `hummingbird.ml.convert(model, "torch")` to compile trees/SVMs into tensor operations for benchmarking.

## 3. Dynamic Input Generation

The hardest part of universal support is knowing what dummy data to feed the model. We will implement an `InputGenerator` pipeline:

1. **ONNX Inspection:** ONNX models explicitly store their expected input shapes and data types in their metadata. We can automatically generate dummy data by reading `session.get_inputs()`.
2. **Hugging Face Tokenizers:** For NLP models, we will automatically generate `input_ids` and `attention_mask` tensors of shape `(1, 128)` using `torch.randint()`.
3. **User Override:** The `--input-shape` CLI argument will be expanded to support multiple inputs (e.g., `--input-shape ids:1,128 mask:1,128`).

## 4. Smart Pre-Flight Checks (OOM Prevention)

Before launching the Docker container or Lite subprocess, the orchestrator will run a `PreFlightAnalyzer`:

1. **Disk Size Check:** Calculate the size of the model file(s).
2. **RAM Estimation:** A general heuristic: `Memory Required = Model Disk Size * 1.5` (accounting for inference buffer overhead).
3. **Profile Filtering:** If `Memory Required > profile.ram_mb`, `lodevem` will immediately mark that profile as `OOM` (Out of Memory) in the results table without attempting to run it. This prevents hard crashes and saves immense amounts of time.

## User Review Required
> [!IMPORTANT]
> **Dependency Strategy:** Do you agree with stripping PyTorch out of the base `lodevem` installation and forcing users to use `pip install lodevem[pytorch]` if they want to evaluate `.pt` files? This is the only way to keep the base package strictly under 10MB.
> 
> **LLM Benchmarking:** Do you want to try and tackle LLM Tokens-Per-Second (TPS) benchmarking, or strictly limit `lodevem` to single-pass models (Vision, Audio, Sequences, Scikit-Learn)? LLM benchmarking requires an entirely different measurement loop.

## Verification Plan
1. Update `pyproject.toml` to use `[project.optional-dependencies]`.
2. Build the `LoaderBridge` and write unit tests for `.pt`, `.joblib`, and `.onnx` loading.
3. Implement `PreFlightAnalyzer` and verify that a 3GB model correctly skips the 1GB RAM Nokia profile.
