"""
guards.py — CPU Isolation and Hardware Guardrails

This ensures that mobile device benchmarking strictly runs in an isolated CPU environment,
preventing silent fallback or accidental dispatch to host GPUs (NVIDIA, AMD/ROCm, Apple Silicon).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

CPU_MASK_ENV = {
    # NVIDIA CUDA
    "CUDA_VISIBLE_DEVICES": "",
    # AMD ROCm / HIP
    "ROCR_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    # Intel oneAPI / Level Zero
    "ONEAPI_DEVICE_SELECTOR": "",
    "ZE_AFFINITY_MASK": "",
    # Apple Silicon Metal Performance Shaders
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
}


def isolate_cpu_pre_import() -> None:
    """
    Sanitize process environment before importing PyTorch, ONNX Runtime, or Transformers.
    Must be called at the very earliest entry point of subprocesses.
    """
    for key, val in CPU_MASK_ENV.items():
        os.environ[key] = val


def get_cpu_isolated_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """
    Return a copy of the environment with GPU visibility masked out across all vendors.
    """
    env = dict(os.environ if base_env is None else base_env)
    env.update(CPU_MASK_ENV)
    return env


def detect_host_gpus() -> list[str]:
    """
    Detect host GPUs across NVIDIA, AMD, Intel, and Apple Silicon without initializing CUDA/ROCm.
    """
    gpus: list[str] = []

    # 1. Check NVIDIA via nvidia-smi (Tesla T4, A100, H100, RTX series, etc.)
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip():
                gpus.extend([line.strip() for line in res.stdout.strip().splitlines() if line.strip()])
        except Exception:
            pass

    # 2. Check AMD ROCm via rocm-smi
    if shutil.which("rocm-smi"):
        try:
            res = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip():
                gpus.append("AMD GPU (ROCm)")
        except Exception:
            pass

    # 3. Check Intel GPUs via xpu-smi
    if shutil.which("xpu-smi"):
        try:
            res = subprocess.run(
                ["xpu-smi", "discovery"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip():
                gpus.append("Intel GPU (Data Center / Arc)")
        except Exception:
            pass

    # 4. Check Apple Silicon MPS (macOS Metal)
    if sys.platform == "darwin":
        try:
            # Check system_profiler on macOS
            res = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if "Apple" in res.stdout or "Chipset Model" in res.stdout:
                gpus.append("Apple Silicon GPU (Metal / MPS)")
        except Exception:
            pass

    # 5. Generic Linux PCI GPU detection fallback (if no vendor CLIs found)
    if not gpus and sys.platform.startswith("linux"):
        try:
            drm_path = Path("/sys/class/drm")
            if drm_path.exists():
                render_nodes = list(drm_path.glob("renderD*"))
                if render_nodes:
                    gpus.append(f"Linux DRI/DRM GPU ({len(render_nodes)} render node(s))")
        except Exception:
            pass

    return gpus


def assert_tensor_on_cpu(tensor: Any, name: str = "tensor") -> None:
    """
    Assert that a given tensor or array resides strictly on CPU.
    """
    if tensor is None:
        return

    # Check PyTorch tensor
    if hasattr(tensor, "device"):
        dev = tensor.device
        dev_type = getattr(dev, "type", str(dev))
        if dev_type != "cpu":
            raise RuntimeError(
                f"CPU Guardrail Violation: {name} is allocated on device '{dev}' instead of CPU. "
                "Simulated mobile benchmarking requires strict CPU execution."
            )


def assert_inputs_on_cpu(inputs: Any, name: str = "inputs") -> None:
    """
    Assert that input structures (tensors, dicts of tensors, lists) reside strictly on CPU.
    """
    if isinstance(inputs, Mapping):
        for k, v in inputs.items():
            assert_inputs_on_cpu(v, f"{name}['{k}']")
    elif isinstance(inputs, (list, tuple)):
        for idx, item in enumerate(inputs):
            assert_inputs_on_cpu(item, f"{name}[{idx}]")
    else:
        assert_tensor_on_cpu(inputs, name)


def assert_model_on_cpu(model: Any) -> None:
    """
    Assert all parameters and buffers of a PyTorch model are strictly on CPU.
    """
    if hasattr(model, "named_parameters"):
        for name, param in model.named_parameters():
            assert_tensor_on_cpu(param, f"Model parameter '{name}'")
    if hasattr(model, "named_buffers"):
        for name, buf in model.named_buffers():
            assert_tensor_on_cpu(buf, f"Model buffer '{name}'")
