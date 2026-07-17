"""
predict.py — Latency Prediction via nn-Meter

What this file does:
    Takes a PyTorch model and a device profile, and returns a predicted
    inference latency in milliseconds.

How nn-Meter works (plain language):
    nn-Meter was built by Microsoft Research. Instead of running your model
    on actual hardware, it:
    1. Breaks your model into individual operations (convolutions, activations, etc.)
    2. Looks up how fast each operation runs on the target chip, using prediction
       models trained from real hardware measurements
    3. Sums everything up, accounting for how the hardware fuses operations together

    It was validated to within ~10-15% MAPE on real devices.
    Paper: "nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model
            Inference on Diverse Edge Devices" — MobiSys 2021.

The scaling_factor:
    nn-Meter's only CPU predictor is for Cortex-A76 (a high-end mobile core).
    Most of our target devices use Cortex-A53, A55, or A7 — older, slower cores.

    We handle this with a scaling factor sourced from ARM's performance data:
        Cortex-A76 → A75:  ×1.2   (same generation, minor difference)
        Cortex-A76 → A55:  ×2.2   (A55 is an efficiency core, significantly slower)
        Cortex-A76 → A53:  ×2.8   (older architecture, ~64% slower than A76)
        Cortex-A76 → A7:   ×5.5–7.0 (very old, used in KaiOS devices)

    So if nn-Meter predicts 300ms on A76, and your device has an A53 (factor 2.8),
    we report 300 × 2.8 = 840ms as the estimated latency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from lodevem.profiles import DeviceProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# nn-Meter predictor name
# ---------------------------------------------------------------------------
# nn-Meter ships with named predictors. We always use the CPU predictor
# (cortexA76cpu_tflite21) because:
# - Our target devices don't have dedicated NPUs we can simulate
# - TFLite 2.1 is the closest runtime to what MobileNetV3 uses on Android
# The scaling_factor in each device profile adjusts for the actual core type.
# ---------------------------------------------------------------------------

NN_METER_PREDICTOR = "cortexA76cpu_tflite21"


def _try_import_nn_meter():
    """
    Lazy import of nn-meter.

    We import it here (not at the top of the file) so that:
    - 'lodevem list' and 'lodevem check' work even if nn-meter isn't installed
    - The error message is specific and actionable
    """
    try:
        from nn_meter import load_lat_predictor
        return load_lat_predictor
    except ImportError:
        raise ImportError(
            "nn-meter is not installed. Run: pip install nn-meter\n"
            "If you've already installed it, make sure you're in the right "
            "virtual environment."
        )


def predict_latency(
    model: nn.Module,
    profile: DeviceProfile,
    input_shape: tuple = (1, 3, 224, 224),
) -> dict:
    """
    Predict inference latency for a model on a given device profile.

    Args:
        model:        Your PyTorch model (nn.Module).
        profile:      The device profile to simulate.
        input_shape:  The shape of one input tensor. Default is (1, 3, 224, 224)
                      which is batch=1, RGB image, 224×224 pixels — standard for
                      MobileNetV3 and most image classifiers.

    Returns a dict with:
        {
            "predictor":          "cortexA76cpu_tflite21",
            "a76_latency_ms":     float,   # Raw nn-Meter prediction (A76 baseline)
            "scaled_latency_ms":  float,   # Adjusted for target core (what we report)
            "scaling_factor":     float,   # The multiplier used
            "target_core":        str,     # e.g. "Cortex-A53"
            "input_shape":        tuple,
        }
    """
    load_lat_predictor = _try_import_nn_meter()

    logger.info(f"Loading nn-Meter predictor: {NN_METER_PREDICTOR}")
    predictor = load_lat_predictor(NN_METER_PREDICTOR)

    # nn-Meter needs the model in eval mode and a sample input to trace the graph
    model.eval()
    dummy_input = torch.zeros(input_shape)

    logger.info(
        f"Predicting latency for '{profile.name}' "
        f"(base: A76, scaling: ×{profile.scaling_factor})"
    )

    # nn-Meter returns latency in milliseconds for the A76 baseline
    a76_latency_ms: float = predictor.predict(model, input_shape)

    # Apply the scaling factor to estimate latency on the actual target core
    scaled_latency_ms = round(a76_latency_ms * profile.scaling_factor, 2)

    return {
        "predictor": NN_METER_PREDICTOR,
        "a76_latency_ms": round(a76_latency_ms, 2),
        "scaled_latency_ms": scaled_latency_ms,
        "scaling_factor": profile.scaling_factor,
        "target_core": profile.core_type,
        "input_shape": input_shape,
    }


def load_model(model_path: str | Path) -> nn.Module:
    """
    Load a PyTorch model from a .pt or .pth file.

    We use weights_only=False because we need to load the full model
    (architecture + weights), not just the state dict.

    If the user saved only a state dict (not the full model), this will fail
    with a clear error telling them what to do.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'\n"
            f"Make sure the path is correct and the file exists."
        )

    logger.info(f"Loading model from: {model_path}")

    try:
        # Try loading as a full model first (most common case)
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(model, nn.Module):
            return model.eval()

        # If it loaded but isn't an nn.Module, it's probably a state dict
        raise TypeError(
            f"'{model_path.name}' contains a state dict, not a full model.\n"
            "To save a full model: torch.save(model, 'path.pt')\n"
            "To save only weights: torch.save(model.state_dict(), 'path.pt')"
        )

    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_path}': {e}") from e


def get_model_size_mb(model_path: str | Path) -> float:
    """Return the size of a model file in megabytes."""
    return round(Path(model_path).stat().st_size / (1024 * 1024), 2)
