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

from pathlib import Path
from typing import Optional, Any
import pickle
import sys
import logging

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

# Cache the predictor so we don't load/download it 16 times per model
_PREDICTOR_CACHE = {}


class _SklearnCompatUnpickler(pickle.Unpickler):
    """
    Custom unpickler that intercepts scikit-learn's Tree class during nn-meter load
    and dynamically injects a missing schema column (missing_go_to_left) to prevent
    modern scikit-learn (1.3+) from crashing on the legacy 0.23 pickle.
    """
    def find_class(self, module: str, name: str) -> Any:
        if module == "sklearn.tree._tree" and name == "Tree":
            try:
                import sklearn.tree._tree
                from sklearn.tree._tree import NODE_DTYPE
                import numpy as np

                RealTree = sklearn.tree._tree.Tree

                class PatchedTree(RealTree):
                    def __setstate__(self, state):
                        try:
                            if isinstance(state, tuple):
                                new_state = list(state)
                                for i, item in enumerate(new_state):
                                    if isinstance(item, np.ndarray) and item.dtype.names and 'left_child' in item.dtype.names:
                                        if item.dtype != NODE_DTYPE:
                                            new_nodes = np.zeros(item.shape, dtype=NODE_DTYPE)
                                            for n in item.dtype.names:
                                                if n in NODE_DTYPE.names:
                                                    new_nodes[n] = item[n]
                                            new_state[i] = new_nodes
                                state = tuple(new_state)
                            elif isinstance(state, dict):
                                if 'nodes' in state:
                                    item = state['nodes']
                                    if item.dtype != NODE_DTYPE:
                                        new_nodes = np.zeros(item.shape, dtype=NODE_DTYPE)
                                        for n in item.dtype.names:
                                            if n in NODE_DTYPE.names:
                                                new_nodes[n] = item[n]
                                        state['nodes'] = new_nodes
                        except Exception as e:
                            logger.debug(f"Failed to patch Tree state during unpickling: {e}")
                            
                        # Pass the patched (or unpatched if it failed) state down to the C-extension
                        super().__setstate__(state)

                return PatchedTree
            except Exception as e:
                logger.debug(f"Failed to setup PatchedTree: {e}")
                
        return super().find_class(module, name)


class _PatchPickleContext:
    """
    Context manager to temporarily override the global pickle.load and joblib.load
    to use our custom unpickler. All exceptions are safely suppressed.
    """
    def __enter__(self):
        self.orig_pickle_load = pickle.load
        
        def patched_pickle_load(f, **kwargs):
            return _SklearnCompatUnpickler(f, **kwargs).load()
            
        pickle.load = patched_pickle_load

        # If nn-meter ends up using joblib, we need to patch its custom NumpyUnpickler
        self.joblib_module = None
        self.orig_joblib_find_class = None
        try:
            import joblib.numpy_pickle
            if hasattr(joblib.numpy_pickle, "NumpyUnpickler"):
                self.joblib_module = joblib.numpy_pickle
                self.orig_joblib_find_class = self.joblib_module.NumpyUnpickler.find_class
                
                # We can't reuse _SklearnCompatUnpickler because joblib uses its own Unpickler inheritance,
                # but we can monkey-patch its find_class method.
                def patched_joblib_find_class(unpickler_self, module, name):
                    if module == "sklearn.tree._tree" and name == "Tree":
                        # Return the exact same PatchedTree logic via our custom unpickler
                        return _SklearnCompatUnpickler(None).find_class(module, name)
                    return self.orig_joblib_find_class(unpickler_self, module, name)
                    
                self.joblib_module.NumpyUnpickler.find_class = patched_joblib_find_class
        except ImportError:
            pass
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pickle.load = self.orig_pickle_load
        if self.joblib_module and self.orig_joblib_find_class:
            self.joblib_module.NumpyUnpickler.find_class = self.orig_joblib_find_class


def _try_import_nn_meter():
    """
    Lazy import of nn-meter.

    We import it here (not at the top of the file) so that:
    - 'lodevem list' and 'lodevem check' work even if nn-meter isn't installed
    - The error message is specific and actionable
    """
    try:
        from nn_meter import load_latency_predictor
        return load_latency_predictor
    except ImportError as e:
        raise ImportError(
            f"nn-meter failed to import. Original error: {e}\n"
            "Run: pip install 'lodevem[pytorch]'. If already installed, check your environment."
        ) from e


def predict_latency(
    backend: Any,
    profile: DeviceProfile,
    input_shape: tuple = (1, 3, 224, 224),
) -> dict:
    """
    Predict inference latency for a model on a given device profile.

    Args:
        backend:      The BenchmarkBackend adapter instance.
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
    if NN_METER_PREDICTOR not in _PREDICTOR_CACHE:
        load_latency_predictor = _try_import_nn_meter()
        
        # Check if the predictor exists locally to warn the user about the download
        predictor_path = Path.home() / ".nn_meter" / "data" / "predictor" / NN_METER_PREDICTOR
        if not predictor_path.exists():
            logger.warning(f"nn-meter predictor dataset (~376MB) not found locally at {predictor_path}.")
            logger.warning("nn-meter will now attempt to download it from GitHub.")
            logger.warning("If you wish to skip this download, cancel and run with --no-predict (or no_predict=True).")
            
        logger.info(f"Loading nn-Meter predictor: {NN_METER_PREDICTOR}")
        try:
            with _PatchPickleContext():
                _PREDICTOR_CACHE[NN_METER_PREDICTOR] = load_latency_predictor(NN_METER_PREDICTOR)
        except Exception as e:
            logger.warning(f"Failed to load nn-meter predictor (scikit-learn incompatibility?): {e}")
            raise
        
    predictor = _PREDICTOR_CACHE[NN_METER_PREDICTOR]

    from lodevem.backends.pytorch import PyTorchBackend
    if not isinstance(backend, PyTorchBackend):
        raise NotImplementedError(f"nn-Meter latency prediction is only supported for PyTorch models. Got: {type(backend)}")
        
    model = backend.model
    model.eval()
    shape_to_use = getattr(model, "expected_input_shape", input_shape)

    logger.info(
        f"Predicting latency for '{profile.name}' "
        f"(base: A76, scaling: ×{profile.scaling_factor})"
    )

    # nn-Meter returns latency in milliseconds for the A76 baseline
    a76_latency_ms: float = predictor.predict(model, shape_to_use)

    # Apply the scaling factor to estimate latency on the actual target core
    scaled_latency_ms = round(a76_latency_ms * profile.scaling_factor, 2)

    return {
        "predictor": NN_METER_PREDICTOR,
        "a76_latency_ms": round(a76_latency_ms, 2),
        "scaled_latency_ms": scaled_latency_ms,
        "scaling_factor": profile.scaling_factor,
        "target_core": profile.core_type,
        "input_shape": shape_to_use,
    }


