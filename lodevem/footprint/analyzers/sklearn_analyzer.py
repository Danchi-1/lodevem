"""
sklearn.py — Scikit-Learn Model Footprint Analyzer

Analyzes Scikit-Learn (.pkl, .joblib) models honestly based on native estimator properties:
- Estimator type (RandomForestClassifier, DecisionTreeRegressor, etc.)
- Number of trees / estimators
- Max depth and total tree node count
- Input feature count
- Disk file size
Explicitly treats neural-net parameter counts and FLOPs as not_applicable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from lodevem.footprint.schema import ModelFootprint
from lodevem.guards import detect_host_gpus

logger = logging.getLogger(__name__)


def analyze_sklearn_footprint(
    model_path: str | Path,
    input_shape: tuple[int, ...] | None = None,
    use_gpu: bool = False,
    allow_untrusted: bool = False,
) -> ModelFootprint:
    """
    Analyze a Scikit-Learn model file and return its native ModelFootprint.
    """
    try:
        import joblib
    except ImportError as e:
        raise ImportError(f"Missing joblib dependency: {e}") from e

    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    file_size_bytes = path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    from lodevem.security import scan_pickle_security, SecurityPolicyError, is_in_docker
    scan_pickle_security(path)

    if not allow_untrusted and not is_in_docker():
        raise SecurityPolicyError(
            f"Security Policy Violation: Scikit-Learn model '{path.name}' uses Python pickle serialization.\n"
            "Analyzing this model in Lite Mode executes arbitrary Python bytecode on your host machine without container sandboxing.\n\n"
            "Remediation Options:\n"
            "  1. Run with Docker for automated sandbox confinement (recommended).\n"
            "  2. Convert model to ONNX or Safetensors (safe, pickle-free formats).\n"
            "  3. If you completely trust this model file, re-run with: --dangerously-allow-untrusted-model"
        )

    try:
        model = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load Scikit-Learn model from '{path}': {e}") from e

    estimator_type = type(model).__name__
    n_estimators = getattr(model, "n_estimators", None)
    max_depth = getattr(model, "max_depth", None)
    n_features_in = getattr(model, "n_features_in_", None)

    # Compute total tree node count across decision trees if present
    total_node_count: Optional[int] = None
    if hasattr(model, "tree_") and hasattr(model.tree_, "node_count"):
        total_node_count = int(model.tree_.node_count)
    elif hasattr(model, "estimators_"):
        node_sum = 0
        found_any = False
        try:
            for item in model.estimators_:
                # Could be a single tree estimator or an array/list of trees per stage
                estimators_list = item if isinstance(item, (list, tuple)) else [item]
                for est in estimators_list:
                    tree = getattr(est, "tree_", None)
                    if tree is not None and hasattr(tree, "node_count"):
                        node_sum += int(tree.node_count)
                        found_any = True
            if found_any:
                total_node_count = node_sum
        except Exception:
            pass

    warnings: list[str] = [
        "Scikit-Learn estimators rely on branching/decision rules; FLOPs metric is not applicable."
    ]
    unavailable_metrics: list[str] = ["parameters", "flops", "precision_breakdown"]
    metrics_status: Dict[str, str] = {
        "parameters": "not_applicable",
        "flops": "not_applicable",
        "precision": "not_applicable",
        "estimator_properties": "exact",
    }

    # Minimum RAM baseline: disk size + python runtime footprint
    estimated_minimum_ram_mb = round(file_size_mb + 25.0, 1)

    host_gpus = detect_host_gpus()

    return ModelFootprint(
        model_path=str(path),
        model_name=path.name,
        backend="sklearn",
        file_size_bytes=file_size_bytes,
        file_size_mb=file_size_mb,
        input_shape=input_shape,
        total_parameters=None,
        trainable_parameters=None,
        parameter_memory_bytes=None,
        parameter_memory_mb=None,
        precision_breakdown={},
        flops=None,
        macs=None,
        flops_status="not_applicable",
        estimator_type=estimator_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        total_node_count=total_node_count,
        n_features_in=n_features_in,
        estimated_minimum_ram_mb=estimated_minimum_ram_mb,
        gpu_diagnostic_mode=use_gpu,
        host_gpus_detected=host_gpus,
        metrics_status=metrics_status,
        unavailable_metrics=unavailable_metrics,
        warnings=warnings,
    )
