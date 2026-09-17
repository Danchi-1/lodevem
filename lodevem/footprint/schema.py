"""
schema.py — Typed Model Footprint Schema

Defines the structured metrics, confidence statuses, and estimator-specific data
produced by the model footprint profiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ModelFootprint:
    """Complete static and architectural footprint of a model."""

    # Basic metadata
    model_path: str
    model_name: str
    backend: str
    file_size_bytes: int
    file_size_mb: float
    input_shape: tuple[int, ...] | None = None

    # Parameter & Weight metrics
    total_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None
    parameter_memory_bytes: Optional[int] = None
    parameter_memory_mb: Optional[float] = None
    precision_breakdown: Dict[str, int] = field(default_factory=dict)

    # Computation metrics
    flops: Optional[int] = None
    macs: Optional[int] = None
    flops_status: str = "not_applicable"  # "exact", "partial", "unsupported_operators", "not_applicable"

    # LLM / Hugging Face specific metrics
    is_llm: bool = False
    context_window: Optional[int] = None
    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    kv_cache_bytes_per_token: Optional[float] = None
    kv_cache_projections_mb: Dict[int, float] = field(default_factory=dict)

    # Scikit-Learn specific metrics
    estimator_type: Optional[str] = None
    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    total_node_count: Optional[int] = None
    n_features_in: Optional[int] = None

    # Memory baseline estimates
    estimated_minimum_ram_mb: Optional[float] = None

    # Host GPU Diagnostic metrics (if --use-gpu diagnostic flag enabled)
    gpu_diagnostic_mode: bool = False
    host_gpus_detected: List[str] = field(default_factory=list)
    gpu_peak_memory_mb: Optional[float] = None

    # Integrity & Status tracking
    metrics_status: Dict[str, str] = field(default_factory=dict)
    unavailable_metrics: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.input_shape is not None:
            data["input_shape"] = list(self.input_shape)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelFootprint:
        d = dict(data)
        if d.get("input_shape") is not None:
            d["input_shape"] = tuple(d["input_shape"])
        return cls(**d)
