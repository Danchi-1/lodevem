"""Backward-compatibility wrapper for lodevem.footprint.analyzers.onnx_analyzer."""
from lodevem.footprint.analyzers.onnx_analyzer import analyze_onnx_footprint, ONNX_TYPE_MAP

__all__ = ["analyze_onnx_footprint", "ONNX_TYPE_MAP"]
