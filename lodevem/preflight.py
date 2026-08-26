from pathlib import Path
from typing import Dict, Any

def get_model_size_mb(model_path: str | Path) -> float:
    """
    Return the size of a model file or directory in megabytes.
    If it's a directory (e.g. Hugging Face), sum the size of all valid files.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")

    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 2)

    # It's a directory (e.g. Hugging Face model)
    total_size = 0
    # Files to ignore (caches, git, etc.)
    ignore_extensions = {".pyc", ".md"}
    ignore_dirs = {".git", "__pycache__"}

    for item in path.rglob("*"):
        if item.is_file():
            # Skip ignored directories in the path
            if any(part in ignore_dirs for part in item.parts):
                continue
            # Skip ignored extensions
            if item.suffix in ignore_extensions:
                continue
            total_size += item.stat().st_size

    return round(total_size / (1024 * 1024), 2)


class PreFlightAnalyzer:
    """
    Analyzes a model before execution to prevent unnecessary OOM crashes.
    Provides a conservative estimate of RAM usage based on disk size.
    """
    
    # We estimate that a model requires ~1.5x its disk size in RAM
    # just to load and execute (weights + activation buffers).
    # This is a heuristic, not a guarantee.
    MEMORY_HEURISTIC_MULTIPLIER = 1.5

    @classmethod
    def analyze(cls, model_path: str | Path, profile_ram_mb: int) -> Dict[str, Any]:
        """
        Estimate RAM usage and determine if the model should be skipped.
        
        Returns:
            A dict containing the preflight results. If status is "preflight_oom",
            the runner should skip execution.
        """
        model_size_mb = get_model_size_mb(model_path)
        estimated_memory_mb = round(model_size_mb * cls.MEMORY_HEURISTIC_MULTIPLIER, 2)
        
        # We strictly fail if the ESTIMATE is strictly greater than the profile limit.
        # If it's exactly at the limit, we let it try.
        will_oom = estimated_memory_mb > profile_ram_mb
        
        if will_oom:
            return {
                "preflight_status": "preflight_oom",
                "model_size_mb": model_size_mb,
                "estimated_memory_mb": estimated_memory_mb,
                "preflight_reason": (
                    f"Pre-flight estimate ({estimated_memory_mb}MB) exceeds "
                    f"profile RAM limit ({profile_ram_mb}MB)."
                )
            }
            
        return {
            "preflight_status": "ok",
            "model_size_mb": model_size_mb,
            "estimated_memory_mb": estimated_memory_mb,
            "preflight_reason": "Memory estimate within limits."
        }
