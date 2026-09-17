"""
cache.py — Model Footprint Cache System

Persists analyzed footprints to .lodevem_cache/footprint/ keyed by a SHA-256 digest
of file path, mtime, file size, backend, input shape, profiler version, and GPU diagnostic mode.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from lodevem.footprint.schema import ModelFootprint

logger = logging.getLogger(__name__)

PROFILER_VERSION = "1.0.0"
CACHE_DIR = Path.cwd() / ".lodevem_cache" / "footprint"


def compute_cache_key(
    model_path: str | Path,
    backend_name: str,
    input_shape: tuple[int, ...] | None = None,
    gpu_mode: bool = False,
) -> str:
    """
    Compute a deterministic SHA-256 cache key based on file metadata and analysis config.
    """
    path = Path(model_path).resolve()
    if path.is_file():
        stat = path.stat()
        mtime = stat.st_mtime
        size = stat.st_size
    elif path.is_dir():
        # Directory model (e.g. Hugging Face repository directory)
        mtimes = [p.stat().st_mtime for p in path.glob("**/*") if p.is_file()]
        sizes = [p.stat().st_size for p in path.glob("**/*") if p.is_file()]
        mtime = max(mtimes) if mtimes else 0.0
        size = sum(sizes)
    else:
        mtime = 0.0
        size = 0

    shape_str = ",".join(map(str, input_shape)) if input_shape else "auto"
    raw_key = (
        f"{str(path)}:{mtime}:{size}:{backend_name}:{shape_str}:{PROFILER_VERSION}:{gpu_mode}"
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def get_cached_footprint(
    model_path: str | Path,
    backend_name: str,
    input_shape: tuple[int, ...] | None = None,
    gpu_mode: bool = False,
    cache_dir: Path | None = None,
) -> Optional[ModelFootprint]:
    """
    Retrieve cached footprint if valid. Returns None on cache miss or corruption.
    """
    target_dir = cache_dir or CACHE_DIR
    key = compute_cache_key(model_path, backend_name, input_shape, gpu_mode)
    cache_file = target_dir / f"{key}.json"

    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return ModelFootprint.from_dict(data)
    except Exception as e:
        logger.debug(f"Failed to read cache file '{cache_file}': {e}")
        return None


def save_cached_footprint(
    footprint: ModelFootprint,
    cache_dir: Path | None = None,
) -> Path:
    """
    Save footprint to disk cache.
    """
    target_dir = cache_dir or CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    key = compute_cache_key(
        footprint.model_path,
        footprint.backend,
        footprint.input_shape,
        footprint.gpu_diagnostic_mode,
    )
    cache_file = target_dir / f"{key}.json"

    cache_file.write_text(
        json.dumps(footprint.to_dict(), indent=2),
        encoding="utf-8",
    )
    return cache_file
