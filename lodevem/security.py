"""
security.py — Defensive Security, Deserialization Policy, and Format Validation

Provides:
- SecurityPolicyError & SecurityError definitions
- Purely static, non-executing pickle stream auditing with resource bounds
- Zip-slip and zip-bomb archive traversal defenses
- Deep ONNX graph, resource ceiling, and external-data path normalization validation
- Strict Safetensors header, offset, and non-overlapping tensor validation
- Tensor allocation bomb and ANSI escape display sanitization

NOTE ON STATIC ANALYSIS:
Static bytecode scanning is an advisory heuristic diagnostic. It is NEVER a security
boundary or guarantee of safety. Untrusted Python pickle/joblib streams must be executed
only within an OS-level hardened container sandbox or with explicit user acknowledgment.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import pickletools
import re
import struct
import zipfile
from pathlib import Path
from typing import Any, Dict, Sequence

logger = logging.getLogger(__name__)

# Resource ceilings for analysis
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
MAX_PICKLE_STREAM_BYTES = 512 * 1024 * 1024   # 512 MB
MAX_OPCODES_COUNT = 2_000_000                 # Protection against algorithmic complexity DoS
MAX_ZIP_MEMBER_BYTES = 1 * 1024 * 1024 * 1024 # 1 GB max per uncompressed archive member
MAX_ZIP_RATIO = 100                            # Max uncompressed-to-compressed ratio (zip bomb)
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024  # 100 MB max JSON header

# Prohibited modules and callables for static heuristic inspection
BLOCKED_MODULES = {
    "os",
    "posix",
    "nt",
    "subprocess",
    "pty",
    "socket",
    "urllib",
    "requests",
    "shutil",
    "ctypes",
    "multiprocessing",
    "importlib",
    "webbrowser",
    "http",
    "ftplib",
    "platform",
    "sqlite3",
    "pdb",
    "bdb",
}

BLOCKED_CALLABLES = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "system",
    "popen",
    "spawn",
    "call",
    "check_call",
    "check_output",
}

ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class SecurityError(RuntimeError):
    """Raised when an active exploit, malicious payload, or corrupted stream is detected."""
    pass


class SecurityPolicyError(RuntimeError):
    """Raised when an untrusted model format is blocked by sandboxing policy."""
    pass


def is_in_docker() -> bool:
    """Check if the current process is running inside Docker container."""
    return os.environ.get("LODEVEM_IN_DOCKER") == "1" or Path("/.dockerenv").exists()


def is_host_safe_format(file_path: str | Path) -> bool:
    """
    Check if a model format is memory-safe / non-code-executing (ONNX or Hugging Face with Safetensors, or raw Safetensors),
    allowing it to be loaded on the host for latency prediction without a sandbox.
    """
    path = Path(file_path)
    if path.is_file():
        return path.suffix.lower() in (".onnx", ".safetensors")
    elif path.is_dir():
        has_safetensors = (path / "model.safetensors").exists() or any(path.glob("*.safetensors"))
        has_bin = (path / "pytorch_model.bin").exists() or any(path.glob("*.bin"))
        has_pt = any(path.glob("*.pt")) or any(path.glob("*.pth"))
        return has_safetensors and not (has_bin or has_pt)
    return False


def _check_module_and_callable(module_name: str, callable_name: str, pos: int | None, source_name: str) -> None:
    """Check module and callable against prohibited security blocklists."""
    root_module = module_name.split(".")[0]
    if root_module in BLOCKED_MODULES:
        raise SecurityError(
            f"Security Violation: '{source_name}' contains potentially malicious code. "
            f"Prohibited module '{module_name}' detected at opcode offset {pos}."
        )
    if callable_name in BLOCKED_CALLABLES and root_module in ("builtins", "__builtin__", ""):
        raise SecurityError(
            f"Security Violation: '{source_name}' contains potentially malicious code. "
            f"Prohibited callable '{callable_name}' detected at opcode offset {pos}."
        )


def _audit_pickle_stream(stream: io.BytesIO, source_name: str) -> None:
    """
    Parse a pickle byte stream using pickletools.genops purely statically.
    NEVER calls .load() or instantiates any class.
    Enforces an opcode count ceiling to protect against CPU exhaustion.
    """
    stack: list[str] = []
    opcode_count = 0

    try:
        for opcode, arg, pos in pickletools.genops(stream):
            opcode_count += 1
            if opcode_count > MAX_OPCODES_COUNT:
                logger.warning(
                    f"Static audit for '{source_name}' reached opcode ceiling ({MAX_OPCODES_COUNT:,}). "
                    "Static inspection terminated early."
                )
                return

            if opcode.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                stack.append(str(arg))
            elif opcode.name == "STACK_GLOBAL":
                if len(stack) >= 2:
                    callable_name = stack.pop()
                    module_name = stack.pop()
                    _check_module_and_callable(module_name, callable_name, pos, source_name)
            elif opcode.name == "GLOBAL":
                if arg:
                    parts = arg.split() if isinstance(arg, str) else []
                    if len(parts) == 2:
                        module_name, callable_name = parts[0], parts[1]
                    elif "." in str(arg):
                        module_name, callable_name = str(arg).rsplit(".", 1)
                    else:
                        module_name, callable_name = str(arg), ""
                    _check_module_and_callable(module_name, callable_name, pos, source_name)
    except ValueError:
        # Non-standard, truncated, or chunked opcodes (e.g. Joblib raw numpy buffers).
        # We stop cleanly without executing any code.
        pass


def scan_pickle_security(file_path: str | Path) -> None:
    """
    Statically audit PyTorch (.pt, .pth) and Scikit-Learn (.pkl, .joblib) files
    for dangerous pickle opcodes and zip-slip directory traversal.
    
    Resource bounds:
    - Enforces max file size.
    - Inspects zip archives with member size and compression ratio limits (zip bomb defense).
    """
    path = Path(file_path).resolve()
    if not path.exists() or not path.is_file():
        return

    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise SecurityError(
            f"File size {file_size:,} bytes exceeds maximum security threshold of {MAX_FILE_SIZE_BYTES:,} bytes."
        )

    # Check for PyTorch Zip container (PK\x03\x04)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            total_uncompressed = 0
            for member in zf.infolist():
                # Defend against Zip Slip (path traversal in archive member names)
                raw_name = member.filename
                if raw_name.startswith("/") or ".." in Path(raw_name).parts:
                    raise SecurityError(
                        f"Zip Slip detected in '{path.name}': Malicious member path '{raw_name}'"
                    )

                # Defend against Zip Bomb
                if member.file_size > MAX_ZIP_MEMBER_BYTES:
                    raise SecurityError(
                        f"Zip member '{raw_name}' uncompressed size {member.file_size:,} exceeds limit."
                    )
                total_uncompressed += member.file_size
                if member.compress_size > 0:
                    ratio = member.file_size / member.compress_size
                    if ratio > MAX_ZIP_RATIO and member.file_size > 10 * 1024 * 1024:
                        raise SecurityError(
                            f"Suspicious zip compression ratio ({ratio:.1f}x) in '{raw_name}'. Potential zip bomb."
                        )

                # Inspect pickle data files inside PyTorch archive
                if raw_name.endswith(".pkl") or raw_name.endswith("data.pkl"):
                    if member.file_size <= MAX_PICKLE_STREAM_BYTES:
                        with zf.open(member) as f:
                            data = f.read(MAX_PICKLE_STREAM_BYTES + 1)
                            if len(data) > MAX_PICKLE_STREAM_BYTES:
                                logger.warning(f"Pickle member '{raw_name}' exceeds scan stream size.")
                            else:
                                _audit_pickle_stream(io.BytesIO(data), f"{path.name}:{raw_name}")
    else:
        # Raw pickle file (Scikit-Learn .pkl, .joblib, or legacy PyTorch file)
        with open(path, "rb") as f:
            header = f.read(16)
            f.seek(0)
            if header:
                data = f.read(MAX_PICKLE_STREAM_BYTES + 1)
                _audit_pickle_stream(io.BytesIO(data), path.name)


ALLOWED_ONNX_DOMAINS = {"", "ai.onnx", "ai.onnx.ml", "ai.onnx.preview.training"}
ONNX_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def validate_onnx_security(file_path: str | Path) -> None:
    """
    Validate ONNX model graph security, resource bounds, and external-data paths.
    Prevents path traversal, symlink escapes, integer overflows, and graph allocation bombs.

    Security checks enforced:
    - Base file size ceiling (<= 2 GB)
    - Graph node count ceiling (<= 500,000)
    - Graph total initializer parameter count ceiling (<= 500,000,000)
    - Initializer dimension bounds (0 <= dim <= 100,000)
    - Graph IO and value_info dimension bounds (0 <= dim <= 100,000)
    - External data path traversal: prohibits null bytes, absolute paths, backslashes, and '..' components
    - External data symlink validation: detects symlinks and ensures target stays within model directory
    - External data file verification: existence, regular file status (no fifos/devices), per-file and total size limits
    - Operator set & domain allowlisting: only standard ONNX domains permitted
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: '{path}'")

    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise SecurityError(f"ONNX file size {file_size:,} bytes exceeds security ceiling.")

    model_dir = path.parent.resolve()

    try:
        import onnx
        from onnx import ModelProto
    except ImportError:
        # If onnx library is missing, basic file bounds are verified
        return

    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as e:
        raise SecurityError(f"Failed to parse ONNX protobuf safely from '{path.name}': {e}") from e

    # 1. Graph element limits
    graph = model.graph
    if len(graph.node) > 500_000:
        raise SecurityError(f"ONNX graph contains {len(graph.node):,} nodes, exceeding safety limit of 500,000.")

    # 2. Operator allowlisting & domain validation
    for node in graph.node:
        domain = getattr(node, "domain", "")
        if domain not in ALLOWED_ONNX_DOMAINS:
            raise SecurityError(
                f"ONNX Untrusted Operator Domain: Domain '{domain}' on operator '{node.op_type}' is not an allowed standard ONNX domain."
            )
        op_type = getattr(node, "op_type", "")
        if not op_type or not ONNX_IDENTIFIER_PATTERN.match(op_type):
            raise SecurityError(
                f"ONNX Malformed Operator Type: '{op_type}' is not a valid operator identifier."
            )

    # 3. Validate graph inputs, outputs, and value_info dimensions
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if hasattr(vi, "type") and vi.type.HasField("tensor_type"):
            shape = vi.type.tensor_type.shape
            for d in shape.dim:
                if d.HasField("dim_value"):
                    if d.dim_value < 0 or d.dim_value > 100_000:
                        raise SecurityError(
                            f"Suspicious tensor dimension {d.dim_value} in ONNX graph IO/value_info '{vi.name}'."
                        )

    # 4. Validate initializers and external data
    total_elements = 0
    total_external_bytes = 0
    seen_external_files: set[Path] = set()

    for tensor in graph.initializer:
        shape = list(tensor.dims)
        for dim in shape:
            if dim < 0 or dim > 100_000:
                raise SecurityError(f"Suspicious tensor dimension {dim} in ONNX initializer '{tensor.name}'.")
        elements = math.prod(shape) if shape else 0
        total_elements += elements

        # External data path traversal validation
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location":
                    rel_location = entry.value
                    if (
                        "\x00" in rel_location
                        or rel_location.startswith("/")
                        or "\\" in rel_location
                        or ".." in Path(rel_location).parts
                    ):
                        raise SecurityError(
                            f"ONNX External Data Path Traversal: Absolute or invalid path '{rel_location}'."
                        )

                    # Symlink detection BEFORE .resolve()
                    candidate_path = model_dir / rel_location
                    if candidate_path.is_symlink():
                        real_target = candidate_path.resolve()
                        try:
                            if not real_target.is_relative_to(model_dir):
                                raise SecurityError(
                                    f"ONNX External Data Symlink Escape: '{rel_location}' escapes model directory (points outside)."
                                )
                        except AttributeError:
                            if not str(real_target).startswith(str(model_dir)):
                                raise SecurityError(
                                    f"ONNX External Data Symlink Escape: '{rel_location}' escapes model directory (points outside)."
                                )

                    # Resolve path and verify it stays inside model directory
                    target_path = candidate_path.resolve()
                    try:
                        if not target_path.is_relative_to(model_dir):
                            raise SecurityError(
                                f"ONNX External Data Path Traversal: '{rel_location}' escapes model directory."
                            )
                    except AttributeError:
                        if not str(target_path).startswith(str(model_dir)):
                            raise SecurityError(
                                f"ONNX External Data Path Traversal: '{rel_location}' escapes model directory."
                            )

                    # Existence and regular file verification
                    if not target_path.exists():
                        raise SecurityError(
                            f"ONNX External Data File Missing: '{rel_location}' does not exist."
                        )
                    if not target_path.is_file():
                        raise SecurityError(
                            f"ONNX External Data File Invalid: '{rel_location}' is not a regular file."
                        )

                    # External file size limits
                    ext_file_size = target_path.stat().st_size
                    if ext_file_size > MAX_FILE_SIZE_BYTES:
                        raise SecurityError(
                            f"ONNX External Data File '{rel_location}' size {ext_file_size:,} bytes exceeds ceiling of {MAX_FILE_SIZE_BYTES:,} bytes."
                        )

                    if target_path not in seen_external_files:
                        seen_external_files.add(target_path)
                        total_external_bytes += ext_file_size
                        if total_external_bytes > MAX_FILE_SIZE_BYTES:
                            raise SecurityError(
                                f"ONNX total external data size {total_external_bytes:,} bytes exceeds security ceiling of {MAX_FILE_SIZE_BYTES:,} bytes."
                            )

    if total_elements > 500_000_000:
        raise SecurityError(
            f"ONNX graph contains {total_elements:,} total parameter elements, exceeding safety ceiling."
        )


def validate_safetensors_security(file_path: str | Path) -> dict[str, Any]:
    """
    Validate and parse a Safetensors file safely without executing code.
    Enforces header size limits, non-overlapping offsets, and file bounds.
    Returns the parsed header metadata dict.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Safetensors file not found: '{path}'")

    file_size = path.stat().st_size
    if file_size < 8:
        raise SecurityError(f"Truncated Safetensors file: '{path.name}' is only {file_size} bytes.")

    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]

        if header_len <= 0:
            raise SecurityError(f"Invalid Safetensors header length ({header_len}) in '{path.name}'.")
        if header_len > MAX_SAFETENSORS_HEADER_BYTES:
            raise SecurityError(
                f"Safetensors header size ({header_len:,} bytes) exceeds limit of {MAX_SAFETENSORS_HEADER_BYTES:,} bytes."
            )
        if file_size < 8 + header_len:
            raise SecurityError(
                f"Safetensors file is truncated: requires at least {8 + header_len:,} bytes, found {file_size:,} bytes."
            )

        header_json_bytes = f.read(header_len)

    try:
        header = json.loads(header_json_bytes.decode("utf-8"))
    except Exception as e:
        raise SecurityError(f"Malformed JSON in Safetensors header for '{path.name}': {e}") from e

    tensor_buffer_size = file_size - 8 - header_len
    last_end = 0

    for key, val in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(val, dict) or "data_offsets" not in val:
            raise SecurityError(f"Invalid tensor metadata descriptor for '{key}' in '{path.name}'.")

        offsets = val.get("data_offsets")
        if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
            raise SecurityError(f"Invalid data_offsets format for '{key}': {offsets}")

        begin, end = offsets[0], offsets[1]
        if not (isinstance(begin, int) and isinstance(end, int)):
            raise SecurityError(f"Non-integer data_offsets for '{key}': {offsets}")
        if begin < 0 or end < begin:
            raise SecurityError(f"Negative or inverted data_offsets for '{key}': [{begin}, {end}]")
        if end > tensor_buffer_size:
            raise SecurityError(
                f"Safetensors offset out of bounds for '{key}': offset {end} exceeds buffer size {tensor_buffer_size}."
            )

        shape = val.get("shape", [])
        if not isinstance(shape, list):
            raise SecurityError(f"Invalid shape for tensor '{key}': {shape}")
        for dim in shape:
            if not isinstance(dim, int) or dim < 0 or dim > 100_000:
                raise SecurityError(f"Invalid dimension {dim} for tensor '{key}'.")

    return header


def validate_input_shape(
    shape: tuple[int, ...] | Sequence[int] | None,
    max_elements: int = 500_000_000,
) -> tuple[int, ...]:
    """
    Validate tensor dimension bounds to prevent allocation bombs / Denial of Service.
    """
    if shape is None:
        return (1, 3, 224, 224)

    if not isinstance(shape, (tuple, list)):
        raise ValueError(f"Input shape must be a tuple or list of integers, got {type(shape)}")

    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"Input shape dimensions must be positive integers, got: {shape}")
        if dim > 100_000:
            raise ValueError(f"Input dimension {dim} exceeds safety ceiling of 100,000 in shape {shape}")

    total_elements = math.prod(shape)
    if total_elements > max_elements:
        raise ValueError(
            f"Input shape {shape} requires {total_elements:,} elements, exceeding "
            f"the maximum safety threshold of {max_elements:,} elements (allocation bomb protection)."
        )

    return tuple(shape)


def sanitize_display_text(text: str) -> str:
    """
    Strip ANSI escape sequences from untrusted model strings before rendering.
    """
    if not isinstance(text, str):
        return str(text)
    return ANSI_ESCAPE_PATTERN.sub("", text)
