"""
reporter.py — Results Formatter and Saver

What this file does:
    Takes the flat list of result dicts from runner.py
    and produces two outputs:
    1. A colored table printed to the terminal (using Rich)
    2. A CSV file saved to the results/ directory

Why Rich for the terminal table?
    Rich is a Python library for beautiful terminal output. It handles
    column alignment, color coding, and Unicode box characters automatically.
    The table you see is exactly what you'd want to screenshot for a presentation.

Why CSV for the file output?
    CSV opens in Excel, Google Sheets, and can be read by pandas, R, LaTeX, etc.
    It's the most portable format for research data.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

logger = logging.getLogger(__name__)

# Path where results are saved (defaults to ./results in current working directory)
RESULTS_DIR = Path.cwd() / "results"

console = Console()


def _format_latency(value_ms: float | None) -> str:
    """Format latency for display. Returns '—' if None."""
    if value_ms is None:
        return "—"
    if value_ms >= 1000:
        return f"{value_ms / 1000:.1f}s"   # Show in seconds if > 1s
    return f"{value_ms:.0f}ms"


def _format_ram(value_mb: float | None) -> str:
    """Format RAM for display."""
    if value_mb is None:
        return "—"
    return f"{value_mb:.1f} MB"


def _fits_symbol(fits: bool, status: str, preflight_status: str | None = None) -> str:
    """Green tick or red cross with OOM label."""
    if preflight_status == "preflight_oom":
        return "[red]✗ PRE-OOM[/red]"
    if status == "oom":
        return "[red]✗ OOM[/red]"
    if fits:
        return "[green]✓[/green]"
    return "[red]✗[/red]"


def print_table(results: list[dict]) -> None:
    """
    Print a formatted results table to the terminal using Rich.

    Groups rows by tier for readability, with a separator between tiers.
    """
    table = Table(
        title="\n[bold cyan]lodevem Benchmark Results[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white",
        border_style="dim",
        row_styles=["", "dim"],   # Alternate row shading
        expand=False,
    )

    # Column definitions
    table.add_column("Model File",         style="cyan",    min_width=16)
    table.add_column("Device",             style="white",   min_width=20)
    table.add_column("Tier",               style="dim",     min_width=6,  justify="center")
    table.add_column("RAM Limit",          style="dim",     min_width=8,  justify="right")
    table.add_column("Latency (pred)",     style="yellow",  min_width=14, justify="right")
    table.add_column("TTFT (ms)",          style="yellow",  min_width=10, justify="right")
    table.add_column("Speed (TPS)",        style="yellow",  min_width=11, justify="right")
    table.add_column("Peak RAM (meas)",    style="magenta", min_width=15, justify="right")
    table.add_column("Fits in RAM",        min_width=11,    justify="center")

    current_tier = None
    for r in sorted(results, key=lambda x: (x["tier"], x["model_file"], x["device_name"])):

        # Add a visual separator when we move to a new tier
        if r["tier"] != current_tier:
            if current_tier is not None:
                table.add_section()
            current_tier = r["tier"]

        is_llm_row = "median_ttft_ms" in r and r["median_ttft_ms"] is not None
        latency_str = "" if is_llm_row else _format_latency(r.get("predicted_latency_ms"))
        ttft_str = _format_latency(r.get("median_ttft_ms")) if is_llm_row else ""
        tps_str = f"{r.get('median_tps'):.1f} t/s" if is_llm_row and r.get("median_tps") is not None else ""

        table.add_row(
            r["model_file"],
            r["device_name"],
            str(r["tier"]),
            f"{r['ram_limit_mb']} MB",
            latency_str,
            ttft_str,
            tps_str,
            _format_ram(r.get("peak_ram_mb")),
            _fits_symbol(r.get("fits_in_ram", False), r.get("measure_status", ""), r.get("preflight_status")),
        )

    console.print(table)

    # Summary line
    total = len(results)
    oom_count = sum(1 for r in results if r.get("measure_status") == "oom" or r.get("preflight_status") == "preflight_oom")
    ok_count = sum(1 for r in results if r.get("measure_status") == "ok")

    console.print(
        f"\n[dim]Total: {total} runs  |  "
        f"[green]{ok_count} passed[/green]  |  "
        f"[red]{oom_count} OOM[/red][/dim]\n"
    )


def save_csv(results: list[dict], output_path: str | Path | None = None) -> Path:
    """
    Save results to a CSV file.

    Args:
        results:     The list of result dicts from runner.py.
        output_path: Where to save the file. If None, auto-generates a
                     timestamped filename in results/.

    Returns:
        The Path where the file was saved.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"benchmark_{timestamp}.csv"

    output_path = Path(output_path)

    fieldnames = [
        "model_file",
        "model_size_mb",
        "device_name",
        "tier",
        "tier_label",
        "core_type",
        "ram_limit_mb",
        "preflight_status",
        "estimated_memory_mb",
        "predicted_latency_ms",
        "median_latency_ms",
        "median_ttft_ms",
        "decode_time_ms",
        "median_tps",
        "generated_tokens",
        "prompt_length",
        "peak_ram_mb",
        "fits_in_ram",
        "measure_status",
        "error",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to: {output_path}")
    console.print(f"[dim]Results saved → [cyan]{output_path}[/cyan][/dim]")

    return output_path


def save_json(results: list[dict], output_path: str | Path | None = None) -> Path:
    """Save results as JSON (useful for programmatic processing)."""
    RESULTS_DIR.mkdir(exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"benchmark_{timestamp}.json"

    output_path = Path(output_path)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return output_path


def _format_count(val: int | None) -> str:
    """Format parameter or element counts with unit suffixes."""
    if val is None:
        return "—"
    if val >= 1_000_000_000:
        return f"{val / 1_000_000_000:.2f}B"
    if val >= 1_000_000:
        return f"{val / 1_000_000:.2f}M"
    if val >= 1_000:
        return f"{val / 1_000:.1f}K"
    return str(val)


def _format_flops(flops: int | None, status: str) -> str:
    """Format FLOPs value with status annotation."""
    if status == "not_applicable":
        return "[dim]N/A[/dim]"
    if status == "unsupported_operators":
        return "[yellow]unsupported[/yellow]"
    if status == "partial":
        if flops is not None:
            return f"{_format_count(flops)} [yellow](partial)[/yellow]"
        return "[yellow]partial[/yellow]"
    if flops is None:
        return "—"
    if flops >= 1_000_000_000:
        return f"{flops / 1_000_000_000:.2f} GFLOPs"
    if flops >= 1_000_000:
        return f"{flops / 1_000_000:.2f} MFLOPs"
    return f"{flops} FLOPs"


def print_footprint_scorecard(footprints: list[Any]) -> None:
    """
    Print rich scorecard tables detailing static model architecture, parameters,
    memory footprint, and FLOPs.
    """
    # 1. Overview Table
    overview_table = Table(
        title="\n[bold cyan]lodevem Model Footprint Scorecard[/bold cyan]",
        box=box.ROUNDED,
        header_style="bold white",
        border_style="dim",
        row_styles=["", "dim"],
    )
    overview_table.add_column("Model File",      style="cyan",    min_width=18)
    overview_table.add_column("Backend",         style="white",   min_width=12)
    overview_table.add_column("Disk Size",       style="dim",     justify="right", min_width=10)
    overview_table.add_column("Parameters",      style="green",   justify="right", min_width=12)
    overview_table.add_column("Weights RAM",     style="magenta", justify="right", min_width=12)
    overview_table.add_column("FLOPs / MACs",    style="yellow",  justify="right", min_width=15)
    overview_table.add_column("FLOPs Status",    style="dim",     justify="center", min_width=12)
    overview_table.add_column("Min RAM Est",     style="blue",    justify="right", min_width=12)

    for fp in footprints:
        params_str = (
            _format_count(fp.total_parameters)
            if fp.total_parameters is not None
            else ("[dim]N/A (Tree)[/dim]" if fp.backend == "sklearn" else "—")
        )
        weights_ram_str = (
            f"{fp.parameter_memory_mb:.1f} MB"
            if fp.parameter_memory_mb is not None
            else "—"
        )
        flops_str = _format_flops(fp.flops, fp.flops_status)
        min_ram_str = (
            f"~{fp.estimated_minimum_ram_mb:.0f} MB"
            if fp.estimated_minimum_ram_mb
            else "—"
        )

        overview_table.add_row(
            fp.model_name,
            fp.backend,
            f"{fp.file_size_mb:.1f} MB",
            params_str,
            weights_ram_str,
            flops_str,
            fp.flops_status,
            min_ram_str,
        )

    console.print(overview_table)

    # 2. Precision Breakdown Table (if any model has precision details)
    has_precision = any(fp.precision_breakdown for fp in footprints)
    if has_precision:
        prec_table = Table(
            title="\n[bold cyan]Weight Precision Breakdown[/bold cyan]",
            box=box.ROUNDED,
            header_style="bold white",
            border_style="dim",
        )
        prec_table.add_column("Model File", style="cyan", min_width=18)
        all_dtypes = sorted(
            list({dt for fp in footprints for dt in fp.precision_breakdown.keys()})
        )
        for dt in all_dtypes:
            prec_table.add_column(dt, justify="right", min_width=10)

        for fp in footprints:
            if fp.precision_breakdown:
                row = [fp.model_name]
                for dt in all_dtypes:
                    cnt = fp.precision_breakdown.get(dt)
                    row.append(_format_count(cnt) if cnt else "—")
                prec_table.add_row(*row)

        console.print(prec_table)

    # 3. LLM Architecture & KV-Cache Table (if any LLM models)
    llm_models = [fp for fp in footprints if getattr(fp, "is_llm", False)]
    if llm_models:
        llm_table = Table(
            title="\n[bold cyan]LLM Architecture & KV-Cache Footprint[/bold cyan]",
            box=box.ROUNDED,
            header_style="bold white",
            border_style="dim",
        )
        llm_table.add_column("Model File",      style="cyan",    min_width=18)
        llm_table.add_column("Ctx Window",      style="white",   justify="right", min_width=10)
        llm_table.add_column("Layers",          style="dim",     justify="right", min_width=8)
        llm_table.add_column("Attn Heads",      style="dim",     justify="right", min_width=10)
        llm_table.add_column("KV Heads",        style="dim",     justify="right", min_width=10)
        llm_table.add_column("KV/Token",        style="yellow",  justify="right", min_width=12)
        llm_table.add_column("KV @ 512 ctx",    style="magenta", justify="right", min_width=12)
        llm_table.add_column("KV @ 2048 ctx",   style="magenta", justify="right", min_width=12)

        for fp in llm_models:
            kv_token_str = (
                f"{fp.kv_cache_bytes_per_token:.0f} B"
                if fp.kv_cache_bytes_per_token is not None
                else "—"
            )
            kv_512 = f"{fp.kv_cache_projections_mb.get(512):.1f} MB" if 512 in fp.kv_cache_projections_mb else "—"
            kv_2048 = f"{fp.kv_cache_projections_mb.get(2048):.1f} MB" if 2048 in fp.kv_cache_projections_mb else "—"

            llm_table.add_row(
                fp.model_name,
                str(fp.context_window) if fp.context_window else "—",
                str(fp.num_layers) if fp.num_layers else "—",
                str(fp.num_attention_heads) if fp.num_attention_heads else "—",
                str(fp.num_key_value_heads) if fp.num_key_value_heads else "—",
                kv_token_str,
                kv_512,
                kv_2048,
            )

        console.print(llm_table)

    # 4. Scikit-Learn Estimator Table (if any sklearn models)
    sklearn_models = [fp for fp in footprints if fp.backend == "sklearn"]
    if sklearn_models:
        sk_table = Table(
            title="\n[bold cyan]Tree Estimator Breakdown (Scikit-Learn)[/bold cyan]",
            box=box.ROUNDED,
            header_style="bold white",
            border_style="dim",
        )
        sk_table.add_column("Model File",      style="cyan",   min_width=18)
        sk_table.add_column("Estimator",       style="white",  min_width=26)
        sk_table.add_column("Trees",           style="green",  justify="right", min_width=8)
        sk_table.add_column("Max Depth",       style="dim",    justify="right", min_width=10)
        sk_table.add_column("Total Nodes",     style="yellow", justify="right", min_width=12)
        sk_table.add_column("Features In",     style="dim",    justify="right", min_width=12)

        for fp in sklearn_models:
            sk_table.add_row(
                fp.model_name,
                fp.estimator_type or "—",
                str(fp.n_estimators) if fp.n_estimators else "—",
                str(fp.max_depth) if fp.max_depth else "None",
                _format_count(fp.total_node_count) if fp.total_node_count else "—",
                str(fp.n_features_in) if fp.n_features_in else "—",
            )

        console.print(sk_table)

    # 5. Diagnostic GPU info (if requested)
    gpu_models = [fp for fp in footprints if getattr(fp, "gpu_diagnostic_mode", False)]
    if gpu_models:
        console.print("\n[bold yellow]Diagnostic Host GPU Information:[/bold yellow]")
        for fp in gpu_models:
            gpus = fp.host_gpus_detected or ["None detected"]
            console.print(f"  • [cyan]{fp.model_name}[/cyan]: {', '.join(gpus)}")

    # 6. Warnings & Notes
    all_warnings = [(fp.model_name, w) for fp in footprints for w in fp.warnings]
    if all_warnings:
        console.print("\n[bold yellow]Footprint Notes & Warnings:[/bold yellow]")
        for mname, w in all_warnings:
            console.print(f"  • [cyan]{mname}[/cyan]: {w}")
    console.print()


def save_footprint_json(footprints: list[Any], output_path: str | Path | None = None) -> Path:
    """Save model footprints to a JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"footprint_{timestamp}.json"

    output_path = Path(output_path)

    data = [fp.to_dict() if hasattr(fp, "to_dict") else fp for fp in footprints]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    console.print(f"[dim]Footprint JSON saved → [cyan]{output_path}[/cyan][/dim]")
    return output_path

