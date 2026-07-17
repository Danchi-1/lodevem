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


def _fits_symbol(fits: bool, status: str) -> str:
    """Green tick or red cross with OOM label."""
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
    table.add_column("Latency (pred.)",    style="yellow",  min_width=14, justify="right")
    table.add_column("Peak RAM (meas.)",   style="magenta", min_width=14, justify="right")
    table.add_column("Fits in RAM",        min_width=10,    justify="center")

    current_tier = None
    for r in sorted(results, key=lambda x: (x["tier"], x["model_file"], x["device_name"])):

        # Add a visual separator when we move to a new tier
        if r["tier"] != current_tier:
            if current_tier is not None:
                table.add_section()
            current_tier = r["tier"]

        table.add_row(
            r["model_file"],
            r["device_name"],
            str(r["tier"]),
            f"{r['ram_limit_mb']} MB",
            _format_latency(r.get("predicted_latency_ms")),
            _format_ram(r.get("peak_ram_mb")),
            _fits_symbol(r.get("fits_in_ram", False), r.get("measure_status", "")),
        )

    console.print(table)

    # Summary line
    total = len(results)
    oom_count = sum(1 for r in results if r.get("measure_status") == "oom")
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

    # Define the columns we want in the CSV (in order)
    fieldnames = [
        "model_file",
        "model_size_mb",
        "device_name",
        "tier",
        "tier_label",
        "core_type",
        "ram_limit_mb",
        "predicted_latency_ms",
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
