"""
cli.py — Command-Line Interface Entry Point

What this file does:
    Defines the 'lodevem' command and its subcommands:
        lodevem start   — run the benchmark
        lodevem list    — show all available device profiles
        lodevem check   — verify the system is ready (Docker, nn-Meter, cgroups)

How it becomes a terminal command:
    pyproject.toml has this line:
        lodevem = "lodevem.cli:main"
    After 'pip install -e .', Python registers 'lodevem' as a script
    that calls the main() function in this file.

We use Python's built-in 'argparse' for argument parsing —
no extra dependencies needed for the CLI itself.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# We use Python's standard logging module.
# By default, only WARNING and above are shown.
# The --verbose flag enables INFO level (more detail).
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        format="%(message)s",
        level=level,
    )


# ---------------------------------------------------------------------------
# Subcommand: lodevem check
# ---------------------------------------------------------------------------

def cmd_check(args: argparse.Namespace) -> None:
    """
    Verify the system is ready to run benchmarks.

    Checks:
    1. Docker is installed and running
    2. cgroups v2 is enabled (needed for memory limits to work)
    3. nn-Meter is installed
    4. PyTorch is installed
    """
    console.print("\n[bold]lodevem system check[/bold]\n")

    all_ok = True

    # --- Docker ---
    # Docker is optional. If it's not available we'll run in "lite" mode
    # (psutil-based measurement). Only warn the user instead of failing.
    try:
        import docker
        client = docker.from_env()
        client.ping()
        console.print("  [green]✓[/green]  Docker        running")
    except ImportError:
        console.print("  [yellow]![/yellow]  Docker SDK    not installed — running in lite mode (no Docker).\n              To enable full mode install: pip install docker")
        # Docker missing is not a fatal error; lite mode will be used.
    except Exception as e:
        console.print(f"  [yellow]![/yellow]  Docker        not reachable — running in lite mode (Docker error: {e})")
        # Docker daemon problems are not fatal for lite mode.

    # --- cgroups v2 ---
    # The file /sys/fs/cgroup/cgroup.controllers only exists on cgroups v2.
    cgroup_v2_path = Path("/sys/fs/cgroup/cgroup.controllers")
    if cgroup_v2_path.exists():
        controllers = cgroup_v2_path.read_text().strip()
        if "memory" in controllers:
            console.print(f"  [green]✓[/green]  cgroups v2    active (memory controller present)")
        else:
            console.print(f"  [yellow]![/yellow]  cgroups v2    active but memory controller missing: {controllers}")
            all_ok = False
    else:
        console.print(
            "  [red]✗[/red]  cgroups v2    not detected\n"
            "              Add 'systemd.unified_cgroup_hierarchy=1' to GRUB_CMDLINE_LINUX\n"
            "              then run: sudo update-grub && sudo reboot"
        )
        all_ok = False

    # --- nn-Meter ---
    try:
        import nn_meter  # noqa: F401
        console.print("  [green]✓[/green]  nn-Meter      installed")
    except ImportError:
        console.print("  [red]✗[/red]  nn-Meter      not installed (run: pip install nn-meter)")
        all_ok = False

    # --- PyTorch ---
    try:
        import torch
        console.print(f"  [green]✓[/green]  PyTorch       {torch.__version__}")
    except ImportError:
        console.print("  [red]✗[/red]  PyTorch       not installed")
        all_ok = False

    console.print()
    if all_ok:
        console.print("[bold green]All checks passed. Ready to benchmark.[/bold green]\n")
    else:
        console.print("[bold red]Some checks failed. Fix the issues above before running 'lodevem start'.[/bold red]\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: lodevem list
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    """
    Print all available device profiles as a table.
    Optionally filter by tier.
    """
    from lodevem import profiles as profile_loader

    try:
        if args.tier:
            tier_num = int(args.tier.replace("tier", ""))
            all_profiles = profile_loader.load_tier(tier_num)
        else:
            all_profiles = profile_loader.load_all()
    except Exception as e:
        console.print(f"[red]Error loading profiles: {e}[/red]")
        sys.exit(1)

    table = Table(
        title="\n[bold cyan]Available Device Profiles[/bold cyan]",
        box=box.ROUNDED,
        header_style="bold white",
        border_style="dim",
    )
    table.add_column("ID",          style="cyan",    min_width=18)
    table.add_column("Device",      style="white",   min_width=24)
    table.add_column("Tier",        min_width=24)
    table.add_column("Core",        style="dim",     min_width=18)
    table.add_column("RAM",         justify="right", min_width=6)
    table.add_column("Cores",       justify="right", min_width=5)

    current_tier = None
    for p in all_profiles:
        if p.tier != current_tier:
            if current_tier is not None:
                table.add_section()
            current_tier = p.tier

        # Color-code the tier label
        tier_colors = {1: "green", 2: "yellow", 3: "red"}
        color = tier_colors.get(p.tier, "white")
        tier_label = f"[{color}]Tier {p.tier} — {p.tier_label}[/{color}]"

        table.add_row(
            p.id,
            p.name,
            tier_label,
            p.core_type,
            f"{p.ram_mb} MB",
            str(p.cores),
        )

    console.print(table)
    console.print(f"\n[dim]{len(all_profiles)} profile(s) total[/dim]\n")


# ---------------------------------------------------------------------------
# Subcommand: lodevem start
# ---------------------------------------------------------------------------

def cmd_start(args: argparse.Namespace) -> None:
    """
    Run the benchmark. This is the main event.

    Orchestration:
        1. Validate the model files exist
        2. Determine which profiles to use
        3. Call runner.run_benchmark()
        4. Call reporter.print_table() and reporter.save_csv()
    """
    from lodevem import runner, reporter

    # --- Validate model files ---
    model_paths = []
    for path_str in args.models:
        path = Path(path_str)
        if not path.exists():
            console.print(f"[red]Model file not found: {path}[/red]")
            sys.exit(1)
        if path.suffix not in (".pt", ".pth"):
            console.print(f"[yellow]Warning: '{path.name}' doesn't look like a PyTorch model (.pt/.pth)[/yellow]")
        model_paths.append(path)

    # --- Determine profile filter ---
    profile_ids = None
    tier = None

    if args.profile:
        profile_ids = [args.profile]
    elif args.tier:
        tier = int(args.tier.replace("tier", ""))

    # --- Banner ---
    console.print(f"\n[bold cyan]lodevem[/bold cyan] — Low-resource Device Virtual Emulator")
    console.print(f"[dim]Models: {[p.name for p in model_paths]}[/dim]")
    console.print(f"[dim]Warmup: {args.warmup} runs  |  Timed: {args.runs} runs[/dim]\n")

    # --- Run ---
    try:
        results = runner.run_benchmark(
            model_paths=model_paths,
            profile_ids=profile_ids,
            tier=tier,
            warmup_runs=args.warmup,
            timed_runs=args.runs,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Benchmark failed: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # --- Display results ---
    reporter.print_table(results)

    # --- Save to file ---
    output_path = args.output if args.output else None
    reporter.save_csv(results, output_path)
    reporter.save_json(results)


# ---------------------------------------------------------------------------
# Argument parser — defines all the flags and subcommands
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lodevem",
        description="Low-resource Device Virtual Emulator — benchmark PyTorch models on simulated device profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  lodevem start models/cocoa_fp32.pt models/cocoa_int8.pt
  lodevem start models/cocoa_int8.pt --tier tier2
  lodevem start models/cocoa_int8.pt --profile nokia_c1
  lodevem list
  lodevem list --tier tier3
  lodevem check
        """,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed logging output")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # --- start ---
    start_parser = subparsers.add_parser(
        "start",
        help="Run the benchmark against device profiles",
    )
    start_parser.add_argument(
        "models",
        nargs="+",
        metavar="MODEL",
        help="One or more .pt model files to benchmark",
    )
    profile_group = start_parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        metavar="ID",
        help="Benchmark against a single device profile (e.g. nokia_c1)",
    )
    profile_group.add_argument(
        "--tier",
        choices=["tier1", "tier2", "tier3"],
        help="Benchmark against all profiles in a tier",
    )
    start_parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        metavar="N",
        help="Number of warmup inference passes (default: 5)",
    )
    start_parser.add_argument(
        "--runs",
        type=int,
        default=50,
        metavar="N",
        help="Number of timed inference passes (default: 50)",
    )
    start_parser.add_argument(
        "--output",
        metavar="PATH",
        help="Save CSV results to this path (default: ./results/benchmark_<timestamp>.csv)",
    )
    start_parser.set_defaults(func=cmd_start)

    # --- list ---
    list_parser = subparsers.add_parser(
        "list",
        help="List all available device profiles",
    )
    list_parser.add_argument(
        "--tier",
        choices=["tier1", "tier2", "tier3"],
        help="Filter by tier",
    )
    list_parser.set_defaults(func=cmd_list)

    # --- check ---
    check_parser = subparsers.add_parser(
        "check",
        help="Verify Docker, cgroups v2, and dependencies are ready",
    )
    check_parser.set_defaults(func=cmd_check)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# This is the function called when someone types 'lodevem' in the terminal.
# pyproject.toml maps 'lodevem' → 'lodevem.cli:main'
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(verbose=getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()
