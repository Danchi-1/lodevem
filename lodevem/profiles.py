"""
profiles.py — Device Profile Loader

What this file does:
    Reads the YAML files in the profiles/ directory and turns them
    into Python objects (DeviceProfile) that the rest of the tool can use.

Why a separate file?
    Every other module (predict.py, measure.py, runner.py) needs device
    information. Rather than each one reading YAML files itself, they all
    ask this module: "give me the profiles I need."
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# The DeviceProfile dataclass
# ---------------------------------------------------------------------------
# A dataclass is just a convenient way to group related data together.
# Think of it as a structured record — one DeviceProfile per device.
#
# Every field here maps directly to a key in the YAML files.
# ---------------------------------------------------------------------------

@dataclass
class DeviceProfile:
    id: str                         # Unique identifier, e.g. "tecno_spark8"
    name: str                       # Human-readable name, e.g. "Tecno Spark 8"
    tier: int                       # 1, 2, or 3
    chipset: str                    # e.g. "MediaTek Helio A22"
    core_type: str                  # e.g. "Cortex-A53"
    cores: int                      # Number of CPU cores
    clock_ghz: float                # Clock speed in GHz
    ram_mb: int                     # RAM in megabytes — Docker will cap memory at this
    nn_meter_predictor: str         # Which nn-Meter predictor to use as the base
    scaling_factor: float           # Multiplier to convert A76 prediction → this chip's speed
    notes: str = ""                 # Optional human-readable notes
    android_edition: Optional[str] = None  # e.g. "Android Go"
    os: Optional[str] = None        # e.g. "KaiOS 2.5"

    @property
    def ram_gb(self) -> float:
        """Convenience: RAM in GB for display."""
        return round(self.ram_mb / 1024, 1)

    @property
    def tier_label(self) -> str:
        """Human-readable tier label."""
        labels = {
            1: "Budget Android",
            2: "Android Go",
            3: "KaiOS / Feature Phone",
        }
        return labels.get(self.tier, f"Tier {self.tier}")

    def __str__(self) -> str:
        return f"{self.name} [{self.core_type}, {self.ram_mb}MB RAM]"


# ---------------------------------------------------------------------------
# Where to find the profile YAML files
# ---------------------------------------------------------------------------
# __file__ is the path to this file (profiles.py).
# We go up one directory (.parent) to reach the project root,
# then down into profiles/.
# This works regardless of where the user runs lodevem from.
# ---------------------------------------------------------------------------

PROFILES_DIR = Path(__file__).parent.parent / "profiles"

VALID_TIERS = {1, 2, 3}
TIER_DIR_MAP = {
    1: "tier1",
    2: "tier2",
    3: "tier3",
}


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> DeviceProfile:
    """
    Read one YAML file and return a DeviceProfile object.

    If the YAML is missing required fields, we raise a clear error
    rather than failing mysteriously later.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    required = ["id", "name", "tier", "chipset", "core_type", "cores",
                "clock_ghz", "ram_mb", "nn_meter_predictor", "scaling_factor"]

    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(
            f"Profile file '{path.name}' is missing required fields: {missing}"
        )

    return DeviceProfile(
        id=data["id"],
        name=data["name"],
        tier=int(data["tier"]),
        chipset=data["chipset"],
        core_type=data["core_type"],
        cores=int(data["cores"]),
        clock_ghz=float(data["clock_ghz"]),
        ram_mb=int(data["ram_mb"]),
        nn_meter_predictor=data["nn_meter_predictor"],
        scaling_factor=float(data["scaling_factor"]),
        notes=str(data.get("notes", "")),
        android_edition=data.get("android_edition"),
        os=data.get("os"),
    )


def load_all() -> list[DeviceProfile]:
    """
    Load every profile YAML across all tiers.

    Returns a list of DeviceProfile objects, sorted by tier then RAM.
    This is the main function everything else uses.
    """
    profiles = []

    for tier_num, tier_dir in TIER_DIR_MAP.items():
        tier_path = PROFILES_DIR / tier_dir
        if not tier_path.exists():
            continue  # Skip if the tier directory doesn't exist yet

        for yaml_file in sorted(tier_path.glob("*.yaml")):
            try:
                profile = _load_yaml(yaml_file)
                profiles.append(profile)
            except Exception as e:
                # Don't crash the whole tool if one profile file is broken.
                # Just warn and continue.
                print(f"[warning] Skipping {yaml_file.name}: {e}")

    if not profiles:
        raise RuntimeError(
            f"No device profiles found in '{PROFILES_DIR}'. "
            "Make sure the profiles/ directory exists and contains YAML files."
        )

    return profiles


def load_tier(tier: int) -> list[DeviceProfile]:
    """
    Load only profiles from a specific tier (1, 2, or 3).

    Example:
        tier2_profiles = load_tier(2)  # Android Go devices only
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"Invalid tier '{tier}'. Must be one of: {sorted(VALID_TIERS)}")

    all_profiles = load_all()
    return [p for p in all_profiles if p.tier == tier]


def load_by_id(profile_id: str) -> DeviceProfile:
    """
    Load a single profile by its ID string.

    Example:
        profile = load_by_id("nokia_c1")

    Raises a clear error if the ID doesn't exist.
    """
    all_profiles = load_all()
    matches = [p for p in all_profiles if p.id == profile_id]

    if not matches:
        available = [p.id for p in all_profiles]
        raise ValueError(
            f"No profile found with id '{profile_id}'.\n"
            f"Available profiles: {available}"
        )

    return matches[0]
