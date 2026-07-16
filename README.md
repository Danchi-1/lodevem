# lodevem

> **Lo**w-resource **De**vice **V**irtual **Em**ulator — A benchmarking harness for evaluating compressed PyTorch models against simulated low-cost Android device profiles, without requiring physical target hardware.

---

## Motivation

Deploying machine learning models to low-resource Android devices (e.g., entry-level smartphones common in West Africa) is challenging to validate without physical access to diverse hardware. Researchers often face a catch-22: claiming a model is "lightweight" without empirical evidence on target hardware leads to paper rejections, yet acquiring many different physical devices is impractical.

`lodevem` solves this by combining:
- **nn-Meter** (peer-reviewed, Microsoft Research) for kernel-level latency prediction on target mobile SoCs
- **Docker + cgroups v2** for real RAM-constrained memory measurement and OOM detection
- A reproducible compression pipeline (FP32 → FP16 → INT8 → pruned) applied to your PyTorch model

The result is a complete, reproducible benchmarking report — suitable for inclusion in academic papers — generated entirely without physical target hardware.

---

## How It Works

```
Your PyTorch model
      │
      ▼
┌─────────────────────────────┐
│   Compression Pipeline      │
│  FP32 → FP16 → INT8 → Pruned│
└─────────────┬───────────────┘
              │
      ┌───────┴────────┐
      ▼                ▼
┌──────────────┐  ┌────────────────────────┐
│  nn-Meter    │  │  Docker + cgroups v2   │
│  Latency     │  │  RAM-constrained       │
│  Prediction  │  │  Memory Measurement    │
│  (per device │  │  + OOM Detection       │
│   SoC)       │  │  (per device profile)  │
└──────┬───────┘  └──────────┬─────────────┘
       └──────────┬──────────┘
                  ▼
        ┌─────────────────┐
        │  Results Table  │
        │  (console, CSV, │
        │   LaTeX-ready)  │
        └─────────────────┘
```

---

## Device Profiles

`lodevem` ships with profiles across four hardware tiers, from mainstream budget Android down to KaiOS feature phones. This range is designed to answer the question: *at what point does the model break?*

### Tier 1 — Budget Android (2–4 GB RAM)

| Profile ID        | Device               | Chipset         | Cores           | Clock    | RAM  |
|-------------------|----------------------|-----------------|-----------------|----------|------|
| `tecno_spark8`    | Tecno Spark 8        | Helio A22       | 4× Cortex-A53   | 2.0 GHz  | 2 GB |
| `itel_a70`        | Itel A70             | Unisoc SC9863A  | 4× Cortex-A55   | 1.6 GHz  | 2 GB |
| `samsung_a03`     | Samsung Galaxy A03   | Unisoc T606     | 2× A75 + 6× A55 | 1.6 GHz  | 3 GB |
| `infinix_hot11s`  | Infinix Hot 11s      | Helio G88       | 2× A75 + 6× A55 | 2.0 GHz  | 4 GB |
| `tecno_pop6`      | Tecno Pop 6 Pro      | Helio A22       | 4× Cortex-A53   | 2.0 GHz  | 2 GB |
| `nokia_g11`       | Nokia G11            | Unisoc T606     | 2× A75 + 6× A55 | 1.6 GHz  | 3 GB |

### Tier 2 — Android Go (512 MB – 2 GB RAM)

Android Go is Google's stripped-down OS variant designed for devices with ≤2 GB RAM. These represent the true lower bound of Android ML inference.

| Profile ID         | Device                  | Chipset         | Cores          | Clock    | RAM    |
|--------------------|-------------------------|-----------------|----------------|----------|--------|
| `nokia_c1`         | Nokia C1 2nd Edition    | MediaTek MT6580 | 4× Cortex-A7   | 1.3 GHz  | 1 GB   |
| `tecno_pop5_go`    | Tecno Pop 5 Go          | Helio A20       | 4× Cortex-A53  | 1.8 GHz  | 1 GB   |
| `itel_p37`         | Itel P37                | Unisoc SC9863A  | 4× Cortex-A55  | 1.6 GHz  | 1 GB   |
| `redmi_a1`         | Xiaomi Redmi A1         | Helio A22       | 4× Cortex-A53  | 1.8 GHz  | 2 GB   |
| `samsung_a03_core` | Samsung Galaxy A03 Core | Unisoc SC9863A  | 8× Cortex-A55  | 1.6 GHz  | 2 GB   |
| `itel_a23_pro`     | Itel A23 Pro            | Unisoc SC9832E  | 4× Cortex-A53  | 1.4 GHz  | 512 MB |

### Tier 3 — KaiOS / Feature Phones (256 – 512 MB RAM)

KaiOS devices are "smart feature phones" — button phones with a browser and basic app runtime. They do not run native Python or PyTorch. These profiles exist to test the **absolute RAM floor**: can your model even be *loaded* within 256–512 MB? This directly answers whether a ONNX or quantized model could theoretically be ported to these constraints.

| Profile ID       | Device              | Chipset           | Cores          | Clock    | RAM    |
|------------------|---------------------|-------------------|----------------|----------|--------|
| `nokia_8110_4g`  | Nokia 8110 4G       | Snapdragon 205    | 2× Cortex-A7   | 1.1 GHz  | 256 MB |
| `jiophone2`      | JioPhone 2          | Snapdragon 205    | 2× Cortex-A7   | 1.1 GHz  | 512 MB |
| `nokia_2720_flip`| Nokia 2720 Flip     | Snapdragon 205    | 2× Cortex-A7   | 1.1 GHz  | 512 MB |
| `itel_it5626`    | Itel it5626 (4G)    | MediaTek MT6739   | 4× Cortex-A53  | 1.3 GHz  | 512 MB |

> **Note on KaiOS profiles:** These profiles measure whether your model fits in memory and completes a forward pass under extreme RAM constraints. Actual KaiOS runtime environments cannot execute PyTorch natively — the test is a proxy for "could a severely quantized version of this model run on this class of hardware?"

Custom profiles can be added via YAML configuration.

---

## Compression Levels

Each benchmark run tests the same model across four compression levels:

| Level       | Method                              | PyTorch Mechanism                        |
|-------------|-------------------------------------|------------------------------------------|
| `fp32`      | Baseline (no compression)           | Default `.pt` / `nn.Module`              |
| `fp16`      | Half-precision weights              | `model.half()`                           |
| `int8`      | Post-training dynamic quantization  | `torch.quantization.quantize_dynamic`    |
| `pruned`    | Structured channel pruning (50%)    | `torch.nn.utils.prune` + fine-tune       |

---

## Benchmark Output

Running `lodevem` produces a results table like this:

```
Device Profile     | Compression | Latency (ms) | Peak RAM (MB) | Fits in RAM | Accuracy (%)
-------------------|-------------|--------------|---------------|-------------|-------------
Tecno Spark 8      | FP32        | 847          | 42.3          | ✓           | 94.1
Tecno Spark 8      | FP16        | 791          | 21.8          | ✓           | 94.1
Tecno Spark 8      | INT8        | 312          | 11.2          | ✓           | 93.4
Tecno Spark 8      | Pruned-50%  | 480          | 22.1          | ✓           | 92.8
Itel A70           | FP32        | 1021         | 42.3          | ✓           | 94.1
...
```

Results are saved as:
- `results/benchmark_results.csv` — machine-readable
- `results/benchmark_results.json` — structured, for programmatic use
- Console table printed directly to stdout

---

## Project Structure

```
lodevem/
├── README.md
├── pyproject.toml              # Package config + 'lodevem' CLI entrypoint
├── requirements.txt
├── Dockerfile                  # ARM-constrained benchmark container
├── docker-compose.yml          # Orchestrates profile runs
│
├── profiles/                   # Device profile definitions (YAML)
│   ├── tier1/                  # Budget Android
│   │   ├── tecno_spark8.yaml
│   │   ├── itel_a70.yaml
│   │   ├── samsung_a03.yaml
│   │   └── ...
│   ├── tier2/                  # Android Go
│   │   ├── nokia_c1.yaml
│   │   ├── tecno_pop5_go.yaml
│   │   └── ...
│   ├── tier3/                  # KaiOS / feature phones
│   │   ├── nokia_8110_4g.yaml
│   │   ├── jiophone2.yaml
│   │   └── ...
│   └── custom_template.yaml
│
├── lodevem/                    # Core Python package
│   ├── __init__.py
│   ├── cli.py                  # CLI entrypoint ('lodevem start', 'lodevem list', etc.)
│   ├── compress.py             # Compression pipeline (FP32→FP16→INT8→pruned)
│   ├── measure.py              # Memory + latency measurement inside container
│   ├── predict.py              # nn-Meter latency prediction wrapper
│   ├── runner.py               # Orchestrates Docker containers per profile
│   ├── reporter.py             # Table generation (console, CSV, JSON)
│   └── profiles.py             # Profile loader and validator
│
├── models/                     # Place your .pt model file here
│   └── .gitkeep
│
└── results/                    # Benchmark output (auto-generated)
    └── .gitkeep
```

---

## Installation

### Prerequisites

- Linux (Ubuntu 20.04+ recommended)
- Python 3.9+
- Docker Engine (with cgroups v2 enabled)
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/lodevem.git
cd lodevem

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install lodevem as a CLI tool
pip install -e .

# Verify the CLI is available
lodevem --help

# Verify cgroups v2 is active on your system
cat /sys/fs/cgroup/cgroup.controllers
# Expected: cpuset cpu io memory hugetlb pids rdma misc
```

### Enable cgroups v2 (if not already active)

```bash
# Check current cgroup version
stat -fc %T /sys/fs/cgroup/
# "cgroup2fs" = v2 (good), "tmpfs" = v1 (needs update)

# If on v1, add to GRUB_CMDLINE_LINUX in /etc/default/grub:
# systemd.unified_cgroup_hierarchy=1
# Then: sudo update-grub && sudo reboot
```

---

## Usage

After `pip install -e .`, the `lodevem` command becomes available globally in your environment.

### Start a full benchmark run

```bash
lodevem start --model models/your_model.pt
```

This single command auto-detects your Docker setup, loads all device profiles, runs the compression pipeline, and prints the results table.

### Target a specific device tier

```bash
lodevem start --model models/your_model.pt --tier tier2        # Android Go only
lodevem start --model models/your_model.pt --tier tier3        # KaiOS / feature phones
```

### Target a single device

```bash
lodevem start --model models/your_model.pt --profile nokia_c1
```

### Limit to specific compression levels

```bash
lodevem start --model models/your_model.pt --levels fp32 int8
```

### List all available device profiles

```bash
lodevem list
lodevem list --tier tier3        # Filter by tier
```

### Check system readiness (Docker, cgroups v2)

```bash
lodevem check
```

### All options

```
lodevem start  --model PATH        Path to your .pt model file (required)
               --profile ID        Run a single device profile
               --tier TIER         Run all profiles in a tier (tier1 | tier2 | tier3)
               --levels LEVELS     Compression levels: fp32 fp16 int8 pruned (default: all)
               --warmup N          Warmup passes before timing (default: 5)
               --runs N            Timed inference passes (default: 50)
               --output PATH       Save CSV results to this path

lodevem list   [--tier TIER]       List all available device profiles
lodevem check                      Verify Docker, cgroups v2, and nn-Meter setup
lodevem --help                     Show this help message
```

---

## Methodology & Academic Citation

### Latency Measurement

Inference latency is predicted using **nn-Meter** (Zhang et al., 2021), a kernel-level latency predictor that models operator fusion and hardware-specific execution units for common mobile SoCs. Predicted latency corresponds to single-threaded CPU inference on the target chipset.

> Zhang, L., et al. (2021). *nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices*. MobiSys '21. https://github.com/microsoft/nn-Meter

### Memory Measurement

Peak RAM usage is measured inside a Docker container with `memory` and `memory-swap` cgroup v2 limits set to match each device profile's RAM specification. Measurement is performed via `/proc/self/status` (`VmRSS` field) at inference peak. OOM kill events are detected and reported as constraint failures.

### Reproducibility

All device profiles, compression configurations, and benchmark parameters are version-controlled. The entire benchmark can be reproduced by any researcher with a Linux host and Docker installed.

---

## Limitations

This tool provides **simulated** results, not physical device measurements. Users should be aware of the following:

1. **Latency** is *predicted*, not measured. nn-Meter has ~10–15% mean absolute percentage error on supported SoCs.
2. **Memory** measurements are real within Docker constraints, but Android's memory manager (LMKD) and JVM overhead are not replicated.
3. **Thermal throttling** is not simulated. Real devices may perform worse under sustained load.
4. **Hardware accelerators** (NPU, DSP, GPU) are not modeled. Predictions correspond to CPU-only inference.
5. For final validation, cross-checking at least one result against a physical device is strongly recommended.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Issues and pull requests are welcome. If you add a new device profile, please include:
- Source for the chipset specifications (e.g., GSMArena, AnTuTu database)
- The nn-Meter hardware predictor ID used
- Verified RAM capacity from the device manufacturer
