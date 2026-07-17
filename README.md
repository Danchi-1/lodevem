# lodevem

> **Lo**w-resource **De**vice **V**irtual **Em**ulator — benchmark PyTorch models against simulated low-cost Android device profiles, without physical target hardware.

[![PyPI version](https://img.shields.io/pypi/v/lodevem)](https://pypi.org/project/lodevem/)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/lodevem)](https://pypi.org/project/lodevem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What It Does

`lodevem` answers the question: *"If a farmer in rural Ghana with a Nokia C1 (1GB RAM) runs my cocoa disease model, will it work? How fast? Will it crash?"*

You bring your model files (already compressed however you like). lodevem benchmarks each one against a library of 16 real device profiles spanning budget Android phones, Android Go devices, and KaiOS feature phones — and produces a results table ready for your paper.

**lodevem does not compress your model.** That is your responsibility as the researcher.

---

## How It Works

```
Your .pt model files  (cocoa_fp32.pt, cocoa_int8.pt, ...)
              │
              ▼
    ┌─────────────────┐
    │    lodevem      │  benchmarks each model against each device profile
    └────────┬────────┘
             │
    ┌────────┴─────────┐
    ▼                  ▼
nn-Meter              psutil / Docker
Latency Prediction    RAM Measurement
(per device SoC)      (per device profile)
    │                  │
    └────────┬─────────┘
             ▼
      Results Table
   (console + CSV file)
```

### Two measurement modes — selected automatically

| Environment | Mode | What it does |
|---|---|---|
| Kaggle, Colab, any machine | **Lite** (default) | Uses `psutil` to measure real peak RAM. No containers needed. |
| Linux with Docker running | **Full** | Spins up a RAM-capped container. Hard OOM detection enforced by the kernel. |

You don't choose the mode — lodevem detects Docker automatically and uses whichever is appropriate.

---

## Installation

```bash
pip install lodevem
```

**That's it.** All 16 device profiles are bundled inside the package. No cloning required.

### Optional: enable full Docker mode (Linux only)

If you're on Linux with Docker installed and want hard OOM detection:

```bash
pip install lodevem[docker]
```

Then make sure Docker is running:
```bash
sudo systemctl start docker
```

lodevem will detect Docker automatically on the next run.

---

## Usage

### Benchmark one model across all 16 device profiles

```bash
lodevem start models/cocoa_int8.pt
```

### Benchmark multiple model variants side by side

```bash
lodevem start models/cocoa_fp32.pt models/cocoa_int8.pt models/cocoa_pruned.pt
```

The filename is the row label in the results table — no extra flags needed.

### Filter by device tier

```bash
lodevem start models/cocoa_int8.pt --tier tier2     # Android Go devices only
lodevem start models/cocoa_int8.pt --tier tier3     # KaiOS / feature phones only
```

### Target a single device profile

```bash
lodevem start models/cocoa_int8.pt --profile nokia_c1
```

### List all available device profiles

```bash
lodevem list
lodevem list --tier tier2
```

### Check system readiness

```bash
lodevem check
```

Verifies Docker (if applicable), cgroups v2, nn-Meter, and PyTorch.

### Use from Python (recommended for Kaggle notebooks)

```python
from lodevem import runner, reporter

results = runner.run_benchmark(
    model_paths=["models/cocoa_fp32.pt", "models/cocoa_int8.pt"],
    tier=2,          # Android Go devices only (optional)
)

reporter.print_table(results)
reporter.save_csv(results)   # → results/benchmark_<timestamp>.csv
```

### All CLI options

```
lodevem start  MODEL [MODEL ...]   One or more .pt model files to benchmark
               --profile ID        Run against a single device profile
               --tier TIER         Run against a full tier (tier1 | tier2 | tier3)
               --warmup N          Warmup passes before timing (default: 5)
               --runs N            Timed inference passes (default: 50)
               --output PATH       Save CSV results to this path

lodevem list   [--tier TIER]       List all available device profiles
lodevem check                      Verify system readiness
lodevem --help                     Show help
```

---

## Device Profiles

16 profiles across 3 tiers, covering the realistic hardware range for low-cost Android and feature phones in West Africa and similar markets.

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

Android Go is Google's stripped-down OS variant for devices with ≤2 GB RAM.

| Profile ID         | Device                  | Chipset         | Cores          | Clock    | RAM    |
|--------------------|-------------------------|-----------------|----------------|----------|--------|
| `nokia_c1`         | Nokia C1 2nd Edition    | MediaTek MT6580 | 4× Cortex-A7   | 1.3 GHz  | 1 GB   |
| `tecno_pop5_go`    | Tecno Pop 5 Go          | Helio A20       | 4× Cortex-A53  | 1.8 GHz  | 1 GB   |
| `itel_p37`         | Itel P37                | Unisoc SC9863A  | 4× Cortex-A55  | 1.6 GHz  | 1 GB   |
| `redmi_a1`         | Xiaomi Redmi A1         | Helio A22       | 4× Cortex-A53  | 1.8 GHz  | 2 GB   |
| `samsung_a03_core` | Samsung Galaxy A03 Core | Unisoc SC9863A  | 8× Cortex-A55  | 1.6 GHz  | 2 GB   |
| `itel_a23_pro`     | Itel A23 Pro            | Unisoc SC9832E  | 4× Cortex-A53  | 1.4 GHz  | 512 MB |

### Tier 3 — KaiOS / Feature Phones (256–512 MB RAM)

These profiles test the **absolute RAM floor**. PyTorch cannot load natively on KaiOS. The test answers: *can a quantized model even fit within 256–512 MB?*

| Profile ID        | Device              | Chipset         | Cores          | Clock    | RAM    |
|-------------------|---------------------|-----------------|----------------|----------|--------|
| `nokia_8110_4g`   | Nokia 8110 4G       | Snapdragon 205  | 2× Cortex-A7   | 1.1 GHz  | 256 MB |
| `jiophone2`       | JioPhone 2          | Snapdragon 205  | 2× Cortex-A7   | 1.1 GHz  | 512 MB |
| `nokia_2720_flip` | Nokia 2720 Flip     | Snapdragon 205  | 2× Cortex-A7   | 1.1 GHz  | 512 MB |
| `itel_it5626`     | Itel it5626 (4G)    | MediaTek MT6739 | 4× Cortex-A53  | 1.3 GHz  | 512 MB |

---

## Benchmark Output

```
Model File         | Device              | Tier | RAM Limit | Latency (pred.) | Peak RAM (meas.) | Fits
-------------------|---------------------|------|-----------|-----------------|------------------|------
cocoa_fp32.pt      | Tecno Spark 8       | 1    | 2048 MB   | 847ms           | 42.3 MB          | ✓
cocoa_fp32.pt      | Nokia C1 (Go)       | 2    | 1024 MB   | 2389ms          | 42.3 MB          | ✓
cocoa_fp32.pt      | JioPhone 2 (KaiOS)  | 3    | 512 MB    | 5104ms          | 42.3 MB          | ✓
cocoa_int8.pt      | Tecno Spark 8       | 1    | 2048 MB   | 312ms           | 11.2 MB          | ✓
cocoa_int8.pt      | Nokia 8110 4G       | 3    | 256 MB    | —               | —                | ✗ OOM
```

Results are also saved to `results/benchmark_<timestamp>.csv`.

---

## Methodology & Citation

### Latency

Predicted using **nn-Meter** (Zhang et al., 2021) — a kernel-level latency predictor trained on real hardware measurements. lodevem uses the `cortexA76cpu_tflite21` predictor as a base and applies per-profile scaling factors to estimate performance on Cortex-A53, A55, and A7 cores (sourced from ARM's published performance data).

> Zhang, L., et al. *nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices*. MobiSys 2021. https://github.com/microsoft/nn-Meter

### Memory

**Lite mode:** Measured via `psutil.Process.memory_info().rss` during inference. Reports actual peak RAM and flags whether it exceeds the device profile's limit.

**Full mode (Docker):** Measured inside a cgroup v2-constrained container with `memory.max` set to the device profile's RAM. True OOM events are detected via kernel kill signal 137.

### Suggested paper methods statement

> *Hardware simulation was performed using lodevem v0.1.0 [cite], which predicts inference latency via nn-Meter [cite] and measures memory footprint under psutil/Docker RAM constraints matching each target device profile. All results are reproducible via `pip install lodevem`.*

---

## Limitations

1. **Latency is predicted**, not measured (~10–15% MAPE on supported SoCs)
2. **Scaling factors** for sub-A76 cores are based on ARM's published benchmarks, not direct measurement
3. **Thermal throttling** is not simulated
4. **Hardware accelerators** (NPU, DSP, GPU) are not modeled — CPU-only inference assumed
5. **Lite mode** cannot enforce a hard RAM cap — `fits_in_ram` is inferred, not enforced

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- No Docker required for basic use

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Contributing

Issues and pull requests are welcome at [github.com/Danchi-1/lodevem](https://github.com/Danchi-1/lodevem).

To add a new device profile, create a YAML file in `lodevem/profiles/tierN/` following the format in any existing profile. Include the chipset source (GSMArena, AnTuTu, or manufacturer spec sheet).
