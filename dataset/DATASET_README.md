# NATS-Bench-MCU Dataset

> **Hardware measurements for all 15,625 NATS-Bench topology-search-space architectures,
> deployed and profiled on a Nordic nRF5340 microcontroller.**

**DOI:** https://doi.org/10.5281/zenodo.20204556  
**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
**Paper:** NATS-Bench-MCU: A Tabular Hardware Benchmark for Neural Architecture Search on Microcontrollers (AutoML 2026)  
**Code:** https://github.com/ankilab/NATS-Bench-MCU

**DOWNLOAD OF THE DATASET**: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20204556.svg)](https://doi.org/10.5281/zenodo.20204556)

---

## Overview

NATS-Bench-MCU augments the [NATS-Bench topology search space](https://github.com/D-X-Y/NATS-Bench)
with end-to-end hardware measurements on a Nordic Semiconductor nRF5340 development kit.
Every architecture is converted to an INT8 TensorFlow Lite flatbuffer, compiled into a
Zephyr/TFLite Micro firmware, flashed onto the MCU, and profiled with a Nordic Power Profiler
Kit II (PPK2) — no simulation, no proxy measurements.

| Scope | Value |
|---|---|
| Search space | NATS-Bench topology search space (15,625 architectures) |
| Device | Nordic Semiconductor nRF5340 (Arm Cortex-M33, 128 MHz, 1 MB flash, 512 KB SRAM) |
| Deployment | Zephyr RTOS + TFLite Micro, INT8 post-training quantization |
| Power instrument | Nordic Power Profiler Kit II (PPK2) at 100 kHz sampling |
| Successfully deployed | 14,279 (91.4 %) |
| Failed (flash overflow) | 1,346 (8.6 %) |
| Metrics | ROM, RAM, inference latency, mean current, mean power, energy per inference |

---

## Dataset Structure

```
zenodo-deposit/
│
├── DATASET_README.md                  This file
├── dataset_example.py                 Minimal Python example — see Quick Start below
├── requirements_dataset.txt           Python dependencies for dataset_example.py
│
├── index.json                         Maps each arch_idx to its archive filename
├── artifacts_00000-00999.tar.gz       Raw artifacts for architectures 0–999
├── artifacts_01000-01999.tar.gz       Raw artifacts for architectures 1000–1999
├── ...                                (16 archives total, ~1,000 architectures each)
└── artifacts_15000-15624.tar.gz       Raw artifacts for architectures 15000–15624
│
└── export/                            Publication CSVs (~5 MB) — sufficient for all paper figures
    ├── hw_metrics.csv                 One row per successfully deployed architecture
    ├── accuracies.csv                 NATS-Bench test accuracies for deployed architectures
    ├── failures.csv                   One row per failed architecture
    ├── coverage.csv                   Pipeline funnel counts
    └── README.md                      Column-by-column documentation and figure→data mapping
```

Each `.tar.gz` archive contains a set of per-architecture subdirectories named with the
5-digit zero-padded architecture index (e.g. `00042/`). The `index.json` file maps every
architecture index to the archive that contains it, so `dataset_example.py` can open only
the relevant archive without downloading the full dataset.

---

## Quick Start

**Requirements:** Python 3.8+. No extra packages for `print`; `pyarrow` for `export`.

```bash
pip install pyarrow   # only needed for the export command

# Print a summary of architecture 42 to stdout
# (reads only artifacts_00000-00999.tar.gz — no full download needed)
python dataset_example.py print 42

# Export architecture 42 as a folder of fully decompressed, directly readable files
python dataset_example.py export 42 --out ./evaluation_42/

# If the dataset is not in the same folder as the script, pass its location explicitly
python dataset_example.py print 42 --dataset /path/to/zenodo_download/
```

The script reads `index.json` to find the right archive, then extracts only the single
architecture directory from that archive — it never loads the full dataset into memory.

The `export` command produces a folder containing:

| File | Description |
|---|---|
| `meta.json` | Request metadata (board, timestamp, status) |
| `results.json` | Aggregated hardware metrics |
| `ppk2_summary.csv` | Per-inference power summary (decompressed from parquet) |
| `ppk2_samples.csv` | Raw PPK2 current trace at 100 kHz (converted from parquet) |
| `uart.log` | TFLite Micro runtime log (operators, arena usage) |
| `flash.log` | Firmware build log (decompressed from .gz) |
| `model.cpp` | Generated C model array (decompressed from .gz) |
| `rom.json` | Zephyr ROM memory report by category (decompressed from .gz) |
| `ram.json` | Zephyr RAM memory report by category (decompressed from .gz) |
| `nats_bench_model_XXXXX_int8.tflite` | INT8 TFLite flatbuffer |

To reproduce all paper figures from the `export/` CSVs alone (no artifacts needed):

```bash
# Clone the analysis code
git clone https://github.com/ankilab/NATS-Bench-MCU.git
cd NATS-Bench-MCU
pip install matplotlib numpy

# Place the export/ folder from this dataset next to the script, then:
python plot_from_export.py --data export/ --out figures/
```

---

## `export/` — Publication CSVs

These five files contain all data needed to reproduce the paper's figures and tables.

### `hw_metrics.csv` — Hardware measurements (14,279 rows)

One row per successfully deployed architecture.

| Column | Unit | Description |
|---|---|---|
| `arch_idx` | — | NATS-Bench TSS index (0-based integer, 0–15624) |
| `arch_str` | — | Cell topology string, e.g. `\|nor_conv_3x3~0\|+\|skip_connect~0\|avg_pool_3x3~1\|+\|...` |
| `int8_kb` | KB | INT8 TFLite flatbuffer file size |
| `rom_kb` | KB | Total on-device ROM (flash) usage |
| `ram_kb` | KB | Total on-device RAM allocation (constant 366.3 KB across all architectures) |
| `latency_s` | s | Mean inference latency (mean of 10 calls) |
| `current_uA` | µA | Mean current draw during inference |
| `power_mW` | mW | Mean power during inference |
| `energy_mJ` | mJ | Energy per inference |
| `board` | — | Board unit: `nrf5340dk_1`, `nrf5340dk_2`, or `nrf5340dk_3` |
| `g_model_kb` | KB | ROM occupied by `g_model` symbol (model weights only) |
| `arena_used_kb` | KB | Runtime tensor arena usage (maximum across 10 calls) |
| `arena_size_kb` | KB | Statically allocated tensor arena (constant 350 KB) |
| `arena_used_pct` | % | `arena_used_kb / arena_size_kb × 100` |
| `operators` | — | Number of TFLite Micro operators in the model |
| `rom_categories_json` | — | JSON dict: ROM [KB] per Zephyr memory-report category |
| `ram_categories_json` | — | JSON dict: RAM [KB] per Zephyr memory-report category |

### `accuracies.csv` — NATS-Bench test accuracies (14,279 rows)

| Column | Unit | Description |
|---|---|---|
| `arch_idx` | — | NATS-Bench TSS index (matches `hw_metrics.csv`) |
| `cifar10` | % | Test accuracy on CIFAR-10 (200-epoch regime) |
| `cifar100` | % | Test accuracy on CIFAR-100 (200-epoch regime) |
| `ImageNet16-120` | % | Test accuracy on ImageNet-16-120 (200-epoch regime) |

Accuracies are sourced from the NATS-Bench API (not re-trained). All values reflect the
H1 (200-epoch) training protocol as defined in Dong et al. (2021).

### `failures.csv` — Failed architectures (1,346 rows)

| Column | Unit | Description |
|---|---|---|
| `arch_idx` | — | NATS-Bench TSS index |
| `stage` | — | Failure category: `precheck` (flash overflow before deployment) |
| `int8_kb` | KB | INT8 TFLite flatbuffer size |
| `overflow_kb` | KB | Size reported in precheck error message (NaN if unavailable) |
| `board` | — | Board that processed the request |

All 1,346 failures occur at the `precheck` stage: the INT8 flatbuffer exceeds the
configured 800 KB threshold, meaning the compiled firmware would not fit in the nRF5340's
1 MB flash. No on-device measurements are available for these architectures.

### `coverage.csv` — Pipeline funnel (5 rows)

| Column | Description |
|---|---|
| `stage` | Pipeline stage label |
| `count` | Number of architectures reaching this stage |
| `pct_of_total` | Percentage of the full 15,625-architecture search space |

### Figure → data mapping

| Paper figure | File(s) | Key columns |
|---|---|---|
| Fig. 1 — Memory composition | hw_metrics.csv | `rom_categories_json`, `ram_categories_json` |
| Fig. 2 — Pipeline coverage | coverage.csv, failures.csv | `count`, `stage` |
| Fig. 3 — Hardware metric distributions | hw_metrics.csv | `rom_kb`, `latency_s`, `power_mW`, `energy_mJ`, `int8_kb` |
| Fig. 4 — Accuracy vs. cost Pareto fronts | hw_metrics.csv + accuracies.csv | `latency_s`, `energy_mJ`, `rom_kb`, `cifar10/100/ImageNet` |

---

## `artifacts/` — Raw Measurement Data

Each subdirectory `artifacts/XXXXX/` (zero-padded 5-digit index) contains the raw outputs
for one architecture evaluation.

### Successful evaluation (14,279 architectures)

| File | Compressed | Description |
|---|---|---|
| `meta.json` | no | Request metadata: board, timestamp, request ID, status |
| `results.json` | no | Aggregated metrics: ROM, RAM, latency, current, power, energy |
| `ppk2_summary.csv` | no | Per-inference summary from PPK2 |
| `ppk2_samples.parquet` | (parquet) | Raw PPK2 current trace at 100 kHz |
| `uart.log` | no | TFLite Micro runtime log: operator count, tensor arena usage |
| `flash.log.gz` | gzip | Firmware build log |
| `model.cpp.gz` | gzip | Generated C array embedding the INT8 model |
| `rom.json.gz` | gzip | Zephyr ROM memory report (symbol tree) |
| `ram.json.gz` | gzip | Zephyr RAM memory report (symbol tree) |
| `nats_bench_model_XXXXX_int8.tflite` | no | INT8 TFLite flatbuffer |

The `.gz` files and `.parquet` file are decompressed automatically by `dataset_example.py export`.

### Failed evaluation (1,346 architectures)

| File | Description |
|---|---|
| `meta.json` | Request metadata (status will be `"error"`) |
| `error.log` | Error details: failure stage and message |
| `nats_bench_model_XXXXX_int8.tflite` | INT8 TFLite flatbuffer (model was too large to deploy) |

---

## Metrics and Units

All on-device metrics are physically measured — no values are simulated or estimated.

| Metric | Unit | Source | Range (deployed archs) |
|---|---|---|---|
| ROM (flash) usage | KB | Zephyr memory report, `rom.json` | 363–1,022 KB |
| RAM allocation | KB | Zephyr memory report, `ram.json` | 366.3 KB (constant) |
| INT8 model size | KB | `.tflite` file size | 2–744 KB (deployed); 360–1,457 KB (failed) |
| Inference latency | s | GPIO timestamps from PPK2 trace, mean of 10 calls | 0.33–6.69 s |
| Mean current | µA | PPK2 current during GPIO-high interval, mean of 10 calls | — |
| Mean power | mW | Mean current × 3.3 V supply voltage | 10.3–12.5 mW |
| Energy per inference | mJ | Time integral of PPK2 current × 3.3 V, mean of 10 calls | 3.8–79.4 mJ |
| Tensor arena used | KB | TFLite Micro runtime report via UART, max of 10 calls | 0–145 KB |

**Note on RAM:** All architectures share the same static RAM allocation because the firmware
pre-allocates the maximum permitted tensor arena (350 KB) at compile time, plus a fixed
OS/application overhead of ~16 KB. The arena's *runtime usage* (how much of the 350 KB is
actually needed) is separately reported in `arena_used_kb`.

---

## Hardware and Firmware

**Device:** Nordic Semiconductor nRF5340 Development Kit  
**Core:** Arm Cortex-M33 application core, 128 MHz, 1 MB flash, 512 KB SRAM  
**Current measurement:** Nordic PPK2 connected to the nRF5340-DK dedicated current-measurement
pins in source-meter mode (100 kHz sampling, ±10% accuracy in the 500 µA–5 mA range)  
**Supply voltage:** 3.3 V nominal  
**Firmware stack:** Zephyr RTOS + TensorFlow Lite Micro (TFLM)  
**Tensor arena:** 350 KB statically allocated  
**Inference protocol:** 10 consecutive inference calls per architecture; reported values are
the arithmetic mean (for latency, current, power, energy) or the maximum (for arena usage)  
**Parallelism:** Three physically identical nRF5340-DK units (nrf5340dk_1/2/3) operated in
parallel; each architecture is measured on one board. Cross-board variability is within 2%
on all metrics (verified in the paper, Section 3.2).

---

## How the Data Was Collected

The full pipeline is described in the paper (Section 3) and in the [GitHub repository](https://github.com/ankilab/NATS-Bench-MCU).

In brief:

1. **Architecture → Keras → TFLite (INT8):** Each NATS-Bench architecture string is
   converted to a Keras model and quantized to INT8 via post-training quantization.
2. **Flash precheck:** Flatbuffers exceeding 800 KB are rejected (1,346 architectures).
3. **MIMaaS deployment:** Accepted flatbuffers are submitted to [MIMaaS](https://github.com/ankilab/mimaas-server),
   which compiles the Zephyr firmware, flashes it to the nRF5340-DK, runs inference, and
   returns the raw PPK2 trace and Zephyr memory reports.
4. **Artifact download:** Raw outputs are saved per-architecture in `artifacts/XXXXX/`.
5. **CSV export:** `export_plot_data.py` aggregates artifacts into the `export/` CSVs.

The full 15,625-architecture campaign took approximately 4 days of wall-clock time on three
boards running in parallel.

---

## Citation

If you use NATS-Bench-MCU in your research, please cite:

```bibtex
@inproceedings{zimmermann2026natsbenchmcu,
  title     = {{NATS-Bench-MCU}: A Tabular Hardware Benchmark for Neural Architecture
               Search on Microcontrollers},
  author    = {Zimmermann, Sebastian and Groh, Ren\'{e} and Kist, Andreas M.},
  booktitle = {AutoML Conference},
  year      = {2026}
}
```

The MIMaaS benchmarking infrastructure used to collect the measurements:

```bibtex
@inproceedings{zimmermann2026mimaas,
  title     = {Elevating {AI} on the Edge: A Demonstration of {MIMaaS}
               (Machine Intelligence with Microcontroller-as-a-Service)},
  author    = {Zimmermann, Sebastian and Groh, Ren\'{e} and Kist, Andreas M.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {48},
  pages     = {41751--41753},
  year      = {2026}
}
```

---

## License

**Dataset** (`export/` and `artifacts/`): [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

You are free to share and adapt this data for any purpose, provided appropriate credit is given.

**Code** (`dataset_example.py` and all scripts in the GitHub repository): MIT License
