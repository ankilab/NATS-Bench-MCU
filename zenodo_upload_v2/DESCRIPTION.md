# NATS-Bench-MCU: Hardware Measurements for NATS-Bench TSS on Nordic nRF5340

This dataset accompanies the paper *"NATS-Bench-MCU: A Tabular Hardware Benchmark for Neural Architecture Search on Microcontrollers"* (AutoML 2026). It provides direct, physically measured hardware metrics for all **15,625 architectures** in the NATS-Bench topology search space (TSS), deployed on a **Nordic Semiconductor nRF5340** microcontroller. No values are simulated or estimated. For each deployable architecture, the dataset reports flash and SRAM footprint, inference latency, mean current draw, mean power, and energy per inference, measured with a **Nordic Power Profiler Kit II** at a 100 kHz sampling rate.

Of the 15,625 architectures, **14,281 (91.4 %)** were successfully deployed and profiled; **1,344 (8.6 %)** are **flash-infeasible** — their INT8 flatbuffer plus the fixed 277.6 KB firmware overhead would exceed the 1,024 KB (1 MB) application-core flash, equivalently a flatbuffer above the 746 KB deployable budget — so no on-device metrics can be produced for them.

## What's new in version 2 (camera-ready corrections)

- **Outcome taxonomy corrected.** All deployment failures now carry a single, well-defined **`infeasible`** label (flash-infeasibility, as defined above). This resolves the failure-size inconsistencies raised in peer review.
- **Counts updated to 14,281 successful / 1,344 infeasible** (previously 14,279 / 1,346), after recovering the measurement for architecture **07650** (left `pending` by a queuing bug and later spuriously rejected) and remeasuring architecture **07651** (which previously died mid-flash).
- **New `corrections.tar.gz` overlay** documents and applies every post-hoc fix, with full per-architecture provenance. The 16 raw `artifacts_*.tar.gz` archives are **unchanged from version 1**.
- **Export CSVs regenerated** with the corrected taxonomy, and extended with the INT8-vs-FP32 ranking study and the hardware-aware NAS demonstration reported in the camera-ready paper.

## About the benchmark

Hardware-aware neural architecture search depends on benchmarks that expose not only predictive performance, but also the deployment costs of candidate architectures on the target device. Existing tabular NAS benchmarks have enabled reproducible algorithm development at low computational cost, and hardware-aware extensions have added latency and energy measurements for several edge-class platforms. However, microcontroller-class devices remain largely absent from these resources, despite being among the most constrained and practically relevant deployment regimes for Edge AI.

NATS-Bench-MCU augments the NATS-Bench topology search space with end-to-end measurements on a Nordic nRF5340. In contrast to simulation- or proxy-based hardware costs, all reported on-device metrics are obtained from real deployments through a reproducible firmware, quantization, and measurement pipeline.

## Files

### `export/` — Publication CSVs (6.9 MB total)

The primary entry point for most users. Contains all data needed to reproduce the paper's figures and tables. No decompression or special tooling required — plain CSV files readable by any spreadsheet application or by pandas. **These CSVs already reflect every correction in version 2.**

- **`hw_metrics.csv`** (6.1 MB, 14,281 rows) — Core hardware measurements for every successfully deployed architecture: architecture index, cell topology string, INT8 model size, total ROM and RAM usage, inference latency, mean current, mean power, energy per inference, board ID, tensor-arena usage, operator count, and Zephyr memory-report breakdowns by category.
- **`accuracies.csv`** (668 KB, 14,281 rows) — NATS-Bench test accuracies (CIFAR-10, CIFAR-100, ImageNet-16-120) for each successfully deployed architecture, sourced from the NATS-Bench API under the 200-epoch training protocol.
- **`failures.csv`** (56 KB, 1,344 rows) — One row per flash-infeasible architecture: index, outcome label (`infeasible`), INT8 model size, the size reported by the flash-feasibility check, and the board.
- **`coverage.csv`** (5 rows) — Pipeline funnel: how many architectures reached each stage (converted → submitted → successful / infeasible).
- **`int8_rank_correlation.csv`** (100 rows) — Per-architecture FP32 vs. INT8 test accuracy for the quantization-ranking study (backs the paper's quantization analysis).
- **`int8_rank_summary.csv`**, **`nas_baselines_summary.csv`** — Summary statistics for the quantization-ranking study and the hardware-aware NAS demonstration reported in the paper.
- **`README.md`** — Column-by-column documentation for the CSVs, including units, measurement method, and a figure-to-column mapping.

### `artifacts_XXXXX-YYYYY.tar.gz` — Raw measurement archives (≈ 2.3 GB each, ≈ 34 GB total)

16 chunked `tar.gz` archives containing the raw per-architecture output directories for all 15,625 architectures (1,000 per archive, except the last, which holds 625). Each successful architecture directory holds: `results.json` (aggregated metrics), `ppk2_summary.csv` (per-inference power summary), `ppk2_samples.parquet` (raw PPK2 current trace at 100 kHz), `uart.log` (TFLite Micro runtime log with operator count and tensor-arena usage), `rom.json.gz` / `ram.json.gz` (Zephyr memory reports), `flash.log.gz` (firmware build log), `model.cpp.gz` (generated C model array), and the INT8 `.tflite` flatbuffer. Flash-infeasible architectures contain `meta.json`, `error.log`, and the `.tflite` file.

These archives are the **unchanged version-1 campaign measurements**. The 117 corrected records are provided separately in `corrections.tar.gz` and can be merged in with a single command (see *How to use this version*).

You do **not** need to download all 16 archives — use `index.json` and `dataset_example.py` to access a single architecture by downloading only its archive (≈ 2.3 GB).

### `corrections.tar.gz` — Corrected-record overlay (new in v2, 2 MB)

An authoritative overlay on `artifacts/`: the recovered measurement for architecture 07650, the remeasurement for 07651, and regenerated `error.log` / `meta.json` for the 115 mid-band (747–791 KB) failures so the raw artifacts agree with the single `infeasible` taxonomy. **No measured hardware value is altered.** Includes `apply_corrections.py` (one-command merge) and a `README.md` documenting the provenance of every corrected record.

### `index.json` (588 KB) and `dataset_example.py`

`index.json` maps every architecture index (0–15624) to the archive filename that contains it, enabling on-demand single-archive access. `dataset_example.py` is a minimal Python example (Python 3.8+, standard library only, with optional `pyarrow` for the parquet traces).

## How to use this version

**Most users — the tabular data.** Download and extract `export.tar.gz`; the CSVs already reflect every correction, so there is nothing to merge. Start from `export/README.md`.

**Reproduce the paper's figures** (export CSVs only, no archives needed):

```bash
git clone https://github.com/ankilab/NATS-Bench-MCU
pip install matplotlib numpy
python plot_from_export.py --data export/ --out figures/
```

**Access a single architecture's raw data** (no need to decompress any archive; place the files from Zenodo in the repo's `dataset/` folder):

```bash
python dataset_example.py print 42
python dataset_example.py export 42 --out ./evaluation_42/
```

**Build the fully corrected raw dataset.** Extract the `artifacts_*.tar.gz` archives and `corrections.tar.gz` side by side, then merge the overlay into one corrected `artifacts/` tree:

```bash
python corrections/apply_corrections.py            # merges the overlay into artifacts/
python corrections/apply_corrections.py --dry-run  # preview the changes first
```

The merge is idempotent and preserves each architecture's original INT8 model file. Equivalently, the analysis tooling applies the overlay on the fly, so you never have to merge manually to reproduce the paper.

## Links

- **Code:** https://github.com/ankilab/mimaas-api
- **Dataset scripts & examples:** https://github.com/ankilab/NATS-Bench-MCU — see the `dataset/` folder for example scripts to access single evaluations and in-depth documentation.

**License:** CC BY 4.0
