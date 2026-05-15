# NATS-Bench-MCU

> **A tabular hardware benchmark for neural architecture search on microcontrollers.**
> Inference latency, flash and SRAM footprint, power draw, and energy per inference for all 15,625
> NATS-Bench TSS architectures, measured directly on a Nordic nRF5340 microcontroller.

[**Paper**](#citation) | [**Dataset (Zenodo)**](#dataset) | **License: Code MIT · Data CC BY 4.0**

---

## Overview

NATS-Bench-MCU augments the [NATS-Bench topology search space](https://github.com/D-X-Y/NATS-Bench)
with end-to-end hardware measurements on a Nordic Semiconductor nRF5340 development kit.
Every architecture is converted to an INT8 TensorFlow Lite flatbuffer, compiled into a
Zephyr/TFLite Micro firmware, flashed onto the MCU, and profiled with a Nordic Power Profiler
Kit II — no simulation, no proxy measurements.

Of the 15,625 architectures in the search space, **14,279 (91.4%) are successfully deployed
and profiled**; the remaining 8.6% are rejected at a flash-feasibility precheck because their
INT8 flatbuffer exceeds the configured 800 KB threshold.
For each deployed architecture we report: total ROM, RAM allocation, inference latency,
mean current, mean power, and energy per inference.

Key findings from the benchmark:
- **Flash, not SRAM, is the dominant deployment constraint.** The nRF5340's 1 MB flash is the
  hard ceiling; SRAM is well below its limit for all deployed architectures.
- **Mean power varies only weakly across the search space** (coefficient of variation 1.7%,
  range 10.3–12.5 mW), making energy essentially latency-driven.
- **Fewer than 0.4% of deployed architectures lie on any accuracy–cost Pareto front** for any
  dataset–cost combination, highlighting the density of dominated candidates.

---

## Quick Start — Reproduce Figures

The figures in the paper can be reproduced from the CSV exports available on Zenodo using only
`matplotlib` and `numpy`.

```bash
# 1. Clone the repository
git clone --recurse-submodules https://github.com/<your-org>/hw-nats-bench.git
cd hw-nats-bench

# 2. Install minimal dependencies
pip install matplotlib numpy

# 3. Download the export/ CSVs from Zenodo (see Dataset section below)
#    and place them in a folder called export/

# 4. Generate all figures
python plot_from_export.py --data export/ --out figures_export/
```

The script produces the following files in `figures_export/`:

| Output file | Paper figure | Description |
|---|---|---|
| `fig1_deployment_yield.pdf` | Pipeline diagram | Deployment funnel + failure breakdown |
| `fig2_metric_dists.pdf` | Figure 2 | Hardware metric distributions (latency, ROM, power, energy) |
| `fig4_metric_scatter.pdf` | Supplemental | Pairwise scatter plots of hardware metrics |
| `fig5_accuracy_pareto.pdf` | Figure 3 | Accuracy vs. cost Pareto fronts (3×3 grid) |
| `fig6_per_board_boxplots.pdf` | Supplemental | Per-board boxplots (cross-board consistency check) |
| `fig7_memory_composition.pdf` | Figure 1 | ROM and RAM composition breakdown |
| `fig8_arena_utilization.pdf` | Supplemental | Tensor arena utilization distribution |

---

## Repository Layout

```
hw-nats-bench/
│
│  Pipeline scripts (run in order for full reproduction):
├── nats2tflite.py            Step 1 — parse NATS-Bench arch strings, build Keras models,
│                                      convert to TFLite (fp32 + INT8 post-training quant)
├── evaluate_nats_models.py   Steps 2–3 — submit INT8 TFLite files to MIMaaS;
│                                         download measurement artifacts
├── export_plot_data.py       Step 4 — aggregate artifacts into publication CSVs
├── plot_from_export.py       Step 5a — reproduce all figures from CSVs (minimal deps)
├── plot_results.py           Step 5b — generate figures directly from artifacts
│
│  Utilities:
├── verify_arch.py            Structural validation of Keras models vs. reference PyTorch
├── analyze_artifacts.py      Full analysis report: coverage, failures, metric distributions
├── submit_missing.py         Resubmit architectures not yet in the artifact set
├── nats2tflite.ipynb         Notebook walkthrough of the TFLite conversion
│
│  Configuration:
├── requirements.txt
├── README.md
│
│  Submodules (cloned automatically with --recurse-submodules):
├── AutoDL-Projects/          D-X-Y/AutoDL-Projects — reference PyTorch TinyNetwork;
│                             required at runtime by verify_arch.py
└── NATS-Bench/               D-X-Y/NATS-Bench — NATS-Bench API and genotype utilities
```

**Not in this repository** (available on Zenodo or from external sources):

| Path | Size | Source |
|---|---|---|
| `artifacts/` | ~17 GB | Zenodo — raw per-architecture measurement outputs |
| `export/` | ~5 MB | Zenodo — publication CSVs derived from artifacts |
| `benchmarks/` | ~3.2 GB | Official NATS-Bench release (see Step 0 below) |
| `tflite_models/` | ~27 GB | Regenerate locally with `nats2tflite.py` |

---

## Full Pipeline

Follow these steps to reproduce the benchmark from scratch. Steps 0–1 obtain the data; Steps
2–3 require MIMaaS hardware access; Steps 4–5 produce the publication outputs.

### Prerequisites

```bash
git clone --recurse-submodules https://github.com/<your-org>/hw-nats-bench.git
cd hw-nats-bench

pip install -r requirements.txt
pip install git+https://github.com/thesebsterseb/mimaas-api
```

> **Note on submodules:** `AutoDL-Projects/` is required at runtime by `verify_arch.py`
> (imports `xautodl.models`). `NATS-Bench/` provides documentation cross-references;
> the runtime API is supplied by the pip package `nats-bench==1.8`.
> Both are cloned automatically by `--recurse-submodules`.

### Step 0 — Download NATS-Bench benchmark files

Download the TSS benchmark files from the
[official NATS-Bench release](https://github.com/D-X-Y/NATS-Bench) and place them in
`benchmarks/`:

```
benchmarks/
└── NATS-tss-v1_0-3ffb9-simple/    # 15,625 per-architecture pickle files
```

The `NATS-tss-v1_0-3ffb9-simple/` directory is produced by decompressing the primary
NATS-Bench TSS archive. Follow the instructions in the NATS-Bench repository to obtain it.

### Step 1 — Convert architectures to TFLite (`nats2tflite.py`)

```bash
python nats2tflite.py --start 0 --end 15625
```

For each architecture index `XXXXX`, the script:
1. Retrieves the architecture string from the NATS-Bench API.
2. Builds the corresponding Keras model (Stem → Cells → Head, C=16, N=5).
3. Converts to a float32 TFLite flatbuffer (reference) and an INT8 flatbuffer
   (post-training quantization with 10 random calibration samples).

Output: `tflite_models/XXXXX/nats_bench_model_XXXXX[_int8].tflite`.
The script writes `tflite_models/manifest.csv` and skips already-converted indices,
so it can be interrupted and restarted safely.

Runtime: approximately 2–3 hours for all 15,625 architectures on a modern workstation.

### Step 2 — (Optional) Validate architecture correctness (`verify_arch.py`)

Run before submitting to MIMaaS to catch any structural divergence from the reference
PyTorch models:

```bash
python verify_arch.py --samples 100 --seed 42   # random sample
python verify_arch.py --indices 0 42 1337        # specific indices
```

The script checks three invariants against the reference `TinyNetwork` built by
`xautodl.models.get_cell_based_tiny_net`: (1) trainable parameter count, (2) leaf
operation count, (3) multiset of `(op_type, output_shape)` pairs. See the
[Technical Reference](#technical-reference) section for details.

### Step 3 — Submit to MIMaaS and download results (`evaluate_nats_models.py`)

```bash
python evaluate_nats_models.py register    # once — create a MIMaaS account
python evaluate_nats_models.py submit      # submit INT8 TFLite files (resume-safe)
python evaluate_nats_models.py download    # fetch artifacts when jobs complete (resume-safe)
```

MIMaaS ([Zimmermann et al., 2026](#citation)) orchestrates firmware compilation, flashing,
and power measurement on the nRF5340-DK. Access requires a running MIMaaS server configured
for the `nrf5340dk` target. See the
[MIMaaS repository](https://github.com/thesebsterseb/mimaas-server) for setup.

Both `submit` and `download` are resume-safe: already-submitted models are skipped and
already-downloaded artifact directories are not re-fetched.

Artifacts are saved to `artifacts/XXXXX/` with the following structure:

```
artifacts/XXXXX/
├── results.json         inference latency (s), mean current (µA), mean power (mW),
│                        energy per inference (mJ)
├── ppk2_summary.csv     per-inference summary from the Power Profiler Kit II
├── ppk2_samples.zip     raw PPK2 current samples at 100 kHz
├── uart.log             TFLite Micro runtime log (operators, arena usage)
├── rom.json             Zephyr memory report — ROM layout by category
├── ram.json             Zephyr memory report — RAM layout by category
├── flash.log            firmware build log
├── model.cpp            generated C model file
└── meta.json            request metadata (board ID, timestamps)
```

### Step 4 — Export publication CSVs (`export_plot_data.py`)

```bash
python export_plot_data.py --source local --out export/
```

Produces five files in `export/`:

| File | Rows | Key columns |
|---|---|---|
| `hw_metrics.csv` | 14,279 | `arch_idx`, `arch_str`, `int8_kb`, `rom_kb`, `ram_kb`, `latency_s`, `current_uA`, `power_mW`, `energy_mJ`, `board`, `arena_used_kb`, `arena_size_kb`, `operators` |
| `accuracies.csv` | 14,279 | `arch_idx`, `cifar10`, `cifar100`, `ImageNet16-120` |
| `failures.csv` | 1,346 | `arch_idx`, `arch_str`, `outcome`, `int8_kb` |
| `coverage.csv` | 6 | Pipeline funnel counts (total → converted → submitted → done / failed) |
| `README.md` | — | Column descriptions and figure→column mapping |

These CSVs are also available directly from Zenodo (see [Dataset](#dataset)).

### Step 5 — Generate figures

```bash
# Lightweight: from CSVs only — no artifacts, no server access required
python plot_from_export.py --data export/ --out figures_export/

# Full: directly from the local artifact tree
python plot_results.py --source local
```

`plot_from_export.py` requires only `matplotlib` and `numpy`. Pass `--no-png` to skip
rasterised copies.

---

## Dataset

> **Dataset available at Zenodo:** [link to be added on publication]

The Zenodo deposit contains:

- **`export/`** — the five publication CSV files (~5 MB). These alone are sufficient to
  reproduce all figures with `plot_from_export.py`.
- **`artifacts/`** — the complete per-architecture measurement tree (~17 GB), including
  raw PPK2 current traces, Zephyr memory reports, and UART logs.

All on-device metrics in the dataset are derived from physical measurements on real
nRF5340-DK hardware using a Nordic Power Profiler Kit II in source-meter mode.
No values are simulated or estimated.

---

## Technical Reference

<details>
<summary>Expand: NATS-Bench TSS architecture, Keras port, TFLite conversion, and verification details</summary>

### 1. The NATS-Bench TSS architecture

NATS-Bench TSS (Dong et al., 2021) defines a **fixed ResNet-style macro skeleton**; the
15,625 architectures differ only in the micro structure of a **single cell template** that
is repeated throughout the network.

Every structural claim in this section has a direct counterpart in the upstream reference code:

| Component | Reference file (PyTorch, upstream) | Our port (Keras) |
|---|---|---|
| Full network (macro skeleton) | [`AutoDL-Projects/xautodl/models/cell_infers/tiny_network.py`](AutoDL-Projects/xautodl/models/cell_infers/tiny_network.py) — class `TinyNetwork` | [`nats2tflite.py`](nats2tflite.py) — `build_model` |
| Normal cell | [`AutoDL-Projects/xautodl/models/cell_infers/cells.py`](AutoDL-Projects/xautodl/models/cell_infers/cells.py) — class `InferCell` | [`nats2tflite.py`](nats2tflite.py) — `build_infer_cell` |
| Reduction cell | [`AutoDL-Projects/xautodl/models/cell_operations.py`](AutoDL-Projects/xautodl/models/cell_operations.py#L216) — class `ResNetBasicblock` (line 216) | [`nats2tflite.py`](nats2tflite.py) — `build_resnet_basicblock` |
| Conv ops (`ReLUConvBN`) | [`cell_operations.py:115`](AutoDL-Projects/xautodl/models/cell_operations.py#L115) | `op_conv1x1`, `op_conv3x3` |
| `avg_pool_3x3` (`POOLING`) | [`cell_operations.py:262`](AutoDL-Projects/xautodl/models/cell_operations.py#L262) | `op_avgpool` |
| `skip_connect` (`Identity`) | [`cell_operations.py:288`](AutoDL-Projects/xautodl/models/cell_operations.py#L288) | `op_skip` |
| `none` (`Zero`) | [`cell_operations.py:296`](AutoDL-Projects/xautodl/models/cell_operations.py#L296) | `op_none` |
| TSS op registry | [`cell_operations.py:15`](AutoDL-Projects/xautodl/models/cell_operations.py#L15) — `OPS` dict, `NAS_BENCH_201` list | `OPS` dict |
| Architecture string parser | [`NATS-Bench/nats_bench/genotype_utils.py`](NATS-Bench/nats_bench/genotype_utils.py) — `TopologyStructure` | `parse_arch` |
| Default hyperparameters `C=16, N=5` | [`AutoDL-Projects/exps/NATS-algos/search-cell.py`](AutoDL-Projects/exps/NATS-algos/search-cell.py) (lines 783, 786) | `build_model(C=16, N=5)` defaults |

#### Macro skeleton

Defined in [`tiny_network.py:16-38`](AutoDL-Projects/xautodl/models/cell_infers/tiny_network.py#L16-L38)
and [`tiny_network.py:53-63`](AutoDL-Projects/xautodl/models/cell_infers/tiny_network.py#L53-L63)
(`TinyNetwork.__init__` and `TinyNetwork.forward`):

```
Input (3×32×32)
  │
Stem: Conv 3×3, C=16, no bias + BatchNorm    (no ReLU)      ← tiny_network.py:16-18
  │
Stage 1: N normal cells @ 16 channels, 32×32                ← cell_infers/cells.py, InferCell
  │
Reduction cell (ResNet basic block, stride 2, 16→32)        ← cell_operations.py:216, ResNetBasicblock
  │
Stage 2: N normal cells @ 32 channels, 16×16
  │
Reduction cell (ResNet basic block, stride 2, 32→64)
  │
Stage 3: N normal cells @ 64 channels, 8×8
  │
BatchNorm + ReLU (lastact)                                  ← tiny_network.py:36
  │
Global Average Pooling                                      ← tiny_network.py:37
  │
Linear (10 classes, no softmax — raw logits)                ← tiny_network.py:38
```

The cell schedule is defined by two lists at
[`tiny_network.py:20-21`](AutoDL-Projects/xautodl/models/cell_infers/tiny_network.py#L20-L21):

```python
layer_channels   = [C]*N + [C*2] + [C*2]*N + [C*4] + [C*4]*N
layer_reductions = [False]*N + [True] + [False]*N + [True] + [False]*N
```

Default hyperparameters: `C = 16` (initial channels), `N = 5` (cells per stage).
This gives **17 cells total** (3·N normal + 2 reduction).

#### Reduction cell (`ResNetBasicblock`)

Defined at [`cell_operations.py:216-257`](AutoDL-Projects/xautodl/models/cell_operations.py#L216-L257).
The shortcut path at [`cell_operations.py:226-232`](AutoDL-Projects/xautodl/models/cell_operations.py#L226-L232)
uses `AvgPool(2, stride=2) → Conv 1×1` and has **no BatchNorm**.

```
     x ─┬──────────────────────────────────────────────┐
        │                                              │
        ├─ ReLU → Conv 3×3 stride 2 → BN ──┐           │
        │                                  │           │
        └─ ReLU → Conv 3×3 stride 1 → BN ──┤           │
                                           │           │
                                           ▼           │
                                          Add ◄── AvgPool 2×2 stride 2
                                                       │
                                                       └──── Conv 1×1 (no BN)
```

#### Normal cell (`InferCell`)

Defined at [`cell_infers/cells.py:13-69`](AutoDL-Projects/xautodl/models/cell_infers/cells.py#L13-L69).
A directed acyclic graph over **4 nodes** (node 0 is the cell input). Each non-input node is
the **sum of its incoming edges** ([`cells.py:64`](AutoDL-Projects/xautodl/models/cell_infers/cells.py#L64)).
**The cell output is `nodes[-1]` — no concat, no projection.**

#### Operations

| op name | implementation | upstream reference |
|---|---|---|
| `none` | Zero tensor (`x * 0`) | [`cell_operations.py:296-318`](AutoDL-Projects/xautodl/models/cell_operations.py#L296-L318) class `Zero` |
| `skip_connect` | Identity (`return x`) | [`cell_operations.py:288-293`](AutoDL-Projects/xautodl/models/cell_operations.py#L288-L293) class `Identity` |
| `nor_conv_1x1` | `ReLU → Conv 1×1 (no bias) → BN` (pre-activation) | [`cell_operations.py:115-145`](AutoDL-Projects/xautodl/models/cell_operations.py#L115-L145) class `ReLUConvBN` |
| `nor_conv_3x3` | `ReLU → Conv 3×3 (no bias) → BN` (pre-activation) | same `ReLUConvBN` |
| `avg_pool_3x3` | `AvgPool 3×3, stride 1, padding 1, count_include_pad=False` | [`cell_operations.py:262-286`](AutoDL-Projects/xautodl/models/cell_operations.py#L262-L286) class `POOLING` |

#### Architecture string format

Each architecture is a string such as
`|nor_conv_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_3x3~0|avg_pool_3x3~1|none~2|`:

- `+` separates nodes (nodes 1, 2, 3)
- each `|op~src|` applies `op` to the output of node `src`
- every non-input node has one edge from each previous node

Retrieval: `api.arch(index)` returns this string; `api.get_net_config(index, "cifar10")`
returns the dict passed to `get_cell_based_tiny_net`.

---

### 2. How the TFLite models are built

[`nats2tflite.py`](nats2tflite.py) is a direct Keras port of the reference PyTorch
`TinyNetwork`. The mapping table in §1 lets you cross-read the Keras code against the
PyTorch original function-by-function.

#### TFLite conversion

Each model is converted twice:

- **Float32**: plain `tf.lite.TFLiteConverter.from_keras_model(model).convert()`.
- **INT8 (post-training quantization)**:

  ```python
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = _representative_dataset  # 10 random samples
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type  = tf.uint8
  converter.inference_output_type = tf.uint8
  ```

  Calibration data quality is **irrelevant** for this benchmark: we measure hardware
  behaviour (ROM, RAM, latency, power), not prediction accuracy. The representative dataset
  exists only because the INT8 converter requires something to trace activation shapes.

---

### 3. MIMaaS submission and download

`evaluate_nats_models.py` wraps the MIMaaS Python client:

```bash
python evaluate_nats_models.py register    # create account
python evaluate_nats_models.py submit      # upload INT8 TFLite files; resume-safe
python evaluate_nats_models.py results     # print raw server results
python evaluate_nats_models.py download    # fetch + extract artifacts; resume-safe
```

**`submit`** reads `submitted_requests.csv`, skips already-submitted models (matched by
file path), and appends new entries. Pass `--no-quantized` to submit float32 models instead.

**`download`** checks that each extracted artifact folder contains all required output files
(`flash.log`, `meta.json`, `model.cpp`, `ppk2_samples.csv`, `ppk2_summary.csv`, `ram.json`,
`results.json`, `rom.json`, `uart.log`, `<model>.tflite`). Incomplete directories are
re-downloaded; complete ones are skipped.

---

### 4. Verifying architectures match the reference (`verify_arch.py`)

`verify_arch.py` compares each generated Keras model against the reference PyTorch model
built by `xautodl.models.get_cell_based_tiny_net(api.get_net_config(index, "cifar10"))`.
It checks **three independent invariants**:

1. **Trainable parameter count** — must match exactly.
2. **Leaf operation count** — number of atomic operations (`Conv`, `BN`, `ReLU`, `AvgPool`,
   `GAP`, `Dense`, `Zero`) must match after filtering framework no-ops.
3. **Multiset of `(op_type, output_shape)` pairs** — for every leaf op, the pair is collected.
   The two multisets must be equal.

The multiset check is used (rather than a sequence check) because PyTorch hook order and
Keras layer order can differ for **parallel sibling edges** inside a DAG node — they sum to
the same result but reach different forward-pass orderings. The multiset is insensitive to
this while still verifying that every op with every shape appears with the same count.

```bash
python verify_arch.py                         # 10 random architectures (default)
python verify_arch.py --samples 100 --seed 42
python verify_arch.py --indices 0 42 1337
python verify_arch.py --indices 42 --show-full  # dump full (op, shape) multiset
```

Run this **before every large regeneration** of the model set.

---

### 5. Quantization and hardware framing

All hardware numbers are for **INT8 post-training quantized** architectures. This is
intentional: MCU deployment is always quantized in practice (float32 is impractical in
both arena size and latency on Cortex-M class cores), and the nRF5340's Cortex-M33 has
DSP extensions tuned for INT8.

When citing results from this benchmark, describe the network set as:
*"NATS-Bench TSS architectures (Dong et al., 2021) as defined by the reference
`TinyNetwork` macro-skeleton, converted to TFLite via Keras and quantized to INT8 for
MCU deployment."* Float32 accuracy for each architecture is provided by NATS-Bench and
can be used as the accuracy axis in accuracy-vs-hardware Pareto curves.

</details>

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
@article{zimmermann2026elevating,
  title  = {Elevating MCU Benchmarking: MIMaaS},
  author = {Zimmermann, Sebastian and others},
  year   = {2026}
}
```

---

## License

**Code** (all `.py` and `.ipynb` files in this repository): [MIT License](LICENSE)

**Dataset** (`export/` and `artifacts/` on Zenodo): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
