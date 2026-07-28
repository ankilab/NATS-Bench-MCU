# NATS-Bench Hardware Evaluation — Plot Data Export

Generated from 15,625 nRF5340-DK deployments (14,281 successful, 1,344 failed).

## Files

### coverage.csv
Pipeline funnel counts. One row per stage.
| column | description |
|--------|-------------|
| stage  | Pipeline stage label |
| count  | Number of architectures reaching this stage |
| pct_of_total | Percentage of the full 15,625-arch search space |

**Used by:** fig1 (left panel — pipeline funnel bar chart)

### failures.csv
One row per failed architecture.
| column | description |
|--------|-------------|
| arch_idx | NATS-Bench TSS index (0-based, zero-padded to 5 digits) |
| stage | Outcome label. All rejections are flash-infeasible and share the single `infeasible` label (model + 277.6 KB firmware would exceed the 1024 KB flash). |
| int8_kb | INT8 TFLite model file size [KB] from manifest |
| overflow_kb | Model size reported in precheck error message [KB] (NaN if not available) |
| board | Board that ran the request |

**Used by:** fig1 (right panel — failure breakdown), fig3 (failed arch sizes vs threshold)

### hw_metrics.csv
One row per successfully deployed architecture. Core hardware measurements.
| column | description |
|--------|-------------|
| arch_idx | NATS-Bench TSS index (0-based) |
| arch_str | Cell topology string, e.g. `|nor~0|+|nor~0|avg~1|+|...` |
| int8_kb | INT8 TFLite model file size [KB] |
| rom_kb | Total on-device ROM (flash) usage [KB] |
| ram_kb | Total on-device RAM usage [KB] (constant 366.3 KB across all archs) |
| latency_s | Mean inference latency [seconds] |
| current_uA | Mean current draw during inference [µA] |
| power_mW | Mean power during inference [mW] |
| energy_mJ | Mean energy per inference [mJ] |
| board | Board unit: nrf5340dk_1 / nrf5340dk_2 / nrf5340dk_3 |
| g_model_kb | Size of the `g_model` symbol in ROM (model weights only) [KB] |
| arena_used_kb | Runtime tensor arena usage reported by TFLM [KB] |
| arena_size_kb | Statically allocated arena size [KB] (constant 350 KB) |
| arena_used_pct | arena_used_kb / arena_size_kb × 100 [%] |
| operators | Number of TFLM operators in the model |
| rom_categories_json | JSON dict of ROM size [KB] per Zephyr memory-report category (keys: `"(no paths)"`, `"WORKSPACE"`, `"ZEPHYR_BASE"`, `"(hidden)"`, etc.) |
| ram_categories_json | JSON dict of RAM size [KB] per Zephyr memory-report category |

**Used by:** fig2 (metric distributions), fig3 (successful arch sizes), fig4 (pairwise scatter), fig6 (per-board boxplots), fig7 (memory composition), fig8 (arena utilization)

### accuracies.csv
Test accuracy from the NATS-Bench simulator for each successfully deployed architecture.
| column | description |
|--------|-------------|
| arch_idx | NATS-Bench TSS index (matches hw_metrics.csv) |
| cifar10 | Test accuracy on CIFAR-10 [%] |
| cifar100 | Test accuracy on CIFAR-100 [%] |
| ImageNet16-120 | Test accuracy on ImageNet-16-120 [%] |

**Used by:** fig5 (accuracy vs hardware cost Pareto fronts)

## Figure → data mapping

| Figure | File(s) | Key columns |
|--------|---------|-------------|
| fig1 — Pipeline coverage & failures | coverage.csv, failures.csv | count, stage |
| fig2 — HW metric distributions | hw_metrics.csv | rom_kb, latency_s, current_uA, power_mW, energy_mJ |
| fig3 — Flash threshold | hw_metrics.csv, failures.csv | int8_kb, overflow_kb, stage |
| fig4 — Pairwise scatter | hw_metrics.csv | int8_kb, rom_kb, latency_s, energy_mJ |
| fig5 — Accuracy vs cost Pareto | hw_metrics.csv + accuracies.csv | latency_s, energy_mJ, rom_kb, cifar10/100/ImageNet |
| fig6 — Per-board boxplots | hw_metrics.csv | board, power_mW, current_uA, latency_s |
| fig7 — Memory composition | hw_metrics.csv | g_model_kb, rom_*_kb, ram_*_kb |
| fig8 — Arena utilization | hw_metrics.csv | arena_used_kb, arena_size_kb, g_model_kb |

## Notes

- `arch_idx` is the zero-based NATS-Bench TSS index (integer 0–15624).
- RAM is constant at 366.3 KB across all architectures (static tensor arena).
- energy_mJ ≈ power_mW × latency_s (Spearman ρ = 0.999 with measured energy).
- Flash usage = a constant 277.6 KB firmware footprint + the INT8 model flatbuffer.
  On the 1024 KB app-core flash this gives a deployable ceiling of 746 KB, which the
  precheck enforces exactly. (The original campaign used a coarser 800 KB precheck; the
  115 architectures in the 747-791 KB band passed it and were rejected one stage later at
  the linker. All are flash-infeasible and are reported here under the single `infeasible`
  label. See corrections/README.md and the camera-ready change log.)
- NATS-Bench paper global best test accuracies: CIFAR-10 94.37%, CIFAR-100 73.51%,
  ImageNet-16-120 47.31% (hp=200 epochs).
