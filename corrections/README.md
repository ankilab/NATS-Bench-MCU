# corrections/ — authoritative overlay on the released artifacts

Applied on top of `artifacts/` at analysis time by `analyze_artifacts.build_records(corrections_iter=...)` (via `export_plot_data.py --corrections`). Each `NNNNN/` folder either overrides that architecture's record or, if its `meta.json` status is `pending`/`excluded`, removes it from the dataset.

This overlay changes **outcome labels and recovered raw files only** — no measured hardware value is altered. See the camera-ready change log for the full rationale.

## Getting the corrected dataset

**Using the tabular data (`export/*.csv`)?** Nothing to do — the released export already
reflects every correction here.

**Working from the raw per-architecture artifacts?** Extract the `artifacts_*.tar.gz`
archives and this `corrections.tar.gz` side by side, then run:

```
python corrections/apply_corrections.py          # merges the overlay into artifacts/
python corrections/apply_corrections.py --dry-run # preview the changes first
```

This overlays the 117 corrected records onto `artifacts/` in place, producing a single
fully corrected tree (2 recovered/remeasured records + 115 relabelled failures). It is
idempotent and preserves each architecture's original INT8 model file. Equivalently, the
analysis tooling applies the overlay on the fly via
`analyze_artifacts.build_records(corrections_iter=...)` / `export_plot_data.py --corrections`,
so you never have to merge manually to reproduce the paper's figures.

## 07650 — recovered measurement
Genuinely measured on 2026-05-04 (server request 8173: 10 clean inferences, PPK2 power trace, ROM/RAM reports) but left `pending` by an orphaned-process queuing bug and later spuriously rejected. Recovered here as a completed record; `results.json` is reproduced from the raw `ppk2_summary.csv` + `rom.json.gz`/`ram.json.gz` exactly as the server computes it. Regenerate with `python recover_corrections.py`.

## 07651 - remeasured on hardware
Built on 2026-05-04 (server request 8174) but the pipeline died mid-flash, so it had no device measurements. Remeasured on 2026-07-24 with remeasure_7651.py on the nRF5340-DK (server request 16402); the recovered measurement is a normal completed record (rom - int8 overhead = 277.6 KB, matching every other build).

## Mid-band precheck rejections (115 folders)
Architectures in the 747-791 KB band that passed the campaign's coarse 800 KB precheck and were rejected one stage later at the linker (`flash_failed`) or by an intermediate tightened precheck. Under the corrected precheck (exact 746 KB = 1024 KB flash - 277.6 KB firmware) they are rejected up front at the precheck stage. Their `error.log` / `meta.json` are regenerated here to the precheck format so the raw artifacts match the single-`infeasible` taxonomy in the exported CSVs. Only the stage label and message change; every architecture is flash-infeasible and produced no device measurements under either configuration. The >800 KB rejections keep their original campaign logs. Regenerate with `python recover_corrections.py`.
