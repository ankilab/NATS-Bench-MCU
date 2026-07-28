#!/usr/bin/env python3
"""
Demonstrate the benchmark's utility for hardware-aware NAS (addresses Reviewer ZdDi:
"utility is stated but never demonstrated; no NAS algorithm is ever evaluated").

We run the three standard NATS-Bench search algorithms — Random Search, Regularized
Evolution, Local Search — as pure table lookups over the corrected benchmark, in a
hardware-aware setting: maximize CIFAR-10 test accuracy subject to an on-device energy
budget. Because the MCU regime is feasibility-dominated (8.6% of the space is
undeployable and only a fraction is feasible under a tight budget), an architecture
that is undeployable or over-budget is a *wasted* evaluation. We report anytime search
curves vs. the oracle optimum and quantify that wasted-evaluation cost — the concrete
argument for hardware-aware search in this regime.

Runs in seconds on the released CSVs (no training, no hardware). Deterministic (seeded).

    python nas_baselines.py                 # energy budget = median deployable energy
    python nas_baselines.py --budget-q 0.25 # tighter budget (25th pct)
"""
import argparse
import csv
import os
import numpy as np

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
REPO = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ arch encoding
def parse_arch(arch_str):
    """Return the 6 ops (node0:1, node1:2, node2:3 edges) in order."""
    ops = []
    for node in arch_str.split("+"):
        for e in node.strip("|").split("|"):
            op, src = e.split("~")
            ops.append((op, int(src)))
    return ops


def build_arch(ops):
    nodes, k = [], 0
    for n in range(3):
        edges = ops[k:k + n + 1]; k += n + 1
        nodes.append("|" + "|".join(f"{op}~{src}" for op, src in edges) + "|")
    return "+".join(nodes)


def neighbors(arch_str, str2idx):
    """All 1-op mutations that exist in the space (24 candidates)."""
    ops = parse_arch(arch_str)
    out = []
    for i, (op, src) in enumerate(ops):
        for new in OPS:
            if new != op:
                m = list(ops); m[i] = (new, src)
                s = build_arch(m)
                if s in str2idx:
                    out.append(s)
    return out


# ------------------------------------------------------------------------- data
def load(dataset, budget_q):
    manifest = {int(r["index"]): r["arch_str"]
                for r in csv.DictReader(open(f"{REPO}/tflite_models/manifest.csv"))}
    hw = {int(r["arch_idx"]): r for r in csv.DictReader(open(f"{REPO}/export_corrected/hw_metrics.csv"))}
    acc = {int(r["arch_idx"]): float(r[dataset])
           for r in csv.DictReader(open(f"{REPO}/export_corrected/accuracies.csv")) if r.get(dataset)}

    idx2str = manifest
    str2idx = {s: i for i, s in manifest.items()}
    energies = np.array([float(hw[i]["energy_mJ"]) for i in hw])
    budget = float(np.quantile(energies, budget_q))

    # reward[idx] = accuracy if deployable AND within energy budget, else -inf
    reward, feasible = {}, {}
    for i in idx2str:
        ok = i in hw and i in acc and float(hw[i]["energy_mJ"]) <= budget
        feasible[i] = ok
        reward[i] = acc[i] if ok else -1e9
    oracle = max(v for v in reward.values() if v > -1e8)
    n_feasible = sum(feasible.values())
    return idx2str, str2idx, reward, feasible, budget, oracle, n_feasible


# --------------------------------------------------------------------- searchers
def _evaluate(idx, reward, feasible, best, curve, wasted):
    if not feasible[idx]:
        wasted[0] += 1
    elif reward[idx] > best[0]:                 # only feasible archs advance the best
        best[0] = reward[idx]
    curve.append(best[0] if np.isfinite(best[0]) else np.nan)


def random_search(rng, all_idx, reward, feasible, n_evals):
    best, curve, wasted = [-np.inf], [], [0]
    for _ in range(n_evals):
        _evaluate(int(rng.choice(all_idx)), reward, feasible, best, curve, wasted)
    return np.array(curve), wasted[0]


def regularized_evolution(rng, all_idx, idx2str, str2idx, reward, feasible,
                          n_evals, pop=25, sample=10):
    best, curve, wasted = [-np.inf], [], [0]
    population = []
    for _ in range(pop):
        i = int(rng.choice(all_idx)); _evaluate(i, reward, feasible, best, curve, wasted)
        population.append(i)
    while len(curve) < n_evals:
        cand = rng.choice(population, size=min(sample, len(population)), replace=False)
        parent = max(cand, key=lambda i: reward[i])
        nb = neighbors(idx2str[parent], str2idx)
        child = str2idx[nb[rng.integers(len(nb))]] if nb else int(rng.choice(all_idx))
        _evaluate(child, reward, feasible, best, curve, wasted)
        population.append(child); population.pop(0)
    return np.array(curve[:n_evals]), wasted[0]


def local_search(rng, all_idx, idx2str, str2idx, reward, feasible, n_evals):
    best, curve, wasted = [-np.inf], [], [0]
    cur = int(rng.choice(all_idx)); _evaluate(cur, reward, feasible, best, curve, wasted)
    while len(curve) < n_evals:
        nb = neighbors(idx2str[cur], str2idx)
        rng.shuffle(nb)
        moved = False
        for s in nb:
            j = str2idx[s]
            _evaluate(j, reward, feasible, best, curve, wasted)
            if len(curve) >= n_evals:
                break
            if reward[j] > reward[cur]:
                cur = j; moved = True; break
        if not moved:  # local optimum -> restart
            cur = int(rng.choice(all_idx))
            if len(curve) < n_evals:
                _evaluate(cur, reward, feasible, best, curve, wasted)
    return np.array(curve[:n_evals]), wasted[0]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--budget-q", type=float, default=0.5,
                    help="Energy budget as a quantile of deployable energy (default: median).")
    ap.add_argument("--evals", type=int, default=300)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-fig", default=f"{REPO}/figures_corrected/fig9_nas_baselines")
    ap.add_argument("--out-csv", default=f"{REPO}/export_corrected/nas_baselines_summary.csv")
    args = ap.parse_args()

    idx2str, str2idx, reward, feasible, budget, oracle, n_feas = load(args.dataset, args.budget_q)
    all_idx = np.array(sorted(idx2str))
    print(f"space={len(all_idx):,}  feasible(deployable & energy<= {budget:.2f}mJ)="
          f"{n_feas:,} ({100*n_feas/len(all_idx):.1f}%)  oracle {args.dataset}={oracle:.2f}%")

    algos = {
        "Random Search": lambda rng: random_search(rng, all_idx, reward, feasible, args.evals),
        "Regularized Evolution": lambda rng: regularized_evolution(
            rng, all_idx, idx2str, str2idx, reward, feasible, args.evals),
        "Local Search": lambda rng: local_search(
            rng, all_idx, idx2str, str2idx, reward, feasible, args.evals),
    }
    results, summary = {}, []
    for name, fn in algos.items():
        curves, wasted = [], []
        for r in range(args.runs):
            c, w = fn(np.random.default_rng(args.seed + r))
            curves.append(c); wasted.append(w)
        C = np.vstack(curves)
        results[name] = C
        final = C[:, -1]
        final = final[np.isfinite(final)]
        # evals to reach within 0.1 pp of oracle (median over runs; inf if never)
        reach = []
        for row in C:
            hit = np.where(row >= oracle - 0.1)[0]
            reach.append(hit[0] + 1 if len(hit) else np.nan)
        summary.append({
            "algorithm": name,
            "final_acc_mean": round(final.mean(), 3),
            "final_gap_to_oracle_pp": round(oracle - final.mean(), 3),
            "median_evals_to_oracle": (int(np.nanmedian(reach)) if np.isfinite(np.nanmedian(reach)) else "n/a"),
            "wasted_eval_frac": round(np.mean(wasted) / args.evals, 3),
        })
        print(f"  {name:24s} final={final.mean():.2f}%  gap={oracle-final.mean():.2f}pp  "
              f"wasted={np.mean(wasted)/args.evals:.0%}")

    # ---- outputs ----
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"wrote {args.out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        WONG = {"Random Search": "#9A9A9A", "Regularized Evolution": "#0072B2",
                "Local Search": "#D55E00"}
        # Standard repo font sizes + centered title so the figure matches the
        # other figures when placed at ~0.75\linewidth in the paper.
        fig, ax = plt.subplots(figsize=(5.4, 3.1))
        x = np.arange(1, args.evals + 1)
        for name, C in results.items():
            m = np.nanmean(C, 0)
            lo, hi = np.nanpercentile(C, [25, 75], axis=0)
            ax.plot(x, m, label=name, color=WONG[name], lw=1.6)
            ax.fill_between(x, lo, hi, color=WONG[name], alpha=0.15, lw=0)
        ax.axhline(oracle, color="#222222", ls="--", lw=1.0)
        # Crop the VIEW to where the dynamics happen; the curves are flat well
        # before the full budget (all statistics still use the full 300-eval run).
        xmax = min(150, args.evals)
        ax.set_xlim(0.5, xmax)
        ax.text(xmax, oracle, "oracle ", va="bottom", ha="right", fontsize=8,
                color="#222222")
        ax.set_xlabel("architectures evaluated", fontsize=9.5)
        ax.set_ylabel(f"best feasible {args.dataset} acc [%]", fontsize=9.5)
        ax.set_title("Hardware-aware NAS", fontsize=10.5)
        ax.tick_params(labelsize=8.5)
        ax.legend(loc="lower right", fontsize=8.5)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_fig + ".pdf"); fig.savefig(args.out_fig + ".png", dpi=150)
        print(f"wrote {args.out_fig}.pdf/.png")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
