#!/usr/bin/env python3
"""
Does INT8 post-training quantization preserve the accuracy *ranking* of
architectures? (Addresses Reviewer ZdDi.)

For a stratified sample of deployable NATS-Bench TSS architectures we:
  1. build the exact Keras model used by the deployment pipeline (nats2tflite.build_model),
  2. train it on CIFAR-10,
  3. evaluate FP32 test accuracy,
  4. quantize to INT8 with the released full-integer TFLite PTQ config
     (calibrated on real CIFAR-10 data — the deployed flatbuffers are untrained,
     so their random calibration is irrelevant to accuracy),
  5. evaluate INT8 test accuracy via the TFLite interpreter.

We then report Spearman rho / Kendall tau between FP32 and INT8 accuracy, the
top-k overlap (do the FP32-best architectures stay best under INT8 — what NAS
actually relies on), the mean |Delta accuracy|, and any architectures whose INT8
accuracy *craters* (flagged separately, e.g. degenerate quantization scales).

Results stream to a CSV so the run resumes if interrupted.

Usage (GPU strongly recommended):
    python int8_rank_correlation.py --n 100 --epochs 12
    python int8_rank_correlation.py --n 100 --epochs 12 --resume     # continue
"""
import argparse
import csv
import os
import numpy as np
import tensorflow as tf

from nats2tflite import build_model  # exact deployment model

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], np.float32)


# --------------------------------------------------------------------------- data
def load_cifar10(data_dir=os.path.expanduser("~/.keras/datasets/cifar-10-batches-py")):
    """Read the raw CIFAR-10 python batches directly (no keras/tfds downloader —
    the source server is throttled, so the archive is fetched once out-of-band
    into data_dir/)."""
    import pickle

    def unpickle(fn):
        with open(os.path.join(data_dir, fn), "rb") as fo:
            return pickle.load(fo, encoding="bytes")

    xs, ys = [], []
    for i in range(1, 6):
        b = unpickle(f"data_batch_{i}")
        xs.append(b[b"data"]); ys.extend(b[b"labels"])
    xtr = np.concatenate(xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    ytr = np.array(ys, dtype="int64")
    tb = unpickle("test_batch")
    xte = tb[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    yte = np.array(tb[b"labels"], dtype="int64")
    xtr = ((xtr.astype("float32") / 255.0) - CIFAR_MEAN) / CIFAR_STD
    xte = ((xte.astype("float32") / 255.0) - CIFAR_MEAN) / CIFAR_STD
    return xtr, ytr, xte, yte


def make_train_ds(xtr, ytr, batch):
    def augment(x, y):
        x = tf.image.resize_with_crop_or_pad(x, 40, 40)
        x = tf.image.random_crop(x, [32, 32, 3])
        x = tf.image.random_flip_left_right(x)
        return x, y
    ds = tf.data.Dataset.from_tensor_slices((xtr, ytr))
    ds = ds.shuffle(10000).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------------------- per-arch
def train_and_eval_fp32(arch_str, xtr, ytr, xte, yte, epochs, batch):
    tf.keras.backend.clear_session()
    model = build_model(arch_str, num_classes=10)
    steps = (len(xtr) // batch) * epochs
    lr = tf.keras.optimizers.schedules.CosineDecay(0.1, steps)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(make_train_ds(xtr, ytr, batch), epochs=epochs, verbose=0)
    logits = model.predict(xte, batch_size=512, verbose=0)
    fp32_acc = float((logits.argmax(1) == yte).mean())
    return model, fp32_acc


def to_int8(model, xtr):
    """Released full-integer INT8 PTQ config, calibrated on real CIFAR-10."""
    calib = xtr[np.random.default_rng(0).choice(len(xtr), 200, replace=False)]

    def rep():
        for i in range(len(calib)):
            yield [calib[i:i + 1].astype("float32")]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    return conv.convert()


def eval_int8(int8_bytes, xte, yte):
    interp = tf.lite.Interpreter(model_content=int8_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    scale, zp = inp["quantization"]
    correct = 0
    for i in range(len(xte)):
        x = xte[i:i + 1]
        if inp["dtype"] == np.uint8:
            x = np.clip(np.round(x / scale + zp), 0, 255).astype(np.uint8)
        else:
            x = x.astype(inp["dtype"])
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        if int(interp.get_tensor(out["index"]).argmax()) == yte[i]:
            correct += 1
    return correct / len(xte)


# --------------------------------------------------------------------------- sample
def sample_archs(n, seed):
    """Stratified sample of DEPLOYABLE archs across the CIFAR-10 accuracy range."""
    import analyze_artifacts as aa
    rows = list(csv.DictReader(open("export_corrected/accuracies.csv")))
    hw = {r["arch_idx"] for r in csv.DictReader(open("export_corrected/hw_metrics.csv"))}
    pairs = [(int(r["arch_idx"]), float(r["cifar10"]))
             for r in rows if r["arch_idx"] in hw and r.get("cifar10")]
    pairs.sort(key=lambda p: p[1])
    # even strata across the sorted-by-accuracy list
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, len(pairs), n + 1).astype(int)
    chosen = [pairs[rng.integers(edges[i], max(edges[i] + 1, edges[i + 1]))]
              for i in range(n)]
    api = aa  # only used for arch strings via NATS-Bench
    from nats_bench import create
    nb = create(str(aa.DEFAULT_BENCH), "tss", fast_mode=True, verbose=False)
    return [(idx, nb.arch(idx), nats_acc) for idx, nats_acc in chosen]


# ----------------------------------------------------------------------------- main
FIELDS = ["arch_idx", "arch_str", "nats_cifar10", "fp32_acc", "int8_acc",
          "delta_pp", "cratered"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="export_corrected/int8_rank_correlation.csv")
    ap.add_argument("--resume", action="store_true",
                    help="Skip architectures already present in --out.")
    args = ap.parse_args()

    print("GPU:", tf.config.list_physical_devices("GPU") or "NONE (CPU — will be slow)")
    xtr, ytr, xte, yte = load_cifar10()
    archs = sample_archs(args.n, args.seed)

    done = set()
    if args.resume and os.path.exists(args.out):
        done = {int(r["arch_idx"]) for r in csv.DictReader(open(args.out))}
        print(f"Resuming: {len(done)} architectures already done.")

    new_file = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    for k, (idx, arch_str, nats_acc) in enumerate(archs):
        if idx in done:
            continue
        try:
            model, fp32 = train_and_eval_fp32(arch_str, xtr, ytr, xte, yte,
                                              args.epochs, args.batch)
            int8 = eval_int8(to_int8(model, xtr), xte, yte)
            delta = (fp32 - int8) * 100.0
            cratered = int(int8 < fp32 - 0.10 or int8 < 0.15)  # >10pp drop or ~random
            w.writerow({"arch_idx": idx, "arch_str": arch_str,
                        "nats_cifar10": round(nats_acc, 4),
                        "fp32_acc": round(fp32, 4), "int8_acc": round(int8, 4),
                        "delta_pp": round(delta, 3), "cratered": cratered})
            fout.flush()
            print(f"[{k+1}/{len(archs)}] arch {idx:5d}  fp32={fp32:.4f}  "
                  f"int8={int8:.4f}  d={delta:+.2f}pp{'  CRATERED' if cratered else ''}")
        except Exception as e:
            print(f"[{k+1}/{len(archs)}] arch {idx}: ERROR {e}")
    fout.close()
    summarize(args.out)


def summarize(path):
    from scipy.stats import spearmanr, kendalltau
    rows = list(csv.DictReader(open(path)))
    fp = np.array([float(r["fp32_acc"]) for r in rows])
    q8 = np.array([float(r["int8_acc"]) for r in rows])
    crater = [r for r in rows if r["cratered"] == "1"]
    rho = spearmanr(fp, q8).correlation
    tau = kendalltau(fp, q8).correlation
    order_fp = set(np.array([r["arch_idx"] for r in rows])[fp.argsort()[::-1]][:10])
    order_q8 = set(np.array([r["arch_idx"] for r in rows])[q8.argsort()[::-1]][:10])
    print("\n================ INT8 vs FP32 ranking ================")
    print(f"  architectures        : {len(rows)}  ({len(crater)} cratered)")
    print(f"  Spearman rho         : {rho:.4f}")
    print(f"  Kendall tau          : {tau:.4f}")
    print(f"  top-10 overlap       : {len(order_fp & order_q8)}/10")
    print(f"  mean |delta| (pp)    : {np.abs(fp - q8).mean()*100:.2f}")
    print(f"  median delta (pp)    : {np.median((fp - q8))*100:.2f}")
    print("=" * 54)


if __name__ == "__main__":
    main()
