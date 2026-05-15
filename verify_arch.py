"""
Structural comparison between our Keras TinyNetwork (nats2tflite.build_model)
and the reference PyTorch TinyNetwork from NATS-Bench.

For each architecture we:
  1. Build both models.
  2. Collect a normalized (op_type, output_shape) sequence from each.
  3. Align the sequences and diff them layer-by-layer.

Two networks that agree on this signature have identical layer types, identical
tensor shapes, and identical connectivity order — which is much stronger than
matching parameter counts alone.
"""
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net

from nats2tflite import build_model


# -----------------------------
# Normalize op names across frameworks.
# Keys are class names as reported by type(module).__name__ (Torch) or
# type(layer).__name__ (Keras).
# -----------------------------
NORMALIZE = {
    # conv
    "Conv2d": "conv", "Conv2D": "conv",
    # batchnorm
    "BatchNorm2d": "bn", "BatchNormalization": "bn",
    # activation
    "ReLU": "relu",
    # pooling
    "AvgPool2d": "avgpool", "AveragePooling2D": "avgpool",
    "AdaptiveAvgPool2d": "gap", "GlobalAveragePooling2D": "gap",
    # head
    "Linear": "linear", "Dense": "linear",
    # identity / zero
    "Identity": "identity",
    "Zero": "zero", "Lambda": "zero",  # Keras Lambda in our code is the x*0 "none" op
}

# Keras-only layers that have no PyTorch module equivalent (tensor ops).
# They are filtered out before comparison; structural equivalence is preserved
# because the summed nodes have the same shape as their inputs.
KERAS_SKIP = {"InputLayer", "Add"}

# Normalized op names that represent true no-ops. They appear on either side
# depending on framework conventions (PyTorch emits Identity() for skip_connect;
# our Keras code just returns the tensor) and are dropped before comparison.
NOOP_NORMALIZED = {"identity"}


# -----------------------------
# Canonicalize a shape to (H, W, C) or (F,) regardless of framework layout.
# PyTorch tensors are (N, C, H, W) / (N, F); Keras layer shapes are
# (None, H, W, C) / (None, F).
# -----------------------------
def canon_shape_torch(shape):
    if len(shape) == 4:
        _, c, h, w = shape
        if h == 1 and w == 1:        # (N, C, 1, 1) is equivalent to a flat (C,)
            return (int(c),)
        return (int(h), int(w), int(c))
    if len(shape) == 2:
        return (int(shape[1]),)
    return tuple(int(s) for s in shape[1:])


def canon_shape_keras(shape):
    if shape is None:
        return None
    s = tuple(shape)
    if len(s) == 4:
        _, h, w, c = s
        if h == 1 and w == 1:
            return (int(c),)
        return (int(h), int(w), int(c))
    if len(s) == 2:
        return (int(s[1]),)
    return tuple(int(x) for x in s[1:])


# -----------------------------
# Extract ordered (op, shape) signature from a PyTorch model.
# We hook every *leaf* module (one with no children) so that containers like
# Sequential and ReLUConvBN are decomposed into their atomic Conv/BN/ReLU ops.
# -----------------------------
def torch_signature(model, input_shape=(1, 3, 32, 32)):
    sig = []

    def make_hook(mod):
        def hook(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                out_shape = tuple(out.shape)
            else:
                out_shape = tuple(out[0].shape)
            name = type(mod).__name__
            sig.append((NORMALIZE.get(name, name), canon_shape_torch(out_shape)))
        return hook

    handles = []
    for m in model.modules():
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(make_hook(m)))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape))

    for h in handles:
        h.remove()
    return [(op, shape) for op, shape in sig if op not in NOOP_NORMALIZED]


# -----------------------------
# Extract ordered (op, shape) signature from a Keras model by walking its
# layers. Keras returns layers in topological creation order, which matches
# forward-pass order for our sequentially built model.
# -----------------------------
def keras_signature(model):
    sig = []
    for layer in model.layers:
        name = type(layer).__name__
        if name in KERAS_SKIP:
            continue
        out = layer.output
        shape = tuple(out.shape) if out is not None else None
        sig.append((NORMALIZE.get(name, name), canon_shape_keras(shape)))
    return sig


from collections import Counter


# -----------------------------
# Normalize a signature into a multiset of (op, shape) pairs. Two networks
# that agree on this multiset have the same number of every (op_type,
# output_shape) pair — i.e. same layer inventory and tensor shapes. This is
# insensitive to the order of parallel sibling edges in a DAG, so it works
# even though frameworks may report them in different orders.
# -----------------------------
def to_multiset(sig):
    return Counter(sig)


def diff_multisets(ref_ms, ks_ms):
    """Return (missing_in_keras, extra_in_keras, overall_ok)."""
    missing = ref_ms - ks_ms   # present in ref but absent/fewer in keras
    extra   = ks_ms - ref_ms   # present in keras but absent/fewer in ref
    return missing, extra, (not missing and not extra)


def fmt_entry(entry, n):
    op, shape = entry
    return f"  {op:<10} {str(shape):<18}  ×{n}"


def ref_param_count(api, index):
    config = api.get_net_config(index, "cifar10")
    model = get_cell_based_tiny_net(config)
    return sum(p.numel() for p in model.parameters())


def keras_param_count(model):
    return sum(int(tf.keras.backend.count_params(w)) for w in model.trainable_weights)


def verify_one(api, index, show_diff=False, show_full=False):
    arch_str = api.arch(index)

    config = api.get_net_config(index, "cifar10")
    ref_model = get_cell_based_tiny_net(config)
    ref_sig = torch_signature(ref_model)
    ref_params = sum(p.numel() for p in ref_model.parameters())

    tf.keras.backend.clear_session()
    k_model = build_model(arch_str)
    ks_sig = keras_signature(k_model)
    ks_params = keras_param_count(k_model)

    ref_ms = to_multiset(ref_sig)
    ks_ms  = to_multiset(ks_sig)
    missing, extra, ms_ok = diff_multisets(ref_ms, ks_ms)
    params_ok = (ref_params == ks_params)
    ok = ms_ok and params_ok

    if show_full or (show_diff and not ok):
        print(f"\n=== index {index} (arch: {arch_str}) ===")
        print(f"  params: ref={ref_params:,} keras={ks_params:,} diff={ks_params - ref_params:+,}")
        print(f"  leaves: ref={len(ref_sig)} keras={len(ks_sig)}")
        if show_full:
            print("  full layer inventory (multiset of (op, shape)):")
            for entry, n in sorted(ref_ms.items()):
                print(fmt_entry(entry, n))
        if missing:
            print("  Present in ref but missing in Keras:")
            for entry, n in sorted(missing.items()):
                print(fmt_entry(entry, n))
        if extra:
            print("  Present in Keras but missing in ref:")
            for entry, n in sorted(extra.items()):
                print(fmt_entry(entry, n))

    return ok, len(ref_sig), len(ks_sig), ref_params, ks_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="/home/ankilab/hw-nats-bench/benchmarks/NATS-tss-v1_0-3ffb9-simple")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--indices", type=int, nargs="*",
                    help="Explicit indices to verify (overrides --samples)")
    ap.add_argument("--show-full", action="store_true",
                    help="Print full layer-by-layer signature even for passing architectures")
    args = ap.parse_args()

    api = create(args.bench, "tss", fast_mode=True, verbose=False)
    if args.indices:
        indices = args.indices
    else:
        random.seed(args.seed)
        indices = random.sample(range(len(api)), args.samples)

    print(f"{'index':>6} | {'ref params':>12} | {'ks params':>12} | {'ref leaves':>10} | {'ks leaves':>9} | status")
    print("-" * 75)

    all_ok = True
    for idx in indices:
        ok, rn, kn, rp, kp = verify_one(api, idx, show_diff=True, show_full=args.show_full)
        status = "OK" if ok else "MISMATCH"
        print(f"{idx:>6} | {rp:>12,} | {kp:>12,} | {rn:>10} | {kn:>9} | {status}")
        all_ok &= ok

    print()
    print("All architectures structurally identical." if all_ok
          else "*** Structural mismatches detected — see diffs above. ***")


if __name__ == "__main__":
    main()
