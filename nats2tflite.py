import os
import csv
import argparse
import traceback
import tensorflow as tf
from tensorflow.keras import layers
from nats_bench import create


# -----------------------------
# Operation definitions — mirror NATS-Bench cell_operations.py.
# Inside a TinyNetwork InferCell, all ops run with C_in == C_out and stride == 1.
# -----------------------------

def op_skip(x, C):
    return x  # Identity (C_in == C_out, stride == 1)

def op_conv1x1(x, C):
    x = layers.ReLU()(x)
    x = layers.Conv2D(C, 1, padding="same", use_bias=False)(x)
    return layers.BatchNormalization()(x)

def op_conv3x3(x, C):
    x = layers.ReLU()(x)
    x = layers.Conv2D(C, 3, padding="same", use_bias=False)(x)
    return layers.BatchNormalization()(x)

def op_avgpool(x, C):
    return layers.AveragePooling2D(3, strides=1, padding="same")(x)


# "none" is handled specially in build_infer_cell (edges filtered out) rather
# than materialized as a zero-producing layer. A Lambda(t * 0.0) converts to a
# TFLite op whose INT8 quantization scale is degenerate (output range = [0,0]),
# which causes TFLM on the MCU to reject the model (the "GPU delegate" errors).
# Because x + 0 = x, dropping "none" edges from the Add is mathematically
# identical to the reference Zero() op.
OPS = {
    "skip_connect": op_skip,
    "nor_conv_1x1": op_conv1x1,
    "nor_conv_3x3": op_conv3x3,
    "avg_pool_3x3": op_avgpool,
}


# -----------------------------
# Parse NATS architecture string
# Format: |op~src|+|op~src|op~src|+|op~src|op~src|op~src|
# -----------------------------

def parse_arch(arch_str):
    nodes = arch_str.split('+')
    arch = []
    for node in nodes:
        node = node.strip('|')
        edges = node.split('|')
        parsed_edges = []
        for e in edges:
            if not e:
                continue
            op, src = e.split('~')
            parsed_edges.append((op, int(src)))
        arch.append(parsed_edges)
    return arch


# -----------------------------
# InferCell — normal cell (matches cell_infers/cells.py InferCell)
# Each node = sum of op(prev_node) over its incoming edges. Output = last node.
# -----------------------------

def build_infer_cell(x, arch, C):
    # nodes[i] is either a tensor or None. None means "this node is 0"; it
    # propagates because every op in this search space (skip, conv, avgpool)
    # maps 0 -> 0 (no biases; BN with default params passes 0 through).
    nodes = [x]
    for edges in arch:
        outs = []
        for op_name, src in edges:
            if op_name == "none" or nodes[src] is None:
                continue
            outs.append(OPS[op_name](nodes[src], C))
        if not outs:
            nodes.append(None)
        elif len(outs) == 1:
            nodes.append(outs[0])
        else:
            nodes.append(layers.Add()(outs))

    out = nodes[-1]
    if out is None:
        # Only the all-"none" architecture hits this (1 of 15,625). Fall back
        # to the Lambda zero tensor so the graph still has a valid output.
        out = layers.Lambda(lambda t: t * 0.0)(x)
    return out


# -----------------------------
# ResNetBasicblock — reduction cell (matches cell_operations.py, stride=2)
# -----------------------------

def build_resnet_basicblock(x, C_out):
    a = layers.ReLU()(x)
    a = layers.Conv2D(C_out, 3, strides=2, padding="same", use_bias=False)(a)
    a = layers.BatchNormalization()(a)

    b = layers.ReLU()(a)
    b = layers.Conv2D(C_out, 3, strides=1, padding="same", use_bias=False)(b)
    b = layers.BatchNormalization()(b)

    d = layers.AveragePooling2D(2, strides=2, padding="valid")(x)
    d = layers.Conv2D(C_out, 1, strides=1, padding="same", use_bias=False)(d)

    return layers.Add()([b, d])


# -----------------------------
# TinyNetwork — full CIFAR network (matches cell_infers/tiny_network.py)
# Stem -> [N normal cells @ C] -> reduction -> [N normal @ 2C] -> reduction -> [N normal @ 4C]
# -> BN -> ReLU -> GlobalAvgPool -> Linear(num_classes).
# -----------------------------

def build_model(arch_str, C=16, N=5, num_classes=10):
    arch = parse_arch(arch_str)
    inputs = tf.keras.Input((32, 32, 3))

    # Stem: Conv + BN, no ReLU
    x = layers.Conv2D(C, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)

    layer_channels   = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    for C_curr, reduction in zip(layer_channels, layer_reductions):
        if reduction:
            x = build_resnet_basicblock(x, C_curr)
        else:
            x = build_infer_cell(x, arch, C_curr)

    # lastact: BN + ReLU
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)  # raw logits, no softmax
    return tf.keras.Model(inputs, outputs)


# -----------------------------
# Convert a single architecture
# -----------------------------

def _representative_dataset():
    for _ in range(10):
        yield [tf.random.uniform((1, 32, 32, 3))]


def convert_one(api, index, output_dir):
    """Build and export float32 + int8 TFLite models. Returns ((fp32_bytes, int8_bytes), error_msg)."""
    #arch_str = "|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|"
    arch_str = api.arch(index)

    model_dir = os.path.join(output_dir, f"{index:05d}")
    tflite_path = os.path.join(model_dir, f"nats_bench_model_{index:05d}.tflite")
    int8_path   = os.path.join(model_dir, f"nats_bench_model_{index:05d}_int8.tflite")

    if os.path.exists(tflite_path) and os.path.exists(int8_path):
        return (os.path.getsize(tflite_path), os.path.getsize(int8_path)), None

    os.makedirs(model_dir, exist_ok=True)

    tf.keras.backend.clear_session()
    model = build_model(arch_str)

    # Float32
    if not os.path.exists(tflite_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
    else:
        tflite_model = open(tflite_path, "rb").read()

    # INT8
    if not os.path.exists(int8_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        int8_model = converter.convert()
        with open(int8_path, "wb") as f:
            f.write(int8_model)
    else:
        int8_model = open(int8_path, "rb").read()

    return (len(tflite_model), len(int8_model)), None


# -----------------------------
# MAIN — batch conversion
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch-convert NATS-Bench TSS architectures to TFLite")
    parser.add_argument("--bench", default="/home/ankilab/hw-nats-bench/benchmarks/NATS-tss-v1_0-3ffb9-simple",
                        help="Path to NATS-Bench TSS benchmark file/directory")
    parser.add_argument("--output", default="tflite_models",
                        help="Root output directory for TFLite models")
    parser.add_argument("--start", type=int, default=0,
                        help="Start architecture index (inclusive)")
    parser.add_argument("--end", type=int, default=None,
                        help="End architecture index (exclusive, default: all)")
    args = parser.parse_args()

    api = create(args.bench, 'tss', fast_mode=True, verbose=True)
    total = len(api)
    end = min(args.end, total) if args.end is not None else total

    os.makedirs(args.output, exist_ok=True)

    manifest_path = os.path.join(args.output, "manifest.csv")
    failures_path = os.path.join(args.output, "failures.log")

    # Load already-processed indices from manifest to allow resuming
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(int(row["index"]))

    manifest_exists = os.path.exists(manifest_path)
    manifest_file = open(manifest_path, "a", newline="")
    writer = csv.writer(manifest_file)
    if not manifest_exists:
        writer.writerow(["index", "arch_str", "tflite_path", "size_bytes", "int8_path", "int8_size_bytes", "status"])

    success_count = 0
    fail_count = 0

    for idx in range(args.start, end):
        if idx in done:
            continue

        arch_str = api.arch(idx)
        try:
            (fp32_bytes, int8_bytes), _ = convert_one(api, idx, args.output)
            tflite_path = os.path.join(args.output, f"{idx:05d}", f"nats_bench_model_{idx:05d}.tflite")
            int8_path   = os.path.join(args.output, f"{idx:05d}", f"nats_bench_model_{idx:05d}_int8.tflite")
            writer.writerow([idx, arch_str, tflite_path, fp32_bytes, int8_path, int8_bytes, "ok"])
            success_count += 1
        except Exception as e:
            writer.writerow([idx, arch_str, "", 0, f"error: {e}"])
            with open(failures_path, "a") as flog:
                flog.write(f"--- index {idx} ---\n{traceback.format_exc()}\n")
            fail_count += 1

        if (idx - args.start + 1) % 100 == 0:
            manifest_file.flush()
            print(f"[{idx+1}/{end}] done={success_count} failed={fail_count}")

    manifest_file.close()
    print(f"\nFinished: {success_count} converted, {fail_count} failed out of {end - args.start} architectures.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()