#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/env_tensorflow_cuda.sh

uv run python - <<'PY'
import tensorflow as tf

print("tensorflow", tf.__version__)
print("built_with_cuda", tf.test.is_built_with_cuda())
print("physical_gpu", tf.config.list_physical_devices("GPU"))
print("logical_gpu", tf.config.list_logical_devices("GPU"))

with tf.device("/GPU:0"):
    a = tf.random.uniform((1024, 1024))
    b = tf.random.uniform((1024, 1024))
    c = tf.matmul(a, b)

print("gpu_matmul_ok", c.shape, float(tf.reduce_sum(c[:2, :2]).numpy()))
PY
