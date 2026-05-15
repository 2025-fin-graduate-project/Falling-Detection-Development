"""
Export gru_v26 as explicit-state TFLite for STM32 deployment via stedgeai.

Non-stateful (current):    input (1, 60, 55) → 60-step unrolling → ~133K C lines
Explicit-state (target):   input (1,  1, 55) + h1 (1,64) + h2 (1,32)
                           output logits (1,2) + new_h1 + new_h2
                           → 1-step code, no resource variables → stedgeai OK

MCU usage:
    - Maintain h1[64], h2[32] float buffers (zero-initialized)
    - Each frame: feed [features, h1, h2] → get [logits, new_h1, new_h2]
    - Update h1/h2 in place
    - Reset to zeros after fall detection window or periodically

Usage:
    python scripts/export_gru_stateful.py
"""

import numpy as np
import tensorflow as tf

ORIG_KERAS = "artifacts/gru_v26_final_notebook/gru_v26_final.keras"
OUT_DIR    = "artifacts/gru_v26_final_notebook"

# ── 1. Load original ──────────────────────────────────────────────────────────
print("Loading original model...")
original = tf.keras.models.load_model(ORIG_KERAS, compile=False)

# ── 2. Build explicit-state model ─────────────────────────────────────────────
# Inputs: one frame + previous hidden states
pose_inp = tf.keras.Input(shape=(1, 55), name="pose_sequence")
h1_inp   = tf.keras.Input(shape=(64,),   name="gru1_h_in")
h2_inp   = tf.keras.Input(shape=(32,),   name="gru2_h_in")

# return_state=True gives (output, new_h)
gru1_seq, gru1_h = tf.keras.layers.GRU(
    64, return_sequences=True, return_state=True,
    activation="tanh", recurrent_activation="sigmoid",
    reset_after=True, use_bias=True,
    unroll=True,  # eliminates TensorListReserve for fixed-length sequences
    dropout=0.0, recurrent_dropout=0.0, name="gru_1"
)(pose_inp, initial_state=h1_inp)

gru2_out, gru2_h = tf.keras.layers.GRU(
    32, return_sequences=False, return_state=True,
    activation="tanh", recurrent_activation="sigmoid",
    reset_after=True, use_bias=True,
    unroll=True,
    dropout=0.0, recurrent_dropout=0.0, name="gru_2"
)(gru1_seq, initial_state=h2_inp)

x      = tf.keras.layers.Dense(32, activation="relu",    name="head_dense")(gru2_out)
logits = tf.keras.layers.Dense(2,  activation="softmax", name="classifier")(x)

explicit_model = tf.keras.Model(
    inputs=[pose_inp, h1_inp, h2_inp],
    outputs=[logits, gru1_h, gru2_h],
    name="gru_explicit_state"
)

# ── 3. Transfer weights ───────────────────────────────────────────────────────
orig_layers = {l.name: l for l in original.layers}
print("Transferring weights:")
for layer in explicit_model.layers:
    if not layer.get_weights():
        continue
    if layer.name in orig_layers:
        w = orig_layers[layer.name].get_weights()
        layer.set_weights(w)
        print(f"  {layer.name}: {[list(ww.shape) for ww in w]}")

# ── 4. Verify equivalence ─────────────────────────────────────────────────────
print("\nVerifying equivalence (random sequence)...")
np.random.seed(0)
seq = np.random.rand(1, 60, 55).astype(np.float32)

orig_out = original(seq, training=False).numpy()

h1 = np.zeros((1, 64), dtype=np.float32)
h2 = np.zeros((1, 32), dtype=np.float32)
for t in range(60):
    frame = seq[:, t : t + 1, :]
    explicit_out, h1, h2 = explicit_model([frame, h1, h2], training=False)
    h1 = h1.numpy()
    h2 = h2.numpy()
explicit_out = explicit_out.numpy()

max_diff = np.max(np.abs(orig_out - explicit_out))
print(f"  Original : {orig_out}")
print(f"  Explicit : {explicit_out}")
print(f"  Max diff : {max_diff:.6f}  {'OK' if max_diff < 1e-4 else 'WARNING: large diff'}")

# ── 5. Convert to TFLite (fp32) ───────────────────────────────────────────────
print("\nConverting to TFLite fp32...")
converter = tf.lite.TFLiteConverter.from_keras_model(explicit_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_fp32 = converter.convert()

fp32_path = f"{OUT_DIR}/gru_v26_stateful_fp32.tflite"
with open(fp32_path, "wb") as f:
    f.write(tflite_fp32)
print(f"  Saved {fp32_path} ({len(tflite_fp32)/1024:.1f} KB)")

# ── 6. Convert to TFLite (int8) ───────────────────────────────────────────────
print("\nConverting to TFLite int8 (dynamic-range)...")
# Full int8 calibration fails with multi-input shape mismatch in the calibrator.
# Dynamic-range quantization (weights → int8, activations → float) works reliably.
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(explicit_model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_int8 = converter_int8.convert()

int8_path = f"{OUT_DIR}/gru_v26_stateful_int8.tflite"
with open(int8_path, "wb") as f:
    f.write(tflite_int8)
print(f"  Saved {int8_path} ({len(tflite_int8)/1024:.1f} KB)")

# ── 7. Verify TFLite fp32 ─────────────────────────────────────────────────────
print("\nVerifying TFLite fp32...")
interp = tf.lite.Interpreter(model_content=tflite_fp32)
interp.allocate_tensors()

inp_details = interp.get_input_details()
out_details = interp.get_output_details()
print(f"  Inputs : {[(d['name'], list(d['shape'])) for d in inp_details]}")
print(f"  Outputs: {[(d['name'], list(d['shape'])) for d in out_details]}")

pose_idx = next(d['index'] for d in inp_details if 'pose' in d['name'])
h1_idx   = next(d['index'] for d in inp_details if 'h1'   in d['name'] or 'gru1' in d['name'])
h2_idx   = next(d['index'] for d in inp_details if 'h2'   in d['name'] or 'gru2' in d['name'])
logit_idx = out_details[0]['index']

h1 = np.zeros((1, 64), dtype=np.float32)
h2 = np.zeros((1, 32), dtype=np.float32)
for t in range(60):
    interp.set_tensor(pose_idx, seq[:, t : t + 1, :])
    interp.set_tensor(h1_idx, h1)
    interp.set_tensor(h2_idx, h2)
    interp.invoke()
    tflite_out = interp.get_tensor(logit_idx)
    # update states
    h1 = interp.get_tensor(out_details[1]['index'])
    h2 = interp.get_tensor(out_details[2]['index'])

max_diff2 = np.max(np.abs(orig_out - tflite_out))
print(f"  TFLite output: {tflite_out}")
print(f"  Max diff vs original: {max_diff2:.6f}  {'OK' if max_diff2 < 1e-3 else 'WARNING'}")

print("\nDone.")
