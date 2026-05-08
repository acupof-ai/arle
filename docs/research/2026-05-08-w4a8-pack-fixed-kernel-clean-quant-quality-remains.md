# W4A8 pack fixed + kernel/reference clean — remaining blocker is quantization quality

## Context

This continues the W4A8 garbage-output chain:

- `81b6481`: W4A8 produced 100% garbage output from token 0.
- `0be5967` + `4aebcec`: pack/unpack round-trip isolated a scale-chain bug.
- `3cee2f0`: H3 was a wrong-class comparison; PR #31 `W4A8Layer` uses the
  4-consecutive row pattern, not the plain `Layer` skip-8 pattern.

The current script state restores the PR #31 `W4A8Layer` layout and fixes
the scale-chain ordering.

## Evidence

### Pack/Unpack Round-Trip

Command:

```bash
.venv/bin/python scripts/diag_w4a8_pack_roundtrip_multishape.py
```

Result:

```text
(256, 128, 128)           2.3733e-02    1.9394e-02     1.2         44         1.143      PASS
(256, 256, 128)           2.3733e-02    1.9809e-02     1.2         22         1.143      PASS
(512, 128, 128)           2.3733e-02    1.9394e-02     1.2         44         1.143      PASS
(512, 512, 128)           2.3733e-02    1.9800e-02     1.2         11         1.143      PASS
(1024, 256, 128)          2.3733e-02    1.9745e-02     1.2         22         1.143      PASS
(1024, 1024, 128)         2.3733e-02    1.9623e-02     1.2          5         1.143      PASS
(2048, 512, 128)          2.3733e-02    1.9776e-02     1.2         11         1.143      PASS

0/8 shapes FAIL pack/unpack round-trip
```

The `(128,128,128)` case remains unsupported by the W4A8 Marlin shape
constraints, not a pack failure.

Verbose `(256,128,128)` check:

- helper qweight matches `pack_w4a8`: true
- PR #31 `W4A8Layer` perm matches current perm: true
- PR #31 qweight matches current qweight: true
- PR #31 `s_channel` max abs diff: 0
- PR #31 `s_group` max abs diff: 0
- reconstructed `s_group_real` vs raw group scale max abs diff: `~1.7e-05`

This eliminates the prior 1.71x / 1.33x scale amplification fingerprint.

### Greedy Gate Status

After re-quantizing `infer/models/Qwen3-4B-W4A8-marlin`, the output is no
longer pure garbage but still fails the BF16 token-diff gate:

```text
BF16: " Paris. The capital of Germany is Berlin..."
W4A8: " Paris. The capital of France is Paris. The capital of France is Paris..."
matched first 5/32 tokens, diff 84.4%
first divergence idx=5
```

A temporary divisor change from `/7.0` to `2/15` made output worse and was
reverted; `/7.0` is the current canonical path.

### Kernel vs Reference

Using PR #31's Python extension built from `/tmp/marlin-w4a8`, layer-0
`q_proj` side tensors from the current W4A8 checkpoint were compared against
a manual recovered-weight reference:

```text
m=1  mean relative error ~= 0.0087
m=4  mean relative error ~= 0.0083
m=16 mean relative error ~= 0.0083
```

That keeps the Marlin W4A8 kernel, FFI argument order, side-tensor loader
layout, and activation quantizer convention below the likely-bug threshold.

### Quantization Error

Recovered current W4A8 weights compared to BF16 originals show roughly:

```text
layer0 q_proj mean relative weight error ~= 12.9%
layer0 o_proj mean relative weight error ~= 12.9%
```

That is plausible for naive per-group max quantization across 36 layers and
matches the observed behavior: semantically related but repetitive output,
not kernel-shaped random garbage.

## Conclusion

The current blocker is no longer the W4A8 pack scale-chain or kernel wiring.
The next fix should reuse calibrated GPTQ/W4A16 side tensors or run a real
calibration path, then repack into W4A8 Marlin layout.

Promising local sources:

- `infer/models/Qwen3-4B-GPTQ-Int4`: GPTQ `qweight/qzeros/scales/g_idx`.
- `infer/models/Qwen3-4B-GPTQ-Int4-marlin`: already converted W4A16
  `marlin_qweight/marlin_scales` plus transformed `qweight/scales`.
- `infer/models/Qwen3-4B-W4A16-sym-g128-marlin`: production-valid W4A16
  Marlin checkpoint.

## Rule

Round-trip self-consistency only proves layout symmetry. For model accuracy,
the quantizer must also preserve layer semantics. Once pack/unpack and
kernel/reference are clean, stop iterating layout hypotheses and move to
calibrated quantization evidence.
