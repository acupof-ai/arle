# dspark_dsa indexer gate passed f32 weights to an ffi that reads bf16 — 2026-09-11

## Context

First GPU parity batch (commit eb7ef6164, 2026-09-11) reported
`dspark_dsa_parity` `[indexer fused_q rows=256] scale_rel_worst=1.00e0
violators=16381 FAIL`. The scale output of
`dsv4_dsa_fused_q_indexer_rope_hadamard_quant` was off by 100% and ~half
the 32768 per-element checks failed.

## Root cause

Gate bug, not a kernel bug. The kernel
(`crates/cuda-kernels/csrc/attention/dsv4_dsa_official.cu:125`) reads the
`weight` argument as bf16:

```cpp
const float weight_val = bf16_to_float(static_cast<const bf16_t*>(params.weight)[work_id]);
...
params.weights_out[work_id] = weight_val * params.weight_scale * scale;
```

and the Rust ffi types it `*const ffi::Half`. Production passes the
indexer `weights_proj` output, which is `HiddenStates` (`CudaSlice<bf16>`,
crates/cuda-kernels/src/tensor.rs:175). The gate instead allocated and
uploaded `Vec<f32>`:

```rust
let weights = vec![1.0f32; works]; // 32-bit elements
```

The kernel reinterpreted the f32 1.0 bytes (`00 00 80 3f` little-endian)
as two bf16 elements: `0x0000` = 0.0 and `0x3f80` = 1.0. Read element by
work_id the buffer alternates, so every even row got `weight_val` exactly
0.0 and every odd row exactly 1.0. `weights_out` was therefore exactly 0
on every even work_id (a 100% scale error, scale_rel 1.0) and correct on
every odd one; the per-element byte comparison scaled through that output
and failed on 16381 of 32768 checks — about half, matching the alternating
pattern. The fp8 byte output itself was unaffected — the bad buffer feeds
only `weights_out`.

The kernel body is byte-identical to upstream SGLang
(python/sglang/kernels/aot/csrc/elementwise/dsv4_norm_rope.cu,
`fused_q_indexer_rope_hadamard_quant_kernel`).

## Fix

`crates/infer-cuda/examples/dspark_dsa_parity.rs`: allocate the gate's
weights as `vec![bf(1.0); works]` (bf16), matching the ffi contract and
the production `HiddenStates` dtype.

pending-remote: rerun `target/release/examples/dspark_dsa_parity` on the
H20 pod; expect `[indexer fused_q rows=256] scale_rel_worst` within
SCALE_REL_MAX and violators=0, plus negative control.
Needle/lever not applicable (example-only parity gate, no serve path).

## Rule

A raw-pointer ffi has no element-type check at the boundary. When the
host wrapper casts to `*const Half`, the test fixture must allocate that
dtype too — a `Vec<f32>` of ones is not a stand-in for a bf16 buffer,
even though both fill memory with nonzero bytes. The scale output
multiplier is the canary: it is the only place the weight buffer enters
this kernel's result.
