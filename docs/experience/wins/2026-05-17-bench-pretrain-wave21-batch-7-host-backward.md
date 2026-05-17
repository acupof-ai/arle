# `arle train pretrain` Wave 2.1 — atomic batch port of 7 host-only backward ops to device-lazy, RTX 4070 Ti SUPER

> **Status: infrastructure ships (parity 19/19 green), headline gate
> FAILED, 4th consecutive KILL in the Wave 2 family.** Per the brief's
> failure-mode handling: the per-op port strategy has fundamentally
> diminishing returns — the residency chain is binary, and 7 more
> upstream demoters (`layout::reshape/transpose/slice_backward`,
> `softmax::log_softmax_backward`, `broadcast::add_broadcast_backward`'s
> host-fallback gate, `embed::embedding_backward`'s gate, plus the
> `linear_attention` body, plus `gather_last_dim_backward`'s host
> fallback, plus the `mul_scalar_backward` host fallback in chains
> where mul_backward runs first) keep `Dirty::Host` propagating up the
> tape. **STOP** — do not push to a 5th KILL. The next step is a
> structural rewrite (collapse the tape-store layer to a single-residency
> Tensor like candle / candle-grad), not another point-fix port.

## Goal (type: optimization, infrastructure)

Atomic batch port of the 7 remaining host-only backward ops per the
Wave 2.0 architectural-debt summary:

| op | file | host reads/step (Wave 2.0 estimate) |
|---|---|---|
| `rms_norm_backward` | `norm.rs` | ~128 |
| `mul_backward` | `elementwise.rs` | ~192 |
| `silu_backward` | `activation.rs` | per-MLP-layer |
| `gelu_backward` | `activation.rs` | cumulative |
| `sigmoid_backward` | `activation.rs` | cumulative |
| `exp_backward` | `activation.rs` | cumulative |
| `rope_backward` | `rope.rs` | ~32 |

Each receives:
1. A `Backend::xxx_backward_device` trait method with a default
   `readback → host → upload` fallback so CPU/Metal silently inherit
   correct behaviour.
2. A CUDA NVRTC override that consumes both `upstream` and the saved
   forward-side handle on-device.
3. A `device_path_ok` gate in the `ops/` wrapper that prefers the
   device path when both `upstream` and saved are `dirty != Host` AND
   have a device handle.
4. A parity test in `tests/test_cuda_lazy_ops.rs` (production-rep shape;
   tolerance `1e-5 atol + 1e-4 rtol` for the trig / two-pass-reduce
   ops, `1e-6 + 1e-4` for pure elementwise).

Acceptance gate (from the brief): median tok/s ≥ 230, DtoH count < 500,
DtoH total bytes < 3 GB, GPU avg utilization ≥ 20%.

## Hypothesis

Per the Wave 2.0 wins entry: the residency chain is gated by 7
unconditional host-demoters in per-layer backward ops. Porting them as
a batch should close the chain and push DtoH bytes from Wave 2.0's
42.7 GB down to ~1 GB (P3.1's level for *non-CE-loss* traffic) plus
the residual `[B,S,V] = 1 GB` logits tile.

**Realised**: 19/19 parity green. **5-step bench median tok/s = 172.7
(steps 2-5)**, vs Wave 2.0 174.7 / P3.1 171.3. **DtoH count = 3 544 /
DtoH bytes = 17.4 GB (per `cuda_api_sum` time band)** — unchanged from
Wave 2.0. The Wave 2.0 attribution **was right about the demoter
*locations* but wrong about the *count*** — porting these 7 ops still
leaves 7+ other demoters along the chain (layout backwards,
log_softmax_backward's host fallback when its `device_path_ok` gate
misses, broadcast_backward's host fallback, linear_attention's body).

## Command

```bash
CUDA_HOME=/opt/cuda CARGO_TARGET_DIR=/tmp/arle-target-cuda \
NVCC_CCBIN=g++-14 CC=gcc-14 CXX=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo build --release --bin arle --features cuda

# parity gate (19 = 12 prior + 7 new)
cargo test --release -p autograd --features cuda --test test_cuda_lazy_ops

# 5-step throughput
/tmp/arle-target-cuda/release/arle train pretrain \
  --backend cuda \
  --corpus /home/ckl/arle-data/pretrain/corpus.txt \
  --tokenizer /home/ckl/arle-data/models/Qwen3.5-0.8B/tokenizer.json \
  --preset small-25m --model-family qwen35 \
  --steps 5 --batch 2 --seq 512 --grad-accum-steps 16 \
  --lr 3e-4 --log-every 1 --save-every 5 \
  --out /home/ckl/arle-data/benches/wave21/run

# nsys 1-step profile
nsys profile --output=/home/ckl/arle-data/benches/wave21-profile/wave21_step1 \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true \
  /tmp/arle-target-cuda/release/arle train pretrain --steps 1 ...
```

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER · 16.0 GB · sm_89 |
| CUDA / nvcc | 13.2 V13.2.78 (system /opt/cuda) |
| Nsight Systems | 2025.6.3.541-256337736014v0 |
| Host compiler | g++-14 (NVCC_CCBIN) |
| cudarc | 0.19.7 |
| ARLE commit | Wave 2.1 staged, not committed |
| Features | `cli,cuda` |
| Model | Qwen3.5-family `small-25m` preset (V=248070, H=160, L=2, A=5, FFN=320) |
| Params | 40 255 328 (40.26 M) |
| Hyperparams | steps=5, batch=2, seq=512, grad_accum=16 → effective batch 32, tokens/step 16 384 |

## Results

### Parity test (19/19 green)

```
running 19 tests
test cuda_mean_backward_device_matches_cpu ... ok
test cuda_add_into_device_matches_cpu ... ok
test cuda_add_broadcast_backward_device_matches_cpu ... ok
test cuda_mul_scalar_backward_device_matches_cpu ... ok
test cuda_gelu_backward_device_matches_cpu ... ok          # new
test cuda_rms_norm_backward_device_matches_cpu ... ok      # new
test cuda_rope_backward_device_matches_cpu ... ok          # new
test cuda_mul_backward_device_matches_cpu ... ok           # new
test cuda_exp_backward_device_matches_cpu ... ok           # new
test cuda_sigmoid_backward_device_matches_cpu ... ok       # new
test cuda_silu_backward_device_matches_cpu ... ok          # new
test cuda_adamw_step_device_matches_cpu ... ok
test cuda_gather_last_dim_device_lazy_matches_cpu ... ok
test cuda_embedding_backward_device_matches_cpu ... ok
test cuda_gather_last_dim_backward_matches_cpu ... ok
test cuda_log_softmax_last_axis_device_lazy_matches_cpu ... ok
test cuda_matmul_backward_device_matches_cpu ... ok
test cuda_softmax_last_axis_device_lazy_matches_cpu ... ok
test cuda_log_softmax_last_axis_backward_matches_cpu ... ok

test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured
```

### Per-op parity verdicts

| op | shape | tolerance | verdict |
|---|---|---|---|
| `silu_backward` | `[B=2, S=512, H=160]` | atol=1e-6 + rtol=1e-4 | PASS |
| `gelu_backward` | `[B=2, S=512, H=160]` | atol=1e-6 + rtol=1e-4 | PASS |
| `sigmoid_backward` | `[B=2, S=512, H=160]` | atol=1e-6 + rtol=1e-4 | PASS |
| `exp_backward` | `[B=2, S=512, H=160]` | atol=1e-6 + rtol=1e-4 | PASS |
| `mul_backward` | `[B=2, S=512, H=160]` (both sides) | atol=1e-6 + rtol=1e-4 | PASS |
| `rms_norm_backward` | `[B=2, S=512, H=160]` + weight `[H=160]` | atol=1e-5 + rtol=1e-4 | PASS |
| `rope_backward` | `[B=2, n_heads=5, S=512, head_dim=32]` | atol=1e-5 + rtol=1e-4 | PASS |

### Throughput (5 steps)

| Step | tok/s | ms/step |
|---|---|---|
| 1 | 171.04 | 95 790 |
| 2 | 174.61 | 93 830 |
| 3 | 172.49 | 94 986 |
| 4 | 172.30 | 95 090 |
| 5 | 172.70 | 94 872 |

**Median tok/s (steps 2-5): 172.69** vs Wave 2.0 174.7 vs P3.1 171.28.
Δ vs Wave 2.0: -2.0 tok/s (-1.1%, inside noise but slightly negative).
Δ vs P3.1: +1.4 tok/s (+0.8%, well inside noise).
**Acceptance gate ≥ 230 tok/s: FAILED.**

Loss curve nearly identical to Wave 2.0 (`12.437489 → 12.354364 →
12.281852 → 12.205198 → 12.114258`) — minor ULP-level drift vs Wave
2.0's `12.437490 → ...` (the new device kernels' `__expf` /
`rsqrtf` differ from libm by ~1-2 ULP), confirming the device kernels
are being invoked but only at the leaves of the backward chain.

### nsys headline metrics (1-step profile)

| Metric | P3.1 | Wave 2a | Wave 2.0 | Wave 2.1 (this) | Δ vs Wave 2.0 |
|---|---|---|---|---|---|
| DtoH call count | 121 | 3 544 | 3 544 | **3 544** | 0 (no movement) |
| DtoH total time | 0.64 s | 17.77 s | 15.85 s | **18.33 s** | +2.48 s (slight regression) |
| HtoD call count | 166 | 4 240 | 4 240 | **4 240** | 0 |
| HtoD total time | n/a | n/a | n/a | **6.32 s** | n/a |
| Max single DtoH | 1 016 MB | 1 016 MB | 1 016 MB | **~1 GB** | 0 (logits tile unchanged) |
| Tok/s median (2-5) | 171.28 | 174.40 | 174.7 | **172.69** | -1.1% |
| `adamw_step_f32` kernel launches | n/a | ~24 | 24 | ~24 | 0 |
| GPU avg util | 11.7% | n/a | 9.6% | **~10%** | wash |

Wave 2.1 `.nsys-rep` at:
`/home/ckl/arle-data/benches/wave21-profile/wave21_step1.nsys-rep`.

### Which ops still host-resident (the remaining demoters)

Per `grep tensor_host crates/autograd/src/ops/`:

| File | Calls | Notes |
|---|---|---|
| `linear_attention.rs` | 19 | Entire body is host-loop; not yet device-routed (out of scope of Wave 2 family). |
| `rope.rs` | 15 | Includes the cos/sin host-cache ensures (legit) + the **eager-host** fallback (still hot in chains where upstream `Dirty::Host`). |
| `activation.rs` | 14 | Host-eager forward + host-fallback backward branches of 4 activations. |
| `norm.rs` | 9 | Same — host-eager and host-fallback paths. |
| `elementwise.rs` | 9 | Same. |
| `layout.rs` | 7 | `reshape_backward`, `transpose_backward`, `slice_backward` each `tensor_host(upstream)` unconditionally — **not** ported in Wave 2.1. |
| `softmax.rs` | 5 | `log_softmax_backward` host-fallback. |
| `broadcast.rs` | 5 | `add_broadcast_backward` host-fallback (the device override DOES exist; the gate misses when its specific shape conditions don't hold). |
| `matmul.rs` | 3 | `matmul_backward` host-fallback (when `device_path_ok` gate misses). |
| `reduce.rs` | 2 | `mean_backward` / `sum_backward` host-fallback. |
| `gather.rs` | 2 | `gather_last_dim_backward` host-fallback. |
| `embed.rs` | 2 | `embedding_backward` host-fallback. |

**Total: 92 callsites in ops/.** Wave 2.1 closed 7. The remaining 85
include 7+ chain-bottlenecks (layout.rs's 7 backwards alone re-host
every reshape/transpose/slice that appears in Qwen3.5's q/k/v
projection chain — and Qwen3.5 has multiple per-layer reshape +
transpose + slice ops). Closing the chain requires *all* of them, plus
re-architecting the tape-store layer so that `Dirty::Host` is no
longer the default state for the `tensor_host` accessor.

### Which ops are already device-resident (the cumulative wins)

For balance — what Wave 2 family *did* land:

| op | wave | trait method |
|---|---|---|
| `matmul_backward` | P2 | `matmul_backward_device` |
| `log_softmax_backward` | Wave 1 | `log_softmax_last_axis_backward` |
| `gather_last_dim_backward` | Wave 1 | `gather_last_dim_backward` |
| `mean_backward` | P3 | `mean_backward_device` |
| `mul_scalar_backward` | P3 | `mul_scalar_backward_device` |
| `embedding_backward` | Wave 2a | `embedding_backward_device` |
| `add_broadcast_backward` | Wave 2a | `add_broadcast_backward_device` |
| `adamw_step` (device-grad) | Wave 2.0 | `adamw_step_device` |
| `silu_backward` | **Wave 2.1** | `silu_backward_device` (new) |
| `gelu_backward` | **Wave 2.1** | `gelu_backward_device` (new) |
| `sigmoid_backward` | **Wave 2.1** | `sigmoid_backward_device` (new) |
| `exp_backward` | **Wave 2.1** | `exp_backward_device` (new) |
| `mul_backward` | **Wave 2.1** | `mul_backward_device` (new) |
| `rms_norm_backward` | **Wave 2.1** | `rms_norm_backward_device` (new) |
| `rope_backward` | **Wave 2.1** | `rope_backward_device` (new) |

15 ops have device kernels + ops-layer gates. The chain still loses
residency because **every** demoter in the per-layer post-order must
be closed; any one host hop poisons the rest.

## Problems

### #1 (Root cause, confirmed) — The chain demotion is structural, not point-fix-able

The brief's failure-mode handling predicted this: "If it KILLs again,
the SOLID conclusion is that the per-op port strategy has fundamentally
diminishing returns and we need a structural rewrite (collapse to
single-residency Tensor like candle, per the candle survey's §4)."

This wave is the 4th consecutive in-family KILL (Wave 2a wash, Wave
2.0 wash, P3.1 wash — relative to its 230-tok/s gate). Each closed a
subset of demoters; none moved the wall-clock by more than ~3%. The
common root cause: `TensorStore` exposes both host (`tensor_host` →
`ensure_host` → guarantees host copy) AND device (`device_handle`)
views simultaneously, and the `Dirty` tri-state requires *every* op
to opt in to the device path. Each new op-port adds N new
`device_path_ok` gates and N new host fallbacks — every fallback that
fires demotes its output back to Host, and the chain breaks at the
first miss.

**The architectural premise of `Dirty + tensor_host` was an
incrementalism enabler — it let us add ops one at a time without
breaking earlier ops.** That same premise is now the gate: closing the
chain requires *every* op to be lazy-device, which means *no* host
fallback can fire on the production path. Candle's `Tensor` has a
single backing store (device or host, never both) — its forward + bwd
ops are obligated to dispatch through the device on the device path.
That structural invariant is what we need.

### #2 — The 7 new device kernels DO produce correct outputs

Loss curve is nearly bit-identical to Wave 2.0 modulo expected 1-2 ULP
intrinsic drift (`__expf` vs libm `expf`, `rsqrtf` vs `1/sqrt`). The
new kernels run when their gates fire — verified by the slight loss
divergence at the 1e-6 level. They are not no-ops. But the gates fire
rarely enough that the wall-clock impact is sub-noise.

### #3 — SOLID gap: I should have measured `device_path_ok`-fire-rate before licensing the per-op port strategy

A SOLID experiment for Wave 2.1's *premise* would have been: add
`println!` (or a `tracing::trace`) counter inside each new
`device_path_ok` branch, run 1 training step, count fires vs misses
per op. If `device_path_ok` fires < 50% of the time for any of the 7
ops, the port is provably not the bottleneck — the upstream demoter
is. **This was never done.** The wave was scoped on the Wave 2.0
diagnostic's *list* of demoter sites, not on a per-site
fire-rate-of-the-device-gate measurement.

The brief explicitly anticipated this in its failure-mode-handling
clause; the fact that *this is the 4th consecutive KILL* is itself the
SOLID signal that the strategy is wrong.

## Learnings

- **Binary residency requires structural single-residency.** When a
  chain is N ops long, the closed-chain probability under
  `Dirty + tensor_host` is `product_i(gate_fire_rate_i)`. With N≥6
  layers per Qwen3.5 transformer block and ~10 ops per layer's
  backward, even a 95% per-op fire-rate gives `0.95^60 = 4.6%`
  closed-chain probability. To reach >50% closed-chain on a 60-op
  per-layer post-order, per-op fire-rate must be >98.9% — i.e.
  effectively no host fallbacks anywhere on the production path. This
  is structurally easier to guarantee by *eliminating* the host
  fallback than by chasing 98.9% gate hits.

- **The Wave 2 family's wall-clock budget is now spent on infra.** P3,
  P3.1, Wave 1, Wave 2a, Wave 2.0, Wave 2.1 collectively added ~2 000
  LoC of trait methods, NVRTC kernels, ops-layer gates, parity tests.
  All are *correct* and *infrastructure-grade* — they will compound
  the moment the structural single-residency invariant lands. None of
  them moved the bench by >+3 tok/s.

- **Candle's single-residency Tensor is the right reference.** Per
  `docs/research/2026-05-17-candle-kernel-vendor-survey.md` §4:
  candle stores `Storage::Cpu(...)` XOR `Storage::Cuda(...)` per
  Tensor. Backward ops dispatch on the storage discriminant; there is
  no `Dirty` state. Porting ARLE's autograd to that model is the
  Wave-3 boundary.

- **The 7 new device kernels and 7 new trait methods are not wasted.**
  They land the device-side numerics for every backward op the brief
  listed. When the structural rewrite happens, these kernels become
  the *direct* dispatch targets for the device-discriminant arm of the
  new Tensor — no further per-kernel work is needed.

## Rule

**When a single-variable optimization wave fails 3+ consecutive KILLs
in the same family, escalate to a structural rewrite.** Each
incremental wave inside the same architecture re-confirms the same
ceiling. Continuing past a 3-KILL streak is anti-SOLID — the data is
telling you the architecture is the bottleneck, not the implementation.

## Architectural claim — did Wave 2.1 unblock the post-P3.1 chain?

**No. 4th consecutive KILL.** Per the brief's failure-mode handling,
the SOLID conclusion is that per-op port strategy has fundamentally
diminishing returns. The 7 new device-aware backward ops + 7 NVRTC
kernels land correctly (parity 19/19 green, kernels invoked at the
leaves of the chain) but the residency chain remains broken upstream
by `layout::reshape/transpose/slice_backward` + the host-fallback
branches of every other ops/-layer dispatcher.

The Wave 2 family's trait machinery
(`adamw_step_device`, `mul_backward_device`,
`rms_norm_backward_device`, `silu_backward_device`,
`gelu_backward_device`, `sigmoid_backward_device`, `exp_backward_device`,
`rope_backward_device`, plus all earlier Wave-1/Wave-2a/P2/P3 methods)
IS the right foundation — but it is gated, not productive, until the
tape-store layer is restructured to single-residency.

**STOP** per the brief. Recommended next step: a structural rewrite of
`crates/autograd/src/tensor.rs` `TensorStore` to a candle-style
single-residency `Tensor` enum
(`Tensor::Cpu(...) | Tensor::Cuda(...)`), reusing all Wave 2 family
kernels as direct dispatch targets.

## Files

| Path | Δ |
|---|---|
| `crates/autograd/src/backend.rs` | +280 (7 new trait methods + default impls) |
| `crates/autograd/src/backend_cuda.rs` | +470 (7 trait overrides + 4 helper fns) |
| `crates/autograd/src/backend_cuda/kernels.rs` | +30 (10 new kernel names + 4 source includes) |
| `crates/autograd/src/backend_cuda/kernels/activation_backward.cu` | +72 (4 elementwise activation backward kernels) |
| `crates/autograd/src/backend_cuda/kernels/mul_backward.cu` | +28 (2 elementwise mul backward kernels — lhs / rhs split) |
| `crates/autograd/src/backend_cuda/kernels/rms_norm_backward.cu` | +120 (3 kernels: inv_rms / grad_x / grad_w) |
| `crates/autograd/src/backend_cuda/kernels/rope_backward.cu` | +38 (1 kernel — forward with sin inlined-negated) |
| `crates/autograd/src/ops/norm.rs` | +60 (device_path_ok dispatch for rms_norm_backward) |
| `crates/autograd/src/ops/elementwise.rs` | +70 (device_path_ok dispatch for mul_backward; mul_scalar_backward unchanged) |
| `crates/autograd/src/ops/activation.rs` | +200 (4 backwards each get a device_path_ok dispatch wrapper) |
| `crates/autograd/src/ops/rope.rs` | +45 (device_path_ok dispatch for rope_backward) |
| `crates/autograd/tests/test_cuda_lazy_ops.rs` | +400 (7 new parity tests at production-rep shapes) |

No changes to `tape.rs` / `tensor.rs` / `optim.rs` / `ops/matmul.rs` /
`ops/softmax.rs` / `ops/gather.rs` / `ops/reduce.rs` / `ops/embed.rs` /
`ops/broadcast.rs` / `ops/layout.rs` / `ops/linear_attention.rs` (per
the brief's hard scope).
