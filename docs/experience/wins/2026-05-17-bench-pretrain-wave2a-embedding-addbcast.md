# `arle train pretrain` Wave 2 Commit A — CUDA device-lazy `embedding_backward` + `add_broadcast_backward`, RTX 4070 Ti SUPER

> **Status: infrastructure-only — headline gate FAILED. STOP and re-profile.**
> Trait + 2 NVRTC kernels (with mandatory `atomicAdd` for embedding) + 2
> parity tests landed clean (11/11 green; loss matches P3.1 to 1e-5).
> 5-step bench median tok/s **174.4** vs P3.1 **171.28** = **+3.1 tok/s
> (+1.8 %)**. Acceptance gate required **≥ +15 tok/s**. nsys reveals a
> **regression** in memcpy count and bytes: DtoH calls 121 → **3 544**
> (+29×), DtoH bytes 1 185 MB → **42 691 MB** (+36×). Root cause:
> `AdamW::step_device` (`crates/autograd/src/optim.rs:227-229`) calls
> `store.to_host(grad_id)` because the `Backend::adamw_step` trait
> signature takes `grad: &[f32]`. Pre-Wave-2a, the embedding & bias
> gradients were already host-resident from the CPU scatter-add /
> broadcast-reduce fallback paths, so `to_host` was a free no-op. With my
> changes those grads are now device-resident → `to_host` does a real
> DtoH per param per grad-accum step. **The bottleneck has moved off
> `embedding_backward` / `add_broadcast_backward` onto AdamW's host-only
> grad ingestion.** Wave 2 cannot proceed without an `adamw_step_device`
> trait variant or equivalent.

## Goal (type: optimization, infrastructure)

Per the [candle vendor survey](../../research/2026-05-17-candle-kernel-vendor-survey.md)
§Wave 2 Commit A, port `embedding_backward` and `add_broadcast_backward`
— the two highest per-step-bytes-moved host backwards in the P3.1 residue
— to device-lazy CUDA paths. Expected gain ~+50 tok/s (171 → ~220). The
survey explicitly falsifies vendoring candle kernels: candle has zero
hand-written backward kernels (graph composition in Rust), and its
`scatter_add` deliberately omits `atomicAdd`, unsafe for duplicate token
ids. Hand-write with mandatory atomics.

## Hypothesis (recorded for SOLID accounting)

Per the survey:

1. `embedding_backward`: replacing the CPU `scatter_add_rows_forward`
   roundtrip with an on-device `atomicAdd` kernel keeps the `[1, S, H]`
   upstream and the `[V, H]` table grad on-device. Predicted: drops ~1
   GB of cumulative DtoH per training step.
2. `add_broadcast_backward`: replacing the host broadcast-offset reduce
   loop with a per-output-element shared-memory device reduction keeps
   the `[B, S, H]` upstream and the `[H]` bias grad on-device. Predicted:
   drops ~50 MB cumulative DtoH per step plus eliminates a serial CPU
   loop hot spot.

**Realised**: parity tests green, loss numerically identical, but the
**gradient produced by both kernels is now device-resident** — and the
downstream `AdamW::step_device` host-side grad ingestion **DtoH-downloads
every gradient** (`store.to_host(grad_id)` at `optim.rs:227-229`). The
net effect is *more* DtoH bytes, not fewer. Same architectural inversion
P3 documented in reverse: pre-fix-host-resident grads were a free
no-op for AdamW; post-fix-device-resident grads force an explicit DtoH.

## Command

```bash
CUDA_HOME=/opt/cuda CARGO_TARGET_DIR=/tmp/arle-target-cuda \
NVCC_CCBIN=g++-14 CC=gcc-14 CXX=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo build --release --bin arle --features cuda

# parity gate (11 tests = 9 prior + 2 new Wave 2a)
cargo test --release -p autograd --features cuda --test test_cuda_lazy_ops

# 5-step throughput
/tmp/arle-target-cuda/release/arle train pretrain \
  --backend cuda \
  --corpus /home/ckl/arle-data/pretrain/corpus.txt \
  --tokenizer /home/ckl/arle-data/models/Qwen3.5-0.8B/tokenizer.json \
  --preset small-25m --model-family qwen35 \
  --steps 5 --batch 2 --seq 512 --grad-accum-steps 16 \
  --lr 3e-4 --log-every 1 --save-every 5 \
  --out /home/ckl/arle-data/benches/wave2a/run

# nsys 1-step profile
nsys profile --output=/home/ckl/arle-data/benches/wave2a-profile/wave2a_step1 \
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
| ARLE commit | post-`fcba268` (Wave 2a staged, not committed) |
| Features | `cli,cuda` |
| Model | Qwen3.5-family `small-25m` preset (V=248070, H=160, L=2, A=5, FFN=320) |
| Params | 40 255 328 (40.26 M) |
| Hyperparams | steps=5, batch=2, seq=512, grad_accum=16 → effective batch 32, tokens/step 16 384 |

## Results

### Parity test (11/11 green)

```
running 11 tests
test cuda_add_broadcast_backward_device_matches_cpu ... ok
test cuda_embedding_backward_device_matches_cpu ... ok
test cuda_mul_scalar_backward_device_matches_cpu ... ok
test cuda_add_into_device_matches_cpu ... ok
test cuda_mean_backward_device_matches_cpu ... ok
test cuda_gather_last_dim_device_lazy_matches_cpu ... ok
test cuda_gather_last_dim_backward_matches_cpu ... ok
test cuda_log_softmax_last_axis_device_lazy_matches_cpu ... ok
test cuda_softmax_last_axis_device_lazy_matches_cpu ... ok
test cuda_matmul_backward_device_matches_cpu ... ok
test cuda_log_softmax_last_axis_backward_matches_cpu ... ok

test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Tolerances:
- `embedding_backward_device`: `atol=1e-4 + rtol=1e-4`. Tested on
  `[B=2, S=512, H=160]` upstream into `[V=248070, H=160]` table. Two
  index distributions: (a) uniform-random (production-ish, most ids
  unique), (b) **deliberate duplicates** (4 target rows × 256 hits each,
  the canonical "`the` appears 800 times" stress case) — the atomicAdd
  correctness gate. Both green; the duplicate-row sum-of-first-column
  cross-check agrees to 1e-3 absolute (host serial-sum vs device
  cross-block atomicAdd reorder).
- `add_broadcast_backward_device`: `atol=1e-5 + rtol=1e-4`. Tested on
  `[B=2, S=512, H=160]` upstream → `[H=160]` reduce. The kernel sums
  1024 elements per output position; `sqrt(1024)*f32_eps ≈ 4e-6` per
  element absorbs the cross-block reorder.

### Throughput (5 steps, drop step 1)

| Step | tok/s | ms/step |
|---|---|---|
| 1 | 169.45 | 96 690 |
| 2 | 175.03 | 93 605 |
| 3 | 175.39 | 93 414 |
| 4 | 173.02 | 94 693 |
| 5 | 173.84 | 94 248 |

**Median tok/s (steps 2-5): 174.40** vs P3.1 baseline 171.28 = **+3.12 tok/s
(+1.82 %)**. Acceptance gate required ≥ 185.0 (+15 tok/s minimum per
the survey's stop-rule). **GATE FAILED.**

### nsys headline metrics (1-step profile, post-checkpoint)

| Metric | P3.1 (baseline) | Wave 2a (mine) | Δ | Verdict |
|---|---|---|---|---|
| DtoH call count | 121 | **3 544** | **+3 423 (+29×)** | **regression** |
| DtoH total bytes | 1 185 MB | **42 691 MB** | **+41.5 GB (+36×)** | **regression** |
| HtoD call count | 166 | 4 240 | +4 074 (+26×) | regression |
| HtoD total bytes | 1 506 MB | 45 365 MB | +43.9 GB (+30×) | regression |
| Max single DtoH | 1016 MB | 1016 MB | 0 (unchanged — still the `[B, S, V]` logits tile) | hold |
| `cuMemcpyDtoHAsync_v2` total time | 0.64 s | 17.77 s | +17.13 s | regression |
| Tok/s median (steps 2-5) | 171.28 | 174.40 | +3.12 (+1.8 %) | **below gate** |

Both P3.1 and Wave 2a `.nsys-rep` available at:
`/home/ckl/arle-data/benches/p3.1-profile/p31_step1.nsys-rep` and
`/home/ckl/arle-data/benches/wave2a-profile/wave2a_step1.nsys-rep`.

## Problems

### #1 (Root cause) — `AdamW::step_device` `to_host(grad_id)` poisons every device-resident gradient

`crates/autograd/src/optim.rs` line 227-229:

```rust
// Grad stays host-side — matmul_backward returns host Vec<f32>.
let grad = store
    .to_host(grad_id)
    .expect("gradient tensor should be readable from the store");
```

The comment "Grad stays host-side — matmul_backward returns host Vec<f32>"
codifies the **assumption that every backward op produces a host
gradient**. That assumption was correct under P3.1 because all weight
gradients in the small-25m hot path landed via host-fallback paths
(`cpu_scatter_add_rows_forward` for embedding, `cpu_add_broadcast` host
reduce loop for biases, the host fallback in `matmul_backward` for the
two linear weights). Each `to_host(grad_id)` resolved to "already on
host, return clone" with zero DtoH.

With Wave 2a, the embedding gradient (`[V=248070, H=160] ≈ 158 MB`)
and every bias gradient (`[H=160] = 0.64 KB`) are now device-resident.
`store.to_host(grad_id)` for the embedding grad triggers a 158 MB
DtoH **per grad-accum step × 16 steps × ~14 params per step ≈ 35 GB
extra DtoH per training step**. Matches the observed 42 GB total.

The `Backend::adamw_step` trait signature itself is the structural
bottleneck:

```rust
fn adamw_step(&self, param: &DeviceHandle, m: &DeviceHandle, v: &DeviceHandle,
              grad: &[f32], ...)
```

`grad: &[f32]` mandates a host slice. There is no `adamw_step_device`
variant that accepts a `DeviceHandle` for the gradient. Until one
exists, **every device-resident gradient pays a full readback** in
`step_device`.

This is the exact architectural inversion P3 documented (in reverse):
- P3: upstream gradient was host → downstream device override never
  fired → no win.
- Wave 2a: upstream gradient is now device → downstream optimizer is
  host-only → forces DtoH → regression.

### #2 — Wave 2 Commit A cannot ship as-specified

The survey doc (commit `fcba268`) ranked `embedding_backward` and
`add_broadcast_backward` as the highest per-step-bytes-moved host ops in
the P3.1 residue. That ranking was correct **in terms of
opportunity-bytes-on-the-table**, but it implicitly assumed the
downstream consumer (AdamW) was already device-aware. The forensic
nsys trace says it isn't.

Wave 2 Commit A as-specified is mechanically correct (kernels parity
green) but **strictly worse** wall-clock than P3.1 baseline because it
trades a free host-loop for an unfree DtoH. The kernels themselves
behave as designed; the bottleneck moved.

### #3 — The 1016 MB max single DtoH is unchanged (still the `[B, S, V]` logits)

Both P3.1 and Wave 2a show the same 1016 MB max single DtoH (the
`[2, 512, 248070] × 4 B` logits tile). Neither Wave 2 Commit A op
touches that tile. The historical attribution from P3.1's "121 memcpys"
was that this single transfer is now the dominant cost; Wave 2a doesn't
move it, and shouldn't be expected to.

## Learnings

- **Per-op device-port wins are coupled by the downstream consumer.**
  A backward op that produces a device-resident gradient is only a
  net win if every downstream consumer (optimizer, grad accumulator,
  cross-step reset) can ingest a device handle. Wave 2 Commit A
  satisfied the **first** half of that contract; the optimizer
  violates the **second** half.
- **The candle survey's per-step-bytes-moved ranking is correct but
  insufficient.** "Bytes that *would* save if all consumers were
  device-aware" ≠ "bytes that *will* save when ported in isolation".
  The next wave-planning iteration must include a downstream-consumer
  audit as a P0 step.
- **Pre-Wave-2 nsys re-profile of P3.1 would have predicted this.**
  P3.1's 121 DtoH calls include the AdamW grad readback path; that
  cost was hidden in "post-fix steady state". A 5-minute `nsys stats`
  inspection of where each of those 121 calls comes from would have
  flagged AdamW's `to_host(grad_id)` as the dominant consumer.
- **The atomicAdd kernel itself is correct under duplicate stress.**
  The parity test deliberately routes 1024 token positions into 4
  vocab slots and checks the host-vs-device sum of the first column
  across those 4 rows — agrees to 1e-3 absolute. The kernel is
  ready to ship the moment AdamW is device-aware.

## Rule

**Before porting a per-op backward to device, verify the downstream
consumer can ingest a device handle.** Concretely: `grep` the trait
signature(s) of every method called on the produced gradient. If any
takes `&[f32]` or equivalent host slice, the device port is a net
regression (forces an explicit readback that the host fallback path
already had for free). The fix is to first add the device-handle
variant of the consumer's trait method, *then* port the producer.

## Next steps (recommended)

1. **STOP Wave 2 Commit A from landing as a perf optimization.** The
   parity kernels + trait methods are correct and reusable; the
   ops-layer dispatch gates can land too. But the wins headline must
   be **rolled back** or relabelled "infrastructure ships, perf gate
   deferred to adamw_step_device".
2. **Wave 2 Commit A-prereq: add `Backend::adamw_step_device`** that
   accepts `grad: &DeviceHandle` and produces device-resident param/m/v
   updates without ever crossing PCIe. The CUDA override reuses the
   existing `adamw_step_f32` NVRTC kernel; the trait default falls back
   to `readback → cpu_adamw_step_in_place → upload` so CPU/Metal
   inherit correct behaviour.
3. **Wave 2 Commit A-rerun**: after the prereq lands, re-run the 5-step
   bench. Expected: tok/s 171 → ~220 (the original survey target),
   memcpy count 121 → ~50, DtoH bytes 1.2 GB → ~250 MB.
4. **Audit other downstream consumers**: `accumulate_grad`,
   `clip_grad_norm`, `grad_norm_compute`, checkpoint save — every
   touch of a gradient tensor.

## Files

| Path | Δ |
|---|---|
| `crates/autograd/src/backend.rs` | +88 (2 new trait methods + doc) |
| `crates/autograd/src/backend_cuda.rs` | +185 (2 method overrides + 2 helper fns) |
| `crates/autograd/src/backend_cuda/kernels.rs` | +9 (2 includes, 2 function names, 2 concat entries) |
| `crates/autograd/src/backend_cuda/kernels/embedding_backward.cu` | +42 (new file, atomicAdd-based scatter) |
| `crates/autograd/src/backend_cuda/kernels/add_broadcast_backward.cu` | +119 (new file, shared-mem reduce along contracted axes) |
| `crates/autograd/src/ops/embed.rs` | +27 (device-path dispatch in `embedding_backward`) |
| `crates/autograd/src/ops/broadcast.rs` | +43 (device-path dispatch in `add_broadcast_backward`) |
| `crates/autograd/tests/test_cuda_lazy_ops.rs` | +140 (2 new parity tests, both with prod shape + duplicates stress) |

No changes to `tape.rs` / `tensor.rs` / `ops/matmul.rs` / `ops/softmax.rs` /
`ops/gather.rs` / `ops/reduce.rs` / `ops/elementwise.rs` (per Wave 2a hard
constraint).
