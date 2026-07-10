# `arle train pretrain` Wave 1 — CUDA device-lazy `log_softmax` / `gather_last_dim` backward, RTX 4070 Ti SUPER

> **Status: infrastructure-only.** Trait + kernel + parity gate
> landed clean. Production hot path **does not yet activate** the new
> device backward — upstream-of-`log_softmax` grad is still produced
> host-side by `mean_backward` / `mul_scalar_backward` (Wave 2 work).
> `tok_per_sec` median **91.41** (M5.3b 92.08; **−0.7 %**, within
> step-to-step noise). nsys max single DtoH **1016.1 MB** — the
> targeted 1 GB transfer is **unchanged**. Real wall-clock win blocked
> on Wave 2.

## Goal (type: optimization, infrastructure)

Per
[`docs/research/2026-05-17-cuda-training-step-nsys-attribution.md`](../../research/2026-05-17-cuda-training-step-nsys-attribution.md)
Wave 1: port the backward of `log_softmax_last_axis` and
`gather_last_dim` to device-lazy on CUDA so the saved
`[B=2, S=512, V=248070] = 1 015 MB` log_softmax output and the
`[B, S, V]` scatter-add grad never round-trip through host.

## Hypothesis (recorded for SOLID accounting)

Per the research doc prediction: porting the two backwards saves the
single largest readback (~1 GB / step), worth ~3-5 % wall-clock
(~5 s / step at the 178 s baseline). **Realised outcome: 0 %** wall
improvement. The gap explanation is in §Problems #1.

## Command

```bash
CUDA_HOME=/opt/cuda CARGO_TARGET_DIR=/tmp/arle-target-cuda \
NVCC_CCBIN=g++-14 CC=gcc-14 CXX=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo build --release -p agent-infer --features cli,cuda --bin arle

# parity gate (5 tests = 3 forward M5.3b + 2 new backward)
cargo test --release -p autograd --features cuda --test test_cuda_lazy_ops

# nsys 1-step profile
CUDA_HOME=/opt/cuda nsys profile \
  --output=/home/ckl/arle-data/benches/wave1-profile/wave1_step1 \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true \
  /tmp/arle-target-cuda/release/arle train pretrain \
    --backend cuda --corpus … --tokenizer … --preset small-25m \
    --model-family qwen35 --steps 1 --batch 2 --seq 512 \
    --grad-accum-steps 16 --lr 3e-4 \
    --out /home/ckl/arle-data/benches/wave1-profile/run

# 5-step throughput bench
/tmp/arle-target-cuda/release/arle train pretrain \
  --backend cuda --corpus … --tokenizer … --preset small-25m \
  --model-family qwen35 --steps 5 --batch 2 --seq 512 \
  --grad-accum-steps 16 --lr 3e-4 \
  --out /home/ckl/arle-data/benches/wave1-throughput/run
```

GPU sampler: `nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw
--format=csv,noheader,nounits` every 2 s → `gpu.csv` (607 samples
covering the full 20:13 bench wall).

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER · 16.0 GB · sm_89 |
| CUDA / nvcc | 13.2 V13.2.78 |
| Host compiler | g++-14 (NVCC_CCBIN) |
| Driver | 595.71.05 |
| cudarc | 0.19.7 |
| ARLE commit | post-M5.3b (last commit on `main`: `docs(experience): record p3.6 ncu profiling blocker`) |
| Features | `cli,cuda` |
| Model | Qwen3.5-family `small-25m` preset (vocab=248070, hidden=160, layers=2, heads=5, kv_heads=5, head_dim=32, ffn=320, max_pos=512, tie_embed=true) |
| Params | 40 255 328 (40.26 M) |
| Hyperparams | steps=5, batch=2, seq=512, grad_accum=16 → effective batch 32, tokens/step 16 384 |

## Results

### Parity test (5 tests, all green)

```
running 5 tests
test cuda_gather_last_dim_backward_matches_cpu ... ok
test cuda_gather_last_dim_device_lazy_matches_cpu ... ok
test cuda_log_softmax_last_axis_device_lazy_matches_cpu ... ok
test cuda_softmax_last_axis_device_lazy_matches_cpu ... ok
test cuda_log_softmax_last_axis_backward_matches_cpu ... ok
test result: ok. 5 passed; 0 failed; finished in 5.56s
```

Both new backward parity tests run on the exact production shape
`[B=2, S=512, V=248070]` (~`254 M` elements). `gather_last_dim_backward`
holds the strict `atol=1e-6 + rtol=1e-4` gate (it's a point-write —
no accumulation). `log_softmax_last_axis_backward` uses
**`atol=1e-5 + rtol=1e-4`** (10× looser absolute floor): the kernel
sums upstream across `vocab = 248 070` per row (`__fadd` rounding
bounded by `sqrt(vocab) × f32_eps ≈ 5e-5`), then multiplies by
`__expf(saved_output)` (~1-2 ULP gap vs `expf`). At small grad
magnitudes that cancellation hits `~4.7e-6` worst-case absolute diff —
well within 1e-5 atol, and far below the 1e-4 rtol practical bound.

### Real-workload bench (5 steps)

| step | loss | grad_norm | ms/step | tok/s |
|---|---|---|---|---|
| 1 | 12.4375 | 0.7718 | 179 016 | 91.52 |
| 2 | 12.3544 | 0.8979 | 178 944 | 91.56 |
| 3 | 12.2819 | 1.0146 | 178 637 | 91.72 |
| 4 | 12.2052 | 1.1590 | 179 523 | 91.26 |
| 5 | 12.1143 | 1.2301 | 179 989 | 91.03 |

- **Median tok/s (steps 2–5, dropping step 1 warmup): 91.41** ·
  range [91.03, 91.72] · std ≈ 0.30
- Loss trajectory: 12.44 → 12.11 — real learning, byte-identical
  through step 5 to the M5.3b bench (same seed, same data) modulo
  fp drift from `__expf` rounding — confirms zero correctness
  regression.

### nsys API delta (1-step profile, post-Wave-1 vs pre-Wave-1)

Captured against `wave1_step1.nsys-rep` with
`nsys stats --report cuda_api_sum / cuda_gpu_mem_size_sum`. Pre-Wave-1
numbers from
[`docs/research/2026-05-17-cuda-training-step-nsys-attribution.md`](../../research/2026-05-17-cuda-training-step-nsys-attribution.md).

| Metric | Pre-Wave-1 | Post-Wave-1 | Δ |
|---|---:|---:|---:|
| DtoH bytes / step | 93.2 GB | 93.2 GB | **0** |
| DtoH calls / step | 4 760 | 4 760 | **0** |
| DtoH avg / call | 19.6 MB | 19.6 MB | 0 |
| **DtoH max single transfer** | **1 015 MB** | **1 016 MB** | **+0.1 %** |
| HtoD bytes / step | 45.4 GB | 45.4 GB | 0 |
| HtoD calls / step | 4 272 | 4 272 | 0 |
| HtoD max single transfer | 1 016 MB | 1 016 MB | 0 |
| DtoD bytes / step | 0.48 GB | 0.48 GB | 0 |

**No movement on any memcpy axis.** The 1 GB single DtoH that Wave 1
targeted is **still there**. See §Problems #1 for the SOLID gap.

### GPU sampler (607 samples × 2 s = 20:13)

| Metric | Wave 1 | M5.3b | Δ |
|---|---|---|---|
| `peak memory.used` | 6 190 MiB | 6 190 MiB | flat |
| `avg memory.used` | 3 879 MiB | 5 022 MiB | −23 % |
| `peak utilization.gpu` | 100 % | 100 % | flat |
| **`avg utilization.gpu`** | **8.08 %** | **11.17 %** | **−3.1 pp** |

(`avg memory.used` and `avg util` are sample-period-dependent; this
bench was 5 steps × ~3 min vs M5.3b's 10 steps × ~3 min, and the
sampler covers checkpoint-save wall too. The directional read is
"flat" — not a regression worth worrying about, the 5-step bench
just has more checkpoint-write overhead per training-step sample.)

Raw artefacts:
- `/home/ckl/arle-data/benches/wave1-profile/wave1_step1.nsys-rep`
- `/home/ckl/arle-data/benches/wave1-throughput/train.log` (5-step JSONL-event stream)
- `/home/ckl/arle-data/benches/wave1-throughput/gpu.csv` (607 samples)

## Δ vs M5.3b

| Metric | M5.3b | Wave 1 | Δ |
|---|---|---|---|
| tok_per_sec (median) | 92.08 | 91.41 | **−0.7 %** (noise) |
| ms/step | 177 745 | 179 222 | +0.8 % |
| DtoH max single | 1 015 MB | 1 016 MB | 0 |
| nsys memcpy count | 9 032 | 9 032 | 0 |

## Problems

1. **The trait + kernel infrastructure landed but the production
   chain doesn't activate it.** Both new dispatchers
   (`ops::softmax::log_softmax_backward` and
   `ops::gather::gather_last_dim_backward`) gate the device path on
   *upstream* being `Dirty::Device`. The CE chain backward order is:

   ```
   loss → mul_scalar_backward → mean_backward → gather_backward →
        log_softmax_backward → matmul_backward → …
   ```

   `mul_scalar_backward` and `mean_backward` are still host-only
   (they live in §Wave 2 — `rms_norm`, `rope`, `silu`, `mul`,
   `mul_scalar`, `mean`, … per the nsys research doc). They
   produce `Dirty::Host` upstream gradients. So when
   `gather_backward` and `log_softmax_backward` check
   `upstream_on_device`, the answer is **always false**, and both
   fall back to the existing host path. Net: the new CUDA kernels
   are dead code in the production workload, and the 1 GB DtoH
   from the pre-backward `flush_to_host_batch` (which flushes the
   saved log_softmax output) is unchanged.

2. **The research-doc Wave-1 prediction missed two extra
   dependencies.** The doc claimed Wave 1 alone would save ~5 s
   per step (~3 % wall). For that to be true, *all* of these must
   also hold:
   - `mean_backward` / `mul_scalar_backward` must produce
     device-resident grads (so `gather_backward`'s `upstream_on_device`
     fires).
   - `clone_tensor` in `merge_grad` must preserve device handles
     (so the device grad from `gather_backward` flows into
     `log_softmax_backward` without `ensure_host`). **This part is
     landed in this commit** (see `tensor.rs::clone_tensor`) but is
     inert without #2a.
   - `matmul_backward` must accept device-resident upstream (today
     it calls `store.tensor(id).clone()`, which panics on
     `Dirty::Device`). Without device-aware matmul backward, the
     log_softmax grad would have to be `ensure_host`'d on the way
     in — restoring the 1 GB DtoH at a different call site, net
     zero.

   **Wave 1 in isolation cannot kill the 1 GB DtoH.** This is a
   discovery — the research doc's per-wave attribution was
   optimistic about op coupling. Per `CLAUDE.md` §0 first-principle
   SOLID, this gap is recorded here, not silently papered over with
   forced uploads.

3. **Acceptance gate "DtoH max single < 100 MB" is unmet.**
   Headline number is 1016 MB, identical to pre-Wave-1.

## Learnings

1. **License-or-kill applies to research-doc ranking, not just
   plans.** The nsys attribution doc gave a clean "Wave 1 = 1 GB
   DtoH kill = 3-5 % wall" prediction; reality shows the kill
   requires Waves 1 + 2 + a `merge_grad` refactor + a
   `matmul_backward` device path. The error mode is the same as G3
   and M5.3b (over-optimistic single-step attribution); the SOLID
   correction is "next time the research doc declares a wave, also
   list the implied prerequisites in adjacent waves".
2. **Infrastructure work is still worth shipping.** The two new
   CUDA kernels + trait methods + parity gate + device-preserving
   `clone_tensor` are necessary preconditions for any Wave 2 / 3
   to actually move the number. Reverting just to avoid the
   "wins entry shows no Δ" framing would force re-doing the
   identical mechanical work later. Better: land it, label it
   infrastructure, point Wave 2 at the activation work.
3. **The tape's pre-backward `flush_to_host_batch` is the actual
   single largest DtoH source for the saved log_softmax output.**
   Skipping that flush requires every consumer of the saved tensor
   (today only `log_softmax_backward`) to use a device-aware path.
   I tried adding a per-op skip filter in `tape::backward` during
   development, then reverted: with `mean_backward` still producing
   host upstream, the device path can't fire, and the skip just
   shifts the DtoH from pre-flush to `merge_grad`'s `clone_tensor`.
   The right place for the flush skip is once Wave 2 makes the
   upstream device-resident.
4. **GPU `__expf` + sum-reduce drift on `vocab = 248 070` is real
   but small** — worst-case absolute diff ~4.7e-6 against
   `cpu_log_softmax_backward`. The strict AdamW gate
   (`atol=1e-6 + rtol=1e-4`) false-positives on small-magnitude
   grads where the cancellation `upstream - exp(y) * sum` lives.
   `atol=1e-5` is the right floor for this op family
   (`torch.allclose` defaults are `atol=1e-8 + rtol=1e-5` for fp64
   and `atol=1e-6 + rtol=1e-4` for fp32 — but the latter assumes
   no `vocab`-wide reduction; PyTorch's own `log_softmax_backward`
   tests use `atol=1e-4` at this shape).

## Rule

**An optimization wave only counts when its dispatch fires in the
production workload.** Shipping kernels + parity tests + trait
methods is necessary infrastructure but not a wall-clock win until
the upstream of the targeted dispatch is also producing whatever the
device path requires (here: a `Dirty::Device` upstream gradient).
Future Wave entries must include a "dispatch activation check" —
either a unit-test that exercises the device path end-to-end through
the autograd tape, or a nsys delta on the targeted DtoH count — and
withhold the wins-entry headline until that gate is green.

## Files changed (this commit, on top of post-M5.3b `main`)

1. `crates/autograd/src/backend.rs` — adds two `Backend` trait
   methods + their CPU reference helpers:
   - `fn log_softmax_last_axis_backward(&self, upstream: &DeviceHandle, log_softmax_output: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle>` (default: `readback → cpu_log_softmax_backward → upload`)
   - `fn gather_last_dim_backward(&self, upstream: &DeviceHandle, indices: &[i32], src_shape: &[usize]) -> Result<DeviceHandle>` (default: `readback → cpu_gather_last_dim_backward → upload`)
   - `pub fn cpu_log_softmax_backward(...)` and `pub fn cpu_gather_last_dim_backward(...)` reference helpers.
2. `crates/autograd/src/backend_cuda.rs` — CUDA overrides:
   - `cuda_log_softmax_last_axis_backward` (single `launch_rows`, 256-thread shared-mem sum reduce, no `synchronize` — caller owns terminal eval).
   - `cuda_gather_last_dim_backward` (single `launch_1d`, one thread per prefix row, `alloc_zeros` + scatter, no `synchronize`).
3. `crates/autograd/src/backend_cuda/kernels.rs` — register two
   new function names + concat their `.cu` sources.
4. `crates/autograd/src/backend_cuda/kernels/log_softmax_backward.cu`
   — new kernel.
5. `crates/autograd/src/backend_cuda/kernels/gather_backward.cu`
   — new kernel.
6. `crates/autograd/src/ops/softmax.rs` —
   `log_softmax_backward` now dispatches on
   `saved.dirty == Dirty::Device && upstream.dirty == Dirty::Device`:
   device path calls the trait method, returns `Dirty::Device` grad
   via `alloc_device_tensor`; host path unchanged.
7. `crates/autograd/src/ops/gather.rs` —
   `gather_last_dim_backward` now dispatches on
   `upstream.dirty == Dirty::Device`: device path calls the trait
   method, returns `Dirty::Device` grad; host path unchanged.
8. `crates/autograd/src/tensor.rs` —
   `TensorStore::clone_tensor` preserves the `DeviceHandle` Arc on
   `Dirty::Device` tensors instead of asserting through the
   panic-on-clone gate. Required for the device-aware backward
   chain to flow grads through `merge_grad` without a forced
   `ensure_host`. Inert today (no dispatcher activates it in the
   production workload) but a prerequisite for Wave 2.
9. `crates/autograd/tests/test_cuda_lazy_ops.rs` — two new parity
   tests at `[B=2, S=512, V=248 070]` shape:
   - `cuda_log_softmax_last_axis_backward_matches_cpu` (atol=1e-5)
   - `cuda_gather_last_dim_backward_matches_cpu` (atol=1e-6)
   - shared `max_err_with_tol` helper to parameterise atol/rtol.

No changes to `crates/autograd/src/tape.rs` — the pre-backward
`flush_to_host_batch` is unchanged. A skip filter for `LogSoftmax`
/ `Gather` entries was prototyped during development and reverted
once the activation gap was understood (see §Problems #1).

## Next attack (cited from this entry)

In strict prerequisite order:

1. **Wave 2 prologue — make `mean_backward` and
   `mul_scalar_backward` device-aware.** Both are tiny pointwise /
   broadcast ops. Once their backwards produce `Dirty::Device`
   grads, the `gather_backward` and `log_softmax_backward`
   dispatchers committed in this entry **activate**, and the 1 GB
   pre-flush DtoH disappears for real.
2. **Device-aware `matmul_backward`.** Either via a new
   `Backend::matmul_backward_device` trait method (taking
   `DeviceHandle` for upstream + saved a/b) or by extending the
   existing `matmul_backward` to accept device-resident upstream
   and route through `cuBLAS sgemm_strided_batched` directly. Until
   matmul backward consumes device upstream, the `[B, S, V]`
   log_softmax grad has to be `ensure_host`'d on entry — no real
   wall improvement.
3. **`tape::backward` pre-flush skip filter.** Re-add the
   `BackwardOp::LogSoftmax` / `BackwardOp::Gather` skip in
   `flush_to_host_batch`'s filter set once #1 + #2 are in place.
   At that point the saved log_softmax output truly stays
   device-resident across backward.
4. **Wave 2 main — batch-port the remaining ~10 host ops** per the
   research doc.
