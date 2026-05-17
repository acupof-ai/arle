# `arle train pretrain` Δ G3 — CUDA M5.3b device-lazy softmax / log_softmax / gather, RTX 4070 Ti SUPER

> **Status: small-win, host-readback chain hypothesis confirmed as
> partial — not the whole picture.**
> Headline: `tok_per_sec` 78.6 → **92.08** (+17.2 % vs baseline, +15.8 %
> vs G3 prerequisite). Parity passes. GPU avg utilization didn't budge
> (12.4 % → 11.17 %). Per-op host-readback removal works, but only on
> the **fraction of ops we ported** — 3 of ~15+ host-bound ops live in
> the per-micro-batch forward+backward chain. Wider port (rope, rmsnorm,
> silu, embedding, mul/add_broadcast, mean, linear_attention) is the
> next required wave.

## Goal (type: optimization)

Wire `Backend::softmax_last_axis`, `Backend::log_softmax_last_axis`,
and `Backend::gather_last_dim` on `CudaBackend` to the existing NVRTC
kernels (`softmax.cu :: softmax_last_axis_f32` and
`log_softmax_last_axis_f32`, `gather.cu :: gather_last_dim_f32`)
**without** going through the default trait fallback's
`readback → compute on host → upload` chain. Once on, the ops/dispatch
layer (`crates/autograd/src/ops/{softmax,gather}.rs`) — which already
detects `Dirty::Device/Both` and calls the lazy device path — activates
automatically.

The hypothesis (carried in from
[`2026-05-17-bench-pretrain-g3-cuda-adamw-step.md`](2026-05-17-bench-pretrain-g3-cuda-adamw-step.md)
§Learnings #4) was that the `[B, S, V] = [2, 512, 248 070] ≈ 1 GB`
materialization in the cross-entropy chain is the biggest single
host-readback in the per-micro-batch hot loop. **If true**, removing
those three ops should yield a measurable tok/s bump and raise GPU
utilization.

## Hypothesis

Concrete pre-bench prediction (recorded for §0 SOLID accounting):

- `tok_per_sec`: 79.5 → 200–500 (3–6× G3) **if** the CE chain was the
  whole show.
- `tok_per_sec`: 79.5 → 100–150 (1.3–1.9× G3) **if** the CE chain was
  one of many host bottlenecks of comparable size.
- `avg utilization.gpu`: 11.96 % → 25–40 % **if** the readback chain
  was the actual hot loop.

## Command

```bash
CUDA_HOME=/opt/cuda CARGO_TARGET_DIR=/tmp/arle-target-cuda \
NVCC_CCBIN=g++-14 CC=gcc-14 CXX=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo build --release -p agent-infer --features cli,cuda --bin arle

# parity gate
cargo test --release -p autograd --features cuda --test test_cuda_lazy_ops
# → 3 tests pass at atol=1e-6 + rtol=1e-4 in ~3.86s

# 10-step bench
/tmp/arle-target-cuda/release/arle train pretrain \
  --backend cuda \
  --corpus /home/ckl/arle-data/pretrain/corpus.txt \
  --tokenizer /home/ckl/arle-data/models/Qwen3.5-0.8B/tokenizer.json \
  --preset small-25m --model-family qwen35 \
  --steps 10 --batch 2 --seq 512 --grad-accum-steps 16 \
  --lr 3e-4 --log-every 1 --save-every 10 \
  --out /home/ckl/arle-data/benches/m53b-device-lazy-ce-v2/run
```

GPU sampler: `nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw
--format=csv,noheader,nounits` every 2 s → `gpu.csv` (886 samples
covering the full 30:00 bench wall).

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER · 16.0 GB · sm_89 |
| CUDA / nvcc | 13.2 V13.2.78 |
| Host compiler | g++-14 (NVCC_CCBIN) |
| Driver | 595.71.05 |
| cudarc | 0.19.7 |
| ARLE commit | post-G3 (last commit: `perf(autograd-cuda): add adamw_step kernel + route CUDA via new_with_device`) |
| Features | `cli,cuda` |
| Model | Qwen3.5-family `small-25m` preset (vocab=248070, hidden=160, layers=2, heads=5, kv_heads=5, head_dim=32, ffn=320, max_pos=512, tie_embed=true) |
| Params | 40 255 328 (40.26 M; vocab dominates) |
| Hyperparams | steps=10, batch=2, seq=512, grad_accum=16 → effective batch 32, tokens/step 16 384 |

## Results

### Parity test

```
running 3 tests
test cuda_gather_last_dim_device_lazy_matches_cpu ... ok
test cuda_log_softmax_last_axis_device_lazy_matches_cpu ... ok
test cuda_softmax_last_axis_device_lazy_matches_cpu ... ok
test result: ok. 3 passed; finished in 3.86s
```

All three at `[B=2, S=512, V=248070] = 254 M element` shape (the
exact production shape), `atol=1e-6 + rtol=1e-4`.

### Real-workload bench (10 steps)

| step | loss | grad_norm | ms/step | tok/s |
|---|---|---|---|---|
| 1 | 12.4375 | 0.7718 | 176 938 | 92.60 |
| 2 | 12.3544 | 0.8979 | 178 210 | 91.94 |
| 3 | 12.2819 | 1.0146 | 176 435 | 92.86 |
| 4 | 12.2052 | 1.1590 | 178 293 | 91.89 |
| 5 | 12.1143 | 1.2302 | 177 923 | 92.08 |
| 6 | 12.0193 | 1.2116 | 178 420 | 91.83 |
| 7 | 11.9344 | 1.2082 | 177 378 | 92.37 |
| 8 | 11.8771 | 1.1806 | 178 093 | 92.00 |
| 9 | 11.8017 | 1.1941 | 177 594 | 92.26 |
| 10 | 11.7312 | 1.2145 | 177 161 | 92.48 |

- **Median tok/s (steps 2–10): 92.08** · range [91.83, 92.86] · std ≈ 0.34
- Wall time: 30:00 for 10 steps → 30:00 × 6 = 3 h projected for 60-step run (compared to baseline projection 11.6 h for 200 steps = ~3.5 h/min)
- Loss trajectory: 12.44 → 11.73 (real learning, not noise)

### GPU sampler (886 samples × 2 s = 30:00 covering full bench wall)

| Metric | M5.3b | Baseline | G3 | Δ vs baseline |
|---|---|---|---|---|
| `peak memory.used` | 6 190 MiB | 5 675 MiB | (sampler stopped early) | +9 % |
| `avg memory.used` | 5 022 MiB | 4 423 MiB | n/a | +13 % |
| `peak utilization.gpu` | 100 % | 100 % | n/a | flat |
| **`avg utilization.gpu`** | **11.17 %** | **12.43 %** | n/a | **−1.26 pp** |

Raw artefacts:
- `/home/ckl/arle-data/benches/m53b-device-lazy-ce-v2/train.log` (full 10-step JSONL-event stream)
- `/home/ckl/arle-data/benches/m53b-device-lazy-ce-v2/gpu.csv` (886 samples)

## Δ vs baseline / vs G3

| Metric | baseline | G3 | M5.3b | Δ vs base | Δ vs G3 |
|---|---|---|---|---|---|
| tok_per_sec (median) | 78.60 | 79.53 | **92.08** | **+17.2 %** | **+15.8 %** |
| ms/step | 208 418 | 206 008 | 177 745 | −14.7 % | −13.7 % |
| avg GPU util | 12.43 % | n/a | 11.17 % | **−1.26 pp** | n/a |
| peak memory | 5 675 MiB | n/a | 6 190 MiB | +9 % | n/a |

## Problems

1. **Avg GPU utilization went the wrong direction.** Hypothesized
   ≥25 % under M5.3b (host readback chain was the dominant cost). Saw
   **11.17 %**, slightly *lower* than the 12.43 % baseline. The 17 %
   tok/s gain came from per-call walltime reduction in the three ops,
   not from saturating the GPU. The dominant cost is still elsewhere.
2. **CE chain ≠ the whole bottleneck.** Per the optimization-roadmap
   table in the baseline doc, this milestone was projected as
   "2–3× → 500-1 200 tok/s". Reality is +17 % → 92 tok/s. Same SOLID
   pattern as G3: the ranking was attribution-inferred, not nsys
   measured. The CE-chain-as-#1 hypothesis is **partially confirmed**
   (real positive Δ, not zero) but **falsified as a single-step-fix**.
3. **Per-step wall is still ~178 s.** At `grad_accum=16`,
   per-micro-batch wall ≈ 11.1 s. The remaining host-readback ops in
   the autograd graph (the ~12 we did NOT port: `rope`, `rmsnorm`,
   `silu`, `mul`, `mul_scalar`, `embedding`, `mean`,
   `linear_attention`, `add_broadcast`, `exp`, `neg`,
   `scatter_add_rows`) each force an `ensure_host` per call. With
   each forward+backward visiting most of them per layer × 2 layers ×
   16 micro-batches, the cumulative chain is still huge.
4. **Memory grew by 9 %.** Expected — intermediates that previously
   round-tripped through host are now device-resident across op
   boundaries. Not a regression at our batch size; at `batch=8` this
   would have helped fit, but the OOM there is dominated by the
   `[B,S,V]` materialization itself (Liger FusedLinearCE territory),
   not the intermediate-tensor count.

## Learnings

1. **"Port one op at a time, measure each, batch-port the rest"
   is the right protocol when the bottleneck is distributed.**
   Removing 3 of ~15 host ops → +17 %. Linear extrapolation says
   porting all 15 ≈ +85 % (cumulative tok/s ≈ 145). That's still 8×
   below industry baseline (≈ 1 200 tok/s on this hardware for a
   ~40 M model). So device-lazy port is **necessary but not
   sufficient**. The next big lever must be structural (FusedLinearCE,
   bf16, CUDA graph, or kernel fusion across consecutive ops), not
   just "port one more op".
2. **GPU utilization is the more reliable bottleneck signal than
   `tok/s` deltas.** A +17 % tok/s with utilization *falling* tells
   us the work that ran faster wasn't the work that pinned the GPU
   high — i.e., the host-bound regime is still firmly in place. The
   next round's acceptance gate should weight `avg utilization.gpu`
   as heavily as `tok_per_sec`.
3. **Cross-cuts to consider before the next port wave**:
   - `linear_attention` (Mamba2 SSM scan) is suspiciously absent from
     this entry's data. With seq=512 and a per-step scan, it's a
     plausible top-3 cost. Profile it first before porting blindly.
   - Most of the remaining host ops are pointwise/reduction kernels
     that already have CUDA implementations under
     `crates/autograd/src/backend_cuda/kernels/` for the `*_forward`
     host-host path. Wiring lazy-device for them is mechanical, same
     pattern as this milestone.
4. **Parity tolerance: re-using the AdamW gate (`atol=1e-6 +
   rtol=1e-4`) worked perfectly for softmax and log_softmax.** The
   GPU `__expf`/`__logf` intrinsics differ from libm by ~1 ULP near
   zero, which the absolute floor catches. No tuning needed.

## Rule

**Optimization-ranking tables generated from a callgraph survey are
hypotheses, not evidence.** This is now the second consecutive
milestone (G3, M5.3b) where the projected magnitude was wrong
(over-optimistic by 3–10×). The right pattern going forward: for each
optimization, pre-record (a) the headline metric, (b) the predicted
Δ%, (c) the SOLID gap if Δ doesn't materialize. When reality
disagrees, retire the original ranking and re-baseline before
proposing the next move.

## Files changed (this commit, on top of HEAD)

1. `crates/autograd/src/backend_cuda.rs` — adds three `Backend` trait
   overrides:
   - `fn softmax_last_axis(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle>`
   - `fn log_softmax_last_axis(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle>`
   - `fn gather_last_dim(&self, src: &DeviceHandle, src_shape: &[usize], indices: &[i32]) -> Result<DeviceHandle>`

   All three reuse the existing kernel-cache module (`softmax_last_axis_f32` /
   `log_softmax_last_axis_f32` / `gather_last_dim_f32`) and return
   unevaluated handles per the M5.3b.11 batched-eval contract. No
   readback inside the override path.

2. `crates/autograd/tests/test_cuda_lazy_ops.rs` — new parity test
   file. Three tests, each on the `[B=2, S=512, V=248070]` production
   shape, gated `#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]`,
   tolerance `atol=1e-6 + rtol=1e-4`.

No dispatch changes in `crates/autograd/src/ops/`. The lazy device path
that was already there (introduced in M5.3a) now activates on CUDA the
same way it does on Metal.

## Next attack (cited from this entry)

In ranked order, **after** the lesson that single-op-Δ is small:

1. **Batch-port remaining host ops** (rope, rmsnorm, silu, mul,
   mul_scalar, embedding, mean, add_broadcast, exp, neg) as one
   coordinated milestone. Each adds an override + parity test; the
   cumulative bench Δ goes in a single wins entry. Expected order of
   magnitude: +50–100 % (median around 140–180 tok/s).
2. **Profile `linear_attention` separately.** Mamba2 SSM scan on
   seq=512 with hybrid Qwen3.5 layers is plausibly a single op that
   dominates wall time per micro-batch. If so, it needs its own kernel
   port + state-on-device, not just a generic lazy-device wire.
3. **Then FusedLinearCE (Liger-style).** Once the per-op chain is
   device-resident, the next structural win is avoiding the
   `[B, S, V]` materialization entirely. This unblocks `batch ≥ 8`
   (which currently OOMs) AND reduces compute on the LM head.
4. **bf16 throughout the activations + grads (master fp32 kept)** —
   2× memory + 1.5–2× compute on `sm_89`. Applies after the chain is
   on device (host bf16 conversion would itself be a per-op cost).
5. **CUDA-graph capture for the training step** — only worth doing once
   shapes are fixed (i.e., after packed-seq). Kills launch overhead at
   the very end of the optimization sequence.

The target stays **industry baseline × 1.3** on the same hardware
(estimate ≥ 15 000 tok/s based on nanochat scaling-law tier × single-
GPU efficiency on 4070 Ti SUPER). Current: 92 tok/s. Gap: **163×**.
The roadmap above is necessary but not sufficient — at some point we
will need a re-evaluation of the autograd contract itself
(e.g., per-op-eager vs whole-graph-lazy), but every step on the
current path narrows the gap and pins one more attribution claim.
