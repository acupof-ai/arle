# `arle train pretrain` Δ baseline — CUDA `Backend::adamw_step` override, RTX 4070 Ti SUPER

> **Status: KILL on acceptance gate (78.5 tok/s vs. ≥ 200 target).**
> Code change shipped + parity-validated, but the headline gain did
> not materialize because AdamW PCIe was *not* the dominant
> per-step cost on this stack. Baseline's "3–5× from AdamW" projection
> was a hypothesis on inferred attribution, not nsys-measured. See
> §Problems and §Learnings for the SOLID gap that retires.

## Goal (type: optimization)

Close the CUDA AdamW PCIe loop: replace the default
`Backend::adamw_step` host-readback fallback
(`readback × 3 + cpu_adamw_step_in_place + upload × 3` per parameter
per step) with a single fused NVRTC kernel launch
(`backend_cuda/kernels/adamw.cu :: adamw_step_f32`), and route the
CUDA backend through `AdamW::new_with_device` (matching Metal).

Target: ≥ 200 tok/s on `--preset small-25m --model-family qwen35
--batch 2 --seq 512 --grad-accum-steps 16` (≥ 2.5× the 78.6 tok/s
baseline).

## Hypothesis

From [`2026-05-17-bench-pretrain-qwen35-25m-cuda-baseline.md`](2026-05-17-bench-pretrain-qwen35-25m-cuda-baseline.md):
"`Backend::optim_adamw_step` + device-resident grads → 3–5×, projected
250–400 tok/s." The expectation was that AdamW's per-parameter PCIe
roundtrip was the dominant pipeline cost. ~200 trainable params × 3
readback + 3 upload + host compute per step at 40 M total params should
have charged measurable wall time per optimizer step.

## Command

```bash
CUDA_HOME=/opt/cuda CARGO_TARGET_DIR=/tmp/arle-target-cuda \
NVCC_CCBIN=g++-14 CC=gcc-14 CXX=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo build --release -p agent-infer --features cli,cuda --bin arle

/tmp/arle-target-cuda/release/arle train pretrain \
  --backend cuda \
  --corpus /home/ckl/arle-data/pretrain/corpus.txt \
  --tokenizer /home/ckl/arle-data/models/Qwen3.5-0.8B/tokenizer.json \
  --preset small-25m --model-family qwen35 \
  --steps 20 --batch 2 --seq 512 --grad-accum-steps 16 \
  --lr 3e-4 --log-every 5 --save-every 20 \
  --out /home/ckl/arle-data/benches/g3-adamw-device/run
```

GPU sampler: `nvidia-smi --query-gpu=memory.used,utilization.gpu
--format=csv,noheader,nounits` every 2 s → `gpu.csv`.

Smoke parity test (separate command, gated to CUDA):

```bash
cargo test --release -p autograd --features cuda --test test_cuda_adamw_step
```

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER · 16.0 GB · sm_89 |
| CUDA / nvcc | 13.2 V13.2.78 |
| Host compiler | g++-14 (nvcc 13.2 + GCC 16 stdlib mismatch; pin via `NVCC_CCBIN=g++-14`) |
| Driver | 595.71.05 (CUDA 13.2 runtime) |
| OS / kernel | Linux 7.0.3-1-cachyos |
| CPU / RAM | AMD Ryzen 7 3700X 8C16T · 31.3 GB |
| Rust toolchain | 1.95.0 stable |
| cudarc | 0.19.7 |
| ARLE commit | `bccabb4` (`docs(experience): record p3.6 ncu profiling blocker`), working tree dirty with this change |
| Features | `cli,cuda` |
| Model | Qwen3.5-family `small-25m` preset (vocab=248070, hidden=160, layers=2, heads=5, kv_heads=5, head_dim=32, ffn=320, max_pos=512, tie_embed=true) |
| Params | 40 255 328 (40.26 M) |
| Hyperparams | steps=20, batch=2, seq=512, grad_accum=16 (effective batch 32, tokens/step 16 384), lr=3e-4 cosine, AdamW (device path, fused NVRTC kernel) |

## Results

### Parity test

```
running 1 test
test cuda_adamw_step_matches_cpu_5_steps ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
finished in 0.22 s
```

`[hidden=128, batch=64] = 8192` elements × 5 sequential AdamW steps,
LR=3e-4, β=(0.9, 0.95), wd=0.01. Combined tolerance gate `atol=1e-6
+ rtol=1e-4` (industry standard, matches `torch.allclose`); pure
1e-4-relative failed only on EMA values ≈ 3e-7 where |dev−host| ≈
7.6e-11 = FMA-contraction rounding noise. Real divergence on
parameter-magnitude values was undetectable (all under the 1e-6
absolute floor).

### Real-workload bench

| Metric | Value | Baseline | Δ |
|---|---|---|---|
| `tok_per_sec` (step 1, warmup) | **79.53** | 78.6 | **+1.2%** |
| `ms_per_step` (step 1) | 206 008 ms (206 s) | 208 418 ms | −1.2% |
| `loss` | 12.437 | 12.437 | parity |
| `grad_norm` | 0.772 | 0.772 | parity |
| `peak memory.used` | (sampler stopped early — n/a) | 5 675 MiB | — |
| `avg utilization.gpu` | (sampler stopped early — n/a) | 12.4 % | — |

Cross-check on a faster shape (`--grad-accum 1 --log-every 1
--steps 2`):

| Step | ms/step | tok/s |
|---|---|---|
| 1 | 13 593 | 75.33 |
| 2 | 13 271 | 77.16 |

→ `tok/s` is held essentially constant at ~78 regardless of
`grad_accum_steps`. Wall scales linearly with `grad_accum_steps`
(13.3 s/micro-step × 16 ≈ 213 s/step, matches the observed 206 s).
**The optimizer step itself is dwarfed by the 16 host-readback-heavy
forward+backward micro-batches.**

Raw artefacts:
- `/home/ckl/arle-data/benches/g3-adamw-device/train.log`
- `/home/ckl/arle-data/benches/g3-adamw-device/gpu.csv`

## Problems

1. **Acceptance gate missed by ~2.5×.** Target ≥ 200 tok/s; achieved
   ~79 tok/s. The change is correct (parity test green) but the
   speedup did not materialize. Per
   [CLAUDE.md §0 SOLID](../../../CLAUDE.md): the change to the
   bench number is **+1.2%**, statistically zero on a single
   non-warmup step measurement. No regression but also no win.
2. **Root-cause hypothesis falsified.** Baseline document attributed
   3–5× headroom to AdamW PCIe, but that attribution was
   *callgraph-inferred*, not nsys-measured. A per-step wall budget
   of 206 s with grad_accum=16 implies ~13 s per forward+backward
   micro-batch; the AdamW optimizer call fires **once** per step. Even
   if it cost a full 1 s (~200 params × 200 KB avg payload × PCIe @
   ~16 GB/s ≈ ~5 ms one-way), that's 0.5 % of step wall, not 50–80 %.
   The dominant cost is the 16 × per-micro-batch host readbacks in
   forward (`rope`, `rmsnorm`, `softmax`, `log_softmax`,
   `cross_entropy`, `gather`, `add_broadcast`,
   `linear_attention` host paths in `crates/autograd/src/ops/`).
3. **`metal_adamw_step_stays_device_resident` style end-to-end
   eval-count test is not yet ported to CUDA.** Parity is covered, but
   there is no assertion that the device path runs in `step_device`
   (as opposed to silently falling back to `step_host`) inside the
   real `AdamW::step` dispatch. On Metal this is asserted by reading
   the lazy-graph eval counter; CUDA has no equivalent host-visible
   counter, so a counter-style assertion would need a custom
   instrumentation hook. Deferred.
4. **GPU sampler stopped logging mid-run.** `gpu.csv` rows = 399 (~13
   min) but bench wall ≈ 14 min. The until-poll loop killed the
   sampler when it noticed `arle train` had exited; this happened
   before the final post-step idle window was captured. Not a code
   bug — the bench wrapper script needs a more defensive teardown.
   Documenting for next iteration.

## Δ vs baseline (cite 78.6 tok/s)

- `tok/s`: 78.6 → 79.5, **+1.2 %** (within run-to-run noise).
- `ms/step`: 208 418 → 206 008, **−1.2 %**.
- `loss / grad_norm` parity preserved (same seed → same numbers).

**Net**: code change is a no-op on the headline metric **on this
stack version**. Numerical correctness preserved; the planned 2.5–5×
gain went elsewhere (or never existed for this configuration).

## Learnings

1. **Attribution without measurement is hypothesis, not evidence.**
   The baseline doc's optimization-ranking table was a useful planning
   artifact but its 3–5× gain estimate for AdamW was not nsys-backed.
   When an "expected win" doesn't show up under controlled A/B, the
   first move is to interrogate the original attribution, not the
   implementation. §0 SOLID: 80 % is not enough — this entry retires
   the AdamW-as-#1 claim with an explicit kill.
2. **`tok/s` is grad-accum-invariant in a host-bound regime.** At
   `grad_accum=1` we measured 75.3 → 77.2 tok/s; at `grad_accum=16`
   we measured 79.5 tok/s. Constant ≈ 78 tok/s says the bottleneck is
   inside the *micro-batch* (forward + backward + ops with
   `ensure_host`), not the optimizer step or grad accumulation
   boundary. Future "AdamW PCIe" hypotheses on this stack must
   condition on this.
3. **Combined `atol + rtol` tolerance is mandatory for fused-FMA
   kernels.** Pure relative-error gates fail on values near zero where
   nvcc's FMA contraction produces ~1 ULP drift vs the host's
   separate mul-add. Industry standard
   (`torch.allclose(atol=1e-6, rtol=1e-4)`) is the right primitive.
4. **The next CUDA-training optimization must target the
   per-micro-batch host-readback chain, not the per-step optimizer.**
   Specifically (in priority order based on this falsification):
   - Device-resident `cross_entropy` / `log_softmax` / `gather`
     (kills the `[B, S, V] = 8 × 512 × 248070 × 4 B = 4 GB` host
     materialization)
   - Device-resident `rope` / `rmsnorm` / `add_broadcast` /
     `linear_attention` (the `crates/autograd/src/ops/*.rs`
     `ensure_host` chain is the actual hot loop)
   - FusedLinearCE (Liger-style, both unblocks larger batches and
     skips the `[B,S,V]` materialization).
   The AdamW kernel that this entry ships **is still a structural
   prerequisite** — once the per-micro-batch readbacks are killed,
   AdamW would become the next bottleneck via Amdahl. Shipping it
   now means future optimizations won't be measuring on a stale
   `step_device → readback → host AdamW → upload` fallback.
5. **Code shipped, gain deferred.** The intent of the brief was 2.5×.
   Outcome is correctness + structural prep. Reasonable engineering
   choice to keep the diff: it eliminates a known PCIe loop, passes
   parity, and clears the way for the real optimization (host-op
   removal). The wins entry must say so explicitly rather than
   marketing it as a win.

## Rule

When the baseline document lists an "expected gain" for an
optimization but the gain is attribution-inferred (not
nsys-measured), the implementer's first acceptance check is **wall
clock at the headline metric, on the exact production shape, A/B
against the baseline snapshot**. If the gain is < 2× of run-to-run
noise (here: ~2 %), treat the entry as a **prerequisite ship**, not a
win, and reframe the optimization-ranking table to retire the killed
hypothesis. Do not amplify a +1 % bench result into "win" language —
that pollutes the experience cache for the next implementer.

## Files changed (this commit)

1. `crates/autograd/src/backend_cuda/kernels/adamw.cu` — new fused
   per-element AdamW kernel (decoupled weight decay + EMA updates +
   bias-corrected step), parity-checked against
   `cpu_adamw_step_in_place`.
2. `crates/autograd/src/backend_cuda/kernels.rs` — register
   `ADAMW_CU` in the NVRTC concat + add `"adamw_step_f32"` to
   `FUNCTION_NAMES` (compiled into the same `KernelCache` module —
   no recompile per call, one PTX per `CudaBackend` lifetime).
3. `crates/autograd/src/backend_cuda.rs` — `CudaBackend::adamw_step`
   override + `cuda_adamw_step` helper (dtod-seed fresh
   param/m/v slices, htod-upload grad, single `launch_1d`,
   return unevaluated handles per the M5.3b.11 batched-eval contract).
4. `crates/train/src/cli_args.rs` — route `Device::Cuda` through
   `AdamW::new_with_device` (was `AdamW::new` host-only).
5. `crates/autograd/tests/test_cuda_adamw_step.rs` — new parity test,
   5-step chain, `atol=1e-6 + rtol=1e-4` tolerance gate.
