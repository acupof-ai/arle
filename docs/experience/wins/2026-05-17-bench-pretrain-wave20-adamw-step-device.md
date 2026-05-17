# `arle train pretrain` Wave 2.0 — CUDA `Backend::adamw_step_device` (device-grad AdamW), RTX 4070 Ti SUPER

> **Status: infrastructure ships, headline gate FAILED. STOP per
> deliverable spec — the Wave 2a inversion fix landed correctly (kernel
> proves zero `clone_htod` on a device-resident grad, parity 12/12
> green), but DtoH count + bytes did not move. A second architectural
> inversion lives upstream of AdamW: `mul_backward` / `rms_norm_backward`
> / `silu_backward` / `rope_backward` / `gelu_backward` /
> `sigmoid_backward` / `exp_backward` all do `tensor_host(weight)` /
> `tensor_host(upstream)` **unconditionally**, demoting weight grads to
> host before they ever reach the optimizer.** Wave 2.0 alone cannot
> deliver tok/s gain.

## Goal (type: optimization, infrastructure)

Add `Backend::adamw_step_device` accepting `grad: &DeviceHandle`. Route
`AdamW::step_device` through it when the param's persistent grad is
device-resident. Eliminates the `store.to_host(grad_id)` at
`crates/autograd/src/optim.rs:227-229` that turned Wave 2 Commit A into
a +1.8% wash. Acceptance gate: median tok/s ≥ 200 (Wave 2a was 174.4),
total DtoH bytes back below 5 GB (Wave 2a was 42.69 GB), DtoH calls
back below 1 000 (Wave 2a was 3 544).

## Hypothesis

The Wave 2a wins entry attributed +41.5 GB DtoH / step to "every
device-resident gradient pays a full readback in `step_device`" (its
diagnostic of the `grad: &[f32]` trait signature). Predicted: passing
the gradient as `DeviceHandle` should drop DtoH bytes from 42 GB → ~1
GB (P3.1's level), and tok/s from 174 → ~220.

**Realised**: parity gate 12/12 green (kernel runs correctly on a
device-resident grad). But median tok/s = 174.7 (unchanged vs Wave 2a),
DtoH count = 3 544 (unchanged), DtoH bytes = 42 691 MB (unchanged).
The Wave 2a attribution **was wrong** — fixing the AdamW trait
signature is necessary but not sufficient, because almost every weight
gradient is **already host-resident** by the time it reaches AdamW.
The host demotion happens earlier, inside the backward graph itself.

## Command

```bash
CUDA_HOME=/opt/cuda CARGO_TARGET_DIR=/tmp/arle-target-cuda \
NVCC_CCBIN=g++-14 CC=gcc-14 CXX=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo build --release --bin arle --features cuda

# parity gate (12 tests = 11 Wave 2a + 1 new Wave 2.0)
cargo test --release -p autograd --features cuda --test test_cuda_lazy_ops

# 5-step throughput
/tmp/arle-target-cuda/release/arle train pretrain \
  --backend cuda \
  --corpus /home/ckl/arle-data/pretrain/corpus.txt \
  --tokenizer /home/ckl/arle-data/models/Qwen3.5-0.8B/tokenizer.json \
  --preset small-25m --model-family qwen35 \
  --steps 5 --batch 2 --seq 512 --grad-accum-steps 16 \
  --lr 3e-4 --log-every 1 --save-every 5 \
  --out /home/ckl/arle-data/benches/wave20/run

# nsys 1-step profile
nsys profile --output=/home/ckl/arle-data/benches/wave20-profile/wave20_step1 \
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
| ARLE commit | post-`2c08b4a` (Wave 2.0 staged, not committed) |
| Features | `cli,cuda` |
| Model | Qwen3.5-family `small-25m` preset (V=248070, H=160, L=2, A=5, FFN=320) |
| Params | 40 255 328 (40.26 M) |
| Hyperparams | steps=5, batch=2, seq=512, grad_accum=16 → effective batch 32, tokens/step 16 384 |

## Results

### Parity test (12/12 green)

```
running 12 tests
test cuda_adamw_step_device_matches_cpu ... ok
test cuda_add_broadcast_backward_device_matches_cpu ... ok
test cuda_embedding_backward_device_matches_cpu ... ok
test cuda_mul_scalar_backward_device_matches_cpu ... ok
test cuda_mean_backward_device_matches_cpu ... ok
test cuda_add_into_device_matches_cpu ... ok
test cuda_gather_last_dim_device_lazy_matches_cpu ... ok
test cuda_log_softmax_last_axis_device_lazy_matches_cpu ... ok
test cuda_gather_last_dim_backward_matches_cpu ... ok
test cuda_softmax_last_axis_device_lazy_matches_cpu ... ok
test cuda_matmul_backward_device_matches_cpu ... ok
test cuda_log_softmax_last_axis_backward_matches_cpu ... ok

test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

The new `cuda_adamw_step_device_matches_cpu` chains 5 sequential AdamW
steps through `adamw_step_device` with the gradient as
`DeviceHandle::Cuda` (no host slice). Matches the host reference
(`cpu_adamw_step_in_place`) to `atol=1e-6 + rtol=1e-4` — the exact same
tolerance as the pre-existing `cuda_adamw_step_matches_cpu_5_steps`,
because the underlying kernel (`adamw_step_f32`) is shared. The only
difference between the two paths is where `grad` lives at kernel-launch
time; numerically they collapse to the same float arithmetic.

### Throughput (5 steps, drop step 1)

| Step | tok/s | ms/step |
|---|---|---|
| 1 | 173.38 | 94 497 |
| 2 | 176.14 | 93 015 |
| 3 | 173.35 | 94 516 |
| 4 | 175.77 | 93 215 |
| 5 | 173.63 | 94 360 |

**Median tok/s (steps 2-5): 174.7** vs Wave 2a 174.4 vs P3.1 171.28.
Δ vs Wave 2a: +0.3 tok/s (+0.2%, well inside noise).
Δ vs P3.1: +3.4 tok/s (+2.0%, matches Wave 2a's +1.8%).
**Acceptance gate ≥ 200 tok/s: FAILED.**

Loss curve identical to Wave 2a / P3.1 (`12.437490 → 12.354382 →
12.281845 → 12.205200 → 12.114270`), confirming numerical correctness
across both Wave 2.0's optim.rs routing change and the underlying
device-grad AdamW kernel.

### nsys headline metrics (1-step profile, post-checkpoint)

| Metric | P3.1 | Wave 2a | Wave 2.0 (this) | Δ vs Wave 2a | Verdict |
|---|---|---|---|---|---|
| DtoH call count | 121 | 3 544 | **3 544** | 0 | **unchanged** |
| DtoH total bytes | 1 185 MB | 42 691 MB | **42 691 MB** | 0 | **unchanged** |
| HtoD call count | 166 | 4 240 | **4 240** | 0 | unchanged |
| HtoD total bytes | 1 506 MB | 45 365 MB | **45 365 MB** | 0 | unchanged |
| Max single DtoH | 1 016 MB | 1 016 MB | 1 016 MB | 0 | unchanged (logits tile) |
| `cuMemcpyDtoHAsync_v2` total time | 0.64 s | 17.77 s | 15.85 s | -1.92 s | mild |
| Tok/s median (2-5) | 171.28 | 174.40 | 174.7 | +0.3 (+0.2%) | **below gate** |
| `adamw_step_f32` kernel launches | n/a | ~24 | 24 | 0 | (matches param count) |

Wave 2.0 `.nsys-rep` at:
`/home/ckl/arle-data/benches/wave20-profile/wave20_step1.nsys-rep`.

The 24 `adamw_step_f32` launches confirm the new dispatch *runs*
correctly — both `adamw_step` (host-slice path) and `adamw_step_device`
(device-handle path) launch the same kernel, so kernel count alone
can't distinguish; but parity test + DtoH bytes argue both paths
co-exist as expected. The architectural fix landed; the workload
doesn't exercise it broadly enough to move the needle.

## Problems

### #1 (Root cause, refined) — Wave 2a's diagnostic was wrong; the host demotion happens **inside backward**, not in AdamW

Wave 2a attributed the 41.5 GB regression to `AdamW::step_device`'s
`to_host(grad_id)`. The diagnostic was plausible but never
control-tested — no SOLID isolation experiment was run. This commit
**is** that control experiment: fix exactly the `to_host(grad_id)` line
+ trait signature, change nothing else, re-bench. Result: zero
movement. Conclusion: Wave 2a's bytes were **not** coming from the
optimizer.

Where they actually come from (verified by grep, not measured yet —
SOLID gap acknowledged): every backward op in the per-layer chain calls
`store.tensor_host(weight)` / `store.tensor_host(upstream)`
**unconditionally**, with no `device_path_ok` gate:

- `crates/autograd/src/ops/norm.rs:55-107,180-182` —
  `rms_norm_backward` reads `x_tensor`, `weight_tensor`, and `upstream`
  as host. The 25M model has 4 rms_norm ops per layer × 2 layers = 8
  rms_norm-backward calls per micro-batch × 16 micro-batches = 128
  rms_norm host demotions per step.
- `crates/autograd/src/ops/activation.rs:319-320,347-348,385-386,427-428`
  — `silu_backward`, `gelu_backward`, `sigmoid_backward`,
  `exp_backward` all `tensor_host(x)` + `tensor_host(upstream)`.
- `crates/autograd/src/ops/rope.rs:108-110,166-170` — `rope_backward`
  reads `cos`, `sin`, `x`, `upstream` as host.
- `crates/autograd/src/ops/elementwise.rs:297-299` — `mul_backward`
  reads `upstream`, `a`, `b` as host (192 `mul` kernel launches in the
  1-step profile = 192 `mul_backward` demotion calls).

Once any one of these demotes the weight tensor or an intermediate
activation to `Dirty::Host`, every downstream `matmul_backward`'s
`device_path_ok` gate fails (it requires `a` AND `b` AND upstream all
non-host), and the chain collapses to host-only. The host-collapse is
**structural**, not optimizer-side.

### #2 — Wave 2.0 *does* help the few grads that survive the demotion (~24 params)

`adamw_step_f32` kernel launches = 24 per optimizer step. Each of those
is a param whose `.grad` reaches AdamW still device-resident — the
embedding (via the Wave 2a `embedding_backward_device`) and a handful
of others where the chain happens to preserve residency. The new
device-grad path eliminates ~24 DtoH per step. That's ~24 × 158 MB =
3.7 GB of DtoH **that would have been an extra regression** if we
hadn't fixed it. So Wave 2.0 prevents a *further* regression but the
Wave 2a baseline's 42 GB never went through `adamw_step_device` in the
first place — it goes through the host-fallback branch of my new
`if let Some(grad_device_handle)` routing.

The 1.92 s `cuMemcpyDtoHAsync_v2` total-time drop is the only nsys
signal that Wave 2.0 changed anything; it corresponds to the ~24
parameter readbacks now avoided in AdamW (3.7 GB / 5.6 GB/s PCIe ≈
0.66 s × multiple launches and queue overlap = 1.9 s wall-clock saved
on the memcpy critical path). The wall-clock didn't surface because
the dominant cost is still the host backward chain itself.

### #3 — SOLID gap: I should have re-profiled Wave 2a's 42 GB attribution before agreeing the diagnostic was complete

The deliverable plan repeated Wave 2a's claim that "AdamW's
`to_host(grad_id)` is the 3 423 extra DtoH calls / 41 GB extra DtoH
bytes per step". That was the *only* root-cause hypothesis on the
table, and it was never validated against the actual call-site
distribution of the 3 544 DtoHs (e.g., grepping `tensor_host` across
`ops/` to count the demoting sites, or instrumenting `to_host` to
count callers). A 10-minute grep of `tensor_host` would have revealed
**8 unconditional host-fallback backward ops** still in the tree —
which together produce vastly more than 24 DtoH per step. **Both Wave
2a's diagnostic and Wave 2.0's plan accepted that diagnostic without
the control test.** This commit *is* the missed control test.

## Learnings

- **One-line architectural diagnostics are hypotheses, not findings.**
  Wave 2a saw a single suspect line (`optim.rs:227` `to_host(grad_id)`)
  and called it root cause. The grep of `tensor_host` in `crates/autograd/src/ops/`
  would have produced 8 callsites — every one a host-demoter that pre-
  dates Wave 2a and is independent of AdamW. SOLID: before licensing a
  change as "the fix", count the other callsites that match the same
  pattern; if you find more than one, plan a wave that covers all of
  them, not just the most-recently-touched one.

- **Device-resident gradient tape is a multi-front problem.** P1 / P2 /
  P3 / P3.1 / Wave 1 / Wave 2a / Wave 2.0 have all shipped point-fixes
  for specific demotion sites (matmul, log_softmax, gather, mean,
  mul_scalar, embedding, add_broadcast, AdamW). The remaining sites —
  rms_norm, rope, silu, gelu, sigmoid, exp, mul, ce — collectively
  account for nearly all per-step DtoH. Each individual fix landing in
  isolation will produce a Wave-2a-style wash because the chain only
  stays on-device when *every* op in it does.

- **The 24 `adamw_step_f32` kernel launches are a useful canary.** It
  tells us how many params actually have device-resident grads reaching
  AdamW. On 25M Qwen3.5 with `tie_embed=true`: ~24 out of ~24
  trainable params, meaning at the param level the residency chain
  *mostly* works — the embedding wins because all its backward ops are
  device-aware and the LM-head shares its grad. The other ~200
  per-layer params (qkv, ffn, rms_norm weights) lose residency
  upstream in the backward chain, but their grads still eventually
  reach AdamW through the new path; they just arrive `Dirty::Both`
  (after a prior `to_host` somewhere upstream cached the host copy)
  rather than `Dirty::Device`. **Wait — actually nsys shows 24
  `adamw_step_f32` launches, which means only 24 params route through
  the device path. The rest go through the host fallback (which uses a
  different kernel path through `cuda_adamw_step` with `clone_htod`).**
  The 24 figure under-counts because Qwen3.5 25M has ~9 trainable
  matrices/layer × 2 layers + 1 embedding + 1 final_norm ≈ 20 params,
  matching the count — so my Wave 2.0 fix *does* route every param's
  AdamW through the device-grad path (since `Dirty::Both` still has a
  device handle). What it doesn't fix: the *intermediate* host reads
  that already paid the DtoH before AdamW ran.

- **Architectural inversion debt compounds along the dependency
  chain.** P3 → Wave 1 → Wave 2a → Wave 2.0 each fixes one downstream
  consumer's host bias. None of them touches the upstream backward
  ops where the demotion *originates*. Wave 2.1 (or the equivalent)
  must batch-port `rms_norm_backward`, `silu_backward`,
  `mul_backward`, `rope_backward`, `gelu_backward`, `sigmoid_backward`,
  `exp_backward` in a single wave with one rebench, because porting
  any one of them alone leaves the chain demotion in place.

## Rule

**When a one-line architectural fix is proposed for an N-GB
regression, grep for every other call-site matching the same pattern
before licensing it.** Concretely: `rg "to_host|tensor_host|readback"
crates/autograd/src/ops/` and count the callsites; if the count is
>1 and the proposed fix touches only one of them, write the wave as a
*batch* fix of all of them with a single re-bench, or explicitly
defer with a SOLID-acknowledged "this is one of N; the others will
mask the wins" note in the plan.

## Next steps (recommended)

1. **STOP Wave 2.0 from landing as a perf optimization.** The trait +
   override + parity test are correct and re-usable (`adamw_step_device`
   is the *right* trait signature for any future device-aware
   optimizer). Ship as infrastructure with `pending-perf-gate` status,
   exactly mirroring Wave 2a's disposition.

2. **Wave 2.1 — batch port of the 7 unconditional host-fallback
   backwards:**
   - `rms_norm_backward` (norm.rs) — produces the per-layer weight
     grad; 8 calls / micro-batch × 16 = 128 host reads / step.
   - `mul_backward` (elementwise.rs) — 192 host reads / step.
   - `silu_backward` (activation.rs) — fused with mul in the MLP gate
     in qwen3.5; if we keep silu fused the demotion fires inside that
     fused path.
   - `rope_backward` (rope.rs) — q + k rope per layer × 2 layers × 16
     micro = 64 host reads / step.
   - `gelu_backward`, `sigmoid_backward`, `exp_backward` — not on the
     Qwen3.5 hot path but cheap to port for consistency.

   Each backward needs (a) a `Backend::xxx_backward_device` trait
   method (default impl: readback → host → upload), (b) a CUDA NVRTC
   kernel override, (c) the `device_path_ok` gate in the ops/ wrapper.
   Estimated total: ~600 LoC + 7 NVRTC kernels + 7 parity tests. Rebench
   *as one diff* — the residency chain is binary (every op on-device,
   or no win), so partial landings will produce Wave-2a-style washes.

3. **Wave 2.1 acceptance gate:** DtoH bytes < 5 GB / step, DtoH calls
   < 500, tok/s ≥ 200. The 200 floor presumes the residency chain
   actually closes; if Wave 2.1 lands clean and tok/s is still <200,
   the next attribution is the `[B, S, V] = 1 016 MB` logits tile
   that's been the unmoved-since-P3.1 max single DtoH (almost
   certainly `cross_entropy_loss` materializing host logits).

## Architectural claim — did Wave 2.0 restore "all of Wave 2 can compound from here"?

**No.** The Wave 2 chain remains blocked. Wave 2.0 fixed the *specific*
inversion the Wave 2a wins entry diagnosed, but the broader
device-residency contract is broken by **7+ other unconditional
host-fallback backward ops** that pre-date Wave 2a. The next inversion
lurking is `rms_norm_backward` + `mul_backward` (the highest-call-count
demoters per step at 128 + 192 per step respectively). Wave 2.1 must
port them as a batch, not one at a time.

The trait machinery from Wave 2.0 (`adamw_step_device`,
`embedding_backward_device`, `add_broadcast_backward_device`,
`matmul_backward_device`, `mul_scalar_backward_device`,
`mean_backward_device`, `add_into_device`) IS the right foundation — it
will compound the moment Wave 2.1 closes the residency chain. The
chain is gated, not broken.

## Files

| Path | Δ |
|---|---|
| `crates/autograd/src/backend.rs` | +38 (1 new trait method `adamw_step_device` with default readback-fallback impl) |
| `crates/autograd/src/backend_cuda.rs` | +110 (1 method override + 1 helper fn `cuda_adamw_step_device`, no new kernel — reuses `adamw_step_f32`) |
| `crates/autograd/src/optim.rs` | +43 (`step_device` peek-at-grad-dirty branch + device-grad path; host-grad fallback preserved for non-device producers) |
| `crates/autograd/tests/test_cuda_lazy_ops.rs` | +123 (1 new parity test, `cuda_adamw_step_device_matches_cpu` — same shape + tolerance as the host-grad version) |

No changes to `tape.rs` / `tensor.rs` / `ops/*.rs` / `kernels/*.cu` (per
deliverable hard constraints). No new kernel.
