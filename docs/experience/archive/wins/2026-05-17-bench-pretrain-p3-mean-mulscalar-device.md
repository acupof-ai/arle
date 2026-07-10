# `arle train pretrain` P3 — CUDA device-lazy `mean_backward` + `mul_scalar_backward`, RTX 4070 Ti SUPER

> **Status: infrastructure-only — headline gate FAILED.** Trait + 2 NVRTC
> kernels + parity gate landed clean (9/9 green). Production hot path
> **does not activate** the new device backwards because the **upstream
> gradient is host-resident from the very seed of every backward walk**.
> tok/s median **91.40** (M5.3b 92.08, Wave 1 91.41, P2 ~91.4; ±1 %
> noise). nsys max single DtoH **1016.1 MB** — **identical** to G3 /
> M5.3b / Wave 1 / P1 / P2. The architectural-correction doc's chain
> attribution is **wrong**: `mean_backward` / `mul_scalar_backward` are
> NOT the host-poison source — `Tape::backward`'s seed `fill_like(loss,
> 1.0)` and `flush_to_host_batch` are. Real wall-clock win blocked on
> a tape-level redesign that the P3 task hard-constraint forbade.

## Goal (type: optimization, infrastructure)

Per the [P3 brief](../../research/2026-05-17-cuda-training-architectural-correction.md)
the head of the CE-loss backward chain — `d_loss → mul_scalar_backward
→ mean_backward → gather_backward → log_softmax_backward → matmul_backward` —
runs host-side because `mean_backward` / `mul_scalar_backward` are the
only two ops in that chain without a device override. Porting both to
device-resident NVRTC kernels was hypothesised to unblock the entire
downstream chain (P1 + P2 + Wave 1 + M5.3b would finally fire on the
production hot path), dropping the headline `1 016 MB` DtoH to `< 100 MB`.

## Hypothesis (recorded for SOLID accounting)

Per the architectural-correction doc: with `mean_backward_device` and
`mul_scalar_backward_device` keeping the upstream device-resident, every
downstream `device_path_ok` gate (P2 matmul, Wave 1 softmax/gather, P1
add_into) would flip from host-fallback to device-lazy. Predicted savings:
`1 016 MB → <100 MB` peak DtoH, throughput +30-40 %. **Realised outcome:
0 % wall improvement, 0 MB peak DtoH movement, every prior gate still
host-fallback.** The gap explanation — forensic dispatch trace — is in
§Problems #1.

## Command

```bash
CUDA_HOME=/opt/cuda CARGO_TARGET_DIR=/tmp/arle-target-cuda \
NVCC_CCBIN=g++-14 CC=gcc-14 CXX=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo build --release --bin arle --features cuda

# parity gate (9 tests = 7 prior + 2 new P3)
cargo test --release -p autograd --features cuda --test test_cuda_lazy_ops

# nsys 1-step profile
nsys profile \
  --trace=cuda,nvtx --gpu-metrics-devices=none \
  --output=/home/ckl/arle-data/benches/profile-p3/p3_step1 \
  --force-overwrite=true \
  /tmp/arle-target-cuda/release/arle train pretrain \
    --backend cuda --corpus … --tokenizer … --preset small-25m \
    --model-family qwen35 --steps 1 --batch 2 --seq 512 \
    --grad-accum-steps 16 --lr 3e-4 \
    --out /home/ckl/arle-data/benches/profile-p3/run

# 5-step throughput bench + GPU sampler (2 s)
/tmp/arle-target-cuda/release/arle train pretrain \
  --backend cuda --corpus … --tokenizer … --preset small-25m \
  --model-family qwen35 --steps 5 --batch 2 --seq 512 \
  --grad-accum-steps 16 --lr 3e-4 \
  --out /home/ckl/arle-data/benches/p3-throughput/run
```

GPU sampler: `nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits`
every 2 s → `gpu.csv` (453 samples over the ~15 min bench wall).

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER · 16.0 GB · sm_89 |
| CUDA / nvcc | 13.2 V13.2.78 (system /opt/cuda) |
| Nsight Systems | 2025.6.3.541-256337736014v0 |
| Host compiler | g++-14 (NVCC_CCBIN) |
| cudarc | 0.19.7 |
| ARLE commit | post-`bccabb4` (P3 staged, not committed) |
| Features | `cli,cuda` (default `arle` feature flip needed; root features = `["cli"]`) |
| Model | Qwen3.5-family `small-25m` preset (vocab=248070, hidden=160, layers=2, heads=5, kv_heads=5, head_dim=32, ffn=320, max_pos=512, tie_embed=true) |
| Params | 40 255 328 (40.26 M) |
| Hyperparams | steps=5, batch=2, seq=512, grad_accum=16 → effective batch 32, tokens/step 16 384 |

## Results

### Parity test (9 tests, all green)

```
running 9 tests
test cuda_mul_scalar_backward_device_matches_cpu ... ok
test cuda_add_into_device_matches_cpu ... ok
test cuda_mean_backward_device_matches_cpu ... ok
test cuda_gather_last_dim_device_lazy_matches_cpu ... ok
test cuda_gather_last_dim_backward_matches_cpu ... ok
test cuda_log_softmax_last_axis_device_lazy_matches_cpu ... ok
test cuda_softmax_last_axis_device_lazy_matches_cpu ... ok
test cuda_matmul_backward_device_matches_cpu ... ok
test cuda_log_softmax_last_axis_backward_matches_cpu ... ok

test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Tolerances: combined `atol=1e-6 + rtol=1e-4` for both new tests (well
within the P3 brief's `≤ 1e-4` mandate). The mean backward is a
broadcast divide and `mul_scalar` backward is a multiply — no
accumulation, no transcendentals.

### Throughput (5 steps, drop step 1)

| Step | tok/s | ms/step |
|---|---|---|
| 1 | 91.22 | 179 618 |
| 2 | 91.24 | 179 568 |
| 3 | 91.54 | 178 978 |
| 4 | 91.25 | 179 546 |
| 5 | 91.55 | 178 969 |

**Median (steps 2-5): 91.40 tok/s.** Step-to-step jitter ±0.4 %; no
trend across the 5-step run.

### GPU utilization (453 samples × 2 s)

| Metric | Avg | Notes |
|---|---|---|
| GPU util | **9.63 %** | host-bound, GPU mostly idle |
| Mem util | 1.47 % | DRAM bandwidth essentially unused |
| Power draw | 22.63 W | idle-floor + occasional kernel bursts (baseline GPU idle ≈ 8 W) |
| Memory used | ~5.9 GB | weights + AdamW state + activations |

GPU sitting at <10 % util confirms the runtime is **host-CPU-orchestration-bound**,
not compute-bound. A successful chain-unblock would drive GPU util into
the 50-90 % range.

### nsys headline metrics (1-step profile)

| Metric | Baseline (m53b) | P3 | Δ |
|---|---|---|---|
| Max single DtoH | **1016.1 MB** | **1016.1 MB** | **0 MB** |
| Total DtoH bytes | 93 245.9 MB | 93 245.9 MB | 0 MB |
| DtoH transfer count | 4 760 | 4 760 | 0 |
| Total HtoD bytes | 45 365.4 MB | 45 365.4 MB | 0 MB |
| HtoD transfer count | 4 272 | 4 272 | 0 |
| `cuMemcpyDtoHAsync_v2` total time | 36.94 s | 37.25 s | +0.3 s (noise) |
| `cuStreamSynchronize` calls | 4 817 | 4 817 | 0 |
| `cudaLaunchKernel` calls | 768 | 768 | 0 |
| ms/step (1-step profile) | 181 149 | 183 165 | +1.1 % (within noise) |

**Headline gate: FAILED.** Max DtoH < 100 MB threshold not met (still
1016 MB). Total memcpy count not reduced. Per the P3 brief's failure-mode
contract: forensic dispatch trace required.

### Cross-milestone summary

| Milestone | tok/s (median) | Max DtoH | Memcpy count | GPU util | Source |
|---|---|---|---|---|---|
| Baseline (M5.3b) | 90.4 | 1 016 MB | 9 032 cumulative | n/a | `profile-m53b/m53b_step1` |
| M5.3b (forward lazy) | 92.08 | 1 016 MB | 4 760 (1 step) | ~9 % | wave1 wins entry |
| Wave 1 (logsoftmax+gather bwd) | 91.41 | 1 016 MB | 4 760 | ~9 % | wave1 wins entry |
| P1 (add_into_device) | ~91 | 1 016 MB | 4 760 | ~9 % | inferred (entry not surfaced this session) |
| P2 (matmul_backward_device) | ~91 | 1 016 MB | 4 760 | ~9 % | g3 wins entry + P2 commit `7bbe995` |
| **P3 (mean+mulscalar bwd)** | **91.40** | **1 016 MB** | **4 760** | **9.63 %** | this entry |

**The plateau is flat from M5.3b onward — five "fixes" to the CE-loss
backward chain have produced zero wall-clock movement, because the
host-poison source is upstream of every gate.**

## Problems

### #1 (Root cause) — Backward seed gradient is host-only; `flush_to_host_batch` demotes every saved device tensor. P3's overrides never fire on the hot path

Forensic dispatch trace (added per P3 brief failure-mode contract,
removed before final commit; re-applied via the diff in the appendix):

```
[P3-trace] mul_scalar_backward device_path_ok=false g.dirty=Host g.dev=false g.shape=[] k=-1
[P3-trace] mean_backward         device_path_ok=false g.dirty=Host g.dev=false g.shape=[]
[P3-trace] gather_last_dim_backward device_path_ok=false g.dirty=Host g.dev=false
[P3-trace] log_softmax_backward  device_path_ok=false y.dirty=Both y.dev=true g.dirty=Host g.dev=false
[P3-trace] matmul_backward       device_path_ok=false a.dirty=Both a.dev=true b.dirty=Both b.dev=true g.dirty=Host g.dev=false
[P3-trace] matmul_backward       device_path_ok=false a.dirty=Both a.dev=true b.dirty=Both b.dev=true g.dirty=Host g.dev=false
... (every backward op in the entire walk shows g.dirty=Host g.dev=false)
```

Two architectural problems jointly poison the chain, both inside
`Tape::backward` and the surrounding `TensorStore`:

1. **`Tape::backward` line 183** (`crates/autograd/src/tape.rs`):
   `store.flush_to_host_batch(&device_ids)?` — every Dirty::Device
   *tape output* is eagerly flushed to host *before* the backward walk
   even starts. This demotes saved activations (e.g. `y` from
   `log_softmax`, `a/b` from `matmul`) from `Dirty::Device` to
   `Dirty::Both`. P2/Wave 1 gates use `dirty != Dirty::Host` so the
   saved tensors DO pass (`a.dirty=Both a.dev=true`) — but…

2. **`Tape::backward` line 203**: `let loss_grad_id =
   store.fill_like(loss_id, 1.0)?` — the seed `d_loss = 1.0` is allocated
   via `store.alloc(Tensor::new(vec![1.0; size], …))`, which is **host-only**.
   Every backward op the walk visits receives a host-resident upstream.
   `mul_scalar_backward` (the very first op, for `loss = mul_scalar(neg_log_lik_mean, -1)`)
   sees `g.dirty=Host g.dev=false`, falls through to its host fallback
   (which P3 left untouched per `mul_scalar_backward` host-path contract),
   and emits another host tensor. The same pattern repeats down the chain.

**Net effect**: P3's new `mul_scalar_backward_device` / `mean_backward_device`
*never fire* on the hot path. Their `device_path_ok` gate is the
correctly-mirrored Wave 1 / P2 pattern, but the test for `dirty != Host`
fails because the upstream is brand-new host-allocated, not previously
device-resident. Same for P2 matmul (`g.dirty=Host`), Wave 1 softmax
(`g.dirty=Host`), Wave 1 gather (`g.dirty=Host`), P1 add_into.

**Where does the 1 GB DtoH actually come from?** From line 183's
`flush_to_host_batch` — when `mean_device_lazy` produces the `[B, S, V]`
log-softmax + gather result as Dirty::Device, the tape's eager pre-flush
batch readback at the *start* of backward downloads it. This is the
exact `1 015 MB` DtoH that every prior milestone misattributed to a
single backward op.

**Attribution claim**: The architectural-correction doc, and the
M5.3b / Wave 1 / P1 / P2 / P3 wins entries that referenced it, all
attributed the `1 GB` peak DtoH to the missing device-resident
backward of one of `log_softmax_last_axis_backward`,
`gather_last_dim_backward`, `matmul_backward`, `mean_backward`, or
`mul_scalar_backward`. **All five attributions are wrong.** The
peak is `flush_to_host_batch` at the *start* of `Tape::backward`,
which exists because `Tape::backward`'s implementation predates the
device-resident gradient tape and pre-flushes every saved activation
to host. None of the five backward overrides can dispute this until
the tape-level redesign lands.

### #2 — Hard constraints make Phase-3.1 (true unblock) impossible from the P3 diff alone

The P3 task brief explicitly forbade modifying `tape.rs` / `tensor.rs`,
which is exactly where the fix must land:

- Replace `flush_to_host_batch(&device_ids)` with a lazy `ensure_host`
  per-id only when a tape op actually needs host data (e.g.
  `to_host` calls inside a backward host-fallback). Saved tensors that
  feed a device-overridden backward should stay `Dirty::Device`.
- Replace `fill_like(loss_id, 1.0)` with a backend-aware seed: when
  `loss` is `Dirty::Device`, upload a rank-0 `1.0` and store it as
  `Dirty::Device` so the head of the chain enters the new device
  overrides.
- Audit `merge_grad` / `accumulate_grad` to confirm they preserve
  `Dirty::Device` on the accumulated grad when both input handles are
  device-resident (P1's `add_into_device` is in place but probably
  also gated by the same upstream-host problem).

These three changes are a **Phase-3.1 follow-up** that must touch
`tape.rs` + `tensor.rs`. P3 as-specified ships infrastructure (kernels
+ trait + dispatch gates + parity gate); the Phase-3.1 follow-up flips
the seed + the pre-flush, at which point P3+Wave 1+P2+P1+M5.3b all
activate simultaneously.

## Learnings

- **The `device_path_ok` gate pattern is consistent and correct across
  M5.3b / Wave 1 / P1 / P2 / P3** — five independent implementations
  of "if upstream + saved tensors are device-resident, dispatch device
  override; else host fallback" all behave correctly under the parity
  tests. The failure is upstream of *all of them*.
- **`flush_to_host_batch` in `Tape::backward` is the single architectural
  poison source.** It pre-flushes every Dirty::Device tape output, which
  for a CE-loss workload includes the `[B, S, V] = 1 GB` logits tensor.
  Removing this is the only intervention that moves the `1016 MB` peak
  DtoH metric.
- **The seed `fill_like(loss, 1.0)` is the secondary poison source.**
  Even after the pre-flush is removed, every backward op starts with
  a host-allocated upstream until the seed becomes device-aware.
- **Infrastructure parity ≠ wall-clock win.** Five clean parity tests
  + five clean device-override kernels can sit in tree for weeks
  contributing zero throughput. The SOLID gate (per CLAUDE.md §0) must
  be **wall-clock framing**, not "parity gate green" or "kernel
  exists" framing. P3 should have run the nsys profile FIRST before
  writing the parity tests — would have surfaced the host-flip in
  20 minutes instead of 3 hours.
- **forensic eprintln on every dispatch gate is cheap (~150 lines for
  1 grad-accum step, 0 perf overhead) and definitive.** Should be the
  first intervention after a failed headline gate, before any further
  code change.

## Rule

When a "device-resident backward" infrastructure milestone lands and the
nsys headline gate doesn't move, **do not ship another backward
override**. Add forensic `eprintln!` to every existing `device_path_ok`
gate and run 1 step with `--grad-accum-steps 1`; the answer to "where
does the host-flip happen?" lands in <100 trace lines. If the trace
shows `g.dirty=Host` at the very first backward op, the fix is
*upstream* of any backward op — it's in `Tape::backward`'s seed +
pre-flush, not in another kernel.

## Files

| Path | Δ |
|---|---|
| `crates/autograd/src/backend.rs` | +57 (2 trait methods + doc) |
| `crates/autograd/src/backend_cuda.rs` | +131 (2 method overrides + 2 helper fns) |
| `crates/autograd/src/backend_cuda/kernels.rs` | +9 (2 includes, 2 function names, 2 concat entries) |
| `crates/autograd/src/backend_cuda/kernels/mean_backward.cu` | +26 (new file) |
| `crates/autograd/src/backend_cuda/kernels/mul_scalar_backward.cu` | +20 (new file) |
| `crates/autograd/src/ops/reduce.rs` | +28 (device-path dispatch in `mean_backward`) |
| `crates/autograd/src/ops/elementwise.rs` | +37 (device-path dispatch in `mul_scalar_backward`) |
| `crates/autograd/tests/test_cuda_lazy_ops.rs` | +82 (2 new parity tests) |

No changes to `tape.rs`, `tensor.rs`, `ops/matmul.rs`, `ops/softmax.rs`,
`ops/gather.rs` (per P3 hard constraint).

## Appendix — forensic trace patch (for re-investigation)

The `eprintln!` patches added to capture the trace at §Problems #1 were
removed before final commit. To re-apply, add this stanza to each of
the five backward dispatch gates (`ops/matmul.rs::matmul_backward`,
`ops/softmax.rs::log_softmax_backward`, `ops/gather.rs::gather_last_dim_backward`,
`ops/reduce.rs::mean_backward`, `ops/elementwise.rs::mul_scalar_backward`)
immediately after the `device_path_ok` computation:

```rust
{
    let g_t = store.tensor(output_grad_id)?;
    eprintln!(
        "[P3-trace] <op_name> device_path_ok={} g.dirty={:?} g.dev={} g.shape={:?}",
        device_path_ok, g_t.dirty, g_t.device_handle.is_some(), g_t.shape
    );
}
```

Run `--grad-accum-steps 1 --steps 1` and pipe stderr — full trace lands
in ~30 lines and pinpoints the host-flip op.
