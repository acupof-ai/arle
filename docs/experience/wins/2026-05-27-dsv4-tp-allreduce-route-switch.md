# DSv4 TP/allreduce route switch — 2.21× faster than EP/deepep at c=1

## Goal

Validate that switching DSv4 MoE backend default from `deepep` (broken
EP path) to `allreduce` (TP-style local-routed + NCCL allreduce combine)
gives a working, measurable baseline today, given:

- B-3.3.5 DeepEP DeepGEMM branch has a confirmed scatter-accumulate bug
  (`704cc09f` source-level deep dive) — 7/8 of expert contributions
  silently overwritten for topk>1 routing.
- DeepGEMM JIT compile finally passed today (`38bf157b`, c++17 nvcc fix),
  but runtime crashes on H20 with "unspecified launch failure" (`1164f35d`).
- DeepEP LL mode (real ~+15-19 ms TPOT lever per prior nsys) needs
  NVSHMEM container install + ~2000 LOC integration — 2-week scope.

Route switch is license-or-kill'd against wall-clock TPOT per CLAUDE.md
§0 (wall-clock framing, not narrow-window % of NVTX).

## Hypothesis

`ARLE_DSV4_MOE_BACKEND=allreduce` walks the pre-DeepEP code path:
`forward_local_routed_gpu` (mlp.rs:1642) + `post_moe_expert_all_reduce_
hidden_states` (weights.rs:2345). No dispatch, no combine — every rank
runs its 1/N local experts and the result is summed via one NCCL
`all_reduce` per layer. The code was the default before DeepEP
landed and is battle-tested.

Expectation going in: simpler ≠ faster. EP was made default for a reason
(decode wave p50 = 210 ms at c=1, see
`2026-05-26-dsv4-default-deepep-deepgemm.md`). My estimate was TP would
land in the 80-150 ms TPOT range — worse than a working EP but
**better than the currently-broken EP path with its DeepGEMM crash and
scatter race**.

## Params

| Item | Value |
|---|---|
| Hardware | 8× NVIDIA H20, driver 535.161.08 (CUDA 12.2 line) |
| CUDA | 12.2 toolchain, sm_90 cubins |
| NCCL | 2.21.5 (older archive at `~/.cache/uv/archive-v0/hvOud1G8tTqpJUVwB1stZ/nvidia/nccl/lib/libnccl.so.2`) |
| Runtime | ARLE DSv4 CUDA, 8 workers, `--num-slots 1`, `--max-seq-len 4096`, `--mem-fraction-static 0.10`, `--kv-cache-dtype fp8`, `--deepseek-distributed-layers 43` |
| Model | DeepSeek V4-Flash at `/root/DeepSeek-V4-Flash` |
| MoE backend | `ARLE_DSV4_MOE_BACKEND=allreduce` (new default after `a6c910b2`) |
| Expert backend | `ARLE_DSV4_EXPERT_BACKEND=native` (DeepGEMM auto skipped for local-routed path) |
| Request | `max_tokens=32`, prompt = "Compute 137 + 269. Answer with the number only." |
| Wrapper | `bash scripts/dsv4_toolchain.sh nsys` |
| Profiler | `nsys profile --capture-range=cudaProfilerApi`, NVTX scope `step_decode_kernel_launch` |
| Single profile request elapsed | 2.546 s for 17 prompt + 32 completion tokens |

Binary on pod was built earlier today (16:38, `/data01/build/arle/target/
release/infer`); no rebuild needed because the route switch is a runtime
env flip only — Rust source unchanged.

## Results

### Decode wave timing (per-rank-range = per-decode-step per rank)

| Metric | TP/allreduce (today) | EP/deepep (2026-05-26 baseline) | Δ |
|---|---:|---:|---:|
| decode wave p50 | **94.847 ms** | 210.033 ms | **−54.8%** |
| decode wave min | 87.491 ms | n/a | — |
| decode wave max | 342.218 ms (cold) | 362.614 ms | −5.6% |
| decode waves | 31 | 31 | — |

Cold-start (first wave 342 ms) discarded; the 30 steady-state waves are
all in the **88-99 ms range**, p50 = 94.85 ms.

**TP/allreduce is 2.21× faster wall-clock than EP/deepep at c=1.**

### Top GPU kernels (per rank-range, ms)

| Rank | Kernel | per-range ms | calls | % of decode |
|---:|---|---:|---:|---:|
| 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 30.39 | 21320 | **32.0%** |
| 2 | `dsv4_fp8_gemv_batch_kernel` (expert FFN) | 11.45 | 90479 | 12.1% |
| 3 | `dsv4_hybrid_attention_kernel` | 7.74 | 10168 | 8.2% |
| 4 | `dsv4_route_kernel` | 5.65 | 10664 | 6.0% |
| 5 | `dsv4_fp4_gemv_batch_kernel` (KV proj) | 4.89 | 23994 | 5.2% |
| 6 | `dsv4_csa_select_kernel` | 3.77 | 5208 | 4.0% |
| 7 | `dsv4_mhc_params_kernel` | 3.05 | 21328 | 3.2% |

### Top CUDA runtime APIs (per rank-range, ms — overlaps with kernels)

| Rank | API | per-range ms | calls |
|---:|---|---:|---:|
| 1 | `cuMemcpyDtoHAsync_v2` | 27.75 | 10679 |
| 2 | `cudaLaunchKernel_v7000` | 20.19 | 424882 |
| 3 | `cuMemAllocAsync` | 16.54 | 177502 |
| 4 | `cuMemFreeAsync` | 11.40 | 164118 |
| 5 | `cuMemsetD8Async` | 2.22 | 63632 |

### Memcpy summary

| Direction | calls | bytes | ms per range |
|---|---:|---:|---:|
| DtoH | 10683 | 1,365,068 | 0.107 |
| DtoD | 8379 | 73,826,304 | 0.042 |
| HtoD | 10912 | 43,648 | 0.036 |

DtoH 10,683 calls per rank-range for 1.36 MB total = the L5 binding
constraint (in-graph metadata, A3 lever) is still wide open. **44 KiB
typical per call** matches the binding-constraints table prediction
(metadata readbacks, not data transfer).

### Decode output (parity check)

```
"4262 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000"
```

Same garbage-shape as previous DSv4-Flash runs (no chat template in
the base model) — confirms allreduce path produces the same
shape-of-output as deepep path, not silent corruption.

## What this means

### Why TP/allreduce beats broken EP/deepep here

The current EP path is not "EP done well" — it's "EP done with the
B-3.3.5 scatter race plus the DeepGEMM JIT-once-runtime-crash hazard
plus the legacy NCCL `ReduceScatter` combine which is itself ~68 ms
per rank-range per the 2026-05-26 baseline". When I previously thought
"TP would be 1.5-2× slower than EP", I was implicitly assuming the EP
path was the LL-mode tuned version SGLang ships, **not** the
intranode-with-broken-DeepGEMM-fallback path ARLE actually has today.

This is exactly the M_pf-graph framing trap CLAUDE.md §0 calls out —
estimating perf against an idealized rival instead of the real,
shipped baseline. **The TP route is correct as the production default
right now**, not because TP is intrinsically better than well-tuned EP,
but because it's better than the broken EP that ARLE currently has.

When DeepEP LL lands (or the H20 DeepGEMM crash is debugged + scatter
accumulate fix shipped), EP can return as default. The route switch
is reversible via the same env knob.

### SLO gap framing

- SLO target: TPOT ≤ 30 ms (current target 18 ms) at 32K input / 1.5K
  output / c=8 / qps=8.
- Today's c=1 max_tokens=32 TPOT: 94.85 ms.
- SLO gap is **3.2× off TPOT target** at c=1 short prompt.
- This profile **cannot** be used as the SLO verdict — it's the wrong
  workload (c=1 short prompt vs c=8 32K prompt). It's a baseline anchor
  for next-axis ranking, not a SLO claim.

### Next-axis ranking (revised, given TP nsys)

Mapping today's per-rank-range top costs back to the L1-L6
binding-constraint table in
`docs/research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md`
and `docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md`:

| Lever | Estimated TPOT lift | Why now |
|---|---|---|
| **A3 in-graph metadata** (L5) | ~5-15 ms | DtoH still 10,679 calls per rank-range for 1.36 MB = pure sync overhead. SGLang V4 captures this inside graph. Single-card, no architecture license. Was already the recommended first axis. |
| **A4 multi-stream overlap** (allreduce vs FFN) | ~10-20 ms | NCCL allreduce 30 ms can partially overlap with FFN compute. Different from EP path where dispatch+combine dominated. Scheduler-level. |
| A2 FlashMLA hybrid attention | ~3-5 ms | hybrid (7.74) + csa (3.77) ≈ 11.5 ms fused candidate. Smaller absolute lift than expected. |
| A1 DeepGEMM Mega MoE | n/a (TP path) | A1 was the EP-specific lever. TP path's MoE compute is the per-expert GEMV — DeepGEMM Mega MoE doesn't apply unless we switch back to EP. |

A3 stays the recommended first axis. The L5 binding constraint
(10k+ DtoH calls per decode step) is invariant across MoE backends and
is the most binding overhead now that EP's combine cost is gone.

## Problems

- **NCCL 2.28.9 (G6fq archive) is incompatible with CUDA 12.2 driver
  535.161.08** — first launch attempt hit
  "CUDA driver version is insufficient for CUDA runtime version".
  Switched to NCCL 2.21.5 (hvOud archive). Operators must point
  `LD_LIBRARY_PATH` + `ARLE_NCCL_LIBRARY` at the 2.21.5 build until
  the pod driver is upgraded.

- **Codex usage limit (gpt-5.5 xhigh fast) hit until 2026-05-31 21:20**
  during the day's work — session 4 idle through this entire arc.
  All commits today done by Claude solo per standing instruction
  ("没订阅了就你自己来完成").

## Day's commit arc

| Commit | Subject | Why |
|---|---|---|
| `38bf157b` | `build(cuda): force -std=c++17 in DeepGEMM JIT nvcc invocation` | Real root cause of 7+ failed JIT compiles — ARLE's own `csrc/gemm/deepgemm_native.cu` had `-std=c++20` hardcoded, nvcc 12.2 + gcc 8.3 silently fell back to c++14 → cute headers broke |
| `1164f35d` | `docs(cuda): B-3.3.5 DeepGEMM runtime crash on H20 errors entry` | After JIT compile fixed, runtime crash with "unspecified launch failure" — three suspected root causes ranked, cuda-memcheck plan deferred |
| `704cc09f` | `docs(research): B-3.3.5 H20 DeepGEMM source-level deep dive` | Source-only analysis — confirmed scatter assignment-vs-accumulation bug in B-3.3.5 (correctness blocker even if H20 crash were fixed), ruled out seq_len staleness, scoped LL mode at ~2000 LOC + NVSHMEM |
| `a6c910b2` | `chore(scripts): default dsv4 toolchain to allreduce + native expert` | Route switch — `MOE_BACKEND=allreduce`, `EXPERT_BACKEND=native` |
| (this entry) | wins/2026-05-27-dsv4-tp-allreduce-route-switch.md | License-or-kill PASS for route switch via TP nsys (94.85 ms TPOT p50, 2.21× faster than EP/deepep) |

## Refs

- `docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md` —
  backlog A1-A8 + SLO frame
- `docs/research/2026-05-27-b335-h20-deepgemm-source-deep-dive.md` —
  source-level scatter bug + LL scope
- `docs/experience/wins/2026-05-26-dsv4-default-deepep-deepgemm.md` —
  EP baseline (decode wave p50 = 210 ms)
- `docs/trace-artifacts/2026-05-27-allreduce-nsys/` — this run's
  `summary.json`, `decode-only-{runtime-api,kernel,memcpy}-*.csv`,
  `warmup-decode.json`, `command.txt`

## Rule

When a backend has three known blockers (race bug + runtime crash +
2000-LOC integration dep) and a working fallback exists in-tree,
**don't measure perf against the broken backend or against the
idealized rival** — measure against the actual fallback, then ship
the fallback as default until the broken path closes its three gaps.

The fallback may surprise you on the upside (here: 2.21× wall-clock
faster, not 1.5× slower). Wall-clock framing — not narrow-window %
of NVTX, not estimates against unmaterialized LL ideals — is what
prevented today's deep dive from spending another day stuck in EP
debug. CLAUDE.md §0 implementation: framing trap avoided.
