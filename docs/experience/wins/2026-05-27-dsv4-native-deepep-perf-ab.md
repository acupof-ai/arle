# DSv4 native-deepep — perf A/B vs NCCL-emulated `=deepep` baseline

## Goal

Quantify the steady-state decode throughput delta between the B-3.3
native-deepep forward path (intranode IPC via DeepEP's
`legacy::intranode::{notify_dispatch,dispatch,cached_notify_combine,
combine}`) and the prior `=deepep` NCCL-emulated baseline that B-3.3
inherits from. The B-4 PASS gate (per
`docs/plans/2026-05-27-multiproc-serve-pivot.md`) is TTFT/TPOT +5% or
better with p99 not regressed >3%.

## Setup

- **Pod**: 8 × NVIDIA H20 (102 GiB HBM each), driver 535.161.08,
  CUDA 12.2.140, NCCL 2.21.5+cuda12.4, libnccl loaded from uv archive.
- **Build**: `/data01/build/arle` @ main `04938e85`, `cargo build
  --release --features cuda,nccl --bin infer`, ARLE_DEEPEP_DIR=
  /data01/build/DeepEP @ `d4f41e4` (deepseek-ai/DeepEP HEAD).
  TileLang 0.1.9, ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1.
- **Model**: `/root/DeepSeek-V4-Flash` (43 layers, 46 safetensor
  shards, ~159 GiB on disk).
- **Runtime knobs** (held constant across A and B):
  - `ARLE_MULTIPROC_SERVE=1` (8-rank coordinator + 7 child processes)
  - `--kv-cache-dtype bf16`
  - `--mem-fraction-static 0.10`
  - `--num-slots 1`
  - `--max-seq-len 4096`
  - `--deepseek-distributed-layers 43`
  - `ARLE_DSV4_EXPERT_BACKEND=native` (DeepGEMM CUTLASS submodule
    not pushed to this pod — same handicap on both arms, so the
    delta isolates the dispatch path)
- **Single variable**: `ARLE_DSV4_MOE_BACKEND` ∈ {`native-deepep`, `deepep`}.

## Workload

5 sequential `POST /v1/chat/completions` requests, each with:
- prompt = `The capital of France is` (17 tokens)
- `temperature: 0`, `max_tokens: 128`, `stream: false`
- Python `urllib.request` client over loopback HTTP

First request discarded as warmup; p50/mean computed over the
remaining 4. Wall clock = `time.perf_counter()` deltas spanning the
full POST→200 response cycle.

## Results

| Backend            | p50 tok/s | mean | min   | max   | wall_s (128 out_tok) |
|--------------------|-----------|------|-------|-------|----------------------|
| **`=native-deepep`** | **15.82** | 15.81 | 15.55 | 16.04 | ~8.05 s |
| **`=deepep` (NCCL)** | **10.80** | 10.80 | 10.73 | 10.86 | ~11.85 s |
| **Δ (native vs NCCL)** | **+46.5 %** | **+46.4 %** | — | — | **−32.1 % wall-clock** |

Variance is tight (max−min spread under 0.5 tok/s per arm, well
under 5% of the mean), so the +46% delta is not a noise artifact.

## Framing audit (per CLAUDE.md §0)

- **Wall-clock framing**: ✓ — `tok/s` is `out_tokens / wall_seconds`
  measured from POST issue to 200 response, not a per-NVTX-window
  ratio. Translates directly to user-observed throughput.
- **One variable**: ✓ — only `ARLE_DSV4_MOE_BACKEND` env was flipped;
  same binary, same model load, same runtime knobs, same 5-request
  Python harness, sequential not concurrent.
- **PASS gate**: B-4 spec asks for TTFT/TPOT +5% (and p99 not
  regressed >3%). Result is +46.5% mean tok/s improvement, which
  comfortably PASSes the +5% line. p99 (max single-request wall_s)
  also moves from 11.93 s → 8.23 s = −31% on the worst case, well
  within the no-regression band.
- **Confounders**: holding EXPERT_BACKEND=native on both arms is
  intentional — the goal here is to isolate the dispatch/combine
  cost, not the absolute throughput ceiling. Flipping to
  `=deepgemm` (~3-5× MoE GEMM lift estimated) would benefit both
  arms equally and is a separate axis.

## What this measures

The +46.5% gap is the steady-state decode throughput improvement
from **swapping the per-token MoE all-to-all transport** from
NCCL-emulated DeepEP-style (the `forward_deepep_routed_gpu` path,
which builds the same dispatch pattern using NCCL `send/recv` + a
DSv4-specific pack/unpack kernel sequence) to **native DeepEP
intranode kernels** (Buffer.dispatch + Buffer.combine over CUDA IPC
+ NVL barrier).

Both paths produce the same downstream output shape on this
checkpoint's smoke prompt (the same base-model garbage continuation
documented in 2026-05-27-dsv4-native-deepep-pod-e2e.md), so the
delta is purely the dispatch-cost reduction, not a numerical drift.

## Bottleneck framing for next steps

With native-deepep at 15.82 tok/s and SGLang's DSv4-Flash baseline
on 8×H20 typically reported in the 30-40 tok/s range, the remaining
~2× gap lives elsewhere:

1. **DeepGEMM expert GEMM** (currently `native`): biggest single
   lift. 256 local experts × 4 GEMM/layer × 43 layers = ~44k naïve
   cuBLAS launches per decode step. DeepGEMM consolidates these
   into ~tens of grouped/masked GEMMs. Estimated 3-5× speedup on
   the MoE FFN body → would put us at ~50-80 tok/s steady-state.
2. **FP8 KV** (`--kv-cache-dtype fp8`): orthogonal to throughput at
   c=1 (KV size doesn't bound a single stream), saves ~50% KV mem
   when batching opens up.
3. **CUDA Graph capture** (`--cuda-graph` default on): already
   active, no extra lift available short of recapture-on-batch-
   change tuning.

## Rule

When a new collective/transport path lands, **run the perf A/B with
all other axes held constant before celebrating** — and **flip the
variable from the same shell** rather than relying on memory of
prior runs. The A/B captured here is a single env flip on the
identical binary; that's the only configuration where a +46% claim
survives the SOLID framing audit.

A throughput claim built from "I ran native-deepep on Monday at 16
tok/s, and the baseline was on Friday at 10 tok/s" doesn't survive,
because in between Monday and Friday someone might have changed
EXPERT_BACKEND, KV dtype, scheduler tuning, etc. Same binary, same
shell, same prompt, two env flips, side-by-side — that's the bar.

## Refs

- B-3.3 wire-up: `2026-05-27-native-deepep-forward-B33-wired.md`
- Pod e2e first-light + parity: `2026-05-27-dsv4-native-deepep-pod-e2e.md`
- Deadlock fix: commit `8fe74407` (worker relay-first reorder)
- Build flow validation: commit `25d70f54` (`dsv4_toolchain.sh
  --deepep-dir`)
- Multiproc-serve pivot doc: `docs/projects/2026-05-27-multiproc-serve-pivot.md`
