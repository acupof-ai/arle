# TileLang HD128 batched prefill fixed on sm_90a — hoist revert; BF16 needle baseline exact 15/15

**Date:** 2026-06-12 · **SKU:** H20 / sm_90a, CUDA 12.9 · **Model:** Qwen3-0.6B BF16
**Commit:** `930cb9d3` (kernel fix + errors-entry root-cause rewrite)

## Context

`batch_prefill_paged_hd128_q16_kv8` spun the device on every serve-driven prefill
since R6 bring-up (GPU 100%, no Xid, request never returns — even at 6 prompt
tokens). The 06-04 session classified it "hard TileLang codegen bug" and routed
around via chunk=1 decode forwards, which were never wired into serve — so H20
dense-Qwen serving was prefill-broken for 8 days.

## What Worked

**Root cause** (CUDA_LAUNCH_BLOCKING + gdb + reading the lowered `.cu`): the
page-lookup fragment hoist (`526515bd`, landed `pending-remote`, never run on H20)
makes TileLang emit a half-warpgroup predicated region
`if ((threadIdx.x >> 6) == 0)` **containing a `__syncthreads()`**. Threads 64–127
skip it, pair with the wgmma block's barrier → barrier slip → partial-warpgroup
`wgmma.mma_async` → device spin. Full chain in the updated
`errors/2026-06-04-tilelang-hd128-prefill-wgmma-hang-sm90a.md`.

**Fix:** inline the divmod + `KV_indices` gather back into the `(j, d)` copy loop
(pre-hoist form; the working decode kernel's pattern). Single-variable experiment:
only the hoist changed; regenerated device source has no predicated barrier.

**Verification ladder (all on the rebuilt pod binary, serve path):**

| Probe | Result |
|---|---|
| 5-tok prefill (`The capital of France is`) | " Paris. The capital of France is also…" — instant |
| 21-tok prefill | coherent continuation — instant |
| 213-tok prefill (4 q-tiles, 14 KV pages) | contextually correct answer |
| needle gate ×3, len=115/300/446/2000/8000 | **exact=3/3 at every length, DET** |
| 8000-tok prefill wall | 0.3–0.5 s (was: infinite hang at 6 tok) |

## BF16 baseline envelope (gate reference for the #68 dtype matrix)

`GATE_PROFILE=generic MODEL=/data01/models/Qwen3-0.6B scripts/lever_gate.sh
qwen06b_bf16_raw RAW=1` → `needle_gate_qwen06b_bf16_raw.log`:

```
SUMMARY len=115  depth=0.00 exact=3 partial=0 miss=0 DET
SUMMARY len=300  depth=0.00 exact=3 partial=0 miss=0 DET
SUMMARY len=446  depth=0.00 exact=3 partial=0 miss=0 DET
SUMMARY len=2000 depth=0.00 exact=3 partial=0 miss=0 DET
SUMMARY len=8000 depth=0.00 exact=3 partial=0 miss=0 DET
```

**Harness note (binding for the matrix):** the default `/v1/chat/completions` gate
path is structurally broken for thinking models at `max_tokens=16` — Qwen3-0.6B's
chat template enters `<think>` and burns the whole budget → all-miss baseline that
cannot discriminate dtypes (first run, label `qwen06b_bf16`, kept as the artifact).
All Qwen3-0.6B matrix runs MUST use `RAW=1` (raw `/v1/completions`; the cue
"The secret access code is" is completion-native). Param alignment: every dtype
run repeats the exact invocation above plus `SERVE_FLAGS="--kv-cache-dtype <dt>"`.

No perf delta claimed vs a prior baseline — there is none: this path never
completed a single request on H20 before. The entry IS the baseline.

## Killed hypothesis: "warmup runs decode with uninit PageMeta"

The debug session flagged garbage meta (`q_indptr=[2092116076, 32591]`) at warmup —
a probe artifact, not a bug. The garbage appears exactly at the second warmup pass
(the graph CAPTURE): under stream capture the `R6_ATTN_DEBUG` dtoh probes are
recorded into the graph instead of executed, so the dump prints the uninitialized
*host* staging Vec. Three-way confirmation: (1) the first (eager) pass dumps clean
meta through all 28 layers; (2) the capture-rejection error "112 host-side memcpy
nodes" = 4 probe arrays × 28 layers exactly; (3) small Vecs show heap garbage while
the 8192-entry `kv_indices` shows zeros — malloc-bin reuse vs fresh-mmap pages,
possible only for a host-side buffer. `stage1_write` initializes device meta before
every pass. **Rule:** a dtoh probe under stream capture reads its own uninit host
buffer — never interpret probe output produced inside a capture region.

## Rule

- A "route-around and defer" classification on a kernel bug is a standing tax —
  here it silently broke the serve path for 8 days because the route-around
  (chunk=1) was never actually wired in. Re-open deferred kernel bugs the moment
  the route-around's wiring status is unverified.
- Needle-gating a thinking model through its chat template with a small token
  budget measures think-mode verbosity, not retrieval. Gate raw completions, or
  budget past the think preamble.
