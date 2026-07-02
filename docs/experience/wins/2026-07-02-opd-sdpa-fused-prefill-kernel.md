# OPD forward 14.4s → 3.8s — training SDPA on the inference prefill kernel

## Context

After the FP8-GEMM fix ([2026-07-02-opd-fp8-gemm-cublas-dequant](2026-07-02-opd-fp8-gemm-cublas-dequant.md))
the masked-writeback forward was 14.4s with full-attn layers at ~770ms
(sync=false wall). The training SDPA composed score-matrix primitives with a
host-built causal mask; b92ec601 routed the forward through the production
inference kernel `nonpaged_prefill_attention_cuda` (bf16, online softmax, GQA
native) behind a new `causal_sdpa_prefill_device` backend method.

**First A/B came back FLAT** (run-sdpafuse-toy1r: forward 14.7s, full-attn
unchanged) — verified negative, correctly not written up as a win. Every
fused-path rejection was a silent `Ok(None)`, so attribution needed a probe:
`ARLE_SDPA_TRACE=1` (8426adb2) prints TAKEN/REJECTED + shapes per call, and
`ARLE_OPD_PROFILE_SYNC=1` gives true per-layer kernel walls.

## What Worked

The trace decoded the flat result in one run (run-sdpatrace-toy1r, 17aeebcc):
**32× `fused=TAKEN q=[1, 24, 1010, 256]`, 0× REJECTED** — the kernel was never
envelope-rejected. The b92ec601 flatness was the head-chunk wrapper
(`ATTN_HEAD_CHUNK=8`) slicing the call 8-heads-at-a-time and stitching with
host `cat_heads` (readback + host copy per chunk) *around* the fused kernel;
~770ms was the wrapper's host work. Deleting the wrappers (4ee47196 — folded
the transient budget into the SDPA entry points, fused-first over all heads)
was the unlocking change, not an optional cleanup.

Same toy config, GPU 7, RUN_EXIT=0, TileLang full AOT regen clean:

| metric | fp8dq baseline | sdpatrace (17aeebcc) | Δ |
|---|---|---|---|
| forward_hidden_states | 14.416s | **3.768s** | **−74%** |
| forward layers sum | 13.496s | 3.180s | −76% |
| full-attn layer wall (31/43/47/63) | ~770ms | **96–97ms** | −87% |
| linear-attn layer wall (1/4/61/62) | 25–27ms (async) | 32.3–32.6ms (sync=true) | true kernel time |
| backward | 38.003s | **26.252s** | −31% |
| loss | 0.279333 | 0.282402 | in 0.24–0.33 band |

Cumulative vs the pre-FP8-fix baseline (137ffb28): forward 122.1s → 3.8s
(**32×**), backward 149.1s → 26.3s (5.7×).

Next wall: backward now dominates (26.3s vs 3.8s forward) — the composed
`causal_sdpa_recompute_backward` gradient math and the rest of the backward
lane; full-attn still carries ~65ms/layer over linear-attn.

## Rule

- A flat A/B on a "wired" fast path needs call-site truth, not code-reading
  truth: one env-gated TAKEN/REJECTED trace line beats re-auditing guards.
  Silent `Ok(None)` fallbacks are unattributable by design — don't ship them
  without a probe switch.
- sync=false per-layer walls attribute queued async work to whichever layer
  syncs first; license per-layer claims only from `ARLE_OPD_PROFILE_SYNC=1`.
- The wrapper *around* an adopted kernel is part of the operator: adopting
  SOTA and leaving a host-stitching chunk wrapper in front of it delivered 0%.
