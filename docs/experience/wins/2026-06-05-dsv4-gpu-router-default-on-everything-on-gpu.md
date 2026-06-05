# DSv4 everything-on-GPU: gpu-router default-on (kills per-layer host-route D2H) + the 4096 non-determinism finding

**Date:** 2026-06-05. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** flip landed (default-on, opt-out `ARLE_DSV4_GPU_ROUTER=0` kept pending
fallback deletion); 64-tok token-exact verified; clean decode `tok/s` A/B
**pending-remote** (the 4096 token-exact A/B is invalid — see below). Task #31.

## Context

Default decode is the **eager** path (`dsv4_decode_graph_enabled()` default-off),
which routed MoE **on the host every layer every token**: `ctx.sync()` +
`clone_dtoh(logits)` + host top-k + `clone_htod` back (`moe.rs` eager ~1119,
deepep ~1524). The on-device router (`dsv4_route_device` → `dsv4_route` kernel)
was built + wired at all route sites but gated off behind `ARLE_DSV4_GPU_ROUTER`.

The prior audit claimed the GPU router "skips group-limited routing → wrong
experts." **That was a source-survey hypothesis and it is false for DSv4-Flash:**
`config.rs:131` + `MoeConfig::dsv4` hardcode `n_group/topk_group = None` ("ships no
group-limited routing"), so `route_token` applies no group mask and the device
kernel's plain bias-corrected top-k (SqrtSoftplus) is **algorithmically identical**
to the host. The decode-graph path already runs this exact kernel unconditionally
and was oracle-verified (#25, 16/16).

## What Worked

- **`use_gpu_router()` default-on** (`moe.rs`), opt-out `ARLE_DSV4_GPU_ROUTER=0`.
  Eager + deepep decode now route fully on-device — no per-layer logits D2H, no
  per-layer `ctx.sync()`. Bias stays a device tensor; only Hash-layer `token_ids`
  keep a tiny [N]-int H2D (input staging, unavoidable).
- **64-token A/B token-exact**: gpu-router default == `ARLE_DSV4_GPU_ROUTER=0`
  host route, 16/16 identical. The on-device path matches the host at a shape
  where the model is deterministic.

## The 4096 non-determinism finding (why token-exact-at-4096 is a dead bar)

The 4096-token A/B looked like a failure — gpu-router `[539]` vs host `[1345]`.
A **same-config determinism control** (scalar default, run twice at 4096, slot
4176) settled it: `clean_tokens` matched (`[344]`/`[344]`) but the **per-layer
attention-output hashes diverged from layer 1 onward** (`ARLE_DSV4_ATTN_DUMP`).
A SlidingWindow layer differing between two identical runs proves **run-to-run
non-determinism** — MoE expert-scatter accumulates with non-associative float
adds. At 64 tokens the noise is sub-argmax-flip; at 4096 it can flip the greedy
argmax. So the 539/1345/344 spread was non-determinism + slot-config sensitivity,
**not a router (or FlashMLA-prefill) bug.** See
[[reference_dsv4_moe_nondeterminism_confounds_4096_parity]].

## Rule

A long-shape token-exact-vs-reference parity is **invalid** when the runtime is
run-to-run non-deterministic — run a same-config-twice control *first*; if it
diverges, switch the verdict to needle-retrieval / quality, not token-exact. And:
a source-survey "kernel is missing stage X" claim is hypothesis-grade — verify it
against the model's actual config (DSv4-Flash `n_group=None`) before believing it
blocks a default flip. The everything-on-GPU win is real (per-layer host sync
removed); the decode `tok/s` magnitude lands once a needle-verified clean A/B runs.
