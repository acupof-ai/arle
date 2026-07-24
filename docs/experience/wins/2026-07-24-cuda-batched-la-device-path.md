# Batched linear-attention device path — B>1 crash fixed, 337 s CPU fallback gone

Commits `ecc058b20` (per-row dispatch + i64 state_history index) +
`5f68d1f6e` (gate models exact LA ctx bytes) + `f05642b68`/`bb17f5332`
(simplify passes). Pod-verified 2026-07-24 on H20 (GPU 0/3, snapshot binary
sha `d6fca3a2…`, Qwen3.5-0.8B rubric batched writeback).

## Context

Batched (B>1) linear attention never had a CUDA path: the device forward
bailed on `batch != 1`, so B=4 writeback ran a pure-CPU LA forward and a
CPU-orchestrated backward whose scan-assist kernel indexes `state_history`
with `int`. At B=4×3153×16heads×128×128 the index space is 3.31e9 > i32::MAX
→ negative offsets → `CUDA_ERROR_ILLEGAL_ADDRESS` at the `linear_attention
dqkv` dtoh (errors/2026-07-24-batched-checkpoint-la-backward-crash). The same
CPU fallback explained the 337 s/micro-batch checkpointed short-B=4 wall.

## What Worked

Per-row dispatch above the proven batch==1 kernels — no kernel changes:
slice batch-leading rows (contiguous D2D), run the B=1 forward/backward per
row, concat per-token results, sum weight grads. The chunked kernels stay
batch==1-only (chunk_state has no batch stride); `la_state_time_base`
widened to `long long` for the remaining non-128-dim fallback lane.

Measured (pod, snapshot binary at `5f68d1f6e`):

| Gate | Result |
|------|--------|
| `cuda_linear_attention_batched_grad_matches_cpu` (B=3, real CUDA) | pass; full file 11/11, batch=1 unregressed |
| Long B=4 seq≈3150 checkpointed (previously ILLEGAL_ADDRESS) | crash-free, **21–23 s/mb** (was: crash 354 s into mb1); phase-C peak 34.6 GiB; loss 0.0616 |
| Short B=4 seq=1040 checkpointed | **6–9 s/mb** (337 s CPU-fallback pathology gone); loss 0.1321/0.1329 vs pre-change 0.1317–0.1323 (parity) |
| `[ckpt-gate]` probe | engage=true at B=4 seq≈3150, modeled 41.26 GB ≈ the commit's 41.2 GB sanity |

Follow-up (issue filed): at B=4 seq=1040 the **full-tape** device footprint
measured ~79 GB vs 13.6 GB modeled (~5.8×) — `engage=false` passed a
385-MiB-headroom near-OOM on a clean GPU and a real OOM under 50 GB
co-tenancy. Gate boundary at mid-length batched shapes needs attribution +
re-tightening; checkpointing now costs only ~1.5–2× there, so the risk
asymmetry favors engaging.

## Rule

A batch>1 feature that "works" via a host fallback is not done — probe which
path actually ran (`[ckpt-gate]`-style one-liners) before trusting wall
times, and route batch through the proven single-row device path (contiguous
per-row dispatch) before writing new batched kernels.
