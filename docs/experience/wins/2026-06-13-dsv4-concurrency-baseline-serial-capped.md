# DSv4 concurrency baseline (d2 spec) — aggregate is SERIAL-CAPPED at 53 tok/s; replicated-attn deleted

## Context

Goal: completely build DP-attention for c>1 throughput, test concurrency, and
deletion-refactor the killed replicated-attn lane. Step 1 is the honest
concurrency baseline — does today's lockstep serve scale aggregate throughput
with concurrent clients at all? Without that number, "DP-attn for throughput"
is a hypothesis, not a licensed lever.

This entry is also the mandatory bench entry for the **replicated-attn
deletion refactor** (below): that refactor removes opt-in (default-OFF) code
only, so the DSv4 default decode path is byte-identical — the baseline here
*is* the post-refactor number.

## Params / Env

- 8×H20, TP=8/EP=8, DSv4-Flash, CUDA 12.9, sm_90a.
- Worktree binary (`/data01/build/arle-dsv4`), `--spec-type mtp
  --mtp-draft-tokens 2` (d2 chain-fold, default-on), `MOE_BACKEND=allreduce`,
  `EXPERT_BACKEND=deepgemm`, native DeepGEMM, incremental KV, auto NUMA pin.
- Client: `conc_bench.py` — warm once, then C concurrent threads, each a
  128-token `temperature=0` completion to `/v1/completions`. Serial GPU
  (no other client; a wedged 2h client had contaminated earlier runs —
  killed, serve restarted clean, gated on a real non-empty completion).

## Results — aggregate is FLAT, wall-clock scales linearly with C

| c | agg tok/s | per-req tok/s | wall (s) | wall vs c×(c=1) |
|---|---|---|---|---|
| 1 | 53.3 | 53.3 | 2.40 | — |
| 4 | 53.2 | 13.3 | 9.62 | 4×2.40 = 9.62 ✓ |
| 8 | 53.2 | 6.6 | 19.25 | 8×2.40 = 19.25 ✓ |
| 16 | 53.2 | 3.3 | 38.49 | 16×2.40 = 38.49 ✓ |

**Aggregate is dead flat at ~53 tok/s regardless of C; wall-clock is exactly
C × single-stream.** per-req = agg / C — the signature of pure serialization.
But the flat cap is **not** "lockstep cannot batch" — it is that **batched
decode is opt-in** (`--dsv4-batched-decode` / `INFER_DSV4_BATCHED_DECODE`,
serve.rs:69) and this serve had it **off**. See the control below.

## Control — batched-decode ON (no spec), same binary, same c-sweep

| c | batched agg tok/s | per-req | vs default (d2 no-batch) |
|---|---|---|---|
| 1 | 44.4 | 44.4 | −17% (no spec) |
| 4 | 53.8 | 13.5 | +1% |
| 8 | 57.4 | 7.2 | +8% |
| 16 | **62.0** | 3.9 | **+17%** |

The batched lockstep lane **works on the current binary**: aggregate *rises*
44→62 with c (not flat), so lockstep **can** co-batch concurrent decodes. Two
things follow:

1. **The immediate throughput win is the batched-decode default flip (#60)** —
   it already exists and lifts c=16 to 62. (c=1 batched 44.4 < d2-spec 53.3
   because this arm ran *no spec*: spec is the single-stream win, batching is
   the concurrency win; composing batched+spec is an open follow-up — they were
   not co-enabled here.)
2. **But batching scales WEAKLY — 1.40× aggregate for 16× the load.** A single
   TP=8 group still pays the per-token attention collectives (Q all-gather + O
   all-reduce) and lockstep skew on every step; those do not amortize with
   batch size, so they cap the scaling. **This weak curve is the real, measured
   motivation for DP-attention** (#89) — DP-attn removes the per-token
   attention-collective floor that batched-TP-decode cannot. The ceiling is the
   admission + attention-collective model, not the GPU.

## Deletion refactor — replicated-attn removed (default byte-identical)

replicated-attn (opt-in `ARLE_DSV4_REPLICATED_ATTN=1`) was correct but −48%
at B=1 ([kill entry](../errors/2026-06-13-dsv4-replicated-attn-bandwidth-kill.md)).
It is the *opposite* of DP-attn (every rank redundantly computes ALL requests
vs. each rank computing DIFFERENT requests), and its scalar grouped
o-projection is the wrong impl for any salvage (DP-attn needs a vectorized
o-proj). Keeping it unwired would be a multi-week half-state. Deleted whole:

- `attention.rs`: `dsv4_replicated_attn_enabled`, the `replicated` gate +
  param threaded through `try_flashmla_decode_attention` / `mla_oproj`, the
  grouped-o-projection branch, all `if replicated`/`!replicated` arms collapse
  to the existing sharded path.
- `dsv4.rs`: `wq_b_full`/`wo_a_groups`/`wo_b_full` fields,
  `replicated_attn_active()`, and the 4 AR-skip gates → unconditional
  all-reduce (the default path always took these).
- `loader.rs`: `load_dsv4_wo_a_groups` + the full-width conditional load.

Every kept branch is the pre-existing non-replicated default; replicated was
default-off, so the default decode path is byte-for-byte unchanged. The
grouped-o-projection design is preserved in the kill entry for re-derivation
(vectorized) when DP-attn is built. Mac CUDA-Rust typecheck clean; pod rebuild
+ needle smoke confirm the default path (see verify section in commit).

## Rule

- **A flat default curve is "a lane is off," not "the lane can't exist" — run
  the opt-in control before attributing root cause.** The default flat-53 looked
  like "lockstep serializes," but batched-decode was simply opt-in and off;
  enabling it (the control) rose 44→62. Attributing the cap to lockstep without
  the `INFER_DSV4_BATCHED_DECODE` control would have mis-licensed DP-attn as the
  *only* fix when the immediate fix is the #60 default flip. (§0 root-cause
  license-or-kill: the root-cause hypothesis itself needs a control experiment.)
- **per-req = agg/C is the serialization fingerprint — but read it per-config.**
  It is true for the *default* (batched off); the batched arm scales (1.40×),
  just weakly, because per-token attention collectives don't amortize with batch.
- **Separate the immediate lever from the structural one.** Batched-decode (#60)
  is the existing, cheap concurrency win (default flip). DP-attn (#89) is the
  structural lever *beyond* it (removes the attention-collective floor that caps
  batched-TP scaling). Don't conflate "DP-attn is the lever" with "DP-attn is
  the only lever."
- **Delete the whole killed lane, not just its gate.** A killed opt-in path
  whose impl is the wrong shape for the successor (scalar vs vectorized
  o-proj) is dead code in the final ideal state — remove it, preserve the
  design in the kill entry, re-derive correctly when the successor lands.
