# DSv4 batched MTP Stage 1 — correct but −44% @c=4 (un-batched the attention per-row MTP already batched)

## Context
Batched MTP (理想态, production steady c≥4 `--spec-type mtp`): batch the per-row
`spec_step`'s draft+verify across N slots ([plan](../../plans/dsv4-batched-mtp-decode.md)).
Stage 1 (`a36725f0`, gated OFF `ARLE_DSV4_BATCHED_MTP`) batched the MoE over all
M=Σ(depth+1) verify rows but kept attention **per-slot**, using the per-row
ring-replay path (sub-mode 2) — which I briefed as "the safe reference."

## What happened (KILL — perf, correctness intact)
Pod 8×H20, same binary, two env flips, steady /v1/stats window:

| arm @ avg_active=4 | agg decode tok/s |
|---|---|
| per-row MTP (control) | 42.37 |
| batched MTP Stage 1 | 23.63 (**−44%**) |

Correctness PASS (the kill is purely perf): decode-read c=4/c=8 both arms 4/4
coherent, **identical answers**, MTP acceptance normal (~1.98). `[dsv4-mtp-batched]`
fired (360× in the coherence run) — the path engaged.

## Root Cause
Per-row MTP's verify (`forward_tokens_verify_scheduled` → `dsv4_mtp_tree_attn_enabled`
default ON) uses the **tree-attn batched lane** (sub-mode 1, `dsv4.rs:2699`): ONE
FlashMLA over the chain's 3-row chunk per layer. My Stage 1 verify used the **per-row
ring-replay** (sub-mode 2, `dsv4.rs:2737`): one FlashMLA PER ROW. So at c=4
(depth=2 → 3 rows/chain):
- per-row MTP attention: 4 slots × 1 tree call = **4 FlashMLA/layer**.
- Stage 1 attention: 4 slots × 3 rows = **12 per-row FlashMLA/layer** (3×, each a
  tiny gridX=1 launch + 3 memcpys).

**The batched lane batched the cheap axis (MoE, 60.8%) while UN-batching the attention
that per-row had already batched.** The MoE amortization (1×12-row GEMM vs 4×3-row)
could not offset the 3× attention-launch explosion. Faster-MoE, slower-overall.

Second finding: the serve **caps decode at c=4** (`max_active=4 max_queue=5`) —
`num_slots` from `max_seq_len=8192` (large per-slot KV). The "steady c≥8" production
scenario needs `max_seq_len` lowered (more slots) to even measure.

## Fix
Swap Stage 1's verify attention to **sub-mode 1 (tree-attn) per slot**: build a
`Dsv4TreeAttnMeta` from each slot's `SpecVerifySchedule`, ONE `mla_attention` over the
slot's chunk/layer (4 tree calls/layer = matches per-row MTP) into the combined
`[M,hidden]` buffer, then the batched MoE over all M rows. Decouples from commit-fold
(keep re-forward commit; tree-attn here writes no rings, persists no `spec_normed`).
Then attention == per-row, MoE+norm+HC+allreduce all amortized → net win. Re-measure at
real c≥8 (`INFER_DSV4_MAX_SEQ_LEN` low enough to uncap slots).

NOTE the residual: Stage 1 batches only the VERIFY; the per-slot DRAFT
(`mtp_forward_level`) and COMMIT re-forward stay looped. The full 理想态 batches all
three — but fixing the attention first is the −44%→win step.

## Rule
- **A "batched" lane must batch the axis that was the bottleneck, not just the easy
  one.** Per-row MTP had ALREADY batched the chain attention (tree-attn); batching the
  MoE while reverting attention to per-row is a net loss. Before batching, check what
  the per-row path already batches — don't un-batch it.
- **"Safe reference sub-mode" ≠ "the sub-mode the reference path uses."** sub-mode 2
  (ring-replay) is *a* validated path but NOT the default per-row MTP verify path
  (sub-mode 1 tree-attn); briefing the slower one cost a build+bench cycle. Match the
  production path's kernel, not just any correct one.
- **Verify the concurrency the SLO needs is reachable before perf-judging it**
  ([[feedback_verify_slo_lane_runs_before_optimizing]]): the serve capped at c=4
  (`max_seq_len`→num_slots); a c=8 verdict needs the slots uncapped.
- Correctness gate held: decode-read coherence at c≥4 confirmed the path is correct;
  the kill is wall-clock perf only (§0 ground truth).
