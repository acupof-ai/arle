# DSpark-for-DSv4 TP=4 lockstep deadlock fixed — rank-0-authoritative proposal

## Context

DSpark speculative decode for DeepSeek-V4-Flash under TP=4 (`--spec-type dspark`,
8×H20, GPUs 3-6). The full pipeline loaded correctly (stages=3 block=5
target_layers=[40,41,42], 294 GB fp8 across 4 ranks) and HTTP came up, but the
first decode **deadlocked at tick #11** (`lockstep ack wait exceeded 120s,
min_acked=7`) — no panic, no CUDA error. Ran ~7-10 ticks in lockstep, then hung.

## Root cause

The draft proposal (`dspark_build_proposal`: semi-AR Markov sampling +
confidence truncation over the draft backbone's `block_hidden`) is computed
**rank-locally**. Under TP the ranks' `block_hidden` drifts by FP; after ~7
tokens the drift crosses a confidence/sampling boundary → each rank's
`draft_len` diverges. A divergent `draft_len` feeds `forward_tokens_verify` a
different token count per rank → the per-forward collective **count** mismatches
across ranks → the lockstep coordinator's all-ranks ack never completes → hang.

The draft forward itself is fixed-shape (always `block_size` positions), so its
collectives stay lockstep — only the **variable-length verify** desyncs.

## What Worked

`TpRuntime::broadcast_rank0_i32` (all_gather + take rank-0's slice; identity on
single-rank / non-NCCL). Two rank-0-authoritative broadcast points in
`dspark_decode_tokens`:
1. **Before verify** — broadcast `{draft_len, chain}` → every rank verifies an
   identical shape+tokens → lockstep collective count.
2. **After accept** — broadcast `{accepted, bonus}` → every rank truncates+commits
   an identical KV tail (the next tick's combined attention reads all ranks' KV
   shards; an inconsistent commit would corrupt it).

Greedy correctness gate: the argmax accept ignores `draft_logits`, dropped on the
rank>0 path when the adopted chain differs. Sampled-mode `q` reconciliation is a
separate lever (noted inline).

### Result (GPUs 3-6, TP=4, `The capital of France is`, greedy, max_tokens=16)

- **Deadlock gone.** Decode completes in ~30 s, 16 tokens, curl exit 0.
- **Output coherent** — begins `" Paris."` (correct), not garbage → the fix is
  correctness-preserving, not a paper-over that breaks output.
- `/v1/stats spec_decode`: `available:true, drafted:4, accepted:0,
  accept_rate:0.0` → the draft path executed and stayed lockstep across all 4
  ranks through verify.

Commit `d4c25f1c2`. Files: `crates/infer-cuda/src/tp.rs`,
`crates/infer-cuda/src/executor/dsv4.rs`.

## Open — draft acceptance is 0%

`accept_rate=0.0`: every draft is rejected, so the base token is committed each
step (output stays correct, zero speedup — spec is pure overhead right now). This
is the separate draft-quality lever (candidate: post-attention inverse-RoPE on
the draft head, or an accept-alignment off-by-one). Attribute at token level
(decode drafts vs verify argmax) before fixing — do NOT blind-fix de-RoPE.

## Rule

TP spec-decode proposals must be **rank-0-authoritative** — any rank-local
data-dependent length (confidence truncation, dynamic draft length, accept count)
feeds a variable-shape verify whose per-forward collective count must match across
ranks, or the lockstep coordinator deadlocks. Broadcast the decision, don't
recompute it per rank. The tell: deadlock after N>1 clean ticks (not tick #1) =
FP drift crossing a data-dependent boundary, not a static branch mismatch.
