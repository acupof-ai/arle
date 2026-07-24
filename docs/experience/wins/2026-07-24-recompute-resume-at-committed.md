# Recompute preemption resumes at the committed position — no duplicate stream (#156)

> Status: Shipped (correctness). Bench: pending-remote — scheduler-path change;
> canonical throughput run vs the rolling champion owed on the H20 pod.

## Context

A `Decoding` request preempted onto the plain-recompute path cleared
`generated_tokens` and restarted from the prompt; the token observer re-forwarded
every regenerated token, so a streaming client saw the generation twice — and a
*different* second copy under temperature>0 (#156, pre-existing recompute
semantics).

## What Worked

vLLM-style resume-at-committed, aligning plain recompute with the whole-slot
swap path (one canonical resume semantics):

- `reset_for_recompute` preserves `generated_tokens`; prefill targets the
  committed stream (prompt + generated) via `committed_len/tokens/slice`
  helpers; the prefill→decode transition fires at committed length, so the only
  observed token after resume is the newly sampled one.
- Admission + both recompute fallbacks radix-match against the committed
  stream — the requeue-time `publish_prefix_blocks` becomes self-serving: the
  resumed request re-attaches through its own published generated blocks.
- max_tokens/EOS accounting unchanged (`generated_tokens` never cleared).
- **Codex-review P1 fixed in the same diff**: a DSv4 finish-write-through
  sidecar could restore the full committed stream and enter `Decoding` with the
  last token's KV already materialized (decode seed re-feeds it → silent KV
  duplication). The restore clamp now holds one token back
  (`restored.min(target-1)`), same contract as the full-match trim.

Gates: new e2e `recompute_preemption_never_reemits_committed_tokens` (observer
stream has zero duplicates and equals final `generated_tokens`) +
`frontier_tail_restore_still_prefills_the_last_token`; infer-core 110/110,
clippy -D warnings clean, cuda-lane typecheck clean.

## Rule

- Any resume path (recompute, swap restore, sidecar) must end with KV
  materialized to exactly committed−1 and the decode seed sampled from real
  logits — a fully-materialized restore double-appends the seed token's KV.
