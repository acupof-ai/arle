# DSpark experience capture: DSv4 lane was 100% dead, and post-bias (#176)

> Status: Fixed. Runtime counter verify **pending-remote** — needs a DSv4
> `--spec-type dspark --dspark-train` serve on the pod to watch the buffer fill.
> Opt-in path only; the default serve is unchanged except for one deleted copy.

## Context

The DSpark RL sidecar is a live loop: the hot path pushes
`(draft_tokens, draft_logits, target_logits, accepted)` into a ring buffer, a
trainer thread drains it, runs acceptance-weighted PG on the Markov head, and
hot-swaps the weights back into the serving engine (`--dspark-train`, wired at
`cli/serve.rs:196`). #176 reported two coupled defects in the producer. Both
confirmed at the token/shape level before any edit:

**1. Every DSv4 push was dropped.** `dsv4/dspark.rs` builds
`chain = [anchor] + drafts` (len `draft_len + 1`) but `draft_logits` with
`draft_len` rows. The call site passes the whole `chain` as `draft_tokens`, so
the guard computed `vocab_size = draft_len·vocab / (draft_len+1)` — never equal
to `vocab` for any `draft_len ≥ 1` — and rejected the push. The DSv4 lane fed
the trainer *nothing*; only the qwen35 full-block same-position lane got through.

**2. The rows were post-bias.** DSv4 retained `corrected` (base + Markov bias)
while qwen35 captures raw `scratch.logits`. The trainer itself re-derives the
bias (`dspark_train.rs`: `corrected = add(logits, matmul_bt(embedding(w1), w2))`),
so fixing (1) alone would have trained a double-biased objective — and against a
*stale* bias, since the serve's `w2` at capture time lags the trainer's live one.

## What Worked

Attribution first, and the fix fell out as a deletion:

- **Raw rows for free.** `base_logits` (`[vocab, block]`, pre-bias) is already
  computed and still alive where `draft_logits` is built — so sourcing from it
  costs nothing. Its only consumer chain was
  `corrected` → `draft_logits` → the capture, so the whole `corrected` retain
  buffer became dead: one `[vocab, block]` allocation and **one vocab-sized D2D
  copy per draft step** removed from the hot path. Both lanes now hand the
  trainer the same thing (raw base rows), which is what it always assumed.
- **Kill the inferred vocab, not just the off-by-one.** The guard's real defect
  was deriving a *width* from a *ratio of two counts that may legitimately
  differ*. `vocab` now comes from the authoritative source — the target's
  `hidden_dim` on the DSv4 path, an explicit parameter on the qwen35 path — and
  the guard only checks that the width divides the draft buffer and that the
  target covers the chain. Row counts are free to differ (confidence truncation
  does that too), and the trainer already recomputes them itself.

Gates: infer-cuda clippy `-D warnings` + fmt clean; `train` dspark tests 4/4
(including `dspark_trainer_serve_frame_and_alignment`, the row-alignment gate).

## Rule

- **A guard that drops 100% of one lane's data is silent by construction.** It
  logged a warning per drop, and nobody was reading serve logs for a
  default-off training sidecar. A producer whose pushes can be rejected needs a
  *counter* the operator can see (pushes vs drops), not just a per-event warn.
- **Never derive a width from a ratio of counts.** `vocab = bytes / rows` is
  only right while the two agree; the moment one path adds an anchor row or
  truncates, it silently becomes wrong. Take the width from whoever owns it.
- **Check for the raw value before allocating to keep it.** The pre-bias rows
  were already resident — the "expensive" fix (retain a second buffer) would
  have added memory to duplicate data that was one slice away.
