# DSv4 auto-context ceiling (`29fdda704`) caps the value but does not fix the crash at TP=4

> Superseded/corrected 2026-07-06: the old path returned an `anyhow::ensure!` error, not a Rust panic. `ba36fbd39` added the missing FlashMLA-band pre-check; infeasible TP4 `max_seq_len=32768` now rejects earlier with an actionable message. Historical boot results below are unchanged.

## Context

Pod-verifying `29fdda704` ("cap DSv4's auto-resolved max-context at a known-safe
ceiling") on real CUDA hardware (8×H20), the follow-up fix to the crash found
pod-verifying `b7cd9a2e7` (documented in
`docs/experience/errors/2026-07-06-dsv4-max-total-tokens-pod-verify.md`). That
prior doc found: DSv4 serve with no `--max-total-tokens`/`--max-prompt-tokens`
flags auto-resolves `max_total_tokens` from the checkpoint's
`max_position_embeddings` (1,048,576 for `DeepSeek-V4-Flash-FP8`), and a single
slot's fixed FlashMLA band at that size (4098 pages) exceeds the pool budget
the scheduler's `affordable` gate left available (3344 pages) — a hard
`ensure!` panic in `crates/infer-cuda/src/attention/kv_layout.rs:1017-1025`
that crashes all worker ranks (TP=4, GPUs 4,5,6,7).

`29fdda704`'s fix: `crates/cli/src/serve.rs`'s auto-context block now clamps
via `infer_api::cuda_model_is_dsv4(&model_path)` →
`max_ctx.min(infer_api::DSV4_AUTO_CONTEXT_CEILING)`,
`DSV4_AUTO_CONTEXT_CEILING = 32768` (`crates/infer-api/src/loaded.rs:1605`) —
restoring the pre-`b7cd9a2e7` default value for the no-flags case.

**Pod state:** synced clean to `29fdda704` (no `.git` loss this time, unlike
the prior session). `cargo build --release --features cuda,nccl,deepep --bin
arle`: `BUILD_EXIT=0` in 47.69s (warm cache). **GPUs:** 0, 2, 3, 4, 5, 6, 7
free; GPU 1 held by a foreign tenant throughout (51 GB, 9-98% util, untouched).
GPU 0 also carried a live foreign tenant partway through
(`arle serve --model-path /host/Qwen3.6-27B-FP8`, PID 1198140, another agent's
terminal-bench session — untouched). Used GPUs 4,5,6,7 (TP=4,
`INFER_CUDA_DEVICES=4,5,6,7 INFER_TP_SIZE=4`) — the same config the prior
verify session used (GPU1 unavailable precluded TP=8).

## Root Cause

Three boots, same model/GPUs/binary:

**Boot 1 — no flags** (the fixed scenario):
```
INFER_CUDA_DEVICES=4,5,6,7 INFER_TP_SIZE=4 ./target/release/arle serve \
  --model-path /host/DeepSeek-V4-Flash-FP8 --backend cuda --port 18191 \
  --max-running-requests 2
```
Log confirms the new clamp fired as designed:
```
DSv4 max context: auto-resolved to 32768 from
/host/DeepSeek-V4-Flash-FP8/config.json (max_position_embeddings=1048576)
```
But the server **still crashes**, via the identical code path as before, just
at smaller numbers:
```
[rank*] DSv4 KV budget: free 24171MB, per_slot 317MB ... affordable 68
[rank*] DSv4 KV budget: requested 256 slots ... clamping num_slots to 68.
[rank*] TokenKVPool: 4736 max tokens (74 pages @ page_size=64) ...
[arle-worker rank=*] failed: worker rank * engine build: DSv4 FlashMLA pool
  page mismatch: page_size=64 pages=74 need page_size=64 pages>=130
worker rank * exited Some(1)   (all 4 ranks)
[ARLE serve] multiproc coordinator setup failed: ... aborting coordinator
RUN_EXIT=1
```
**Server never reaches ready state — hard crash on all 4 ranks, same as
before the fix.** The `affordable` gate (`dsv4.rs:1867`) computed
`affordable=68 > 0` (so its own reject-below-fixed guard passes), but the
FlashMLA pool it sized only has 74 pages while a single slot's fixed band at
`max_seq_len=32768` needs 130 — the exact reconciliation gap the prior verify
session identified and explicitly deferred as out-of-scope, still present,
just triggered by 32768 instead of 1,048,576.

**Boot 2 — explicit `--max-total-tokens 16384 --max-prompt-tokens 16000`**
(unaffected-path regression check): boots clean, `CUDA engine: executor
clamped slots 256 -> 121; scheduler follows`, all 4 ranks
`engine-ready ack sent`, `all 4 worker engines ready; opening HTTP`,
`serving OpenAI v1 on http://127.0.0.1:18192`. **Decode-check** ("What is the
capital of France? Answer in one word.", `max_tokens=50`, `temperature=0`):
```json
{"content":"Paris","reasoning_content":"The user asks for the capital of
France and specifies to answer in one word. The answer is straightforward:
Paris."}
```
Correct, unaffected by the fix — as expected (16384 < 32768, never hit the
new ceiling branch's clamp; behavior identical pre/post `29fdda704`).

**Boot 3 — explicit `--max-total-tokens 2000000 --max-prompt-tokens 1999000`**
(ceiling-bypass check, mirroring the C4 script's intent): confirms the
ceiling did **not** leak into the explicit-value path — `max_ctx` is never
clamped when a flag is passed (`serve_args.max_total_tokens.is_none()` guards
the whole block). But at TP=4 this does **not** hit the graceful
"rejected startup... affordable 0" path documented in
`docs/experience/wins/2026-06-12-...` (that doc's clean reject was TP=8, a
much larger free-VRAM-per-rank config): `affordable=1 > 0` here too, so the
same hard `ensure!` panic fires — `pages=3085 need pages>=7815`. This is
**not a new regression** (the explicit-value path was never in scope for the
ceiling fix and its crash-vs-reject behavior is unchanged pre/post
`29fdda704`), but it does confirm the underlying reconciliation gap is not
narrow — it reproduces across three very different `max_seq_len` values
(32768, 1048576 from the prior doc, 2000000) whenever TP=4's tighter
per-rank free VRAM (weights alone use 74745 MB/97 GB per GPU at TP=4, leaving
only ~22–24 GB free) lands `affordable` at a small-but-nonzero count.

All three boots self-terminated cleanly — GPUs 4-7 back to 0 MiB after each,
no zombie processes, no manual cleanup needed.

## Fix

**Not yet fixed.** `29fdda704` correctly restores the old 32768 auto-resolve
default and correctly scopes the clamp to only the auto-resolve path (verified
the explicit-override path is untouched) — but 32768 is **not actually safe**
on this checkpoint at TP=4: the crash the fix set out to resolve is still
reproducible, just requiring a smaller ceiling breach than before. The
"known-safe ceiling" premise in the commit message assumes 32768 was
previously exercised successfully at TP=4 with this reconciliation gap
present — that assumption is unverified and this test suggests it's false for
this GPU-count/free-VRAM combination.

The actual fix needed is what the prior doc already deferred: reconcile
`dsv4_kv_budget_plan`'s `affordable` gate (`crates/infer-cuda/src/dsv4.rs`)
with `flashmla_slot_pages` (`crates/infer-cuda/src/attention/kv_layout.rs`) so
that a single slot's fixed FlashMLA band size — which scales with
`max_seq_len` alone, independent of `num_slots` — is checked *before* clamping
`num_slots`, and fails closed with the same graceful "rejected startup...
affords 0 slots" message instead of a hard `ensure!` panic when even one slot
doesn't fit. A context ceiling on the auto-resolve path is a reasonable
mitigation for the common case but does not close the gap; the fix should
target the reconciliation, not just pick a smaller default.

## Not yet pinned down (deferred)

- Whether TP=8 (still untested — GPU1 occupied by a foreign tenant throughout
  this session) clears 32768 cleanly. TP=8 halves the per-rank weight
  footprint (vs. TP=4's 74745 MB/rank here), leaving far more free VRAM per
  rank — plausible that 32768 (and even larger values) fit fine at TP=8. If
  DSv4's only supported/expected production topology is TP=8/EP=8 (per
  `CLAUDE.md`'s support matrix), this ceiling fix may be adequate for the
  production shape and only failing on the non-production TP=4 shape used
  here for GPU-availability reasons — but that has not been measured, only
  hypothesized from the free-VRAM arithmetic above. Needs a real TP=8 boot to
  confirm or kill this hypothesis.
- The exact `affordable` threshold at TP=4 below which the FlashMLA band
  stops fitting (somewhere between 16384 — works, affordable presumably
  higher and pool_per_layer/pages sufficient — and 32768 — affordable=68,
  pool=74 pages, needs 130). Not bisected.

## Rule

- **A "restore the old default" fix is not the same as "fix the bug" when the
  old default was never verified against the specific reconciliation gap in
  play.** `29fdda704` correctly reverts the *symptom trigger* (the auto-resolved
  value) back to its pre-refactor size, but the underlying two-gate
  disagreement (`dsv4.rs` `affordable` vs. `kv_layout.rs` FlashMLA pool
  `ensure!`) is untouched and still reachable at the "known-safe" value on a
  tighter-VRAM topology (TP=4) — confirmed by reproducing the identical crash
  signature at 32768, not just at 1,048,576 or 2,000,000.
- **A CLI-driven default-value fix needs the same pod-verification standard as
  the original bug it followed up on** — "the log line prints the right
  number" is not "the server boots"; both were checked here and only the
  first passed.
