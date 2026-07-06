# DSv4 `--max-total-tokens` unification: no-flags auto-context crashes on FlashMLA pool sizing

## Context

Pod-verifying `b7cd9a2e7` ("delete `INFER_DSV4_MAX_SEQ_LEN` — DSv4 serve arena
unifies onto `--max-total-tokens`") on real CUDA hardware (8×H20). Only a Mac
`cuda,no-cuda` cross-typecheck had passed before this — no real-GPU boot.

**Pod state before verify:** `/host/arle-build` had **no `.git` at all**
("fatal: not a git repository") — the working tree existed but history was
gone, so `scripts/pod.sh sync` (which needs a resolvable pod HEAD) failed
outright. Re-provisioned via a full bundle (`git bundle create ... HEAD`,
`tn push`, then `git init && git fetch <bundle> HEAD && git reset --hard
FETCH_HEAD` in place — preserves the 931M `target/` build cache since
`reset --hard` only touches tracked files). Pod HEAD after: `3a1cf9806`
(includes `b7cd9a2e7`, 3 commits ahead of the `497e81473` floor asked for).

**GPUs:** 0, 2, 3, 4, 5, 6, 7 free; GPU 1 held by another tenant (51 GB, ~95%
util throughout — a foreign PID, left untouched). Used GPUs 4,5,6,7 (TP=4) for
DSv4 (`INFER_CUDA_DEVICES=4,5,6,7`, `INFER_TP_SIZE=4`). GPU 0 also carried an
unrelated `arle serve --model-path /host/Qwen3.6-27B-FP8` (another agent's
terminal-bench session, PID 1198140) — untouched.

**Build:** `cargo build --release --features cuda,nccl,deepep --bin arle` —
`BUILD_EXIT=0 (compiled 22 crates)` in 1m48s (warm 931M target/ cache).
Confirms no missed caller of the deleted `dsv4_max_seq_len()`/
`DSV4_DEFAULT_MAX_SEQ_LEN` — the Mac typecheck cannot see this class of error
since `cuda,no-cuda` compiles a different code path than real `cuda`.

Checkpoint `/host/DeepSeek-V4-Flash-FP8/config.json`:
`max_position_embeddings = 1048576`.

## Root Cause

Two boots, same model/GPUs/binary, differing only by the flag this refactor
changed:

**Boot 1 — no flags at all** (the case whose behavior actually changed):
```
INFER_CUDA_DEVICES=4,5,6,7 INFER_TP_SIZE=4 arle serve --model-path \
  /host/DeepSeek-V4-Flash-FP8 --backend cuda --port 18191 --max-running-requests 2
```
Log: `DSv4 max context: auto-resolved to 1048576 from
/host/DeepSeek-V4-Flash-FP8/config.json (max_position_embeddings)` — confirms
the refactor's new pathway fired as designed. `EngineLoadConfig{ num_slots:
256, max_total_tokens: 1048576, ... }` on all 4 ranks. Then per-rank:

```
DSv4 KV budget: free 24171MB, per_slot 7654MB (...), pool_per_layer 119MB, affordable 2
DSv4 KV budget: requested 256 slots ... clamping num_slots to 2.
[arle-worker rank=N] failed: worker rank N engine build: DSv4 FlashMLA pool page mismatch: page_size=64 pages=3344 need page_size=64 pages>=4098
worker rank N exited Some(1)   (all 4 ranks)
[ARLE serve] multiproc coordinator setup failed: ... aborting coordinator
```
**Server never reaches ready state — hard crash on all 4 ranks at engine
build.** Not an OOM at the CUDA-alloc level (no `cudaMalloc` failure); it's an
`ensure!` in `crates/infer-cuda/src/attention/kv_layout.rs:1017-1025`:
`pool.max_total_pages >= flashmla_slot_pages` failing (3344 available < 4098
needed).

**Why:** two independent budget computations are not reconciled at large
`max_seq_len`. `dsv4.rs`'s `dsv4_kv_budget_plan` (the `affordable`
gate, `dsv4.rs:1794-1874`) computes `affordable=2` (nonzero → passes its own
`ensure!(affordable > 0, ...)` reject-below-fixed guard) and derives
`pool_budget_bytes_per_layer` from *what's left* after reserving 2 slots'
worth of per-slot state. But the FlashMLA shared pool's page requirement
(`flashmla_slot_pages`, computed independently in `kv_layout.rs` via
`Dsv4FlashMlaDecodeShape::new(..., max_seq_len, ...)`) is **a fixed cost that
scales with `max_seq_len` alone, not with num_slots** — it's the ring+
compressed band size for a *single* slot at the full context length. At
`max_seq_len=1048576` that single-slot band alone needs 4098 pages, but the
"coherent remainder" budget the `dsv4.rs` gate left for the pool only fits
3344. The `dsv4.rs` `affordable` gate never sees `flashmla_slot_pages`, so it
can (and did) say "yes, admit 2 slots" while the pool sizing that depends on
the *same* `max_seq_len` is simultaneously infeasible. This gap is orthogonal
to what `b7cd9a2e7` touched — it's a pre-existing reconciliation hole between
`dsv4.rs`'s budget-plan and `kv_layout.rs`'s FlashMLA pool `ensure!` — but it
was **unreachable before this refactor** because the old
`INFER_DSV4_MAX_SEQ_LEN` default (32768) never drove `max_seq_len` anywhere
near the ~1M range where a single slot's band exceeds the whole pool budget.
Post-refactor, **the default no-flags case now always resolves to the
checkpoint's native context** (here 1048576), so this crash is now the
default behavior for anyone booting DSv4 serve with no explicit
`--max-total-tokens` on this checkpoint/GPU-count combination.

**Boot 2 — explicit `--max-total-tokens 16384 --max-prompt-tokens 16000`**
(the old script pattern, now via flag instead of env var):
```
INFER_CUDA_DEVICES=4,5,6,7 INFER_TP_SIZE=4 arle serve --model-path \
  /host/DeepSeek-V4-Flash-FP8 --backend cuda --port 18192 \
  --max-running-requests 2 --max-total-tokens 16384 --max-prompt-tokens 16000
```
Boots clean: `CUDA engine: executor clamped slots 256 -> 121; scheduler
follows`, all 4 ranks `engine built; entering lockstep driver` /
`engine-ready ack sent`, `all 4 worker engines ready; opening HTTP`, `serving
OpenAI v1 on http://127.0.0.1:18192`. No FlashMLA pool mismatch — at this
`max_seq_len`, a single slot's band comfortably fits.

**Decode-check** (boot 2, greedy, `max_tokens=200`): request "What is the
capital of France? Answer in one word." →
```json
{"content":"Paris","reasoning_content":"The user asks for the capital of France and specifies \"Answer in one word.\" The answer is straightforward: \"Paris.\"","finish_reason":"stop"}
```
Correct.

## Ruled out

- Not a GPU-sharing/VRAM-pollution artifact: GPU1 (the shared tenant) was
  never in the device set (`INFER_CUDA_DEVICES=4,5,6,7`); GPUs 4-7 showed
  0 MiB used before both boots.
- Not a build miss: `BUILD_EXIT=0`, both boots ran the freshly built binary
  (mtimes/PID confirm), and boot 2 with the same binary/GPUs/model boots and
  decodes correctly — isolates the failure to the `max_seq_len` value alone.
- Not the intended "reject cleanly" path (`dsv4.rs:1852`'s
  `affordable > 0` ensure, which the wins/2026-06-12 doc previously validated
  produces a graceful "rejected startup" message at `affordable=0`, TP=8,
  `max_seq_len=2000000`). Here `affordable=2 > 0`, so that gate passes and a
  *different*, later `ensure!` (kv_layout.rs) is what actually kills the
  process — a harder, less informative failure than the documented graceful
  reject.

## Not yet pinned down (deferred)

- Whether TP=8 (not tested here — GPU1 unavailable) also hits this at
  `max_seq_len=1048576`, or whether more GPUs' extra free VRAM per rank
  raises `affordable`/the pool budget enough to clear 4098 pages. The
  wins/2026-06-12 doc's successful graceful-reject case was at TP=8 with an
  even larger `max_seq_len=2000000` but landed `affordable=0` (clean reject)
  rather than this `affordable=2`-but-pool-too-small crash — suggesting the
  crash window is a specific band between "some slots affordable" and
  "single-slot FlashMLA band still doesn't fit," not simply "huge
  `max_seq_len` always crashes."
- The actual fix (reconciling `dsv4_kv_budget_plan`'s `affordable` gate with
  `flashmla_slot_pages`, e.g. failing closed with the same graceful
  "rejected startup" message instead of the FlashMLA `ensure!` panic) is out
  of scope for this verify pass — reporting the finding, not patching it.

## Rule

- **A CLI-driven default that now auto-resolves to a checkpoint's native
  context (here 1,048,576) crosses budget-computation boundaries that were
  never exercised at the old hardcoded default (32768) — verify the new
  default's actual boot on real hardware, not just "the flag threads
  through correctly."** The refactor's own code review was right that the
  behavior changes for the no-flags case; pod verification is what turned
  "genuine default-behavior change, need to re-confirm" into "here is the
  exact crash and its file:line."
- **Two independent VRAM-budget computations for the same `max_seq_len`
  (scheduler-level `affordable` gate vs. executor-level FlashMLA pool
  `ensure!`) can each individually look correct and still disagree** — the
  first says "admit N>0 slots," the second says "not even one slot's fixed
  band fits." A budget gate that doesn't know about every downstream
  consumer of the same input is not a complete gate.
- **`scripts/pod.sh sync`'s bundle-based re-sync assumes the pod tree is
  already a git repo with a resolvable HEAD** — if the tree lost its `.git`
  entirely (observed here), `sync` fails outright ("pod HEAD unknown locally
  — re-provision the pod tree" via a `set -e`-terminated pipeline). Recovery
  is a one-liner (`git init && git fetch <bundle-of-local-HEAD> HEAD &&
  git reset --hard FETCH_HEAD` in place) that preserves the existing
  `target/` build cache — worth keeping as the standard re-provision recipe
  rather than re-cloning from scratch.

## Cleanup

Boot 1 (crash): self-terminated, all 4 ranks + coordinator exited, GPUs 4-7
back to 0 MiB — no manual cleanup needed. Boot 2: `kill <serve-explicit16k
coordinator PID>`, confirmed zombie + reaped, GPUs 4-7 back to 0 MiB.
GPU1 (foreign tenant) and GPU0's `Qwen3.6-27B-FP8` serve (another agent's
terminal-bench session) were never touched.
