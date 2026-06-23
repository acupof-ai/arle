# Session KV-recall: ARLE core landed + e2e effect on real Qwen3.6

## Context

Infinite-memory feature (per `docs/plans/2026-06-23-session-infinite-kv-memory.md`):
honor `session_id` so a session's context can exceed the GPU working set by
offloading old KV and recalling the relevant blocks (sink + local + top-k mean-key).
This entry lands the two algorithmic cores + validates the effect end-to-end.

## What Worked

**#3 recall planner** (`infer-core/src/recall.rs`, device-neutral, 6 tests):
`plan_recall(cache_len, block_scores, cfg)` → ascending merged token ranges
`sink ∪ top_k blocks ∪ local`. Returns the single contiguous range when the session
fits → default path byte-identical. The `working_set_is_budget_bounded` test proves
the invariant: `token_count` is constant (= `working_set_tokens`) whether the
session is 1K or 1M tokens.

**#4 page-gather** (`infer-metal/executor.rs` `gather_kv_ranges` /
`bf16_recall_read_inputs`, tested): decode reads K/V from selected contiguous token
ranges by slicing each + `concatenate_axis` — reuses shipped `slice_kv_tokens` +
`concatenate_or_single`, **no new FFI**.

**e2e effect** (`scripts/kv_recall_quality_eval.py`, real `Qwen3.6-35B-A3B-4bit`,
ctx~5684, mid-depth passkey):

| mode | KV attended | % | acc |
|---|---|---|---|
| full (attend all) | 5691 | 100% | 1.00 |
| stream (sink+local) | 288 | 5.1% | **0.00 (MISS)** |
| recall (sink+top_k+local) | 544 | 9.6% | **1.00 (= full)** |

Recall gets the **exact** full-attention answer at 9.6% KV; streaming at the same
sink+local budget but no recall **misses** — recall is what makes the budget work.
The harness's recall policy is identical to `plan_recall`, and its `kv=544` equals
`RecallConfig::working_set_tokens()` (32 + 8×32 + 256) — mechanism and Rust impl both
validated on the same number.

## Per-slot vs per-layer (license-or-kill)

The harness recall is **per-layer** (each full-attn layer scores `query·mean-key`
with its own Q/K and picks its own top-k). The Rust `plan_recall` is **per-slot**
(one selection reused by all layers) — simpler + shares the gather, but unvalidated.
`--shared` mode (one selection, first full-attn layer decides) tested it:

| depth | full | stream | recall (per-slot) |
|---|---|---|---|
| 0.25 / 0.5 / 0.75 | OK | MISS | **OK** (544 = 9.6% KV) |

Per-slot **LICENSED** for passkey retrieval — per-layer complexity not needed. Caveat:
validated for retrieval/uniform; harder tasks (aggregation, diverse, multi-hop) would
need their own per-slot re-check before claiming parity there.

**Stale-Q** (`--stale`, score the current token's blocks with the *previous* step's
query) also LICENSED — per-slot + stale-Q retrieves at 0.25/0.5/0.75, acc=1.00, 544 KV.
This settles the architecture: the live recall is **Rust-orchestrated** — the C++ step
outputs layer-3 block scores, Rust runs `plan_recall` → sets `recall_ranges` for the
*next* step → the #4 gather + #5 dispatch. No mid-step interception, no C++-internal
recall, and the committed Rust path is the live path. The C++ only adds a per-block
score output at the first full-attn layer.

## Rule

The recall *planner* and *gather* are validated; the served ARLE-native e2e still
needs device-side mean-key scoring (expose the decode query from the C++ step), an
infer-metal model-load needle harness, and tier(L3)/Engine wiring — those are the
next phase. Default-off (planner returns contiguous when it fits, gather unused
until wired) → no baseline bench needed; correctness via the harness + unit tests.
