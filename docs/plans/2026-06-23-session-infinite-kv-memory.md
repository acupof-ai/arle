# Session-scoped infinite KV memory (infer-api feature)

Goal: honor `CompletionRequest.session_id` as a **per-session infinite-memory**
namespace. When a session's context exceeds the GPU working set, the runtime
offloads old KV blocks to the cheap tier and **recalls** the relevant ones
(sink + local window + top-k mean-key) instead of failing or dropping them — so a
session can grow far past the model's native window. Opt-in; default byte-for-byte
unchanged. Decided 2026-06-23 (ckl): **option A** — `session_id` is the
recall-store namespace + isolation key; prefix reuse stays token-matched.

Algorithm risk is **retired**: the mean-key top-k recall was verified offline this
session on Qwen3.6 (recall = full answer quality at ~2% of KV; diverse distractors;
agentic 3-hop; aggregation degrades gracefully). See
`wins/2026-06-22-kv-recall-quality-qwen36-phase0.md`. What remains is plumbing +
one new attention primitive.

## Tier placement (ckl 2026-06-23)

The budget regions map onto the HiCache tier ladder — **long-term memory lives in
L3**, not held in DRAM:

| region | tier | medium | role |
|---|---|---|---|
| working set (sink + local + recalled) | **L1** | GPU HBM | what decode attends; fixed budget |
| current-prompt overflow | **L2** | host DRAM | hot, this-request long input |
| **long-term memory (history)** | **L3** | **NVMe / remote** | cold, episodic, unbounded — cheapest tier (doc §4 INT4 economics) |

Recall = promote the selected blocks L2/L3 → L1. So `SessionMemory` (#2) demotes
turn-boundary history blocks straight to **L3** (`kv-native-sys` SSD / remote), and
intra-prompt overflow to L2; #5 promotes the top-k back. The recall *planner* (#3,
`recall.rs`) is tier-agnostic — it only emits token ranges; the fetch tier is #2/#5.

## Scoring: resident reps (load-bearing — do not skip)

To recall you must score **all** middle blocks every step. If offloaded blocks were
gone you could never rediscover them → recall collapses to a fixed subset. So each
block keeps a tiny **mean-key rep resident** (one `[nkv, hd]` vector, mean-pooled at
offload); only the full KV goes to L3. Scoring = `q · resident-reps` over all blocks
(cheap). This keeps the live path **Rust-orchestrated**, reusing #4/#5:

1. C++ `full_attn_step` (`mlx_qwen35_model.cpp:905`, layer 0) emits the decode
   **query vector** (small). One-step stale → fine (`--stale` licensed).
2. Rust: `score = q · reps` (resident) → `plan_recall` → `recall_ranges`.
3. #4 gather over the resident working set + promote selected full KV from L3.

`--shared --stale`'s `reps` (mean-pooled block K) **is** the resident rep — that's the
validated math. The C++ change is small (emit the query), NOT an in-attention recall.

## Components (file:line)

| # | Component | Status | Home |
|---|-----------|--------|------|
| 1 | `session_id` on the request | ✅ exists (`infer-api/types.rs:104`, "advisory") | honor it |
| 2 | Resident mean-key reps + `q·rep` scoring (per-block, layer-0) | ✅ live (Metal) `infer-metal/executor.rs` `block_reps` + `recompute_recall_plan` | device-side reps, host scoring |
| 3 | Recall orchestration: when resident > working-set, keep sink+local, recall top-k by `query·rep` | ✅ live (Metal, **resident variant**); ⏳ L3 tier offload TODO | `infer-core::plan_recall` + `recall_ranges`; L3 demote/promote (`radix.rs`/`kv-native-sys`) deferred |
| 4 | **Executor page-gather**: decode attends a *selected* page list (sink∪local∪recalled), not `[0..cache_len]` | ✅ done+tested `gather_kv_ranges`/`bf16_recall_read_inputs` | `infer-metal/executor.rs`; CUDA mirror TODO |
| 5 | Wire `session_id` → recall → assemble → page-gather | ✅ live: C++ layer-0 query emit + Rust scoring + `recall_ranges`, gated `--kv-recall` (bf16-only), default off | `mlx_qwen35_model.cpp` emit + `executor.rs` `maybe_recompute_recall` |

**Implementation status (2026-06-23):** Pieces 1–5 of the live recall path land
in this pass — C++ emits the layer-0 decode query each step
(`qwen35_compiled_take_recall_query`), Rust mean-pools resident per-block
mean-key reps (`block_reps`), scores `q·rep`, runs `plan_recall`, and sets
`recall_ranges` for the next step (stale-Q, licensed). Recall is **bf16-only**
(int8 KV falls back to full attention, logged once) and gated behind
`--kv-recall` (default off → baseline byte-identical). What's deferred:
component #3's **L3 tier offload** — the current implementation is the
**resident variant** (full KV stays in HBM `slot.kv_flat`; recall restricts
*attention* to the selected ranges, saving decode compute). Freeing the
offloaded blocks' full KV to L3 while keeping only the resident rep (the
flat-VRAM-vs-history win) needs the per-block Metal-slot KV ↔ tier wiring and is
marked `// TODO(kv-recall L3)` in `recompute_recall_plan`. The CUDA #4 mirror is
also deferred (recall is Metal-only, cfg-gated).

## Critical path / DAG

`#4 page-gather` is load-bearing: recall can select blocks (#3) but can't attend
them without it. #2 (store) and #3 (selection) are device-neutral and independent
of #4 until #5. So: **#4 first (derisk), then #2+#3 in parallel, then #5.**

## #4 spec — Metal decode page-gather

- Today: `bf16_prefix_read_inputs(cache_len)` / `int8_prefix_read_inputs` slice
  `[0..cache_len]` (`executor.rs:2357-2430`). Add a variant taking
  `selected_pages: &[u32]` and gathering K/V from those pages (the pool already
  addresses pages; `page_indices(slot)` returns the resident list). Build
  `k_full`/`v_full` by concatenating the selected page slices in Rust, then the
  existing C++ `step_session_paged_*` (unchanged).
- Thread the per-row page list: `ForwardPlan::DecodeRow` (`infer-plan/lib.rs:36`)
  gains `Option<Vec<u32>>` recalled-pages (None = today's contiguous path).
- Default path (`None`) is byte-identical. Unit-test: decode over a chosen subset
  attends only those pages (compare logits vs full-attend on a planted-needle
  toy, mirroring the offline harness).

## #2/#3 — SessionMemory + recall (reuse, device-neutral)

- `SessionMemory` in `infer-core`: `HashMap<SessionId, SessionEntry>`;
  `SessionEntry { resident_pages, offloaded: Vec<(tier_key, MeanKeyRep)>,
  sink_pages, l_bs, top_k }`. The mean-key rep = mean-pooled K per block (the
  Metal-feasible representative — prefill scores unreachable, validated offline).
- On a request with `session_id`: prefix-match (RadixCache, token-keyed, A);
  if resident tokens > working-set budget → demote LRU/oldest blocks to the
  session's tier slot (`prefix.rs` demote + kv-native-sys), record their mean-key
  rep; assemble decode page set = sink ∪ local-window ∪ top-k(query·rep) recalled;
  pass to #4.
- `top_k` default generous ("多召回些" — ckl) since agent history is the workload.

## Gates (mandatory)

- Correctness: **correct-inference needle gate, NOT byte-identity** (recall +
  any offload deliberately deviates from a single resident-KV run) —
  `[[feedback_correct_inference_not_baseline_identity]]`.
- Bench: wins/ entry per backend+model; flat-VRAM curve vs session length is the
  §6 decisive evidence. Default-off → baseline unaffected.
- Multi-tenant: assert session A's tier keys never recall into session B.

## Increments

1. #4 Metal page-gather primitive + unit test (derisk). ← start here
2. #2 SessionMemory store + #3 recall selection (device-neutral), gated off.
3. #5 wire `session_id` → assemble → #4; needle gate; flat-VRAM bench.
4. CUDA #4 mirror.
5. Point eli at `arle serve` (local profile) + relax eli's `char_limit` hard-stop
   (`../eli/crates/nexil/src/llm/tool_loop.rs:374`) — length now handled by ARLE.
