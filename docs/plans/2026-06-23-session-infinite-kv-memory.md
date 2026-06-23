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

## Components (file:line)

| # | Component | Status | Home |
|---|-----------|--------|------|
| 1 | `session_id` on the request | ✅ exists (`infer-api/types.rs:104`, "advisory") | honor it |
| 2 | `SessionMemory` store: `SessionId → {offloaded tier-keys, per-block mean-key rep}` | 🔨 new | `infer-core` (device-neutral), keyed by session → multi-tenant isolation (doc §5) |
| 3 | Recall orchestration: when resident > working-set, keep sink+local, recall top-k by `query·rep` | 🔨 reuse | `infer-core/prefix.rs` demote/promote + `kv-native-sys` tier |
| 4 | **Executor page-gather**: decode attends a *selected* page list (sink∪local∪recalled), not `[0..cache_len]` | 🔨 **only hard piece** | `infer-metal/executor.rs:2357-2430`; CUDA mirror |
| 5 | Wire `session_id` → SessionMemory → assemble → page-gather | 🔨 | gated, default off |

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
