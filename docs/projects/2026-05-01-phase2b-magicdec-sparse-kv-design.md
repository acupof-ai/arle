# Phase 2.B MagicDec Sparse-KV Design

> **Status:** design only, no implementation.
> **Mission parent:** `docs/projects/2026-04-30-longctx-32k-128k-leadership.md`
> **Prior regression:** `docs/experience/errors/2026-05-01-phase2-real-spec-regression.md`
> **External sources:** MagicDec ICLR 2025 paper (`https://arxiv.org/abs/2408.11049`), MagicDec repo (`https://github.com/Infini-AI-Lab/MagicDec`)

## Goal

Phase 2 plain external-draft speculative decode is now a correctness foundation,
not a speed path. The measured Qwen3-0.6B external draft run accepted only 12%
of draft tokens and regressed longctx c=4 throughput by 62.8-80.4%, depending
on headline vs successful-only accounting. The next W2 lever is MagicDec-style
self-speculation with sparse KV: keep the target model as the draft source, but
make draft attention cheap by restricting the draft pass to a fixed sparse KV
view.

Entrance target for Phase 2.B:

- longctx-32k c=4 remains valid and successful-only;
- `spec_draft_k` in the 4-6 range;
- rolling acceptance rate >= 0.40, target >= 0.60;
- effective output throughput >= Phase 1 close baseline `26.169 tok/s`;
- only claim W2 lift once throughput exceeds the Phase 1 baseline, not from
  acceptance rate alone.

## MagicDec Mechanism

The paper's load-bearing observation is that speculative decoding becomes more
attractive for long sequence and large batch serving because decode shifts from
parameter-load dominated to KV-load dominated. The abstract states that
MagicDec uses a sparse-KV draft model to address a KV bottleneck that scales
with both sequence length and batch size, and reports up to 2.51x speedup on
LLaMA-3.1-8B for batch sizes 32-256.

Core mechanics:

| mechanism | MagicDec behavior | ARLE implication |
|---|---|---|
| Self draft | The target model itself can draft; the repo's self-speculation path does not require configuring a separate draft model. | Avoid the Qwen3-0.6B second-model memory tax and eliminate extra target/draft vocab mismatch risk. |
| Sparse draft KV | Draft pass attends a bounded KV budget instead of full long context. The repo supports StreamingLLM and SnapKV draft budgets; examples use `--draft_budget 257` for some longspec runs and 4097 defaults for self-spec scripts. | The draft must be made cheap by attention metadata, not by adding a second model. |
| Full verifier | Target verification still uses full KV and verifies K draft positions plus a bonus token. | Keep ARLE's current verifier/rollback/bonus-token lifecycle, but replace the draft source. |
| Acceptance at long context | The project page reports LLaMA-3.1-8B StreamingLLM self-spec acceptance around 0.84 at 4k, 0.81 at 32k, and 0.79 at 100k with a 512-token budget. | A Qwen3-4B first cut should treat >=0.40 as a gate, >=0.60 as useful, and >=0.75 as MagicDec-quality. |
| Batch behavior | MagicDec reports speedup rising with batch size and notes that optimal speculation length can rise with batch size. | Prioritize c=4/c=8 longctx envelopes over c=1 when evaluating W2; c=1 remains a no-regression guard. |

The cloned repo confirms the operational shape:

- `tests/StreamingLLM/selfspec_benchmark.py` drafts `gamma` tokens in a loop,
  calls `engine.verify(tokens_buffer)`, compares `target_tokens[:, :gamma]`
  with draft tokens, rolls target cache length back by `gamma + 1`, then commits
  accepted tokens plus bonus.
- `Engine/StreamingLLM/backend.py` keeps separate draft page metadata bounded by
  `draft_budget`; once `draft_cachelens` reaches the budget, it clamps draft
  length instead of growing with the full context.
- `Engine/SnapKV/model.py` builds a draft KV cache from selected context tokens
  using a `draft_budget - window_size` top-k region plus a recent window.

## ARLE Current State

Reusable pieces already landed:

| file | current state | reuse in Phase 2.B |
|---|---|---|
| `infer/src/speculative.rs` | `verify_tokens_greedy` accepts until the first target/draft argmax mismatch; `AcceptanceTracker` records rolling accepted/drafted counts. | Reuse unchanged. |
| `infer/src/speculative/cuda.rs` | `DraftEngine` owns external draft state, but plain external draft is not the desired performance path. | Keep external mode as a fallback; add a self sparse-KV draft engine instead of extending external first. |
| `infer/src/scheduler/cuda/spec_path.rs` | `draft_then_verify` owns eligibility, page accounting, verifier dispatch, target KV rollback, draft commit, metrics, and disable threshold. | Reuse control flow. Replace `DraftMode::External(_)`-only dispatch with `DraftMode::SelfSpec` when sparse view is enabled. |
| `infer/src/model/qwen3/forward.rs` | `forward_spec_verify_batch` verifies K+1 positions, currently by sequential paged decode steps. | Keep as correctness baseline; later optimize to a packed verifier once sparse draft is profitable. |
| `infer/src/prefix_cache.rs` | Radix nodes track `ref_count`, LRU, cascade-evictable blocks, and evictable page counts. | Reuse page/block selection and pin discipline for a sparse view, not for semantic importance scoring. |
| `infer/src/scheduler/cuda/core.rs` | `effective_pool_free_pages()` already includes evictable prefix GPU pages. | Sparse draft must not double-count pages or physically evict full target pages. |

Missing pieces:

- a read-only sparse KV view over an active request's full paged KV;
- per-request sparse page selection metadata;
- model/attention APIs that can decode against a supplied subset of pages;
- scheduler eligibility gates that require sparse-KV cheapening before allowing
  `DraftMode::SelfSpec` with `spec_draft_k > 1`;
- sparse-draft metrics: selected pages, selected tokens, draft/full KV ratio,
  sparse draft latency, verifier latency, acceptance rate.

## Sparse-KV Options

### Option A: Radix/PagedKV Sparse View

Use the existing RadixCache/PagedKV page infrastructure to construct a per-slot
`SparseKvView`:

- selected sink pages: first N pages from the request's materialized context;
- selected recent pages: last M pages before the current decode position;
- optional prefix-cache pages: when an attached prefix is radix-backed, reuse
  block metadata and lock-ref discipline so sparse view pages cannot race with
  eviction;
- no physical copy and no page release during drafting; the view is just
  temporary metadata passed to attention planning.

This is the recommended first implementation because it keeps the target full
KV intact, preserves Phase 1's eviction invariants, and avoids a second KV pool.
It is closer to StreamingLLM than full SnapKV, but the same interface can later
carry SnapKV-selected page IDs.

Important nuance: RadixCache LRU is not token importance. It is acceptable for
pinning and page identity, but not as a final top-k attention scorer. The first
landing should use deterministic sink+recent page selection; a later P2.B.x
slice can add SnapKV-like score collection.

### Option B: Streaming Attention Mask

Keep full page metadata but add an attention mask that hides all non-selected
tokens during draft decode.

Pros:

- simplest logical model for correctness tests;
- no sparse page index remapping needed.

Cons:

- if the kernel still streams all full-context pages, it does not reduce the KV
  memory-load bottleneck;
- ARLE's FlashInfer paged decode interface is page-index driven, so a mask-only
  approach risks being a correctness scaffold without speed.

Option B is useful as a CPU/unit-test oracle, but not the first performance
implementation.

## Proposed ARLE Design

Add a sparse view instead of a second draft model:

```text
normal decode state:
  full target KV pages in PagedKVPool

Phase 2.B self sparse draft:
  SparseKvView { slot_idx, selected_page_ids, logical_positions, sink_pages, recent_pages }
  forward_sparse_decode(token, state, sparse_view)
  repeat K times to draft tokens using target weights + sparse attention

full verifier:
  existing forward_spec_verify_batch against full PagedKVPool
  verify_tokens_greedy
  rollback full target KV to original + accepted + bonus
```

Scheduler policy:

- `DraftMode::SelfSpec` with `spec_draft_k > 1` remains illegal unless
  `spec_sparse_kv_enabled` is true.
- Eligible rows must be greedy, penalty-free, no stop sequences, and have
  context length >= `spec_sparse_min_context_tokens` so short prompts do not pay
  sparse-view overhead.
- Mixed decode+prefill pressure can disable sparse speculation for the step if
  verifier pages do not fit under the existing decode retraction policy.
- A request is disabled only per request through `AcceptanceTracker`, never by
  flipping global `spec_enabled`.

Forward API:

- Add model-level sparse draft API, not scheduler model-specific branches:
  `forward_sparse_decode(token, state, sparse_view, draft_ctx)`.
- For Qwen3 first, the sparse path should reuse paged decode buffers with a
  reduced `kv_indices`/`kv_indptr` table.
- Qwen3.5 should start disabled because hybrid linear-attention recurrent state
  cannot be approximated by dropping full-attention KV pages without a separate
  recurrent-state proof.

Correctness rule:

- Sparse draft may be approximate.
- Full verifier is authoritative.
- Output tokens must remain bit-identical to normal greedy decode for the same
  sampling mode.

## Engineering Slices

| slice | files likely touched | deliverable | estimate |
|---|---|---|---:|
| P2.B.1 sparse-KV view interface | `infer/src/model.rs`, `infer/src/scheduler/types.rs`, `infer/src/scheduler/cuda/request.rs`, `infer/src/scheduler/cuda/spec_path.rs` | Types/config only: `SparseKvView`, `SparseKvPolicy`, config gates, metrics labels. No behavior change. | 180-260 LoC |
| P2.B.2 page selection policy | `infer/src/prefix_cache.rs`, `infer/src/scheduler/cuda/core.rs`, `infer/src/scheduler/cuda/spec_path.rs` | Deterministic sink+recent page selector using existing PagedKV/Radix block metadata and ref discipline. | 220-360 LoC |
| P2.B.3 Qwen3 sparse forward path | `infer/src/model/qwen3/batch_decode.rs`, `infer/src/model/qwen3/forward.rs`, `infer/src/model/qwen3/decode_buffers.rs`, possibly `crates/cuda-kernels/src/flashinfer.rs` | Decode one token against reduced page metadata. TP=1 only first. | 350-650 LoC |
| P2.B.4 spec_path self-sparse dispatch | `infer/src/scheduler/cuda/spec_path.rs`, `infer/src/speculative/cuda.rs`, `infer/src/metrics.rs` | Replace external draft source with target sparse draft for `DraftMode::SelfSpec`; keep verifier unchanged. | 240-420 LoC |
| P2.B.5 tests | `infer/tests/spec_decode_correctness.rs`, scheduler unit tests | Greedy spec on/off bit-ident; sparse view never mutates full KV; disable when acceptance below threshold; Qwen3.5 disabled. | 180-320 LoC |
| P2.B.6 bench + wins/errors | `docs/experience/wins/`, maybe `docs/experience/errors/` | longctx-32k c=4 300s, mixed-mode, c=1 guard. | docs only |

Suggested commit order:

1. `feat(scheduler): add sparse-KV view config and request state`
2. `feat(scheduler): select sink-recent sparse KV pages for self-spec`
3. `feat(qwen3): add sparse paged decode draft path`
4. `feat(scheduler): route MagicDec self-spec through sparse draft`
5. `test(scheduler): cover sparse self-spec correctness gates`
6. `docs(scheduler): record phase 2.B sparse-KV bench`

## Bench Matrix

Minimum before claiming W2:

| workload | command envelope | success criterion |
|---|---|---|
| longctx-32k c=4 | Phase 1 S5 c=4 300s equivalent, `--spec-draft-k 4/5/6` | successful-only tok/s >= 26.169, no invalid run |
| mixed-mode c=4 | half long prompt, half short prompt | short rows do not starve; no TTFT tail blow-up vs spec off |
| agent-loop c=4 | short prompts with prefix hits | prefix reuse remains intact; sparse view does not evict hot cache |
| c=1 guard | short and long decode-only | no severe TTFT/ITL regression; not a Phase 2 gate |

Parameter scan:

- `spec_sparse_sink_pages`: 1, 2, 4
- `spec_sparse_recent_pages`: 16, 32, 64
- `spec_draft_k`: 4, 5, 6
- `spec_acceptance_threshold`: 0.3 first, 0.5 after acceptance stabilizes

Record all runtime changes under `docs/experience/wins/` or `errors/` per the
bench-and-trace spec.

## Risks

| risk | why it matters | mitigation |
|---|---|---|
| Sparse path still loads full KV | Mask-only implementation would repeat the external-draft failure pattern: correct but slow. | Require page-count and sparse/full KV ratio metrics before bench claims. |
| Qwen3.5 hybrid correctness | Recurrent state is not page-droppable like attention KV. | Gate Phase 2.B to Qwen3 first; add Qwen3.5 only after a recurrent-state design. |
| Prefix-cache eviction race | Sparse view may reference radix-backed pages while eviction sees them as free. | Reuse existing attached-prefix lock/ref discipline; never physically drop pages for draft view. |
| Packed verifier still missing | Sequential verifier may cap speedup even with sparse draft. | P2.B first proves draft cheapness/acceptance; packed verifier becomes P2.C if sparse draft succeeds but net lift is small. |
| Acceptance below 0.4 | Qwen3-4B may not match LLaMA-3.1 MagicDec behavior with a simple sink+recent view. | Add SnapKV page scoring after the deterministic StreamingLLM-style first cut. |

## Decision

Proceed with Option A as Phase 2.B. It is the only path that directly addresses
the measured bottleneck from Phase 2 external draft: draft must become cheap
without consuming extra KV pool capacity. The implementation should land in
small independent commits and keep the existing external draft path as a
diagnostic fallback, but W2 performance claims should come only from
MagicDec-style self-spec sparse KV.
