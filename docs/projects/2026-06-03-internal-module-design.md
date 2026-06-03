# Internal module design — per-crate cohesion refactor (for review)

> Companion to [`2026-06-03-ideal-inference-engine-architecture.md`](2026-06-03-ideal-inference-engine-architecture.md) §8.
> §8 covered the **cross-crate** boundaries; this doc covers the **inside** of each
> crate. Produced from a parallel 5-agent cohesion analysis (Ousterhout deep-module
> standard, file:line-grounded). **Same numerics — pure reorganization, no behavior change.**
> Review gate before execution.

## Principle

A crate should be a **deep module**: narrow public seam, deep hidden implementation,
one-reason-to-change per *file*. The analysis found every runtime crate currently has
a **wide-shallow god-file** behind its narrow seam — the interface is clean but the
implementation tangles 5–7 concerns in one file, so a reader must grok all of them to
change any one. Fix = split each god-file by concern into cohesive files; the crate's
public seam is unchanged.

---

## 1. `infer-core` — split the 1854-line `lib.rs` god-module (highest severity)

`lib.rs` tangles admission + planning + output + prefix-reuse + slot/page lifecycle +
queue discipline. Two `high` issues: `admit_waiting()` (69-line god-fn mixing governor /
radix / page accounting / priority / attach), and planning+preemption interlocked
(`build_forward_plan` → `retract_decode_to_fit` → `requeue_preempted_decode` = retroactive
repair instead of upfront-valid plans).

| File | Holds | One reason to change |
|---|---|---|
| `engine.rs` | `Engine<E,K>` thin orchestrator: `step()` = poll→admit→plan→alloc→submit, public API, handle/completed map | the step workflow |
| `admission.rs` | `RequestAdmitter`: governor gate + slot availability + prefix match + page budget + priority insert | admission policy |
| `planner.rs` | `ExecutionPlanner`: `build_forward_plan` + retract/victim + `plan_mode` — emits an **already-valid** plan | which rows run this tick |
| `output.rs` | `OutputProcessor`: `apply_output`, token accumulation, phase advance, `finish_reason_for` | output/finish handling |
| `prefix_mgr.rs` | `PrefixManager` wrapping `RadixCache`+KV: `attach/publish/release/evict` — hides radix↔kv plumbing | prefix-reuse strategy |
| `slot.rs` | `SlotManager`: free-slot/page-budget helpers, `finish_slot` cleanup | slot & page lifecycle |
| `queue.rs` | `RequestQueue`: priority/bias ordering of the waiting `VecDeque` | queue discipline |
| `config.rs`, `radix.rs` | `SchedulerConfig`; `RadixCache` (exists) | — |

Result: `Engine` drops from ~700 mixed lines to a ~200-line coordinator.

## 2. `infer-cuda` — split `model.rs` (1347L, 5 concerns)

Confirmed monolith. (Codex just wired sampling into it — split happens *after* that lands.)

| File | Holds |
|---|---|
| `model.rs` | `CudaModel`/`TransformerBlock`/`Attention`/`Mlp` + the forward dataflow only |
| `ops.rs` | kernel-call wrappers: `embedding_batch`/`rms_norm*`/`gemm*`/`silu_mul`/`add_batch` |
| `attention.rs` | `prefill_attention`/`decode_attention`/`run_tilelang_paged` (one paged-attn path) |
| `loader.rs` | `SafetensorLoader`/`SafetensorIndex`/`OwnedTensor` + `from_safetensors` |
| `paging.rs` | `PageMeta` + `for_slot` |
| `executor.rs` | `RealCudaExecutor` (+ `sample_cuda_token`) |
| `lib.rs` | `CudaExecutor` + `CudaKvPool` seam impls, module decls |

## 3. `infer-metal` — split `lib.rs` + `qwen35.rs` (leaky/confused, not the clean module it looked)

`lib.rs` mixes the seam impls, `RealMetalExecutor`, slot state, and page store; `qwen35.rs`
mixes weight structs + loader + the C++ bridge.

- `lib.rs` → `executor.rs` (`MetalExecutor`/`MetalInflight`/`sample_inflight`) · `executor_impl.rs` (`RealMetalExecutor`) · `slot.rs` (`MetalSlotState`) · `page_store.rs` (`MetalPageStore`/prefix blocks) · `kv_pool.rs` (`MetalKvPool`).
- `qwen35.rs` → `weights.rs` (expand: weight structs) · `qwen35_loader.rs` (load fns) · `qwen35_cpp.rs` (`CppQwen35Model` bridge).

## 4. `infer-server` — split the `engine_loop` mixed module

`engine_loop` interleaves submission-drain + step + delivery; tokenizer/state leak.

- `submission.rs` (`Submission`, `admit_submission`) · `execution.rs` (`engine_loop`, `deliver_completions`, `IDLE_PARK`) · `tokenizer.rs` (`OpenAiTokenizer`: encode/decode/render_chat) · `http.rs` (axum router + handlers + `HttpState`) · `openai_schema.rs` (wire types) · `error.rs` (`ApiError`/IntoResponse).

## 5. `infer-seam` + `infer-plan` — the contract crate (the deepest fix)

**KvPool is a 18-method god-trait** mixing 3 concerns + leaking `page_indices`/`slot_epoch`
to the scheduler. Refined split (3-way, composed):

```rust
trait KvQuery     { is_active, page_size, free_pages, free_tokens, seq_len, page_indices, ... }
trait KvAllocator { alloc, alloc_detached_pages, free_slot, truncate_slot, append_pages_needed }
trait KvPrefixStore { retain_pages, release_pages, retained_count, attach_pages, page_indices_for_token_range }
trait KvPool: KvQuery + KvAllocator + KvPrefixStore {}   // composition; backends impl all three
```
The scheduler depends on `KvQuery + KvAllocator`; `PrefixManager` on `KvPrefixStore` — paging
no longer leaks to admission/planning.

Other contract moves:
- `infer-plan`: extract `argmax_logit`/`splitmix64`/`sample_token`/tests → `sample.rs` (pure data vs algorithm).
- `infer-seam`: `executor.rs` (BackendExecutor) · `kv.rs`+`allocator.rs`+`prefix_store.rs`+`kv_query.rs` · `lower_seam.rs` (Communicator/Sampler/GraphRunner/ModelArch — **hypothesis-grade, not engine-facing**) · `governor.rs`.
- **`infer-api` merge** (kill the `seam` jargon, one contract node): fold the above `infer-plan` + `infer-seam` modules into a single `infer-api` crate **once the internal splits settle** — last step, pure mechanical rename + Cargo rewire.

---

## Sequencing (review the order too)

Interdependence forces *mostly sequential* (infer-core/cuda touched by Fix 0 first), but
the per-crate **internal** splits are independent once the contract is frozen:

1. **(done/landing)** Fix 0 sampling — plan+core+metal committed; CUDA landing via Codex.
2. **Contract first** — KvPool 3-way split + `sample.rs` extract (touches seam + core + both executors). One coherent commit, verify greedy parity.
3. **Per-crate internal splits — parallelizable** (different crates, worktree-isolated subagents): `infer-core` (7-file), `infer-cuda` (7-file, after Codex's sampling), `infer-metal` (8-file), `infer-server` (6-file). Each: same numerics, `cargo test` + parity green, own commit.
4. **`infer-api` merge** — last, mechanical rename.

## Open decisions for you

1. **KvPool: 3-way (`KvQuery`/`KvAllocator`/`KvPrefixStore`) vs 2-way (`KvAllocator`/`KvPrefixStore`)?** 3-way is cleaner (scheduler reads `KvQuery` only) but one more trait.
2. **`infer-api` merge now or defer?** It's pure churn touching every consumer; recommend **defer to last** (step 4) so the internal splits land first on stable names.
3. **`infer-core` 7-file split — full or partial?** Full is the elegant target; a partial (just extract `admission.rs` + `prefix_mgr.rs`, the two `high` issues) is a cheaper first cut. Recommend **full** for elegance, but it's the biggest single change.
