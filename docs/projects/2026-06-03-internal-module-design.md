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

## v2 — churn-weighted re-design (SUPERSEDES the per-crate sections below)

The v1 sections below split each god-file by *concern*. Re-reviewed against **(a) cohesion +
the other half of Ousterhout — deep modules, NOT many shallow ones; (b) the actual workload
(AI-PC c=1 agent, prefix-reuse, growing context); (c) where future iteration/optimization
volume actually lands** — v1 **over-decomposes**: it shatters cold, stable code into shallow
modules whose inter-file interfaces cost more than they save. v2 isolates one **deep module per
HOT axis** and consolidates the **COLD** code into the coordinator.

**Churn map (grounded in this repo's bench/opt history — `docs/experience/{wins,errors}`):**

| HOT — frequent perf/feature change → isolate as a deep module | COLD — stable once correct → consolidate |
|---|---|
| attention kernels (paged/decode/prefill, TileLang/FlashMLA) | IR data types (ForwardPlan/StepOutput) |
| KV format/quant + prefix cache / radix | request lifecycle, queue discipline, normalization |
| scheduling (chunked prefill, overlap, retract/clamp) | safetensors loading |
| sampling features (penalties, grammar) | config |
| MoE/EP, new backends, new models | HTTP OpenAI wire schema (fixed external contract) |
| | output / finish handling |

**v2 per-crate target (fewer, deeper files than v1):**

- **`infer-core` → 4 files** (was 7): `planner.rs` (HOT: build_forward_plan + chunked + retract/preempt + mixed) · `prefix.rs` (HOT: radix policy + attach/publish/release/evict choreography over `KvPrefixStore`) · `radix.rs` (the trie data structure, exists) · `engine.rs` (COLD coordinator: orchestration + admission + output/finish + queue + slot/page helpers + RequestState + config). **Drop** v1's separate admission/output/slot/queue files — they're cold and cohesive with orchestration. The two `high` issues still get fixed: the `admit_waiting` god-fn shrinks once prefix-match moves to `prefix.rs`; planning legality moves to `planner.rs`.
- **`infer-cuda` → 6 files** (was 7): `attention.rs` (HOT) · `ops.rs` (HOT: gemm/rms/silu/embedding/add) · `model.rs` (forward dataflow) · `loader.rs` (COLD: safetensors + PageMeta + config-validate folded) · `executor.rs` (RealCudaExecutor + `sample_cuda_token`) · `lib.rs` (CudaExecutor + CudaKvPool seam).
- **`infer-metal` → split only the `lib.rs` god-file** (was 8-file blow-up): `executor.rs` (MetalExecutor + MetalInflight + sample_inflight + RealMetalExecutor + MetalSlotState + MetalPageStore — the session machine is one cohesive unit) · `kv_pool.rs` (MetalKvPool seam) · keep existing config/loader/mlx/qwen35/weights/wired_limit. Splitting `qwen35.rs`'s cpp-bridge is deferred unless FFI churn justifies it (it's stable).
- **`infer-server` → 5 files** (was 6): `execution.rs` (HOT-ish: engine loop) · `http.rs` (router + handlers) · `tokenizer.rs` (COLD) · `schema.rs` (COLD: OpenAI types) · `lib.rs` (ServeHandle + error + submission folded).
- **`infer-seam`/KvPool** — already done (Step 1, `bca44fa9`); the 3-way split maps to churn (KvPrefixStore = the prefix-cache axis).

Net: infer-core 7→4, cuda 7→6, metal 8→2-new, server 6→5. Same parity gates. The rule: **a file boundary must earn its interface — either by cohesion you can't otherwise get, or by isolating an axis you'll re-edit often.**

---

## 1. `infer-core` — split the 1854-line `lib.rs` god-module (highest severity)  [v1 raw analysis]

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
