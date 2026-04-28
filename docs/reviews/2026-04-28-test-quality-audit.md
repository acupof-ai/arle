# Test Quality Audit — 2026-04-28

> **2026-04-28 follow-up (codex review + execution corrections).** During
> Phase 2 execution three audit entries were reclassified to EXEMPT:
> (a) `infer/src/distributed/init_method.rs:257 unique_id_size_constant` —
> `UNIQUE_ID_BYTES` is the rendezvous wire payload size matching NCCL's
> 128-byte `ncclUniqueId`; the other rendezvous tests use the constant
> via `[u8; UNIQUE_ID_BYTES]` so they would still pass if it drifted off
> spec. (b) `crates/cuda-kernels/src/collective.rs:361/366 dtype_enum_size,
> reduce_op_enum_size` — DType / ReduceOp are cast to the matching NCCL
> enum types across the FFI boundary (same FFI-ABI rationale as the
> `nccl.rs` size-of pins). (c) `crates/cuda-kernels/src/tensor.rs:1387
> kernel_alignment_names_scale_layout_explicitly` — the asserted strings
> are public-API field values on `WeightKernelAlignment` and are
> consumed by kernel-dispatch routing, not Debug text. Net: Phase 2
> shipped 10 deletions + 1 refactor (AdamW `is_device_backed()` getter
> replacing the Debug-string lock in `crates/train/src/cli_args.rs`),
> not 18.
>
> The per-file inventory below was generated against a working tree
> that included unrelated dirty state; counts in
> `infer/src/backend/metal/scheduler.rs` and the omission of the root
> `tests/` directory (`tests/cli_smoke.rs` 8, `tests/cli_agent_live.rs`
> 3, `tests/cli_tiny_fixture_live.rs` 1 — all behavioral CLI smoke and
> live-tool tests, EXEMPT) reflect that snapshot. Treat the inventory
> as a guidepost; trust ripgrep at HEAD for absolute counts.

## Summary

- Total `#[test]` / `#[tokio::test]`: **~1255** (1230 in `infer/src` + `crates/*/src` + `src`, plus 25 in `infer/tests/`)
- Total files declaring tests: **~167**
- Tests / file ratio: **~7.5 / file**

Per-class counts (proposed, **tag-only**):

| Class | Count | Notes |
|-------|-------|-------|
| **A** Tautological / trivial | **9** | Hashable+Copy round-trips, enum size_of, defaulted struct identity, etc. |
| **B** Pure constructor coverage | **6** | `new(...)` + field-default reads, `is_quantized()` flag check, etc. |
| **C** Mock-the-world | **0 confirmed** | The two cases I found (`MockEngine`, `MockDraftModel`, `FakeEngine`) all back **real** state machines: `RadixCache`+`DiskStore`, real `verify_tokens`, real `AgentSession`. None are pure mock-flow checks. |
| **D** Implementation-detail locks | **3** | Asserts on `format!("{:?}", ...)` of an optimizer, internal kernel layout-name strings, internal `Debug` non-emptiness. |
| **E** Behavioral end-to-end (KEEP) | **~620** | Scheduler, kv_tier, http_server, agent session, cpu/metal backend, tensor_parallel SGLang parity, scheduler/policy admission/eviction, prefill, decode, requests, request_state, distributed rendezvous, etc. |
| **F** Numerical / golden (KEEP) | **~470** | Autograd ops, sampler penalties, MLX bridge, GPU kernels (rms_norm, gemv, attention, q4k), spec-decode prob math, RoPE, body_span tokenizer math, FlashInfer plan/run, varlen masks, tied-weight loaders, AdamW state, LR schedule. |
| **G** Property / fuzz (KEEP if exists) | **2** | `sample_from_uniform_dist` (statistical), `hybrid_downgrade_fires_on_every_partial_hit` (loop-quantified). No `proptest` / `quickcheck` yet. |
| **U** Unclassified / needs human eyes | **18** | Listed below — mostly small but ambiguous between A and E. |
| **Exempt — contract** | **~530** | See next section. Excluded from delete consideration. |

> Class totals add to ~1128. Remaining ~127 are inside exempt-contract files
> (e.g. `args.rs`, `repl.rs`, `prefix_cache.rs`, `scheduler/types.rs`,
> `openai_v1.rs`, `infer/tests/*`). They are *individually* still behavioral
> but blanket-exempt by file rule.

## Methodology

For every `#[test]` / `#[tokio::test]` I read the function body (Read tool with offset/limit) before tagging — names alone never decided a class. Bodies > 15 lines that exercise real call paths default to **E**. When intent is unclear, the test goes to **U**, never **A–D**. Bias: **false-negative on delete** — when in doubt, keep.

Decision rules per class:

- **A — Tautological / trivial**: body ≤ 6 lines AND assertions reference fields just set, or `Default::default()` field-by-field, or `mem::size_of` of a `#[repr(...)]` enum, or pure `Copy`/`Hash`/`PartialEq` round-trips on a struct without invariants beyond derive.
- **B — Pure constructor coverage**: body essentially `let x = X::new(...); assert!(...)` with assertions only on the values just passed in or simple flag bits.
- **C — Mock-the-world**: file defines a `Fake*` / `Mock*` / `Stub*` (or local trait impl returning canned data) AND the test only verifies the canned data flowing back — no real engine/scheduler/model code is exercised. Zero confirmed in the workspace.
- **D — Implementation-detail locks**: assertions on `format!("{:?}", x)`, exact internal log strings, exact internal error text, or layout-name strings that are not part of any documented public contract.
- **E — Behavioral**: calls into `Engine`, `Scheduler::step`, `ModelForward`, `BlockManager::*`, `RadixCache::*`, `Coordinator::run_once`, `axum::Router::oneshot`, real `MlxArray` bridge, etc.
- **F — Numerical**: loads from `infer/test_data/`, uses `assert_close` / numerical tolerance, or computes against a hand-derived reference.
- **G — Property / fuzz**: looped quantification or stochastic distribution check.

Note: **if a test could plausibly be E or F, it stays a KEEP.** This audit deliberately under-targets deletes.

## Exempt — contract tests (NEVER auto-delete)

Required by the brief:

- `crates/qwen3-spec/**` — tensor-name + shard-spec contract for Qwen3 (every tensor name + sharding rule pinned)
- `crates/qwen35-spec/**` — same for Qwen3.5 (full + linear attention + MoE coverage)
- `crates/cli/src/args.rs` — clap CLI surface; flags are user-visible
- `crates/cli/src/repl.rs` — REPL state machine; recently regressed (commit `00e3927`)
- `infer/src/prefix_cache.rs` — RadixCache invariants; correctness gate
- `infer/src/http_server/openai_v1.rs` — OpenAI v1 wire contract; downstream clients
- `infer/src/scheduler/types.rs` — slot lifecycle + `SchedulerConfig::validate` invariants
- `infer/tests/**` — E2E surface; regenerate baselines, do not delete
- `crates/autograd/tests/test_backend.rs` — numerical reference (CPU vs Metal vs CUDA backend parity for autograd ops)

Pattern-matched as exempt-contract:

- Any test with name pattern `*_matches_(cpu|reference|mlx|cuda|metal)`, `*matches_sglang*`, `*matches_hand_computed*`, `_round_trip` against an external format (safetensors, GGUF, tokenizer.json), `*sglang_docstring*`, `*matches_reference*`.

Additional exempt-contract files I found while walking the tree:

- `crates/chat/src/protocol.rs` — ChatML wire format, `<tool_call>` block contract; downstream training data + tokenizer offsets depend on byte-exact strings
- `crates/chat/src/lib.rs` — OpenAI chat → ChatML adapter contract
- `infer/src/http_server/sessions.rs` — KV-tier session save/load wire format (RadixCache fingerprint + disk payload integrity)
- `infer/src/http_server/tests.rs` — HTTP server integration tests (54 of them) — full request/response lifecycle
- `infer/src/scheduler/tests.rs` — scheduler integration tests via real `Scheduler::step`
- `infer/src/scheduler/cuda/runtime/tests.rs` — CUDA scheduler runtime; rendezvous spec-lock cited in `b6e06fa`
- `infer/src/scheduler/cuda/{budget,prefill,decode,request,execution,core}.rs` — admission math + page math + retract math; numerical contract on the hot path
- `infer/src/types.rs` (`fingerprint_compute_*`) — `BlockFingerprint` cross-version stability is a persistence contract; KV-tier entries at rest depend on it
- `crates/cuda-kernels/src/{paged_kv,flashinfer,tensor,graph_pool}.rs` — kernel-level numerical correctness + page math
- `crates/qwen3-spec/**` and `crates/qwen35-spec/**` (already listed)
- `crates/autograd/src/safetensors_io.rs` — safetensors round-trip contract w/ infer's bf16 reader (`save_from_bf16` is the path infer can actually consume — see source comment)
- `crates/autograd/tests/m*.rs` and `test_adamw_state.rs`, `test_lr_schedule.rs`, `test_optimizer_trait.rs`, `test_device_handle.rs`, `test_linear_attention.rs` — numerical reference for the autograd stack
- `crates/train/tests/*.rs` integration tests — checkpoint v2 wire format, GRPO/SFT reward + verifier numerical reference
- `infer/src/sampler.rs` — sampler is the closest thing to vLLM/SGLang sampler contract; `apply_penalties` formula must stay exact
- `infer/src/speculative.rs` — speculative-decode probability math is a numerical contract (verify_tokens is from the speculative-decode paper)
- `infer/src/tensor_parallel.rs` — TP/PP/EP/DP rank-decomposition is a wire-compat contract with SGLang docstrings cited verbatim
- `infer/src/distributed/init_method.rs` — rendezvous protocol; remote nodes interop
- `infer/src/weight_loader.rs` — quant layout + shard discovery; reading real HF model checkpoints
- `infer/src/gguf.rs` — GGUF on-disk format
- `infer/src/quant.rs` — quantization metadata parsing; reading external configs
- `infer/src/model_registry.rs` — architecture detection from external `config.json`
- `infer/src/trace_reporter.rs` — OTLP wire format + envvar contract

## Class A — Tautological / trivial

Group: `infer/src/types.rs`

- `infer/src/types.rs:198` `request_id_is_hashable_and_copy` — body 5 lines: `let a = RequestId(7); let b = a; assert_eq!(a, b);`. Tests derive(Copy, Eq), no invariant. — body length: 5
- `infer/src/types.rs:205` `block_id_is_copy_and_ordered` — same pattern, plus `BlockId(1) < BlockId(2)`; tests `derive(Ord)`. — body length: 5
- `infer/src/types.rs:213` `block_fingerprint_round_trips` — constructs `BlockFingerprint(bytes)`, asserts `.0 == bytes` and self-equality. No invariant. — body length: 4
- `infer/src/types.rs:303` `request_event_kind_progression_example` — builds an array of 7 enum variants and asserts `events.len() == 7`. — body length: 10
- `infer/src/types.rs:327` `session_id_is_hashable_and_cheap_clone` — inserts into HashSet, asserts contains. Tests derive(Hash, Eq, Clone). — body length: 7

Group: `infer/src/distributed/init_method.rs`

- `infer/src/distributed/init_method.rs:257` `unique_id_size_constant` — `assert_eq!(UNIQUE_ID_BYTES, 128);`. Pure literal-vs-literal. — body length: 1

Group: `infer/src/kv_tier/tier.rs`

- `infer/src/kv_tier/tier.rs:110` `tier_variants_round_trip` — for each `Tier` variant: clone, equality, `format!("{:?}").is_empty()`. Tests `derive(Copy, Debug)`. — body length: 6

Group: `crates/cuda-kernels/src/collective.rs`

- `crates/cuda-kernels/src/collective.rs:361` `dtype_enum_size` — `assert_eq!(std::mem::size_of::<DType>(), 4);`. — body length: 1
- `crates/cuda-kernels/src/collective.rs:366` `reduce_op_enum_size` — same shape. — body length: 1

> Note: `crates/cuda-kernels/src/ffi/nccl.rs:154` `unique_id_size` and
> `:159` `nccl_result_size` are structurally identical to the two
> `collective.rs` size_of asserts BUT they pin the **NCCL FFI ABI**
> (`ncclUniqueId` size and `ncclResult_t` size). A wrong size would silently
> corrupt distributed init. These move to **EXEMPT — FFI contract**, not
> Class A. Treat them as cousins of the qwen3-spec wire-format tests.

Total Class A: **9**

## Class B — Pure constructor coverage

Group: `infer/src/scheduler/policy.rs`

- `infer/src/scheduler/policy.rs:526` `eviction_candidate_new_defaults` — `EvictionCandidate::new(7, 42)` then asserts each `slot/last_access_step/tokens=0/hit_count=0/prefix_depth=0/!pinned`. Tests `new()` field-by-field, no behavior. — body length: 8

Group: `infer/src/quant.rs`

- `infer/src/quant.rs:659` `quant_format_display` — asserts each `QuantFormat` variant's `to_string()` matches a literal. Borderline B vs D — these strings are user-visible diagnostic labels and might be considered surface contract. **Default to KEEP if you treat them as user-facing.** — body length: 5
- `infer/src/quant.rs:667` `quant_format_is_quantized` — flag-bit check across 3 enum variants. — body length: 4

Group: `infer/src/model_registry.rs`

- `infer/src/model_registry.rs:386` `implemented_models` — `assert!(ModelArch::Qwen3.is_implemented()); assert!(ModelArch::Qwen35.is_implemented());`. — body length: 2
- `infer/src/model_registry.rs:392` `llama_not_yet_implemented` — `assert!(!ModelArch::Llama.is_implemented()); assert!(!ModelArch::DeepSeekV3.is_implemented());`. — body length: 2

Group: `crates/train/tests/test_metrics.rs`

- `crates/train/tests/test_metrics.rs:30` `null_sink_emit_does_not_panic` — constructs `NullSink`, calls `emit()` + `flush()`, no assertion. — body length: 9. *Borderline* — public-API "doesn't panic" smoke. Could move to E if you treat `MetricSink` as a contract.

Total Class B: **6**

## Class C — Mock-the-world

**None confirmed.** Several files contain `Mock*` / `Fake*` types but in every case the test mocks a *narrow* boundary while exercising real code on the inside:

- `infer/src/http_server/sessions.rs` — `MockEngine` is a thin `SessionPersistence` adapter; the tests run real `RadixCache::insert_with_fingerprints`, real `DiskStore`, real serde — verifying actual save/load semantics. **Keep as E.**
- `infer/src/speculative.rs:576` `mock_draft_model` — `MockDraftModel::new(42, 0.8, 0.9).draft_batch(...)` returns a real `TokenProposal`; the test verifies the draft probability shape comes out correctly. **Keep as E.**
- `crates/agent/src/lib.rs` `FakeEngine` — wraps a Vec<&str>, but the surrounding `AgentSession::run_turn` test exercises the full session loop, prompt formatting, tool-call recovery, persistence — that's real behavior, the engine is just a deterministic stub. **Keep as E.**

Replacement strategy column not applicable — no deletion target.

## Class D — Implementation-detail locks

- `crates/train/src/data_adapter.rs:245` `adamw_for_backend_keeps_cpu_host_backed` — `assert!(format!("{optim:?}").contains("device_backed: false"))`. Asserts on `Debug` substring of an internal optimizer; should be a real getter. — body length: 5
- `crates/train/src/data_adapter.rs:255` `adamw_for_backend_uses_device_path_on_metal` (under `#[cfg(feature = "metal")]`) — same `format!("{optim:?}").contains(...)` shape. — body length: 7
- `crates/cuda-kernels/src/tensor.rs:1387` `kernel_alignment_names_scale_layout_explicitly` — pins exact layout-name strings (`"wN.row_major.group_packed"`, `"bf16[row, k/group_size]"`, etc.). These look internal — but they may be load-bearing for kernel-dispatch routing. **Recommend asking before delete.** Could move to "exempt" if those strings are part of the cuda-kernels prelude API surface. — body length: 10

Total Class D: **3** (with one debatable)

> No assertions on exact public-API error message text (those would move to exempt). The error-message asserts I found (`err.to_string().contains("publish-last")`, `err.to_string().contains("model.safetensors")`, etc.) test invariants documented in the source comment as load-bearing — keep as E.

## Class E — Behavioral end-to-end (KEEP)

Counted by file (canonical example per file):

- `infer/src/scheduler/tests.rs` — 17 tests; canonical: full `Scheduler::step` cycle with chunked prefill + decode batch
- `infer/src/scheduler/cuda/runtime/tests.rs` — 14 tests; canonical: rendezvous spec-lock test (R4 [P3], cited in `b6e06fa`)
- `infer/src/scheduler/cuda/budget.rs` — 15 tests; canonical: `additional_pages_needed` math + admission shortage
- `infer/src/scheduler/cuda/core.rs` — 5 tests; canonical: `host_spill_target_bytes` watermark policy
- `infer/src/scheduler/cuda/decode.rs` — 3 tests; canonical: `retract_victim_score` ordering invariant
- `infer/src/scheduler/cuda/prefill.rs` — 7 tests; canonical: `should_downgrade_partial_hit_to_miss` hybrid downgrade quantified loop
- `infer/src/scheduler/cuda/execution.rs` — 8 tests; canonical: prefill / decode plan execution
- `infer/src/scheduler/cuda/request.rs` — 5 tests; canonical: `cached_prompt_to_publish` lifecycle
- `infer/src/scheduler/policy.rs` — 19 tests; canonical: prefix-aware admission + eviction policy
- `infer/src/kv_tier/coordinator/tests.rs` — 18 tests; canonical: store roundtrip through real DiskStore
- `infer/src/kv_tier/host_pool.rs` — 10 tests; canonical: pinned-host pool reserve/release
- `infer/src/kv_tier/transport/disk.rs` — 10 tests; canonical: disk store round-trip
- `infer/src/kv_tier/transport/{nixl,shared_fs,local_cuda}.rs` — 12 total
- `infer/src/kv_tier/{io,lookup,chunk,backend,tier,readmission}.rs` — ~16 total; canonical: `advise_recompute` cost model in `lookup.rs`
- `infer/src/http_server/sessions.rs` — 14 tests; canonical: `save_then_load_round_trips_radix_and_payloads`
- `infer/src/backend/cpu.rs` — 7 tests; canonical: `cpu_backend_respects_max_new_tokens_budget` real generate path
- `infer/src/backend/runtime.rs` — 6 tokio tests; canonical: scheduler runtime guard join
- `infer/src/backend/metal/scheduler.rs` — 14 tests; canonical: continuous-batching slot lifecycle
- `infer/src/backend/metal/qwen35.rs` — 20 tests; canonical: hybrid attention forward parity
- `infer/src/backend/metal/dflash.rs` — 14 tests; canonical: dispatcher Flash attention path
- `infer/src/backend/metal/gdr.rs` — 12 tests; canonical: graph dispatcher routing
- `infer/src/backend/metal/kv_pool.rs` — 6 tests; canonical: slot ledger refcount + share-prefix
- `infer/src/backend/metal/{config,prefix_cache,request_state/tests,generate}.rs` — ~17 total
- `infer/src/backend/metal/mlx.rs` — 11 tests; canonical: `batched_sdpa_2pass_matches_causal_sdpa_for_m16`
- `infer/src/backend/cuda/bootstrap.rs` — 1 test; canonical: scheduler runtime guard joins
- `infer/src/server_engine.rs` — 9 tests; canonical: end-to-end engine shutdown semantics
- `infer/src/block_manager.rs` — 8 tests; canonical: alloc/free + ref-count invariants
- `infer/src/sampler.rs` — 12 tests (numerical, see F)
- `infer/src/speculative.rs` — 21 tests; canonical: `partial_acceptance` real verify_tokens
- `infer/src/tokenizer.rs` — 5 tests; canonical: ChatML special-token resolution
- `infer/src/main.rs` — 4 tests (CLI flag parsing, surface)
- `infer/src/events.rs` — 1 test (sink emit)
- `infer/src/metrics.rs` — 6 tests; canonical: prometheus-text round-trip
- `infer/src/trace_reporter.rs` — 7 tests; canonical: OTLP endpoint normalization
- `infer/src/distributed/init_method.rs` — 6 (excluding `unique_id_size_constant` Class A); canonical: `rendezvous_world_size_4`
- `infer/src/model_source.rs` — 2; canonical: prefers runtime-assets sidecar
- `infer/src/model/qwen3/{config,lora}.rs` + `model/qwen35/{config,weights}.rs` — 12 total
- `infer/src/weight_loader.rs` — 7 tests (canonical: tied-weight Qwen3 4B shard discovery)
- `infer/src/gguf.rs` — 6 tests
- `infer/src/hf_hub.rs` — 9 tests; canonical: snapshot-path decode, repo-id round-trip
- `infer/src/quant.rs` — 11 (excl 2 Class B); canonical: GPTQ + AWQ config parsing
- `infer/src/tensor_parallel.rs` — 24 tests; canonical: SGLang docstring 1749-1756 group derivation
- `infer/src/model_registry.rs` — 13 (excl 2 Class B); canonical: arch detection across Qwen / Llama / Phi / Gemma / DeepSeek
- `infer/src/bin/metal_bench.rs` — 9 tests; canonical: bench-output formatter
- `infer/src/backend/metal.rs` — 2 ignored bench tests
- `crates/agent/src/lib.rs` — 15 tests; canonical: `tool_call_messages_are_not_duplicated_in_followup_prompt`
- `crates/cli/src/{model_catalog,hardware,hub_discovery,doctor,welcome,startup,model_picker,serve,hf_search,tps,train_cli}.rs` — ~75 tests
- `crates/cli/src/repl.rs` — 24 (exempt)
- `crates/cli/src/args.rs` — 27 (exempt)
- `crates/chat/src/{lib,protocol}.rs` — 19 (exempt — wire format)
- `crates/tools/src/lib.rs` — 11 tests; canonical: sandbox-exec profile + bash command timeout
- `crates/kv-native-sys/src/lib.rs` — 11 tests; canonical: WAL append + replay round-trip
- `crates/cuda-kernels/src/{graph_pool,paged_kv,flashinfer,kv_types}.rs` + `ffi/nccl.rs` — ~46 (excl Class A pair); canonical: `bf16_budget_has_no_work_buffer_component`
- `crates/cuda-kernels/src/tensor.rs` — 6 (excl Class D); canonical: `test_device_matrix_from_safetensors_matches_from_host`
- `crates/autograd/src/{module,tensor,tape}.rs` — 5; canonical: `backward_on_empty_tape_does_not_panic`
- `crates/train/src/*.rs` (per-file lists already enumerated above) — ~57 unit tests
- `crates/train/tests/*.rs` — integration tests (exempt)

(Counts above approximate; the canonical example per file is enough for a delete-blockers list.)

## Class F — Numerical / golden (KEEP)

By file, with a canonical example each:

- `infer/src/sampler.rs` — `test_apply_repetition_penalty_positive_logit` (vLLM-equivalent formula)
- `infer/src/speculative.rs` — `partial_acceptance` (speculative-decode paper math)
- `infer/src/ops/tests.rs` — `test_rms_norm_batch_multi_tile` (CUDA kernel vs CPU reference, `assert_close`)
- `infer/src/backend/metal/mlx.rs` — `batched_sdpa_2pass_matches_causal_sdpa_for_m16`, `build_varlen_verify_mask_b2_matches_reference`
- `infer/src/backend/metal/qwen35.rs` — multiple `*_matches_reference*` (~20)
- `infer/src/backend/metal/dflash.rs` — flash-attention numerical match
- `crates/autograd/src/{module,safetensors_io}.rs` — bit-exact and bf16-tolerant round-trips
- `crates/autograd/tests/test_backend.rs` — 64 tests, full numerical reference for autograd backend
- `crates/autograd/tests/m*.rs` — gradient checks against numerical-grad probes
- `crates/autograd/tests/test_adamw_state.rs` — AdamW step math
- `crates/autograd/tests/test_lr_schedule.rs` — schedule curves
- `crates/autograd/tests/test_linear_attention.rs` — linear-attention numerical reference
- `crates/cuda-kernels/src/{paged_kv,flashinfer,tensor}.rs` — kernel-aligned page math, plan/run reuse, safetensors→DeviceMatrix matches host upload
- `crates/cuda-kernels/src/graph_pool.rs` — graph dispatch padding math
- `infer/src/quant.rs` — config-parsing fixtures (exempt-class numeric)
- `infer/tests/q4k_kernel_correctness.rs`, `infer/tests/ground_truth_q4k.rs`, `infer/tests/carnice_*` — GGUF Q4_K dequant numerical golden (in exempt `infer/tests/`)
- `infer/tests/greedy_consistency.rs` — greedy-decode determinism gate (exempt)
- `crates/train/src/sampling.rs` — `sampled_log_prob_matches_reference_at_temperature`
- `crates/train/src/sft_data.rs` — body_span byte-offset math (boundary-crossing BPE merge)
- `crates/train/tests/test_qwen3{,5}_forward.rs`, `test_qwen35_hybrid_*.rs` — Qwen3/Qwen3.5 forward numerical parity (exempt)

## Class G — Property / fuzz (KEEP if exists)

No `proptest` / `quickcheck` integration exists. Two loop-quantified tests behave as ad-hoc properties and are valuable:

- `infer/src/scheduler/cuda/prefill.rs:723` `hybrid_downgrade_fires_on_every_partial_hit` — quantifies over a 9×16 grid of `(raw, prompt)` shapes plus the diagonal exempt set.
- `infer/src/speculative.rs:608` `sample_from_uniform_dist` — empirical statistical check (1000 draws, ±3σ bound).

## Class U — Unclassified / needs human eyes

These are tests where intent / value is ambiguous from the body alone. **Do not delete without a human review** — listed here so they're visible:

- `infer/src/backend/metal/mlx.rs:908` `lifecycle` — three trivial assertions on a tiny `MlxArray`. Smoke for the FFI bridge initialization, not for behavior. Could be A but it does cross a real FFI; keep U.
- `infer/src/backend/metal/mlx.rs:916` `add_basic` — 1+2 + 3+4 → 4 elementwise. Trivial *math* but real *bridge* exercise. U.
- `infer/src/backend/metal/mlx.rs:926` `matmul_basic` — same shape, smoke for matmul. U.
- `infer/src/backend/metal/mlx.rs:1022` `dtype_supports_int64_roundtrip` — `as_dtype(&ints, Int64).dtype() == Int64`. Tests one FFI call. Could be A but it actually crosses the dtype-cast path. U.
- `infer/src/backend/metal/generate.rs:303–323` — three flag-parser tests on `metal_kv_pool_flag_is_truthy` / `resolve_metal_kv_pool_enabled`. Looks B-like, but the "truthy parser" set is part of the env-var contract and overlaps with `feedback_use_industry_env_vars.md`. Lean exempt-contract. U.
- `infer/src/scheduler/policy.rs:373` `scheduler_signals_default_is_cold` — calls `SchedulerSignals::default().is_cold_request()`. Tests one branch of one method on the default value. Borderline A / B. U because `is_cold_request` is a key admission-policy primitive.
- `infer/src/scheduler/policy.rs:378–402` (3 tests) `scheduler_signals_warm_on_*` — set one field, assert `!is_cold_request()`. Each tests a different warm-signal branch (prefix hit, session affinity, turn depth). The branches *are* the policy contract. Lean E. U for safety.
- `infer/src/types.rs:319` `session_id_round_trips_from_str_and_string` — `SessionId::from(&str) == SessionId::from(String)`. Tests the `From` impls converge. Borderline A. U because session_id round-trip is part of the HTTP session API.
- `crates/cli/src/hardware.rs:208–230` (3 tests) — detect_system / effective_memory / compiled_backend "sensible values"; reads real OS state. Doctor-output contract; keep but tag U because they could be tightened.
- `crates/cli/src/model_catalog.rs:191` `small_metal_system_gets_small_models` — recommendation logic but the catalog is data; could be a spec test or a placeholder. U.
- `crates/cli/src/model_catalog.rs:217` `catalog_entries_have_valid_data` — non-emptiness loop over CATALOG. Borderline B. U because catalog drives `arle list-models`.
- `infer/src/distributed/init_method.rs:257` is in **A** above; the *other* 6 tests in that file are E.
- `infer/src/quant.rs:659` `quant_format_display` — listed in B but flagged for human review (user-visible label).

Total Class U: **18** (some files contribute multiple).

## Proposed delete order

Recommended phasing for the cleanup pass:

1. **A first.** Lowest risk: pure tautologies on derives. None of them encode behavior. ~9 deletes.
2. **B next.** Constructor-coverage tests still test something but redundantly with the constructor's documentation. ~6 deletes; flag the borderline `null_sink_emit_does_not_panic` for human eyes.
3. **D third.** Replace each `format!("{:?}", ...)` assertion with a real getter or move the asserted attribute into a public API. Deletes only when there's no covering test.
4. **C last.** Deferred: zero-confirmed candidates means there's nothing to do here. If a future audit finds them, replace with real backend / axum test client / Engine fixture **before** deleting.

**Rationale:** A → B → D → C orders by safety. Class A's removal cannot regress any contract. Class B's removal mildly weakens documentation-by-test for constructors. Class D requires a concrete refactor (add a getter) and so is more invasive. Class C deletes need a real-backend replacement and so are most invasive.

## Per-file inventory (full list)

Full enumeration of every file with `#[test]` / `#[tokio::test]` markers, with
counts and a one-line summary. Files marked **EXEMPT** are excluded from
delete consideration regardless of class (see *Exempt — contract tests*
above).

### `infer/src/`

| File | Tests | Class summary |
|------|-------|---------------|
| `infer/src/types.rs` | 11 | 5×A (lines 198, 205, 213, 304, 327), rest E (`fingerprint_compute_*`) — fingerprint stability is a persistence contract → **EXEMPT** for those |
| `infer/src/sampler.rs` | 12 | F (numerical penalties + greedy) |
| `infer/src/main.rs` | 4 | E (CLI flag parsing surface) |
| `infer/src/server_engine.rs` | 9 | E + tokio (engine lifecycle, shutdown) |
| `infer/src/scheduler/types.rs` | 24 | **EXEMPT** (slot lifecycle invariants) |
| `infer/src/scheduler/policy.rs` | 19 | E + 1 B (`eviction_candidate_new_defaults`); 4 in U (warm-signal flag tests) |
| `infer/src/scheduler/tests.rs` | 17 | E (full Scheduler::step integration) |
| `infer/src/scheduler/cuda/budget.rs` | 15 | E (page-budget math) |
| `infer/src/scheduler/cuda/core.rs` | 5 | E (watermark + sealed prefix) |
| `infer/src/scheduler/cuda/decode.rs` | 3 | E (retract math) |
| `infer/src/scheduler/cuda/prefill.rs` | 7 | E + 1 G (hybrid downgrade quantified loop) |
| `infer/src/scheduler/cuda/execution.rs` | 8 | E |
| `infer/src/scheduler/cuda/request.rs` | 5 | E |
| `infer/src/scheduler/cuda/runtime/tests.rs` | 14 | E (CUDA scheduler runtime, rendezvous spec-lock) |
| `infer/src/prefix_cache.rs` | 55 | **EXEMPT** (cache invariants) |
| `infer/src/block_manager.rs` | 8 | E (allocator) |
| `infer/src/speculative.rs` | 21 | E + F + 1 G (sample_from_uniform_dist) |
| `infer/src/tensor_parallel.rs` | 24 | E (TP/PP/EP/DP rank decomposition; SGLang docstring parity) |
| `infer/src/tokenizer.rs` | 5 | E (special-token resolution) |
| `infer/src/events.rs` | 1 | E (sink emit) |
| `infer/src/metrics.rs` | 6 | E (prometheus-text round-trip) |
| `infer/src/trace_reporter.rs` | 7 | E (OTLP wire format) |
| `infer/src/distributed/init_method.rs` | 7 | 1×A (`unique_id_size_constant`), 6 E (rendezvous protocol) |
| `infer/src/model_registry.rs` | 15 | 13 E (arch detect) + 2 B (`implemented_models`, `llama_not_yet_implemented`) |
| `infer/src/model_source.rs` | 2 | E (sidecar resolution) |
| `infer/src/model/qwen3/config.rs` | 3 | E |
| `infer/src/model/qwen3/lora.rs` | 4 | E |
| `infer/src/model/qwen35/config.rs` | 4 | E |
| `infer/src/model/qwen35/weights.rs` | 1 | E |
| `infer/src/ops/tests.rs` | 16 | F (CUDA kernel numerical reference) |
| `infer/src/quant.rs` | 13 | 11 E (config parsing) + 2 B (`quant_format_display`, `quant_format_is_quantized`) |
| `infer/src/weight_loader.rs` | 7 | E (quant layout + shard discovery) |
| `infer/src/gguf.rs` | 6 | E (GGUF format) |
| `infer/src/hf_hub.rs` | 9 | E (snapshot path round-trip) |
| `infer/src/kv_tier/coordinator/tests.rs` | 18 | E (real coordinator + DiskStore) |
| `infer/src/kv_tier/host_pool.rs` | 10 | E (pinned pool) |
| `infer/src/kv_tier/io.rs` | 2 | E (KVPayloadRef) |
| `infer/src/kv_tier/lookup.rs` | 2 | E (cost model) |
| `infer/src/kv_tier/chunk.rs` | 2 | E (KVSpan / KVHandle) |
| `infer/src/kv_tier/backend.rs` | 2 | E |
| `infer/src/kv_tier/tier.rs` | 2 | 1×A (`tier_variants_round_trip`) + 1 E (`block_location_reports_its_tier`) |
| `infer/src/kv_tier/readmission.rs` | 3 | E |
| `infer/src/kv_tier/transport/disk.rs` | 10 | E (file format) |
| `infer/src/kv_tier/transport/local_cuda.rs` | 3 | E |
| `infer/src/kv_tier/transport/nixl.rs` | 4 | E |
| `infer/src/kv_tier/transport/shared_fs.rs` | 5 | E |
| `infer/src/http_server/openai_v1.rs` | 29 | **EXEMPT** (OpenAI v1 wire) |
| `infer/src/http_server/sessions.rs` | 14 | **EXEMPT** (KV-tier session save/load wire) |
| `infer/src/http_server/tests.rs` | 54 | **EXEMPT** (HTTP integration) |
| `infer/src/backend/cpu.rs` | 7 | E (real cpu generate path) |
| `infer/src/backend/metal.rs` | 2 | E (ignored bench tests) |
| `infer/src/backend/metal/config.rs` | 6 | E |
| `infer/src/backend/metal/dflash.rs` | 14 | E + F (flash attention numerical) |
| `infer/src/backend/metal/gdr.rs` | 12 | E (graph dispatcher routing) |
| `infer/src/backend/metal/generate.rs` | 3 | U (env-var truthy parser; possibly EXEMPT) |
| `infer/src/backend/metal/kv_pool.rs` | 6 | E (slot ledger) |
| `infer/src/backend/metal/mlx.rs` | 11 | F + 4 U (smoke FFI bridge tests) |
| `infer/src/backend/metal/prefix_cache.rs` | 4 | E |
| `infer/src/backend/metal/qwen35.rs` | 20 | F (Qwen3.5 numerical reference) |
| `infer/src/backend/metal/request_state/tests.rs` | 4 | E |
| `infer/src/backend/metal/scheduler.rs` | 14 | E (continuous-batching) |
| `infer/src/backend/runtime.rs` | 6 | E + tokio |
| `infer/src/backend/cuda/bootstrap.rs` | 1 | E |
| `infer/src/bin/metal_bench.rs` | 9 | E (bench-output formatter) |

### `infer/tests/` — all **EXEMPT** (E2E surface)

| File | Tests | Description |
|------|-------|-------------|
| `infer/tests/e2e.rs` | 1 | Qwen3 end-to-end |
| `infer/tests/e2e_qwen35.rs` | 1 | Qwen3.5 end-to-end |
| `infer/tests/qwen3_8b_regression.rs` | 3 | 8B regression baseline |
| `infer/tests/test_qwen3_lora_loader.rs` | 3 | LoRA adapter loader |
| `infer/tests/bench_prefill.rs` | 2 | Prefill bench (#[test]-bearing utility per `project_tests_dir_convention.md`) |
| `infer/tests/greedy_consistency.rs` | 1 | Greedy determinism gate |
| `infer/tests/q4k_kernel_correctness.rs` | 5 | F (Q4_K kernel) |
| `infer/tests/ground_truth_q4k.rs` | 1 | F |
| `infer/tests/carnice_dtype_audit.rs` | 1 | F |
| `infer/tests/carnice_tensor_probe.rs` | 1 | F |
| `infer/tests/carnice_real_tensor_dequant.rs` | 1 | F |
| `infer/tests/smoke_qwen3_4b_gguf.rs` | 1 | smoke |
| `infer/tests/smoke_qwen35_gguf.rs` | 1 | smoke |
| `infer/tests/smoke_carnice_27b_q4k.rs` | 1 | smoke |
| `infer/tests/gen_test_data_35.rs` | 1 | utility-named, but #[test]-bearing per convention |
| `infer/tests/regen_test_data.rs` | 1 | utility-named, but #[test]-bearing per convention |

### `crates/agent/`

| File | Tests | Class |
|------|-------|-------|
| `crates/agent/src/lib.rs` | 15 | E (full agent session loop, tool-call recovery, persistence) |

### `crates/autograd/`

All numerical / behavioral; collectively **EXEMPT** (autograd contract).

| File | Tests | Class |
|------|-------|-------|
| `crates/autograd/src/module.rs` | 2 | F (Linear forward against hand-derived) |
| `crates/autograd/src/safetensors_io.rs` | 5 | F (round-trip f32 + bf16) — **EXEMPT** wire format |
| `crates/autograd/src/tape.rs` | 1 | E (empty-tape backward) |
| `crates/autograd/src/tensor.rs` | 2 | E (alloc/free reuse, from_slice tracking) |
| `crates/autograd/tests/m0_ops.rs` | 4 | F |
| `crates/autograd/tests/m0_toy.rs` | 1 | F |
| `crates/autograd/tests/m1_adamw.rs` | 1 | F |
| `crates/autograd/tests/m1_exp.rs` | 1 | F |
| `crates/autograd/tests/m1_layout.rs` | 3 | F |
| `crates/autograd/tests/m1_mlp.rs` | 1 | F |
| `crates/autograd/tests/m1_ops.rs` | 10 | F |
| `crates/autograd/tests/m1_sigmoid.rs` | 1 | F |
| `crates/autograd/tests/m1_slice.rs` | 2 | F |
| `crates/autograd/tests/test_adamw_state.rs` | 8 | F |
| `crates/autograd/tests/test_backend.rs` | 64 | **EXEMPT** F (backend parity reference) |
| `crates/autograd/tests/test_device_handle.rs` | 25 | E |
| `crates/autograd/tests/test_linear_attention.rs` | 2 | F |
| `crates/autograd/tests/test_lr_schedule.rs` | 8 | F |
| `crates/autograd/tests/test_optimizer_trait.rs` | 1 | F |

### `crates/chat/` — all **EXEMPT** (ChatML wire format)

| File | Tests |
|------|-------|
| `crates/chat/src/lib.rs` | 4 |
| `crates/chat/src/protocol.rs` | 15 |

### `crates/cli/`

| File | Tests | Class |
|------|-------|-------|
| `crates/cli/src/args.rs` | 27 | **EXEMPT** (clap surface) |
| `crates/cli/src/repl.rs` | 24 | **EXEMPT** (REPL state machine, recent regressions) |
| `crates/cli/src/doctor.rs` | 6 | E |
| `crates/cli/src/hardware.rs` | 3 | U (sensible-values smoke) |
| `crates/cli/src/hf_search.rs` | 4 | E |
| `crates/cli/src/hub_discovery.rs` | 5 | E |
| `crates/cli/src/model_catalog.rs` | 4 | E + 1 U (`catalog_entries_have_valid_data`) |
| `crates/cli/src/model_picker.rs` | 3 | E |
| `crates/cli/src/serve.rs` | 3 | E |
| `crates/cli/src/startup.rs` | 2 | E (XDG marker path) |
| `crates/cli/src/tps.rs` | 5 | E (formatter + stream-record) |
| `crates/cli/src/train_cli.rs` | 9 | E (invocation resolution + JSON report serialization) |
| `crates/cli/src/welcome.rs` | 3 | E (XDG-config path) |

### `crates/cuda-kernels/`

| File | Tests | Class |
|------|-------|-------|
| `crates/cuda-kernels/src/collective.rs` | 2 | **2×A** (enum_size) |
| `crates/cuda-kernels/src/ffi/nccl.rs` | 2 | **EXEMPT** (NCCL FFI ABI; size-of pin) |
| `crates/cuda-kernels/src/flashinfer.rs` | 5 | E (plan/run reuse + decode plan) |
| `crates/cuda-kernels/src/graph_pool.rs` | 16 | E (graph dispatch math) |
| `crates/cuda-kernels/src/kv_types.rs` | 2 | E |
| `crates/cuda-kernels/src/paged_kv.rs` | 17 | E (page math) |
| `crates/cuda-kernels/src/tensor.rs` | 6 | 5 E (round-trip) + 1 D (`kernel_alignment_names_scale_layout_explicitly`) |

### `crates/kv-native-sys/`

| File | Tests | Class |
|------|-------|-------|
| `crates/kv-native-sys/src/lib.rs` | 11 | E (WAL + shm + mmap + arena lifecycle) |

### `crates/qwen3-spec/`, `crates/qwen35-spec/` — all **EXEMPT**

| File | Tests |
|------|-------|
| `crates/qwen3-spec/src/lib.rs` | 11 |
| `crates/qwen35-spec/src/lib.rs` | 21 |

### `crates/tools/`

| File | Tests | Class |
|------|-------|-------|
| `crates/tools/src/lib.rs` | 11 | E (sandbox-exec + bash + bare-command timeout) |

### `crates/train/` — most **EXEMPT** (numerical + checkpoint wire)

| File | Tests | Class |
|------|-------|-------|
| `crates/train/src/checkpoint.rs` | 7 | E (latest-symlink + publish-last contract) |
| `crates/train/src/cli_args.rs` | 9 | E (arg parser) |
| `crates/train/src/commands/pretrain.rs` | 11 | E |
| `crates/train/src/commands/train_grpo.rs` | 5 | E |
| `crates/train/src/commands/train_multi_turn.rs` | 7 | E |
| `crates/train/src/commands/train_sft.rs` | 6 | E |
| `crates/train/src/data_adapter.rs` | 9 | 7 E + **2 D** (`adamw_for_backend_*`) |
| `crates/train/src/eval_lm.rs` | 2 | E |
| `crates/train/src/multi_turn.rs` | 1 | E |
| `crates/train/src/qwen3_checkpoint.rs` | 5 | E |
| `crates/train/src/qwen3.rs` | 1 | F |
| `crates/train/src/qwen35_checkpoint.rs` | 1 | E |
| `crates/train/src/sampling.rs` | 3 | F (sampled log-prob ref) |
| `crates/train/src/sft_data.rs` | 5 | F (body_span byte math) |
| `crates/train/src/tokenizer.rs` | 4 | E |
| `crates/train/tests/test_checkpoint_v2.rs` | 6 | **EXEMPT** (v2 wire format) |
| `crates/train/tests/test_control.rs` | 6 | E |
| `crates/train/tests/test_convergence_smoke.rs` | 1 | F |
| `crates/train/tests/test_curriculum.rs` | 8 | E (TaskPool lifecycle) |
| `crates/train/tests/test_grad_accum.rs` | 4 | F |
| `crates/train/tests/test_grad_clip.rs` | 10 | F |
| `crates/train/tests/test_grpo.rs` | 6 | F (GRPO numerical) |
| `crates/train/tests/test_lm.rs` | 1 | F |
| `crates/train/tests/test_metrics.rs` | 16 | E + 1 B (`null_sink_emit_does_not_panic`) |
| `crates/train/tests/test_multi_turn_qwen3.rs` | 1 | F |
| `crates/train/tests/test_multi_turn.rs` | 6 | E |
| `crates/train/tests/test_qwen_lora.rs` | 2 | F |
| `crates/train/tests/test_qwen3_forward.rs` | 1 | F (**EXEMPT** parity) |
| `crates/train/tests/test_qwen35_forward.rs` | 2 | F (**EXEMPT** parity) |
| `crates/train/tests/test_qwen35_hybrid_backend_parity_metal.rs` | 6 | F (**EXEMPT** parity) |
| `crates/train/tests/test_qwen35_hybrid_forward_metal.rs` | 1 | F (**EXEMPT** parity) |
| `crates/train/tests/test_qwen35_hybrid_sft_loop_metal.rs` | 1 | E |
| `crates/train/tests/test_qwen35_hybrid_sft_loop.rs` | 1 | E |
| `crates/train/tests/test_qwen35_sft_loop_metal.rs` | 1 | E |
| `crates/train/tests/test_reward.rs` | 7 | F |
| `crates/train/tests/test_sft_data.rs` | 4 | F |
| `crates/train/tests/test_sft_loop_metal.rs` | 1 | E |
| `crates/train/tests/test_sft_loop_smoke.rs` | 1 | E |
| `crates/train/tests/test_task_gen.rs` | 5 | E |
| `crates/train/tests/test_trainer_loop.rs` | 21 | E |
| `crates/train/tests/test_verifier.rs` | 11 | F |

## Risk register

False-positive risk per class:

- **A**: Very low. The 9 tests assert structural derives; deleting them removes ~50 LOC and no behavioral coverage.
- **B**: Low–medium. `null_sink_emit_does_not_panic` is a "doesn't panic" smoke for a public sink — if `MetricSink` adds a panicking debug assert later, this test would have caught it. Either keep or replace with a `#[should_panic = ""]`-style negative.
- **C**: N/A (none confirmed).
- **D**: Medium. The two `data_adapter.rs` tests do test a *real distinction* (host-backed vs device-backed AdamW) — the lock is the wrong way to test it, but the *fact* is load-bearing for Metal training. Replace with `optim.is_device_backed()` (add the getter) before deleting.
- **U**: High by definition. Do not delete without a human pass.

Mitigation: `cargo test --workspace` after each phase + `codex review --uncommitted` on the deletion commits.

### Smoke list — 10 tests that, if accidentally deleted, would cost us the most

1. `infer/src/scheduler/cuda/prefill.rs:723` `hybrid_downgrade_fires_on_every_partial_hit` — quantified safety-invariant for hybrid prefill.
2. `infer/src/sampler.rs:233` `test_apply_repetition_penalty_positive_logit` — vLLM-parity formula. Wire-compat with downstream prompt engineers.
3. `infer/src/scheduler/cuda/budget.rs:347` `estimated_request_pages_ignores_max_tokens_at_admission` — locks SGLang-style admission against accidental "reserve full decode tail" regression.
4. `infer/src/scheduler/cuda/budget.rs:339` `estimated_request_pages_charges_prompt_only_plus_one` — pair with the above.
5. `infer/src/http_server/sessions.rs:766` `load_errors_on_tampered_disk_payload` — KV-tier persistence integrity contract.
6. `infer/src/http_server/sessions.rs:795` `load_rejects_kv_format_tag_mismatch` — kv_format_tag wire-compat across pool format flips.
7. `infer/src/tensor_parallel.rs:751` `tp2_pp4_groups_sglang_docstring_1749_1756` — TP/PP rank-decomposition vs SGLang reference.
8. `infer/src/distributed/init_method.rs:262` `rendezvous_world_size_2` — basic distributed bring-up; broken means multi-GPU broken.
9. `infer/src/scheduler/cuda/runtime/tests.rs:380` rendezvous spec-lock R4 [P3] — the exact regression noted in `b6e06fa` (recent test_infer R4 [P3]).
10. `crates/autograd/src/safetensors_io.rs:363` `roundtrip_bf16_via_save` — the bf16 path infer can actually consume; deletion would silently regress checkpoint→inference.

(Bonus 11th: `infer/src/scheduler/tests.rs:557` is the integration test that locks "max_new_tokens budget honored across chunked prefill + decode" — also load-bearing.)
