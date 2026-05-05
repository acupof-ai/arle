# 2026-05-05 — CUDA kernel substrate unification on TileLang

**Status:** drafted, awaiting 3 user decisions (D1/D2/D3) before Phase 1 starts.

User direction (verbatim, 2026-05-05): "优化下梳理下当前的代码架构和依赖关系
理想状态是一套代码支持所有的后端 以tilelang 为后端 编译器 阶段性metal独立支持。
删除 triton 全部用 tilelang 也删除 flashinfer 全部用tilelang 就可以
能用tilelang 性能做好的都用"

Goal: tidy architecture+deps; ideal end-state is one codebase, all backends,
with **TileLang as the backend compiler**. Metal stays **independently
supported** for now (phased). **Delete Triton entirely → TileLang for
activation/utility kernels.** **Delete FlashInfer entirely → TileLang for
attention.** Use TileLang wherever it can deliver competitive perf.

Composes with `docs/plans/2026-05-05-multi-backend-tilelang-rocm-vulkan.md`
(same date) which covers the ROCm + Vulkan multi-backend addition. THIS plan
is narrower: it is the CUDA-side Triton-and-FlashInfer-deletion that the
multi-backend plan §10 / §12 Phase C presupposes is already complete.

---

## Section 1 — Current state map

### 1.1 Live Triton AOT kernels (8 live + 4 dead scaffolding)

| # | Kernel | Caller | Live? |
|---|---|---|---|
| 1 | `silu_mul_triton_aot_cuda` | `infer/src/ops/elementwise.rs:74`, `infer/src/ops/linear.rs:436,527` | **YES** |
| 2 | `add_cuda` (Triton AOT variant) | none — shadowed by csrc `add_cuda` at `crates/cuda-kernels/csrc/misc/elementwise_basic.cu:54` | **DEAD** |
| 3 | `embedding_decode_cuda` (Triton variant) | none — shadowed by csrc | **DEAD** |
| 4 | `embedding_batched_cuda` (Triton variant) | none — shadowed by csrc | **DEAD** |
| 5 | `flash_attention_prefill_hd256_cuda` (Triton) | none — superseded by FlashInfer + TileLang HD256 | **DEAD** |
| 6 | `gated_delta_rule_prefill_chunk_prepare_cuda` | `infer/src/ops/recurrent.rs:378` | **YES** |
| 7 | `gated_delta_rule_prefill_chunk_cumsum_cuda` | `recurrent.rs:408` | **YES** |
| 8 | `gated_delta_rule_prefill_chunk_a_cuda` | `recurrent.rs:433` | **YES** |
| 9 | `gated_delta_rule_prefill_chunk_solve_cuda` | `recurrent.rs:457` | **YES** |
| 10 | `gated_delta_rule_prefill_chunk_recompute_cuda` | `recurrent.rs:490` | **YES** |
| 11 | `gated_delta_rule_prefill_chunk_state_cuda` | `recurrent.rs:534` | **YES** |
| 12 | `gated_delta_rule_prefill_chunk_o_cuda` | `recurrent.rs:579` | **YES** |

Build glue: `crates/cuda-kernels/build.rs:225-785` —
`probe_triton_python`, `find_triton_python`,
`generate_triton_artifacts_per_sm`, `format_dispatch_wrapper`,
`build_triton_kernel`, `compile_triton_aot_kernels`.

Tools dir: `crates/cuda-kernels/tools/triton/`.

### 1.2 FlashInfer surface area

FFI: `crates/cuda-kernels/src/ffi/attention.rs:46-484` declares 12 extern
symbols. Caller-meaningful set:

- Single-seq prefill HD128/HD256 (`ops/attention.rs:128-215`) — legacy non-paged path; no TileLang twin today.
- Batched paged prefill HD128/HD256 (`flashinfer.rs:344+412`, `:371+453`; dispatched at `attention.rs:586-851`) — has TileLang twin gated by `tilelang-attn` feature (default-on under `cuda`).
- Batched decode HD128 (`attention.rs:1141`) — no TileLang twin (today aliased via TC-decode at `:1217-1310` "Tranche 4" to the prefill HD128 cubin).
- Batched decode HD256 (`attention.rs:1508`) — TileLang twin exists but gated behind `tilelang-decode-hd256` feature, **build-broken on sm_89** (`docs/experience/errors/2026-04-29-tilelang-decode-hd256-sm89-build.md`).
- TC-decode (`attention.rs:1253`) — uses FlashInfer PrefillPlan as decode; aliased to TileLang HD128 prefill cubin under `tilelang-attn`.
- `flashinfer_append_last_token_indices_cuda` — utility, not perf-critical.

CSRC: `crates/cuda-kernels/csrc/attention/flashinfer_*.cu` — 8 files
(prefill + prefill_paged + decode + tc_decode + metadata, HD128 and HD256
variants).

Rust integration: `crates/cuda-kernels/src/flashinfer.rs` (1311 lines) —
`FlashInferWorkspace`, `FlashInferDecodeMetadata`, `BatchPrefillPagedPlan`,
plan/run/update/mixed-batch staging, decode-plan reuse heuristics.

Header search: `find_flashinfer_include` in `build.rs:1230` (env →
`pip show flashinfer-python` → import).

### 1.3 TileLang AOT kernels currently shipping

Build: `build.rs:1094-1203` (`compile_tilelang_aot_kernels`), gated by
`tilelang-attn` feature (which `cuda` implies). Head configs:

- `TILELANG_PREFILL_HD128_HEAD_CONFIGS = [(16,8), (32,8), (40,8), (64,8)]`
- `TILELANG_PREFILL_HD256_HEAD_CONFIGS = [(8,2), (16,2), (16,4)]`
- `TILELANG_DECODE_HD256_HEAD_CONFIGS = [(8,2), (16,2), (16,4)]` (gated, broken on sm_89)

Tools: `crates/cuda-kernels/tools/tilelang/{batch_prefill_paged_hd128.py, batch_prefill_paged_hd256.py, batch_decode_paged_hd256.py, gen_tilelang_aot.py}`.

FFI: `crates/cuda-kernels/src/ffi/attention.rs:582-710` (per-shape extern decls).

**Perf parity status (the gating evidence for Phase 3):**
- HD128 prefill: **conditional parity, workload-shape-sensitive.**
  - 2026-04-28 entry shows TileLang ahead -82% TTFT p50 / +5.1% out tok/s (Qwen3-4B / L4 / c=16, with patches A+C).
  - 2026-04-29 paired ON/OFF rerun shows TileLang **3.3-3.8% slower** out tok/s and 6.9-9.9% slower TTFT p50 vs FlashInfer, same shape. Contradiction unresolved.
- HD256 prefill: stub-pending-remote benches only.
- HD256 decode: build-broken on sm_89 (BLOCK_M=1 / `GemmWarpPolicy`); pending-remote benches only.
- TC-decode HD128 (alias to HD128 prefill cubin): neutral-to-slight-positive on L4.

### 1.4 Metal backend (out of scope)

`crates/mlx-sys/` (vendored MLX, cmake+cc, opaque `mlx_array*`),
`infer/src/backend/metal/` (scheduler, varlen decode, paged KV pool, prefix cache).
TileLang Metal status: dev-shim only
(`scripts/tilelang_metal_dev_backend.{sh,py}`). Production Metal stays on
MLX-bridge composition. **No edits this refactor.**

### 1.5 Cross-backend module entry points

- `infer/src/server_engine/types.rs:92-119` — `InferenceEngine` trait (4 methods). **Unchanged this refactor.**
- `infer/src/server_engine/loaded.rs` — `LoadedInferenceEngine` enum dispatch. **Unchanged.**
- `infer/src/backend.rs` — `InferenceBackend` / `StreamingInferenceBackend` traits. **Unchanged.**
- `infer/src/backend/cuda.rs` — 11-line `pub use` shim (post-extraction). **Unchanged.**

What changes is the **kernel substrate that the CUDA backend consumes** — `cuda_kernels` crate's `csrc/`, `ffi/`, `flashinfer.rs`. The trait surfaces stay flat.

---

## Section 2 — Target end-state

### 2.1 Trait contract (unchanged)

`InferenceEngine` trait shape stays. CUDA backend remains plumbed through
`LoadedInferenceEngine` enum arms; Metal stays through
`BackendInferenceEngine<MetalBackend>`. No new trait surface this round —
keeps the diff small and frees the multi-backend plan to add ROCm/Vulkan
arms later without conflict.

### 2.2 Kernel surface table after refactor

| Family | Replaces | Today | After refactor | Phase |
|---|---|---|---|---|
| `tilelang_silu_mul_*` | Triton silu_mul | Triton AOT (1 kernel) | TileLang AOT, per-shape cubins | 1 |
| `tilelang_gdr_chunk_*` (1 fused or 7 stages) | Triton GDR pipeline | Triton AOT (7 kernels) | TileLang AOT (1 fused if expressible, else 7 stages 1:1) | 2 |
| `tilelang_batch_prefill_paged_hd128_*` | FlashInfer batched paged prefill HD128 | TileLang + FlashInfer fallback | TileLang only | 3 |
| `tilelang_batch_prefill_paged_hd256_*` | FlashInfer batched paged prefill HD256 | TileLang + FlashInfer fallback | TileLang only | 3 |
| `tilelang_batch_decode_paged_hd128_*` | FlashInfer batched decode HD128 + TC-decode | TC-decode-only TileLang alias to prefill HD128 cubin | TileLang dedicated decode HD128 + alias retained | 4 |
| `tilelang_batch_decode_paged_hd256_*` | FlashInfer batched decode HD256 | Behind `tilelang-decode-hd256` feature, build-broken on sm_89 | TileLang only — fix codegen first | 4 |
| `tilelang_single_prefill_hd128_*` / `_hd256_*` | FlashInfer single-seq prefill | n/a | TileLang single-seq variants | 4 |
| `append_last_token_indices` | FlashInfer utility | utility C | Move into csrc `misc/`, drop FlashInfer header dep | 5 |

**Stays as csrc CUDA C** (kept; not part of this refactor):
`decode_attention_varlen_fp8.cu`, `decode_attention_int8.cu`,
`decode_attention_fp8.cu`, `prefill_attention_paged_prep.cu` (RMS+RoPE+KV
write — explicitly out of scope per `tilelang-integration.md` §2),
`decode_prep_paged*.cu`, `mla_decode.cu` (DeepSeek V4 P0''),
`decode_attention_quantized.cu`, `decode_attention_turboquant.cu`,
`gemm/marlin_*.cu`, `quantized_gemv.cu`, `kv/*` (scatter, quant, paged
append, kv_cache_to_paged), `quant/*`, `misc/elementwise_basic.cu`
(`add_cuda`, `embedding_decode_cuda`, `embedding_batched_cuda` — already
the live path; only Triton scaffolding is dead).

### 2.3 Build / dependency simplification

**Drop:**
- `tools/triton/` directory (entire).
- ~290 LoC in `build.rs` (Triton helpers).
- `INFER_TRITON_PYTHON` env var.
- `triton` Python package from build deps.
- `flashinfer-python` Python package.
- `find_flashinfer_include` (`build.rs:1230`); `-Iflashinfer_inc` nvcc plumbing (`build.rs:1346`); `FLASHINFER_INCLUDE_DIR` env var.
- 8 `.cu` files in `csrc/attention/flashinfer_*`.
- `crates/cuda-kernels/src/flashinfer.rs` (1311 lines) — see D3 below.
- 12 FlashInfer extern decls in `ffi/attention.rs`.
- `tilelang-attn`, `tilelang-decode-hd256` cargo features (collapsed into `cuda`).

**Keep:**
- `tools/tilelang/` directory.
- `find_tilelang_python`, `probe_tilelang_python`, `tilelang_include_dirs`, `generate_tilelang_artifacts_per_sm`, `build_tilelang_kernel`, `compile_tilelang_aot_kernels` (`build.rs:820-1203`).
- All `csrc/` outside `attention/flashinfer_*.cu`.
- `infer/src/backend/cuda.rs` 11-line shim.

**Add:**
- `tools/tilelang/{silu_mul.py, gated_delta_rule.py, batch_decode_paged_hd128.py, single_prefill_hd128.py, single_prefill_hd256.py}`.
- `cuda_kernels::tilelang::TileLangDecodeMetadata` (or rename of
  `FlashInferDecodeMetadata` per D3).
- New extern decls in `ffi/{attention,elementwise,recurrent}.rs`.

---

## Section 3 — Phased plan (risk-ascending)

### Phase 0 — Triton scaffolding deletion (zero-risk) **— LANDED `38d4d773` (2026-05-05)**

**Final scope (extended after Round 3 codex review):** delete 5 dead Triton spec
blocks from build.rs (`silu_mul`, `add`, `embedding_decode`, `embedding_batched`,
`flash_attn_prefill_hd256`) — all 5 had csrc native replacements at
`csrc/misc/elementwise_basic.cu` keeping the same C ABI symbols, OR had zero
runtime callers. The original 4-spec scope exposed a multiple-definition link
error on `silu_mul_triton_aot_cuda` (Round 3 verdict); silu was added to scope
to resolve. Live Triton AOT pipeline reduces to the 7-stage
`gated_delta_rule_chunkwise` Qwen3.5 hybrid kernels.

**Files actually changed (8):** `crates/cuda-kernels/build.rs` (-104 lines),
`crates/cuda-kernels/src/ffi/attention.rs` (-14), `csrc/attention/prefill_attention.cu`
(prose), `crates/cuda-kernels/AGENTS.md` (prose), and 4 `.py` deletes in
`tools/triton/`: `basic_kernels.py`, `silu_mul_kernel.py`,
`gen_silu_mul_aot.py`, `flash_attention_prefill_hd256_kernel.py`.

**Verification:** Round 1 self-review PASS, Round 2 `cargo check
--no-default-features --features no-cuda` PASS in 30s, Round 3 `codex review
--uncommitted` (with full release CUDA cargo build) initially CAUGHT the silu
collision; post-fix re-review PASS on no-cuda check; Round 4 codex@0 peer
review PASS on code structure (REQUEST_CHANGES on stale docs only — addressed
in follow-up commit). Bench: marked `pending-remote` regression-check at
`docs/experience/wins/2026-05-05-bench-tilelang-phase0-pending-remote.md`.

**Sequencing:** independent — runs in parallel with codex@2:0 W4 H5 work.

### Phase 1 — silu_mul Triton → TileLang

**Files (~6):** new `tools/tilelang/silu_mul.py`; `build.rs` swap
spec; `ffi/elementwise.rs` extern rename; delete
`tools/triton/silu_mul_kernel.py`; update 3 call sites in
`infer/src/ops/{elementwise,linear}.rs`.

**Decision D1 below** — extend `silu_mul_fused_cuda` (gate_up
combined) or write fresh matching `silu_mul_triton_aot_cuda` shape
(gate, up separate)?

**Gate:** `cargo test --features cuda` smoke + Qwen3-4B JSON substring
match; guidellm c=16 / Qwen3-4B / L4 — Δ ≤ 2% out tok/s.

### Phase 2 — gated_delta_rule_chunkwise Triton → TileLang

**Files (~14):** new `tools/tilelang/gated_delta_rule.py`; `build.rs`
swap 7 specs; `ffi/recurrent.rs` extern rename ×7; delete
`tools/triton/gated_delta_rule_chunkwise_kernels.py`; port 7 call
sites in `infer/src/ops/recurrent.rs:353-579`.

**Decision D2 below** — single fused kernel or literal 7-stage 1:1
port?

**Gate:** `cargo test --release --test e2e_qwen35` green;
Qwen3.5-4B JSON substring match; guidellm Qwen3.5-4B c=16 — Δ ≤ 5%
out tok/s.

### Phase 3 — FlashInfer prefill HD128 + HD256 → TileLang (HIGHEST RISK)

**Files (~12, each large):** `tools/tilelang/{batch_prefill_paged_hd128.py, batch_prefill_paged_hd256.py}` (review/tune); `build.rs` (drop flashinfer include + nvcc); delete 4 `csrc/attention/flashinfer_prefill*.cu`; surgery on `crates/cuda-kernels/src/flashinfer.rs` (drop `BatchPrefillPagedPlan`); `ffi/attention.rs` delete 8 extern decls; major edits in `infer/src/ops/attention.rs` (drop all prefill `cfg(not(feature = "tilelang-attn"))` arms, drop single-prefill HD128/HD256 helpers); `infer/src/model/{qwen3,qwen35}/prefill*.rs` drop `BatchPrefillPagedPlan` field; root + `infer/Cargo.toml` drop `tilelang-attn` feature.

**Gate (HARD):**
- `cargo test --release --test {e2e,e2e_qwen35}` green on L4 sm_89.
- guidellm full sweep on Qwen3-4B + Qwen3.5-4B at c=16/4096 with
  out tok/s ≥ 138.0 / 151.0 and TTFT p50 ≤ 13.3s / 12.4s
  (2026-04-29 baselines).
- Repeat at c=32 and c=64.
- Run on **at least two** SM tiers (L4 sm_89 + H100 sm_90 OR + A10 sm_86).

**Rollback:** if any tested SM/concurrency regresses beyond gate,
revert the entire phase commit. **Do NOT** keep FlashInfer as a
feature flag (no half-states).

**Sequencing:** must follow Phases 0-2; must precede Phases 4-5.

**Hard stop:** if after one tuning iteration TileLang HD128 prefill
cannot match FlashInfer at L4 c=16/4096-in within 2%, **Phase 3 does
not ship this round**. Phases 0-2 still ship; Phase 3 reopens after
prefill kernel work. The user direction "wherever TileLang can deliver
competitive perf" is the explicit license to back off.

### Phase 4 — FlashInfer batched decode HD128/HD256 → TileLang

**Files (~10):** new `tools/tilelang/{batch_decode_paged_hd128.py, single_prefill_hd128.py, single_prefill_hd256.py}`; fix `batch_decode_paged_hd256.py` BLOCK_M=1 codegen failure; `build.rs` delete `tilelang-decode-hd256` gate + add HD128 decode head configs; delete 4 `csrc/attention/flashinfer_{decode*,tc_decode,metadata}.cu`; **delete `crates/cuda-kernels/src/flashinfer.rs` entirely** (or rename per D3 below); `ffi/attention.rs` delete 6 extern decls; `crates/cuda-kernels/src/lib.rs` remove `pub mod flashinfer`; `prelude.rs` swap `FlashInferDecodeMetadata` for `TileLangDecodeMetadata`; `infer/src/ops/attention.rs` delete all decode `cfg(not(feature = "tilelang-attn"))` arms; `infer/src/model/qwen3/batch_decode.rs` drop `FlashInferWorkspace`.

**Decision D3 below** — rename + preserve metadata staging logic, or
fresh module?

**Gate (HARD):** same as Phase 3, plus decode tok/s ≥ 99% of post-Phase-3 baseline.

**Rollback:** revert; restores `flashinfer.rs` + 4 `.cu` + FFI surface.

**Sequencing:** strictly **after** Phase 3 (shared FlashInfer header dep).

### Phase 5 — Triton + FlashInfer dependency strip-out

**Files (~10):** `crates/cuda-kernels/build.rs` (drop all Triton/FlashInfer helpers); `crates/cuda-kernels/Cargo.toml` (drop `tilelang-attn` feature); `infer/Cargo.toml`; root `Cargo.toml`; `pyproject.toml` (drop `triton`, `flashinfer-python`); delete `tools/triton/` directory; rewrite CLAUDE.md lines 22 + 26 (Triton + FlashInfer references); `docs/{architecture.md, codebase-map.md, environment.md, support-matrix.md}`; `crates/cuda-kernels/AGENTS.md`.

**Gate:** clean-box build succeeds without `triton` or
`flashinfer-python` Python deps; `cargo check -p infer
--no-default-features --features cuda,no-cuda` green on Mac;
`cargo check --features metal` green; full e2e + guidellm regression-check.

**Sequencing:** must be **last**.

---

## Section 4 — Risk register (mitigations)

### R1 — TileLang AOT pipeline maturity

Past blockers: LayoutInferencer InternalError (fixed 4d9c65f0 by mirroring upstream canonical layout); `T.symbolic` arg auto-promotion (fixed via closure-int + scalar-runtime); HD256 decode `M=1` violation (`GemmWarpPolicy`) **unfixed on sm_89**; sm_89 build-broken (`docs/experience/errors/2026-04-29-tilelang-decode-hd256-sm89-build.md`).

**Mitigation:** pin TileLang version BEFORE Phase 1; Phase 1 (silu_mul, smallest kernel) is the pipeline-maturity smoke test; Phase 4 must include the HD256 decode codegen fix as scope; if upstream-first attempt fails, fall back to BLOCK_M ≥ 16 with replication padding per the existing error entry's plan.

### R2 — Numerical parity vs FlashInfer (Phases 3, 4)

FP8 KV path (`decode_attention_varlen_fp8_cuda` stays csrc). RoPE per-row offsets (must match `prefill_attention_paged_prep_cuda` + `decode_prep_paged_cuda`). Paged layouts (page_size=16, HND, BF16 — already TileLang-aligned).

**Mitigation:** add per-token logit delta gate ≤ 1e-3 in FP32 vs FlashInfer for the Phase 3 commit; new unit test in `infer/tests/` running 256 random prompts on both kernels and asserting max abs delta on logits ≤ 1e-3; re-run the existing FP8 KV regression entry as canary.

### R3 — Perf parity vs FlashInfer (Phase 3 GATE)

Per §1.3, the 2026-04-29 paired bench shows TileLang HD128 prefill 3.3-3.8% slower than FlashInfer at L4 c=16/4096-in/Qwen3-4B and Qwen3.5-4B. Regression source unresolved.

**Mitigation:**
- Pre-Phase-3 root-cause investigation. Reproduce 2026-04-28 bench shape exactly on latest `main`. If TileLang still slower, instrument with `ncu` per `docs/reviews/2026-04-14-cuda-kernel-six-principles-review.md`.
- Multi-SM gate: L4 sm_89 + A10 sm_86 + H100 sm_90 (or whichever are reachable). All three ≤ 0% out tok/s regression to ship.
- Workload-shape gate: c=16/4096-in, c=32/2048-in, c=8/8192-in, c=64/1024-in. Any shape >5% regression → hold.
- Hard stop: if TileLang can't match FlashInfer at L4 c=16/4096-in within 2% after one tuning iteration, Phase 3 does not ship this round.

### R4 — Backend isolation discipline

**Mitigation:** pre-merge type-check guards `cargo check -p infer --no-default-features --features cuda,no-cuda` and `cargo check --features metal` for every Phase 3 / Phase 4 commit. `infer/src/backend.rs` must NOT change in any phase.

### R5 — Pending-work conflicts

codex@2:0 (W4 H5 scheduler fix) touches `infer/src/scheduler/cuda/*` — Phase 0-2 don't touch these. Phase 3 changes `model/{qwen3,qwen35}/prefill.rs` plumbing (model-layer not scheduler-layer). Recommend: hold Phase 3 commit until codex@2:0 lands, OR rebase Phase 3 onto codex@2:0 head.

codex@0:0 (W3 vLLM-target research) is research-only; should not interfere.

---

## Section 5 — What is NOT in this refactor

- Metal backend (MLX bridge, `infer/src/backend/metal/*`) — zero edits this round per "Metal 阶段性独立支持".
- Model-side code (`infer/src/model/*`, qwen-spec / qwen35-spec / deepseek-spec crates) — only call-site signature changes from FlashInfer/Triton symbol deletions; model arch unchanged.
- Scheduler, KV-tier, HTTP server, agent loop, distributed/NCCL, train, autograd — unchanged.
- MLA / DeepSeek V4 path stays in design (`csrc/attention/mla_decode.cu`, `crates/deepseek-spec/`).
- CUDA C kernels outside attention (gemm/marlin, kv/scatter, quant, misc/elementwise_basic) — kept; not part of this refactor.
- `add_cuda` / `embedding_*_cuda` already live via csrc — only the dead Triton scaffolding goes in Phase 0.

---

## Decisions awaiting user input

### D1 — silu_mul TileLang signature shape

Today there are two C ABIs:
- `silu_mul_triton_aot_cuda(gate, up, out, n, stream)` — used by 3 sites in `ops/{elementwise,linear}.rs` (Triton AOT — to-be-deleted).
- `silu_mul_fused_cuda(gate_up, out, batch_size, inter_dim, stream)` — defined at `csrc/misc/`, declared at `ffi/elementwise.rs:39`, **zero callers** in `infer/src/`.

**Option A (preserve existing call shape):** TileLang kernel mirrors `silu_mul_triton_aot_cuda` (gate, up separate). 3 caller sites change one symbol name; no model code touches. **Lowest blast radius.**

**Option B (consolidate to fused):** TileLang kernel mirrors `silu_mul_fused_cuda` (gate_up combined). 3 caller sites change BOTH the symbol AND the upstream tensor packing (gate/up split happens earlier in MLP). Wins one extra kernel-launch reduction, but is a model-code change as well — drifts outside kernel-substrate scope.

**Recommendation:** A. Kernel-substrate swap, not model rewrite. If fused MLP is the long-term direction, that's a separate optimization commit AFTER Phase 5 closes.

### D2 — gated_delta_rule TileLang shape

**Option A (literal 7-kernel port):** safe, mirrors current contract bit-for-bit; fits Phase 2's 5% gate; zero TileLang authoring risk.

**Option B (fused single kernel):** higher reward (could be 10-20% Qwen3.5 hybrid TTFT/decode improvement — this pipeline IS the hybrid hot path), but TileLang must express the cumulative-sum + triangular-solve + matmul chain. The triangular-solve in stage 4 is the hard part.

**Recommendation:** A as Phase 2; B as a follow-on Phase 2.5 if Phase 2 lands clean and the delta justifies it. Don't conflate.

### D3 — `cuda_kernels::flashinfer.rs` rename + preserve, or fresh delete?

`flashinfer.rs` is 1311 lines. ~150 lines are FlashInfer-specific (FlashInferWorkspace, plan/run wrappers). The remaining ~1100 lines are paged-KV decode metadata staging logic with unit tests at lines 1265-1311 — semantically reusable for any TC-style attention kernel.

**Option A (rename + delete plan/run):** rename to `tilelang_decode.rs`, rename `FlashInferDecodeMetadata` → `TileLangDecodeMetadata`, delete `BatchPrefillPagedPlan`, `FlashInferWorkspace`, `flashinfer_plan*` (~400 LoC delete). Keep staging logic + tests. Phase 4 becomes a rename-and-delete-section commit.

**Option B (fresh module, full delete):** delete `flashinfer.rs` outright; write a fresh `tilelang_decode.rs`. Risk: regression in staging logic that has been tuned over many iterations.

**Recommendation:** A. Preserve battle-tested staging logic and its tests; module name lies but logic is sound; rename is mechanical and the diff stays focused.

---

## Composition with the multi-backend plan

This plan's Phases 0-4 must complete **before** the multi-backend plan
(`docs/plans/2026-05-05-multi-backend-tilelang-rocm-vulkan.md`) §10.1
small-kernel inventory becomes relevant (that plan presupposes Triton is
gone). Phases A0/A1 (Vulkan smoke / minimal Vulkan backend) of the
multi-backend plan are independent and can run in parallel with these
phases. The `tools/tilelang/` directory remains the single source of
truth — kernels added in Phases 1, 2, 4 will be symlinked into
`crates/{rocm,vulkan}-kernels/tools/tilelang/` per multi-backend §7.6.

---

## Pointers

- Composition: `docs/plans/2026-05-05-multi-backend-tilelang-rocm-vulkan.md`
- TileLang AOT recipe history: `memory/project_tilelang_0p1p9_aot_blocker.md`
- 2026-04-29 prefill paired bench (the gating evidence): `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-tilelang-on-vs-off.md`
- 2026-04-28 patches A+C entry: `docs/experience/wins/2026-04-28-bench-guidellm-cuda-l4-tilelang-prefill-causal-bound.md`
- HD256 decode codegen blocker: `docs/experience/errors/2026-04-28-tilelang-hd256-decode-m1-codegen-failure.md`
- Build.rs Triton + FlashInfer + TileLang regions: `crates/cuda-kernels/build.rs:225-1271`
