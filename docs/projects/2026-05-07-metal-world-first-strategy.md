# Metal World-First Strategy (Consolidated, 2026-05-07)

> Consolidates three 2026-05-07 strategy docs into a single source of truth:
> the SOTA gap analysis, the backend-unification recalibration, and the
> sequencing master analysis. The originals are removed in the same commit
> that introduces this file. Current state of execution lives in
> [`../../ROADMAP.md`](../../ROADMAP.md) (P3 Metal track) and
> [`mlx-backend-roadmap.md`](mlx-backend-roadmap.md); this entry is the
> dated synthesis, not a moving status board.

This entry is the synthesis the next several /loop ticks (or any fresh
session) should read FIRST before touching Metal code. It folds together
the morning's SOTA gap analysis, the unification recalibration, the M_d.1
namespace closure, the M_e.1 paged-KV plan (with errata), the
apples-to-apples c-sweep, and the c=1 isolation decomposition into one
coherent picture.

Every numerical claim below is cited to a wins entry committed earlier on
2026-05-07.

## 1. Where ARLE-Metal stands today (empirical)

All measurements: M4 Pro (20-core), `models/Qwen3.5-0.8B-MLX-4bit`,
guidellm 0.6.0, 30 s `--fast`-style cells, sequential server runs.

### 1.1 Apples-to-apples by workload

| Cell | ARLE ITL p50 | mlx-lm ITL p50 | ARLE TTFT p50 | mlx-lm TTFT p50 | ARLE out tok/s | mlx-lm out tok/s |
|---|---:|---:|---:|---:|---:|---:|
| c=1 short (128/2048) | 3.95 ms | 3.18 ms | **37.4 ms** | 165.1 ms | 245.8 | 308.2 |
| c=1 long (4096/256) | 4.37 ms | 3.38 ms | **920 ms** | 1048 ms | 129.5 | 136.9 |
| c=4 long (4096/256) | 19.34 ms | 7.17 ms | **1.20 s** | 4.51 s | 147.5 | 196.1 |
| c=8 long | 39.77 ms | — | 7.61 s | — | 144.1 | — |
| c=16 long | 82.49 ms | 18.97 ms | 5.18 s | 12.69 s | 78.2 | **467.9** |

Sources: `2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md`,
`2026-05-07-bench-guidellm-metal-c4-apples-vs-mlxlm.md`,
`2026-05-07-bench-guidellm-metal-c1-isolation-decomposition.md`.

### 1.2 Algebraic decomposition

The c=4 long-context ITL gap of 2.70× decomposes structurally:

```
ARLE     c=4 / c=1 = 19.34 / 4.37 = 4.43×
mlx-lm   c=4 / c=1 =  7.17 / 3.38 = 2.12×

Per-token kernel gap        = 4.37 / 3.38 = 1.29×
ARLE-specific batching gap  = 4.43 / 2.12 = 2.09×

2.70×  =  1.29×  ×  2.09×
```

The per-token kernel gap is small. The batching gap is the dominant
structural problem.

### 1.3 What ARLE wins on

- **Prefill / TTFT.** 1.14× (c=1 long) to 4.4× (c=1 short) faster than
  mlx-lm. The chunked prefill + decode-priority scheduler interleave
  (locked by `decode_priority_holds_under_c4_mixed_traffic` in
  `infer/src/backend/metal/scheduler.rs`) delivers regardless of concurrency.
- **ITL p95 stability at sweet spot.** ARLE c=4 ITL p95 = 19.74 ms;
  mlx-lm c=16 ITL p95 = 33.86 ms. ARLE has tighter tails when run inside
  its hot-path envelope.

## 2. Why the batching multiplier is 2.09× worse

`infer/src/backend/metal/request_state.rs::Qwen35PackedDecodeBatch`
(line 773-784) carries:

```rust
struct Qwen35PackedDecodeBatch<'a> {
    batch_cache_len: i32,             // shared column cursor
    left_padding: Vec<i32>,           // per-row pad
    packed_kv_flat: Vec<MlxArray>,    // ONE shared cache, per layer
    packed_gdr_flat: Vec<MlxArray>,
    ...
}
```

Every row's valid KV data sits in `[left_padding[i], batch_cache_len)`.
At c=N with prompt-length variance δ, every row pays for the LONGEST
in-flight prompt's left padding. mlx-lm's decode does not left-pad — each
request maintains its own KV slice and packing happens only at the SDPA
call with no pre-padding.

This kernel architecture choice is the root of the 2.09× ARLE-specific
batching multiplier. Plumbing knobs (e.g. `--max-running-requests` flag
shipped in commit `bbc484c`) cannot fix it; only a kernel rewrite can.

## 3. Substrate audit — what's already in tree

| Component | Path | State |
|---|---|---|
| Token-level KV pool | `infer/src/backend/metal/kv_pool.rs` | **Fully built**: alloc/share/free/write/gather APIs |
| Per-driver use under `--kv-pool` | `request_state.rs:3270-3371` | **Wired but per-driver**: each `Qwen3StepDriver` has its own pool, all writes use `METAL_REQUEST_STATE_ID` singleton key |
| Cross-request shared pool | — | **Not yet** — `MetalKVPool` API supports it (request_id parameter) but no caller threads real IDs |
| Qwen3.5 packed decode → pool | — | **Not yet** — `Qwen35PackedDecodeBatch` bypasses the pool entirely, uses left-pad concat |
| RadixCache namespace (M_d.1) | `prefix_cache.rs` + `tokenizer.rs` | **Closed**: tokenizer SHA-256 + 32-byte namespace + load_snapshot bypass guard + isolation test (commits `fc68450`–`0e1bc3d`) |

Audit-error history: the original M_e.1 plan claimed "MetalKVPool is fully
built but unused on the hot path"; that grep missed `request_state.rs`.
Errata in §7 of the plan + a feedback memory
(`feedback_substrate_audit_grep_full_tree.md`) capture the lesson.

## 4. Gap → fix mapping

The three structural axes of the gap:

| Axis | Magnitude | Owner track | Effort | Status |
|---|---|---|---|---|
| **A. Batching/padding** | 2.09× | M_e.1 paged-KV | L (multi-commit) | Plan ready, errata applied, acceptance numbers tightened (217f1f8) |
| **B. Per-token kernel** | 1.29× | M_e.0 profile pass | M | Demoted from blocker; scope bench-only first |
| **C. Long-context KV scaling** | 1.05× (4096 vs 128) | Composes with A | — | Will partially close as paged-KV lands; full attention-compute optimization is post-A |

## 5. Sequencing — by leverage, ordered

The atomic-commit sequence below is the actual work plan. Each step has a
well-defined acceptance number; landing them in order keeps the tree
coherent and lets each commit revert independently.

### Phase 1 — preparation (no behavior change)

**P1.1 — Promote MetalKVPool from per-driver to runtime-shared.**
Move ownership from `Qwen3StepDriver.kv_pool` to the scheduler runtime;
each driver borrows the shared pool. Every request gets a unique
`request_id` (replace `METAL_REQUEST_STATE_ID` singleton).
- Effort: M
- Files: `metal/runtime.rs`, `metal/request_state.rs`
- Acceptance: Qwen3 `--kv-pool` path under c=4 retains today's ITL ± 5%.
  Tests still pass.

**P1.2 — Slot lifecycle wired to scheduler admit/finish.**
On admit: `pool.alloc_tokens(req_id, prompt_len)`. On finish:
`pool.free_request(req_id)`. Reads still come from the legacy concat
cache; pool slots are tracked but their data is not yet consumed by
attention.
- Effort: S
- Files: `metal/runtime.rs`, `metal/request_state.rs`
- Acceptance: prefill alloc overhead < 1 ms per request.

### Phase 2 — Qwen3.5 dual-write under opt-in flag

**P2.1 — Qwen3.5 packed decode dual-write.** Behind a new
`--metal-qwen35-pool` flag (or reuse `--kv-pool` with model dispatch),
make `Qwen35PackedDecodeBatch` write per-step K/V to BOTH the legacy
left-pad concat cache AND the shared pool via `pool.write_kv_slots`.
Attention still reads from concat (legacy correctness). Add a property
test: `pool.gather_kv(layer, req_id)` matches the concat slice for the
same request, byte-for-byte.
- Effort: M
- Files: `metal/qwen35.rs`, `metal/request_state.rs`, plus a `#[cfg(test)]`
  parity test
- Acceptance: dual-write does NOT regress c=4 ITL beyond 1 ms.

### Phase 3 — the unlock

**P3.1 — Kernel cutover under flag.** SDPA input K/V comes from
`pool.gather_kv_rows(layer, request_ids)` instead of the concat cache.
Each row's gathered tensor has exactly `current_seq_len` keys — no
left-pad. Attention mask becomes per-request scalars again, RoPE offsets
simplify.
- Effort: L
- Files: `metal/qwen35.rs`, `metal/runtime.rs`, possibly
  `crates/mlx-sys/src/mlx_qwen35_model.cpp`
- **Acceptance** (tightened 217f1f8 per c=1 isolation):
  - c=4 ITL p50 ≤ 9.3 ms (currently 19.34 ms)
  - c=16 ITL p50 ≤ 12 ms (currently 82.49 ms)
  - c=16 ITL p95 ≤ 15 ms
  - c=16 output tok/s ≥ 350 (currently 78)
  - c=1 long ITL p50 ≤ 1.05× of pre-commit 4.37 ms

### Phase 4 — promotion

**P4.1 — Flip default `max_running_requests` from 4 to 16.** After P3.1
lands and a c-sweep confirms monotonic scaling.
- Effort: S
- Files: `metal/scheduler.rs`
- Acceptance: c-sweep at default config shows ITL p50 monotonically
  bounded as c grows; c=16 within ±25% of mlx-lm's 467 tok/s.

**P4.2 — Retire the concat path.** Once P4.1 has shipped one bench window
without rollback, drop the legacy concat cache code path entirely. Pure
deletion-style refactor per `feedback_no_half_states.md`.
- Effort: S
- Files: `metal/request_state.rs`, `metal/qwen35.rs`, `metal/runtime.rs`

### Phase 5 — per-token kernel polish (the residual 1.29×)

**P5.1 — c=1 profile pass.** Use `mlx instruments` / metal capture + the
existing `metric.set_memory_bytes` trace at `runtime.rs:2879` to identify
the dominant per-token cost on ARLE single-stream. Hypotheses to test:
- Extra `eval()`/`item()` boundaries on the Rust hot path
- Per-driver step path missing `mx.compile`-style fusion
- Fixed C++-bridge per-call cost
- Effort: M (profile + targeted fix)
- Acceptance: c=1 long ITL p50 ≤ 3.50 ms (was 4.37 ms; closes the 1.29×
  gap to ≤ 1.04× of mlx-lm).

## 6. Quantitative definition of "world #1 on Metal"

After Phase 4 lands:

- **c=4 long-context output tok/s** ≥ mlx-lm c=4 of 196 (≈ parity)
- **c=16 long-context output tok/s** ≥ 0.75× mlx-lm c=16 of 467 (= 350;
  ARLE acceptable up to 25% behind, given matched-c is rare in production
  agent traffic)
- **TTFT at every workload** ≤ mlx-lm TTFT (already true; preserve)
- **ITL p95 stability** ≤ mlx-lm ITL p95 at the same c (already true at
  c=4; should hold at c=16 post-paged-KV)
- **Long-context (W6, 32k prompt)** added to M6 Metal snapshot per
  `docs/plans/M6-metal-world-rank-snapshot.md`; output tok/s within ±15%
  of best Apple-Silicon baseline

After Phase 5 lands:

- **c=1 long-context ITL p50** ≤ 3.5 ms (parity with mlx-lm 3.38 ms ±5%)
  — closes the residual per-token gap

## 7. Risks + unknowns

1. **P3.1 is L effort and high-risk.** The kernel cutover touches the
   C++ bridge and the model step path. A single-tick attempt is not
   safe. Plan to land in 2–3 sessions with strong dual-write
   property-test backing.
2. **C++ bridge per-call cost is unverified.** The hypothesis is in §5
   P5.1; until measured, we cannot rule out that per-token cost includes
   >1 ms fixed overhead that no kernel rewrite reduces.
3. **MLX 0.31.1 fast::rope quirk** (`feedback_mlx_rope_axis.md`,
   `feedback_mlx_rope_layout.md`) requires `[B, H, S, D]` ordering AND
   array-form RoPE offsets; switching to per-request gathered tensors
   must carefully preserve this. Tripwire tests in `metal::mlx::tests`
   should be re-run after P3.1.
4. **CUDA path stays untouched.** All Metal commits in this plan keep
   `infer/src/scheduler/cuda/*` and `crates/cuda-kernels/*` bit-identical.
   Verified by `cargo check -p infer --no-default-features --features
   cuda,no-cuda` + clippy at every commit, per CLAUDE.md.
5. **bench host availability.** Each phase needs a real Metal bench; they
   happen on this M4 Pro host. If the host changes, re-snapshot c=1
   baseline first to recalibrate the 4.37 ms ITL anchor.

## 8. What "start optimizing" means now, concretely

Next bench-track action: implement Phase 1 (P1.1 then P1.2) as two atomic
commits + bench regression check. P1.1 is M effort and slightly above the
12-min cron tick safety margin; expect to land it across two ticks (first
tick: read substrate + draft change + cargo check; second tick: tests
pass + commit + bench regression + push).

Next CUDA-track action: nothing (out of scope per user directive).

---

## 9. SOTA Gap Audit (folded in from morning gap-analysis)

Two parallel research subagents surveyed Apple Silicon SOTA on 2026-05-07.
This section synthesizes their findings against the ARLE Metal backend
code state and ranks the gaps by leverage. It is the input that the
sequencing in §5 already consumed; preserved here for reference so future
readers can recover the upstream-landscape framing without chasing
a separate file.

### 9.1 Method

- Kernel-track subagent: mlx / mlx-lm / llama.cpp Metal / candle /
  mistral.rs Metal kernel advances (PagedAttention, simdgroup-MMA,
  TurboQuant 4-bit KV, MTPLX, tree spec-decode).
- Serving-track subagent: vllm-mlx, oMLX, SGLang RadixAttention, llama.cpp
  Metal slots, mistral.rs PagedAttention, chunked prefill, disaggregated
  prefill, multi-LoRA, structured outputs.
- Cross-checked against current code:
  - [`infer/src/backend/metal/AGENTS.md`](../../infer/src/backend/metal/AGENTS.md)
    invariants 1–8 and Active Priority section.
  - [`infer/src/scheduler/AGENTS.md`](../../infer/src/scheduler/AGENTS.md).
  - `metal/scheduler.rs`, `metal/runtime.rs::execute_prefill_chunk`,
    `metal/prefix_cache.rs`, `metal/kv_pool.rs`, `metal/dflash.rs`.

### 9.2 What is already in tree

- Decode-first continuous batching loop (`run_metal_scheduler_runtime`).
- **Chunked prefill is already wired** (`execute_prefill_chunk` +
  `prefill_chunk(budget)` in `runtime.rs`) — research-track item #2
  (chunked prefill / decode-priority interleave) is *partially*
  implemented, not absent.
- Variable-length decode via `Qwen35PackedDecodeBatch` (left-padding +
  additive mask + per-row RoPE offsets).
- DFlash speculative decode dispatched through the scheduler runtime.
- `metal/prefix_cache.rs` (always-on) and `metal/kv_pool.rs` (always-on,
  not yet on the hot path).

### 9.3 Ranked gap backlog

#### Tier A — biggest leverage per unit effort

1. **Token-level radix prefix cache.** `metal/prefix_cache.rs` today is
   accounting only; SGLang RadixAttention + oMLX block-CoW-radix gives
   2–5× TTFT on shared-prefix agentic workloads. (L-of-S #1)
   - Effort: M (CPU-side radix tree + page handle integration).
   - Prereq for paged KV (Tier B #1) — token-level identity is needed
     before block tables can attach pages cross-batch.

2. **Q8 / FP8 KV cache + Metal-aware wire-down cap** copying mistral.rs's
   `cap = max_seq × max_batch` policy. KV today is BF16. (L-of-S #3,
   L-of-K #4)
   - Effort: M (`metal/ops.rs::extend_kv_cache` + qwen35 cache path).
   - Win: 2× context @ iso-RAM, +10–25% decode @ long ctx.
   - Stacks linearly with Tier B #1.

3. **Decode-priority interleave proof.** Chunked prefill exists, but no
   test asserts that prefill yields to decode under c≥4 mixed traffic.
   Without that gate we cannot claim parity with SwiftLM's
   `--prefill-step-size 512` HOL-blocking story. (L-of-S #2)
   - Effort: S (regression test + scheduler invariant doc).
   - Win: surfaces TTFT p99 regressions that today are silent.

#### Tier B — large but high-cost

1. **Paged-attention block tables on Metal.** Replaces left-padding with
   block tables; EricLBuehler's reported numbers: +77% Qwen3-30B-A3B 4-bit
   and +131% Llama-3.2-3B-8bit decode tok/s vs llama.cpp continuous
   batching. (L-of-K #1, L-of-S #1 paged-KV half)
   - Effort: L (allocator, page-aware mask, per-page RoPE offsets).
   - Win: +30–80% decode @ varlen batches ≥ 4; 2–4× max ctx @ iso-VRAM.
   - **Wire `kv_pool.rs` onto the hot path as the substrate.**
   - Stacks with Tier A #2 (Q8 KV) and Tier A #1 (radix).

2. **MTP / EAGLE speculative decoding integrated as default for Qwen3.5.**
   DFlash is in tree but experimental; Qwen3.5 ships native MTP heads.
   MTPLX shows ~2.24× decode tok/s on M5 Max. (L-of-K #2, L-of-S #5)
   - Effort: M (Qwen3.5 MTP head wiring + scheduler verifier slot).
   - Win: 1.8–2.3× decode tok/s @ temp ≤ 0.7, lossless under residual
     correction.

#### Tier C — smaller but cheap follow-ups

1. **Custom simdgroup-MMA M=16 quantized matmul** (`mma2big`,
   `mma2big_pipe` patterns) for the verify/draft step and MoE token-level
   paths. MLX default `quantized_matmul` is M=1 tuned. (L-of-K #3)
   - Effort: M (in `crates/mlx-sys/src/mlx_bridge.cpp`).
   - Win: +10–40% decode under spec/MoE; stacks with B #2.

2. **Tree-attention spec-decode mask** (DDTree pattern) once B #2 lands.
   (L-of-K #5)
   - Effort: S.
   - Win: +10–15% on top of B #2 for code / structured outputs.

3. **Two-tier prefix cache (RAM + SSD persistent across restarts)** —
   oMLX-style. Critical for `arle` agent loops where TTFT 22 s → 0.2 s
   warm-start matters. (L-of-S #4)
   - Effort: M (file-backed page tier behind A #1 radix).
   - Best deferred until A #1 lands.

#### Tier D — known frontier, not catch-up

- **Multi-LoRA Punica/S-LoRA on Metal.** No one ships it. Not a gap; if
  we build it, we lead.
- **Disaggregated ANE-prefill / GPU-decode** (Squeezebits Yetter). Not
  production-ready upstream; do not adopt.
- **M5 TensorOps via Metal 4.** Automatic via `mlx ≥ 0.31` dep bump;
  reported 3.3–4× TTFT on M5. Not a code change for us, but a release
  gate (verify the `mlx-sys` pin picks it up when M5 hardware lands).

### 9.4 Recommended sequencing (Tier ranking → §5 phases)

The cheapest world-first arc is **A1 → A2 → A3 → B1 → B2 → C1**.

- A1 (radix) is the keystone — both B1 (paged KV) and C3 (two-tier)
  depend on it.
- A2 (Q8 KV) is independently shippable and unblocks long-context
  benchmarks the scheduler-track report calls out as our weakest axis
  (BF16 KV is no longer competitive vs Q8 default in mlx-lm / llama.cpp /
  mistral.rs).
- A3 (decode-priority test) is the smallest commit and locks in a
  regression we will otherwise hit during A1+B1.

### 9.5 Reference benchmark targets

From the serving-track survey — to claim "world #1" on Metal we need to
beat or match these on M-series:

| Metric | Reference | Source |
|---|---|---|
| Qwen3-0.6B-8bit single-stream | **417.9 tok/s** M4 Max | `vllm-mlx` |
| Llama-3.2-3B-4bit single-stream | **205.6 tok/s** M4 Max | `vllm-mlx` |
| Qwen3-30B-A3B-4bit single-stream | **127.7 tok/s** M4 Max | `vllm-mlx` |
| DeepSeek-V3 Q4 c=32 aggregate | **1,150 tok/s** M4 Pro | `vllm-mlx` |
| Cached-prompt TTFT | **0.08 s** Gemma-4-e2b | `SwiftLM` |
| Qwen-32B Q4 ctx=32K | **19.0 tok/s** M3 Ultra | mlx-lm reference |

Our current single-request Qwen3.5-0.8B MLX 4bit on M4 Pro 20c is
**305.5 tok/s** at `1024/256` step-driver
([`mlx-backend-roadmap.md`](mlx-backend-roadmap.md)).

### 9.6 Delta footnote — 2026-05-07 (later same day)

A second research pass narrowed to the past 7 days surfaced one release
that changes flavor (not order) of the Tier B priorities:

- **oMLX v0.3.9.dev1** (2026-05-06) shipped DeepSeek V4 plumbing, **native
  MTP for Qwen 3.5 / 3.6 on Apple Silicon**, and an SSD prefix-cache tier.
  Source: <https://github.com/jundot/omlx/releases>.
- Implication: Tier B #2 (MTP / EAGLE spec-decode default for Qwen3.5)
  now has a public Apple-Silicon reference implementation with native MTP,
  not just MTPLX's prototype. The bar for "world #1 on Metal MTP" rose;
  the priority order does not change but the reference target for our
  verifier integration is now oMLX, not the older MTPLX repo.

mlx, mlx-lm, mistral.rs, llama.cpp Metal, vllm-mlx kernel surfaces had no
new releases in the 7-day window. Tier-A ranking stands.

### 9.7 Sources

Kernel track: `ml-explore/mlx#2228`, `MTPLX`, `ddtree-mlx`, `dflash-mlx`,
`vllm-mlx`, mlx-lm releases, mlx releases, Apple ML M5 post, mlx-mfa,
TurboQuant on MLX (Antonrozanov 2026-03), llama.cpp flash-attention
DeepWiki.

Serving track: `waybarrios/vllm-mlx`, `jundot/omlx`,
`macgpu.com/2026-mac-inference-framework-benchmark`,
`ggml-org/llama.cpp#20574`, `EricLBuehler/mistral.rs/PAGED_ATTENTION.md`,
`lmsys.org/2024-01-17-sglang`, `ml-explore/mlx-lm#630`, `SharpAI/SwiftLM`,
`raullenchai/Rapid-MLX`, `ml-explore/mlx#3209`,
`roborhythms.com/reduce-local-llm-ttft-mac-studio`, `Aryagm/dflash-mlx`,
`youssofal/MTPLX`, Punica `arxiv:2310.18547`, S-LoRA `arxiv:2311.03285`,
`AmesianX/TurboQuant`, `ggml-org/llama.cpp#20969`, vLLM chunked-prefill
docs.

---

## 10. Backend-Unification Recalibration (folded in from recalibration doc)

> Note: recalibrated vs the backend-unification M1–M5 milestones that
> landed on 2026-05-07. Current execution state lives in
> [`../../ROADMAP.md`](../../ROADMAP.md); this section is the dated
> mapping between the morning Tier ranking and the unification spine.

The morning analysis (§9) was framed in isolation: "what to add to Metal
to match SOTA". The correct frame is the unification's: "Metal is missing
what CUDA already has; share the layer once instead of writing twice".

### 10.1 State as of 2026-05-07 evening

Origin/main shipped, in one day, the entire unification spine:

| Milestone | Win entry | What it landed |
|---|---|---|
| **M1** Unified Backend Telemetry + Engine Lifecycle | [`2026-05-07-m1-unified-backend-telemetry.md`](../experience/wins/2026-05-07-m1-unified-backend-telemetry.md) | Backend-agnostic engine trait; both CUDA + Metal report through one telemetry path |
| **M2** KV-tier Policy Adapter for Metal | [`2026-05-07-m2-metal-kv-tier-adapter.md`](../experience/wins/2026-05-07-m2-metal-kv-tier-adapter.md) | `KvTierAdapter` + `MetalTierAdapter`; Qwen3.5 SSD prefix snapshot is the first T2 disk persistence path |
| **M3** Unified Scheduler Decision Layer (Logical IR) | [`2026-05-07-m3-unified-scheduler-ir.md`](../experience/wins/2026-05-07-m3-unified-scheduler-ir.md) | Cross-backend logical schedule IR; scheduler decision and execution split; slot-recovery path unified |
| **M4** Unified Op Trait + Metal `crate::ops::*` | [`2026-05-07-m4-unified-ops-backend.md`](../experience/wins/2026-05-07-m4-unified-ops-backend.md) | Metal taps into the shared `infer/src/ops/*` layer instead of carrying a parallel surface in `metal/ops.rs` |
| **M5** Unified `ModelForward` + Qwen3 Cross-Backend | [`2026-05-07-m5-modelarch-trait.md`](../experience/wins/2026-05-07-m5-modelarch-trait.md) | Qwen3 forward path is one trait implementation, dispatched per backend |
| **M6** Cross-Backend Bench Matrix + World-#1 Gauntlet | [`m6-cuda-vllm-gap-followups.md`](../plans/m6-cuda-vllm-gap-followups.md), [`2026-05-07-m6-world-rank-snapshot-cuda.md`](../experience/wins/2026-05-07-m6-world-rank-snapshot-cuda.md) | **In flight.** CUDA M6 snapshot taken; Metal A4 entry gated on M5 ripple effects |

M_e ([`M_e-world-first-bench-gauntlet.md`](../plans/M_e-world-first-bench-gauntlet.md))
is the cross-vendor bench plan that sits on top of M6 once M_a/M_b.2/M_c/M_d
all land — it adds the spec-decode dimension and 5-baseline (vLLM / TGI /
SGLang / TRT-LLM / mlx-lm) cross-vendor matrix.

### 10.2 Cross-walk: morning Tier ranking → unification milestones

| Morning Tier | Item | Status now | Where it lands in the unification frame |
|---|---|---|---|
| A1 | Token-level radix prefix cache for Metal | **Partial.** `RadixCache` exists at `infer/src/prefix_cache.rs` (CUDA); Metal has `backend/metal/prefix_cache.rs:38` instantiation per [`M_d.1`](../plans/M_d.1-tokenizer-fingerprint-radix-namespace.md). Tokenizer-fingerprint namespace fix is the next required step. | **Not a Metal-only gap.** RadixCache is shared; the gap is wiring + the namespace hole M_d.1 is closing. |
| A2 | Q8 / FP8 KV cache + wire-down cap | **Open.** `metal/ops.rs::extend_kv_cache` is unquantized BF16; `metal/kv_pool.rs` is a real token-level KV substrate now (post-M2) but no quant path. | **M4-shaped.** Lives on the unified `Op` trait — the quantized KV cache extension belongs in `infer/src/ops/kv_ops.rs` so both backends pick it up. |
| A3 | Decode-priority interleave regression test | **Done.** Landed in tick 2 commit `199a0a8` — three c=4 invariants in `MetalScheduler` test mod. | Locked. Same invariant should be re-asserted on the M3 logical IR side once Metal traffic flows through it. |
| B1 | Paged-attention block tables on Metal | **Substrate landed, hot path TBD.** `metal/kv_pool.rs` already mirrors CUDA `TokenKVPool` with MLX `Array` tensors; gather/scatter API is in tree. | Direct M3/M4 ripple — the moment Metal ops adopt the unified attention interface, paged-KV becomes a configuration toggle, not a port. |
| B2 | MTP / EAGLE spec-decode default for Qwen3.5 | **Hooks present, not default.** DFlash is wired through scheduler runtime but experimental. M_b/M_c plans (`M_b-tilelang-fused-draft-verify-kernel.md`, `M_c-hybrid-spec-rollback.md`) drive the next steps. | Owned by M_a/M_b/M_c sub-plans in [`longctx-spec-tilelang-combo.md`](../plans/longctx-spec-tilelang-combo.md), with the world-first claim sealed by M_e. |
| C1 | Custom simdgroup-MMA M=16 quantized matmul | **Open.** Outside unification — pure Metal kernel work in `crates/mlx-sys/src/mlx_bridge.cpp`. | Standalone after M4 lands (call site is the unified `Op::quantized_matmul`). |
| C2 | Tree-attention spec-decode mask | Gated on B2 (DFlash default). | Same plan as B2. |
| C3 | Two-tier prefix cache (RAM + SSD) | **Mostly landed via M2.** `MetalTierAdapter` already routes Qwen3.5 SSD prefix snapshot through the disk store. | M2 closed most of this. RAM-tier hot policy and persistent-restart cross-run hit-rate ≥ 50% are the M2 acceptance gates still to verify. |

Net: every morning Tier item maps onto a unification milestone or ripple.
**Three items remain genuinely open and fit the post-M5 phase:**

1. **A2 — Q8 KV via the unified `Op` layer.** Belongs on
   `infer/src/ops/kv_ops.rs`, not `metal/ops.rs`. Implementing it
   Metal-only would re-create the M4 fork that just got closed.
2. **B1 hot-path wiring of `metal/kv_pool.rs`.** Substrate is in tree;
   the wiring crosses the M3 IR + M4 ops + M5 ModelForward triplet. Best
   landed as a Metal-side ripple of M5 once Qwen3.5 follows Qwen3 across
   the unified ModelForward path.
3. **C1 simdgroup-MMA M=16 quant matmul.** Strictly C++ kernel work in
   `mlx-sys`; orthogonal to unification but only worth landing after M4
   makes its call site stable.

### 10.3 Implications for the /loop ticks ahead

- **Stop thinking in Metal-only Tiers.** Frame work as "unification
  milestone ripple" or "post-M5 fold-in".
- **Q8 KV (task #4)** should pivot from "opt-in flag on Metal" to
  **"design `Op::quantized_kv_cache` on the shared ops layer"**. Land the
  trait + scaffolding first; backend implementations follow.
- **The next bench-driving commit is the c≥4 Metal entry in the W1–W8
  gauntlet (M_e A4 row)**, gated by M5's ModelForward Metal ripple. The
  morning's roadmap referred to vllm-mlx's 1,150 tok/s c=32 DeepSeek-V3
  number; under the unification frame, that target belongs in M_e's
  matrix, not in a Metal-isolated bench.

### 10.4 What the recalibration deliberately does NOT do

- Rewrite the morning gap analysis. §9 above is a snapshot of external
  SOTA against ARLE pre-rebase; this section is the reconciliation, not a
  replacement.
- Pre-empt M6 acceptance numbers. Whatever numbers M6 produces over the
  next few days is authoritative — this entry just maps the scaffolding.
- Touch `M_a`/`M_b`/`M_c`/`M_d` — those sub-plans of
  [`longctx-spec-tilelang-combo.md`](../plans/longctx-spec-tilelang-combo.md)
  are owned by their own ledgers.

---

## 11. References

Bench evidence (chronological, all 2026-05-07):
- `2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md` (c-sweep, surfaces
  the c=16 collapse)
- `2026-05-07-bench-guidellm-metal-c4-apples-vs-mlxlm.md` (matched c=4,
  surfaces "two distinct gaps")
- `2026-05-07-bench-guidellm-metal-c1-isolation-decomposition.md` (c=1
  isolation, surfaces 1.29× × 2.09× decomposition)

Plans:
- `docs/plans/M_e1-metal-paged-kv-hot-path.md` (the load-bearing
  optimization plan)
- `docs/plans/M6-metal-world-rank-snapshot.md` (canonical bench protocol)
- `docs/plans/backend-unification.md` §M-series (master roadmap; M1–M5
  landed 2026-05-07; M6 in flight)

Project context:
- `docs/projects/mlx-backend-roadmap.md` (current Metal master roadmap)

Memory:
- `feedback_metal_unification_frame.md`
- `feedback_substrate_audit_grep_full_tree.md`
- `feedback_no_speculative_interface_shaping.md`
- `feedback_no_half_states.md`
