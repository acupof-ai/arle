# Metal World-First Recalibration vs Backend Unification — 2026-05-07

This entry recalibrates the morning's Tier-A/B/C gap analysis
([`2026-05-07-metal-world-first-gap-analysis.md`](2026-05-07-metal-world-first-gap-analysis.md))
against the master roadmap
([`docs/plans/backend-unification.md`](../plans/backend-unification.md))
after M1–M5 landed on origin/main earlier the same day.

The morning analysis was framed in isolation: "what to add to Metal to
match SOTA". The correct frame is the unification's: "Metal is missing
what CUDA already has; share the layer once instead of writing twice".

## State as of 2026-05-07 evening

Origin/main has shipped, in one day, the entire unification spine:

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
all land — it adds the spec-decode dimension and 5-baseline (vLLM / TGI / SGLang
/ TRT-LLM / mlx-lm) cross-vendor matrix.

## Cross-walk: morning Tier ranking → unification milestones

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

Net: every morning Tier item maps onto a unification milestone or
ripple. **Three items remain genuinely open and fit the post-M5 phase:**

1. **A2 — Q8 KV via the unified `Op` layer.** Belongs on
   `infer/src/ops/kv_ops.rs`, not `metal/ops.rs`. Implementing it
   Metal-only would re-create the M4 fork that just got closed.
2. **B1 hot-path wiring of `metal/kv_pool.rs`.** Substrate is in tree;
   the wiring crosses the M3 IR + M4 ops + M5 ModelForward triplet.
   Best landed as a Metal-side ripple of M5 once Qwen3.5 follows
   Qwen3 across the unified ModelForward path.
3. **C1 simdgroup-MMA M=16 quant matmul.** Strictly C++ kernel work
   in `mlx-sys`; orthogonal to unification but only worth landing
   after M4 makes its call site stable.

## Implications for the /loop ticks ahead

- **Stop thinking in Metal-only Tiers.** Frame work as "unification
  milestone ripple" or "post-M5 fold-in".
- **Q8 KV (task #4)** should pivot from "opt-in flag on Metal" to
  **"design `Op::quantized_kv_cache` on the shared ops layer"**. Land
  the trait + scaffolding first; backend implementations follow.
- **The next bench-driving commit is the c≥4 Metal entry in the
  W1–W8 gauntlet (M_e A4 row)**, gated by M5's ModelForward Metal
  ripple. The morning's roadmap referred to vllm-mlx's 1,150 tok/s
  c=32 DeepSeek-V3 number; under the unification frame, that target
  belongs in M_e's matrix, not in a Metal-isolated bench.
- **ELI integration ([`eli-integration.md`](../resources/eli-integration.md),
  [`2026-05-07-eli-arle-native-provider-design.md`](2026-05-07-eli-arle-native-provider-design.md))**
  is unaffected by unification — it sits above the engine boundary.
  Layer 1 + Layer 2 ship independently of M-series progress.

## What this entry deliberately does NOT do

- Rewrite the morning gap analysis. That doc is a snapshot of
  external SOTA against ARLE pre-rebase; this entry is the
  reconciliation, not a replacement.
- Pre-empt M6 acceptance numbers. Whatever numbers M6 produces over
  the next few days is authoritative — this entry just maps the
  scaffolding.
- Touch `M_a`/`M_b`/`M_c`/`M_d` — those sub-plans of
  [`longctx-spec-tilelang-combo.md`](../plans/longctx-spec-tilelang-combo.md)
  are owned by their own ledgers.
