# M_pf-fuse — Fuse prefill QKV + gate-up GEMMs (industry-standard)

> Created 2026-05-07 EOD+3 from FFN survey. ARLE prefill currently
> runs Q + K + V as 3 separate GEMMs and gate + up as 2 separate
> GEMMs per layer. vLLM, SGLang, TRT-LLM all fuse these into 1
> GEMM each. ARLE's `fused_mlp_into()` kernel exists but is used
> only in the decode path. M_pf-fuse extends fusion to prefill.

## Priority & ROI

**Priority**: **P1**. After M_pf-gemm Phase 0 KILLED
([`267fcfa`](../experience/wins/2026-05-07-m_pf-gemm-phase0-killed-cublas-heuristic-already-optimal.md)),
the long-ctx 4k/c=4 prefill TTFT 1.65× gap to vLLM cannot be
closed via cuBLAS algo selection. The next-cheapest fix is
operator fusion — industry-standard, well-understood, no kernel
authoring required (cuBLAS already supports the larger-N GEMM
shape).

**ROI basis**:

| Aspect | Current (unfused) | After M_pf-fuse | Δ |
|---|---:|---:|---:|
| GEMMs per layer | 7 (Q, K, V, gate, up, down, o) | 4 (QKV, gate_up, down, o) | **-43%** |
| Prefill GEMM calls per cycle (36 layers × 4 reqs × 2 chunks) | 2016 | 1152 | **-43%** |
| Attention QKV M-dim (Qwen3-4B) | 4096 + 1024 + 1024 = 6144 separate launches | 6144 single launch | better tensor-core util |
| FFN gate+up output dim | 11008 + 11008 separate | 22016 single | better tensor-core util |
| Per-call launch overhead at ~5–10 µs/call | 2016 × 7.5 µs = 15 ms | 1152 × 7.5 µs = 8.6 ms | -6.4 ms |
| Expected GEMM throughput gain on large M (industry data) | baseline | +15% (fewer dispatches, better SM occupancy) | -56.7% × 15% ≈ **-8.5% TTFT** |
| **Combined estimate** (call-count + throughput) | 1976 ms TTFT | ~1700 ms TTFT | **-14%** |

ARLE TTFT 1976 → ~1700 ms = closing 35% of the 800 ms gap to
vLLM 1177 ms. Doesn't reach parity alone; combine with later
TileLang prefill GEMM port (Phase 2 of M_pf-gemm) for the rest.

**Negative case**:
- Weight layout change (`gate_proj.weight` + `up_proj.weight` →
  concat at load time). Touches every Qwen3 model checkpoint
  load path — risk of breaking other tests / models.
- The output of gate_up GEMM is `[seq, 2 × intermediate]`; silu_mul
  kernel needs to read `silu(out[:, :intermediate]) * out[:, intermediate:]`
  — this is the standard "split-then-multiply" pattern but ARLE's
  current `silu_mul_native_kernel` may not handle the strided
  layout. New kernel or kernel-arg variant required.
- QKV fusion: output is `[seq, num_q_heads*head_dim + 2*num_kv_heads*head_dim]`.
  Splitting into Q, K, V views needs care — Q is RoPE-rotated, K
  is RoPE-rotated AND written to KV cache, V is written to KV cache.
  ARLE has these as separate buffers today; fusion needs to add
  splitter logic in the qk_norm_rope kernel call.

**Kill criteria**:
- After implementation, bench at long-ctx 4k/c=4 shows < 8% TTFT
  improvement → ROLLBACK or keep behind a feature flag for
  further investigation.
- e2e or greedy_consistency tests fail → ROLLBACK; the fusion
  introduced numerical divergence.
- Memory regression > 5% (unlikely; fusion typically saves
  intermediate buffers).

## Phase 0 — License-or-kill experiment (~1-2 days)

**Single shape proof of concept**: Implement gate-up fusion ONLY
for FFN (skip QKV fusion in Phase 0). Smaller scope, validates
the approach.

Tasks:
1. **Weight load path change**: At load time, allocate one
   `gate_up_proj.weight` tensor of shape `[hidden, 2 × intermediate]`
   and copy `gate_proj.weight` into the first half, `up_proj.weight`
   into the second half. Keep `gate_proj` / `up_proj` removed from
   the module.
2. **Prefill forward pass change**: Single `linear_batch_into` call
   producing fused output. Then call a new
   `silu_mul_split_native_kernel` that reads
   `silu(fused[:, :inter]) * fused[:, inter:]`.
3. **Decode forward pass**: Already uses `fused_mlp_into()` —
   verify the existing fused path also benefits from the new
   weight layout (it should — the fused decode kernel likely
   already treats gate+up as concatenated).
4. **Verify e2e + greedy_consistency**: numerical equivalence
   to current state.
5. **Bench at long-ctx 4k/c=4 + high-conc 1k/256/c=64** to
   measure TTFT and per-row decode impact.

LOC est: ~150-200 (weight loader + forward pass + new silu_mul
kernel variant).

**License decision**:
- ≥ 8% TTFT reduction at longctx 4k → PROCEED to Phase 1 (add
  QKV fusion, ~150 more LOC)
- < 5% improvement → ABANDON; maintain unfused state
- 5-8%: borderline, ship as opt-in flag, defer Phase 1 decision

## Phase 1 — QKV fusion (~1-2 days, after Phase 0 license)

Same pattern for attention:
1. Load `qkv_proj.weight` as one concatenated tensor
2. Single `linear_batch_into` for QKV projection
3. Existing `prefill_attention_paged_qk_norm_rope_hd128_kernel`
   needs to be split into "fused QKV split" (reading from one
   contiguous buffer) + "qk_norm_rope" (existing logic). Or a
   new kernel that takes the fused buffer and produces normed
   Q + normed K + V cache fill.
4. Bench + verify.

LOC est: ~150-250.

## Phase 2 — Down-proj + o-proj? (P3, conditional)

These are typically NOT fused because their inputs are different
(down's input is silu_mul output, o's input is attention output).
Skip unless Phase 1 plateaus and we need every last ms.

## Acceptance

- Long-ctx 4k/c=4 TTFT ≥ 8% lower than current 1976 ms (target ≤ 1818 ms)
- Long-ctx 8k/c=4 TTFT improvement ≥ 5%
- High-conc 1k/256/c=64 out tok/s no regression (decode is unaffected
  except via the existing `fused_mlp_into` path)
- All e2e + greedy_consistency tests pass with bit-exact equivalence
- Wins entry with per-shape data table cross-referenced to bench
  artifacts and sha256

## Tasks

| # | Task | File(s) | LOC est. | Owner | Trigger |
|---|---|---|---|---|---|
| Phase 0.1 | Weight load: concat gate+up | `infer/src/model/qwen3/weights.rs` | ~30 | Codex (after M_b.1) | M_b.1 commits |
| Phase 0.2 | Module struct: replace gate_proj+up_proj with gate_up_proj | `infer/src/model/common.rs`, `qwen3/{prefill,decode,forward}.rs` | ~50 | Codex | 0.1 done |
| Phase 0.3 | New silu_mul_split kernel | `crates/cuda-kernels/csrc/elementwise/silu_mul.cu` (or similar) + ffi binding | ~80 | Codex | 0.2 done |
| Phase 0.4 | Verify e2e + greedy + bench | scripts | 0 | Claude | 0.3 commits |
| Phase 0.5 | License decision + wins entry | `docs/experience/wins/...` | 0 | Claude | 0.4 done |
| Phase 1.1 | QKV weight concat | weights.rs | ~30 | Codex | License fires |
| Phase 1.2 | QKV forward + qk_norm_rope handoff | prefill.rs + kernel | ~100 | Codex | 1.1 done |
| Phase 1.3 | Bench validation + wins | bench scripts | 0 | Claude | 1.2 done |

## Cross-references

- M_pf-gemm Phase 0 KILLED: [`267fcfa`](../experience/wins/2026-05-07-m_pf-gemm-phase0-killed-cublas-heuristic-already-optimal.md)
- H_LP3 finding: [`cae08b7`](../experience/wins/2026-05-07-h_lp3-diagnosed-cutlass-small-tile-gemm-bottleneck.md)
- Existing decode-path fused MLP: `infer/src/ops/linear.rs:403-478` (`fused_mlp_into`)
- Existing weight loader: `infer/src/model/qwen3/weights.rs:268-270`
- Existing prefill FFN forward: `infer/src/model/qwen3/prefill.rs:697-714`
- Existing QKV proj (separate): `infer/src/model/qwen3/prefill.rs:596-598`
- Industry references: vLLM `merge_qkv` (default), SGLang `Linear.is_qkv`, TRT-LLM `unified_attention`

## Rule

- **Operator fusion is industry-standard and untapped in ARLE
  prefill.** Evidence: 3 separate GEMMs for QKV + 2 separate
  GEMMs for gate+up = 5 GEMMs that ALL competitors fuse to 2.
  Closing this gap is "do what everyone else does", not novel
  kernel work.
- **Decode-path fusion already exists**, validated by greedy_
  consistency tests, used in production. Prefill-path fusion
  is a parallel structural change with the same conceptual
  pattern.
- **Trace evidence + survey ranking**: H_LP3 found GEMM-axis
  bottleneck. M_pf-gemm Phase 0 ruled out cuBLAS algo selection.
  M_pf-fuse is the next layer down — kernel call structure
  itself.
