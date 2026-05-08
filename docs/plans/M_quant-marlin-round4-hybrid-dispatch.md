# M_quant Marlin Round 4 — hybrid W4A16 dispatch override (decode-only)

> Execution-ready plan from Round 4 prep
> ([`b3f22ea`](../experience/errors/2026-05-08-marlin-w4a16-bench-implementation-gap.md)).
> Trigger: codex's W4A8 commit lands (10 WIP files include `infer/src/ops/linear.rs`,
> conflict gate); GPU free; Claude executes single-file iteration ≤ 30 LOC.
> Skill applied: `kernel-optimization` Phase 1-8 explicit per section.
> Cron protocol §7 work pool item #1.

## Phase 1 — Target (skill mandatory)

| Field | Value |
|---|---|
| Metric | decode ITL p50 (Qwen3-4B-GPTQ-Int4-marlin, 4k longctx, c=4, BF16 KV) |
| Baseline A | ARLE BF16 baseline 19.27 ms (`786a20a`) |
| Baseline B | ARLE Marlin (current dispatch) 18.13 ms (Round 1) |
| **License** | **≥ 1.5× vs BF16-A** (≤ 12.85 ms) per `M_quant` §9.2 |
| Soft win | ≥ 1.4× (≤ 13.76 ms) — minor improvement worth landing if no regression |
| Kill | < 1.06× (no Δ vs Marlin baseline B) — confirms hybrid hypothesis wrong |
| Wall-clock budget | 5 min code edit + build + bench |

## Phase 2 — Hardware (skill mandatory)

sm_89 RTX 4070 Ti SUPER · 100 KB smem/SM · 88.5 BF16 / 706 FP8 TFLOPS · HBM 672 GB/s. (Same sheet as Round 1-3.)

## Phase 3 — Binding constraint (grounded)

From Round 4 prep (`b3f22ea` survey):

| Path | Launches/call | Source |
|---|---:|---|
| `MarlinW4Gemm` | 6 | `linear.rs:660-739` (alloc_zeros×3 + bf16_to_fp16 + marlin + fp16_to_bf16) |
| `W4A16BatchGemv` | **1** | `linear.rs:909-911` (BF16-native single FFI call) |

After Round 2 elimination of `alloc_zeros` overhead (cudarc pool elides cudaMemsetAsync), surviving Marlin surplus = 2 conversion launches per call.

For decode batch=4: 252 GEMMs × ~15us per surplus launch pair = ~3.8 ms saved per token if W4A16BatchGemv path used instead.

## Phase 4 — Formula prediction (skill mandatory)

```
predicted_decode_ITL = baseline_marlin - (252 × per_call_save_us / 1000)
where per_call_save_us = 2 × launch_overhead_us
launch_overhead empirical range: 5-10 us per elementwise on Ada
=> per_call_save_us = 10-20 us
=> total save per token = 2.5-5.0 ms
=> predicted ITL: 18.13 - 5.0 to 18.13 - 2.5 = 13.13 to 15.63 ms
=> predicted ratio vs BF16-A: 19.27 / [13.13, 15.63] = 1.23× to 1.47×
=> straddles 1.4× soft-win threshold; misses 1.5× hard license at the high end
```

Honest expectation: this is an **incremental** win, not a magnitude win. The bigger lever is M_quant Phase 1 (W4A8 combined = 1.86× decode + 7.9× prefill) which is codex's work. Round 4 #6 lands the surviving Marlin-implementation overhead at zero LOC budget so it doesn't shadow W4A8's signal.

## Phase 5 — Implementation (single-variable, matched controls)

### File: `infer/src/ops/linear.rs:65-93`

Current dispatch in `LinearKernelPlan::batched`:

```rust
fn batched(weight: &DeviceMatrix, batch: usize) -> Self {
    if batch > 1 && marlin_prefill_aligned(weight).is_ok() {
        return Self::MarlinW4Gemm;          // <-- engages for ALL batch>1
    }
    // ...
    match (batch, weight.weight_format()) {
        // ...
        (_, WeightFormat::W4A16) => Self::W4A16BatchGemv,    // fallback
        // ...
    }
}
```

**Single-variable change** (add batch threshold to Marlin gate):

```rust
const MARLIN_DECODE_BATCH_THRESHOLD: usize = 8;
// Round 4 #6: Marlin's BF16↔FP16 conversion overhead (2 elementwise
// launches/call) costs more than its tensor-core advantage for small
// batches. W4A16BatchGemv is BF16-native (1 launch/call). Use Marlin
// only when batch is large enough that tensor-core throughput dominates.
// Threshold matches existing batched-decode convention `(2..=8, GgufQ4K)
// => Q4KBatchGemv`.

fn batched(weight: &DeviceMatrix, batch: usize) -> Self {
    if batch > MARLIN_DECODE_BATCH_THRESHOLD
        && marlin_prefill_aligned(weight).is_ok()
    {
        return Self::MarlinW4Gemm;
    }
    if batch > 1
        && weight.has_marlin()
        && let Err(reason) = marlin_prefill_aligned(weight)
    {
        log::trace!("Marlin W4 fallback: {reason}");
    }
    match (batch, weight.weight_format()) { /* unchanged */ }
}
```

LOC delta: ~10 (constant + 1 line edit + comment block). Single file. Single variable: the threshold value.

### Matched controls (skill checklist v1.2.0 — post-`2853551` correction)

**3-arm A/B at production-default auto-FP8 KV** (no `--kv-cache-dtype` override).
Round 1 baseline correction at `2853551` proved BF16-forced KV vs FP8-default-baseline
was anti-pattern #8 forward-direction. R4 #6 must NOT repeat the trap.

- [ ] Same checkpoint across all 3 arms: `Qwen3-4B-W4A16-sym-g128-marlin`
- [ ] **NO `--kv-cache-dtype` flag** — let auto resolve to FP8 KV on this GPU
- [ ] Same `--num-slots 8 --max-seq-len 5120`
- [ ] Same data spec (4096 in / 256 out, c=4, max-seconds=120, warmup=10)
- [ ] No other GPU process during run (single-card serial)
- [ ] σ < 5% across n=3 — n=1 only sufficient if σ < 2% (Round 1-3 bench std was 0.02-7.7 ms = 0.05-0.4%)

The 3 arms:

| Arm | Marlin dispatch | Reference |
|---|---|---|
| A — BF16 baseline | n/a (full BF16 weight) | re-bench from `786a20a` (auto-FP8 KV) → expected ITL 19.27 |
| **B — Marlin all-batch (current head)** | `batch > 1` triggers Marlin | matches codex `f6f3af3` setup → expected ITL 11.76 (1.64× vs A) |
| **C — Marlin batch>8 hybrid (R4 #6 treatment)** | `batch > 8` triggers Marlin; small-batch → `W4A16BatchGemv` | predicted ITL ~8.26 (2.33× vs A) |

### Build path

```bash
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo build --release --features cuda 2>&1 | tail -3
```

### Bench command (production-default auto-FP8 KV)

```bash
# Server (NO --kv-cache-dtype override → auto-FP8 KV on sm_89)
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 --num-slots 8 --max-seq-len 5120 &

# Bench
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh marlin-w4a16-round4-hybrid-c4-4k \
  --model Qwen3-4B-W4A16-sym-g128-marlin \
  --processor /home/ckl/projects/arle/infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
```

For arm A (BF16 baseline) and arm B (Marlin all-batch) — re-bench can be
skipped if `f6f3af3` numbers are recent enough; otherwise n=1 quick reproduction
each at `marlin-w4a16-round4-baseline-bf16` and `marlin-w4a16-round4-baseline-marlin`.

## Phase 6 — Combinational A/B (optional, post-license)

If Phase 5 wins ≥ 1.4×, run threshold sweep:

| Threshold | Hypothesis |
|---|---|
| 4 | maybe-too-low — short prefill chunks revert to BF16 GEMV early |
| 8 | proposed default (current plan) |
| 16 | maybe-too-high — small prefill chunks miss tensor cores |
| 32 | upper bound — confirms tensor-core advantage at large M |

n=3 each, total ~12 bench × 2 min = 24 min. Optional — skipping is OK if 8 wins.

## Phase 7 — Tradeoffs (skill mandatory)

| Axis | Status | Note |
|---|---|---|
| LOC | ✅ ~10 | Single-file, single-variable |
| Hardware specificity | ✅ none added | Threshold benefits Ada specifically (Marlin conversion overhead worse on Ada vs A100); Hopper may need different value but no regression on Hopper |
| Compiler/runtime | ✅ none | No new FFI / kernel work |
| Maintainability | ⚠ minor | One new constant; tests must not pin to Marlin path for batch ≤ 8 (verify e2e/greedy still pass) |
| Numerical correctness | ⚠ unverified pre-bench | greedy_consistency required — Marlin and W4A16BatchGemv use different code paths, may produce different rounding |
| Generality | ⚠ multi-shape required | high-conc (1k/256/c=64, batch=64 → Marlin) must NOT regress (it should not since batch>8 still Marlin); multi-tenant must NOT regress (small chunks may now route W4A16BatchGemv — verify) |
| Memory budget | ✅ no change | |
| Scheduling impact | ✅ no change | dispatch decision only |
| **Tensor-core advantage at small batch** | ❌ sacrificed | Hypothesis: Marlin's tensor-core throughput at M=4-8 is dominated by launch overhead. If wrong, Round 4 #6 KILLs and hybrid hypothesis is refuted. |

**No-tradeoff axes**: LOC, HW, compiler, scheduling, memory. **Real tradeoffs**: maintainability (minor), correctness (verifiable), generality (multi-shape A/B mandatory), tensor-core trade.

The "tensor-core advantage at small batch" axis is the one that turns into a hypothesis test. If Phase 5 KILLs (NULL Δ vs Marlin), this axis was overestimated and Marlin's tensor cores DO matter at decode batch=4. Both outcomes are knowledge.

## Phase 8 — License decision (post-`2853551` reframing)

Decision is **Δ vs arm B (Marlin all-batch)**, not vs BF16. Production Marlin
already at 1.64× vs BF16 (`f6f3af3`); R4 #6 question is whether hybrid
dispatch incrementally improves on top.

| Arm C (hybrid) ITL Δ vs Arm B (Marlin all-batch) | Action |
|---|---|
| Δ ≥ −20% (≤ 9.4 ms ITL) → 2.0× absolute vs BF16 baseline | **LAND** + optional Phase 6 threshold sweep |
| Δ −5% to −20% (9.4-11.2 ms, 1.7×-2.0× vs BF16) | LAND with note: incremental over Marlin |
| Δ −5% to +5% NULL band | **KILL** — tensor-core advantage at batch=4 dominates the launch overhead; hybrid hypothesis refuted at this threshold; try Phase 6 sweep with threshold ∈ {16, 32} |
| Δ > +5% regression | KILL hard — small-batch BF16-native is slower; Marlin all-batch is the right dispatch |
| TTFT regression > +5% vs Arm B | KILL — hybrid broke prefill (shouldn't, but verify) |
| greedy_consistency divergence > 1% | KILL — two paths producing different numerical outputs |

**Multi-shape gate (mandatory before LAND)**:

- high-conc 1k/256/c=64 (batch=64 → Marlin path,应 unchanged)
- multi-tenant prefix-cache shape (batch varies)

If any shape regresses > 5%, threshold tuning needed (Phase 6) before LAND.

## Conflict gate (cron protocol §5)

This plan touches `infer/src/ops/linear.rs`. Codex's current WIP includes the same file (W4A8 dispatch additions). **Cannot execute Phase 5 until codex commits W4A8.**

Trigger sequence:
1. Codex commits W4A8 → push (file leaves WIP)
2. Claude rebases → pulls codex's W4A8 dispatch logic
3. Apply this plan's edit (10 LOC) on top of W4A8 code
4. Build + greedy_consistency + bench
5. License/kill per Phase 8

If codex's W4A8 already includes a similar batch threshold for W4A8 routing, this plan extends the same threshold to W4A16 — single edit, zero new design.

## Tick log (audit trail)

Each tick that revisits this plan logs the conflict-gate status here so the
"Codex W4A8 commit" trigger fires unambiguously when the file leaves WIP.

| Tick (ISO) | linear.rs WIP? | Codex state | Notes |
|---|---|---|---|
| 2026-05-08 ~11:55 | YES | `codex review --uncommitted` (16m) | Plan written (`1c534e6`) |
| 2026-05-08 ~12:07 | YES | `codex review --uncommitted` (28m, codex edited linear.rs:136-138 group_size constraint) | Conflict still active; brief queued to codex tmux 0:0 |
| 2026-05-08 ~12:12 | YES (5 files: linear.rs + gemm.rs ffi + quant.rs + 2 new C kernels) | review continues (32m) | Self-loop ScheduleWakeup armed; auto re-checks every ~270s |
| 2026-05-08 ~12:32 | YES (12 files; codex review continues 40m) | review continues; **codex acknowledged R4 #6 brief**: "W4A8 没 small-batch hybrid threshold,R4 #6 后续单独 tranche 覆盖" → R4 #6 stays valid | Codex 本地 commit `1e713de` (unpushed): KV W4A8 plan (orthogonal axis to weight, master §1.2.1.B P0). R4 #6 (W4A16 weight axis) and KV W4A8 (KV axis) compose: combined Phase 1.B bench should be 3-way A/B (W4A16-only / KV-W4A8-only / both). Plan separately. |
| 2026-05-08 ~12:43 | YES (12 files) | codex e2e tests passed 2/2; review at 47m | Major: read codex KV W4A8 plan §4 stack table → discovered Round 1 anti-pattern #8 violation (BF16-forced vs FP8 baseline). Self-correction `2853551` issued. R4 #6 reframed: 3-arm A/B at auto-FP8 KV. Skill bumped v1.2.0 (`4add8d7`) with isolation-motive callout. |
| 2026-05-08 ~12:45 | YES (14 files; cosmetic edits in marlin_w4a8_kernel.cu) | review wrapping at 51m | W4A8 substrate read-only survey: `gemm_w4a8_marlin_cuda` 987 LOC + activation quant 59 LOC. R4 #6 hybrid threshold extends to W4A8 dispatch (zero extra LOC). `4bbe246` |
| 2026-05-08 ~12:55 | YES (14 files) | codex 3rd-round review 53m, push imminent | This plan's Phase 5 + Phase 8 updated post-`2853551`: 3-arm A/B at production-default auto-FP8 KV; license decision against Arm B (Marlin all-batch). Bench commands drop `--kv-cache-dtype bf16`. |

When this table shows `NO` for `linear.rs WIP`, Claude proceeds Phase 5
without further confirmation: rebase → apply 10-LOC edit → `cargo build`
→ bench → license-or-kill per Phase 8.

## Cross-references

- Round 4 prep: [`docs/experience/errors/2026-05-08-marlin-w4a16-bench-implementation-gap.md`](../experience/errors/2026-05-08-marlin-w4a16-bench-implementation-gap.md) (`b3f22ea`)
- Skill: [`.claude/skills/kernel-optimization/SKILL.md`](../../.claude/skills/kernel-optimization/SKILL.md) (`73bb506` v1.1.0 with anti-patterns 11-13)
- M_quant plan: [`M_quant-fp8-w4-magnitude-path.md`](M_quant-fp8-w4-magnitude-path.md) §9.2
- ARLE Marlin path: `crates/cuda-kernels/csrc/gemm/marlin_kernel.cu` + Rust wrapper at `linear.rs:660-739`
- ARLE W4A16BatchGemv: `crates/cuda-kernels/src/ffi/gemm.rs:149` + dispatch at `linear.rs:909-911`
