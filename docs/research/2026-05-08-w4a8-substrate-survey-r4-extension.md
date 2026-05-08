# W4A8 substrate survey — R4 #6 hybrid dispatch will extend to W4A8

> Read-only survey while codex 0:0 in /review (51m, GPU released after
> e2e tests passed 2/2). Purpose: ground the future R4 #6 hybrid
> dispatch implementation so it covers both W4A16 (current) and W4A8
> (codex incoming) entry points.
> Tick `2026-05-08 ~12:45` self-loop.

## W4A8 kernel surface (codex WIP, untracked, on-disk)

| File | LOC | Role |
|---|---:|---|
| `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` | 987 | W4A8 Marlin GEMM kernel |
| `crates/cuda-kernels/csrc/gemm/w4a8_activation_quant.cu` | 59 | BF16 → INT8 row-wise activation quant |

External surfaces:

```cpp
// marlin_w4a8_kernel.cu:964
extern "C" int gemm_w4a8_marlin_cuda(
    const void* A,        // INT8-quantized activation
    const void* B,        // W4 weight (Marlin-packed)
    void* C,              // intermediate
    void* D,              // BF16 output
    void* s1, *s2, *s3,   // 3 scales (act / weight / global)
    int prob_m, prob_n, prob_k,
    void* workspace,
    int groupsize, int dev,
    cudaStream_t stream,
    int thread_k, int thread_n, int sms, int max_par
);

// w4a8_activation_quant.cu:45
extern "C" cudaError_t quantize_bf16_rows_to_int8_cuda(...);
```

## Per-call launch density (anticipated)

Mirroring W4A16 Marlin's wrapper pattern (`linear.rs:660-739`), the W4A8
Rust wrapper will need:

1. `alloc act_int8` (m × k bytes)
2. `quantize_bf16_rows_to_int8` ← **new** (vs W4A16's `bf16_to_fp16`)
3. `alloc workspace`
4. `alloc out_int32` (intermediate, m × n × 4)
5. `gemm_w4a8_marlin_cuda`
6. `dequant int32 → BF16 D` ← potentially fused via `D` arg, may not need separate launch

Per-call: **5-6 launches** vs cuBLAS BF16 single-launch GEMM. Same
order-of-magnitude as W4A16 Marlin (6 launches/call per Round 4 prep
`b3f22ea`).

## Anti-pattern #12 mapping (skill v1.2.0)

W4A8 inherits the same decode-vs-prefill duality:
- **Decode** (M ≤ 8): launch overhead dominates → 5-6 launches per linear hurts
- **Prefill** (M = 2048): tensor-core throughput (FP8 mma 706 TFLOPS) wins

W4A8 Marlin alone gives the prefill compute throughput axis but pays
the per-call launch cost at decode. Same hybrid-dispatch hypothesis as
R4 #6: small-batch decode → W4A16BatchGemv (BF16 native, 1 launch/call);
large-batch prefill → W4A8 Marlin (FP8 mma).

## R4 #6 implementation extension (no extra LOC)

The R4 #6 plan `docs/plans/M_quant-marlin-round4-hybrid-dispatch.md`
(`6781f46`) adds `MARLIN_DECODE_BATCH_THRESHOLD=8` to the dispatch
match. When codex's W4A8 lands, the *same* threshold applies to the
W4A8 dispatch arm — the dispatch logic at `linear.rs:65-93` will read
something like:

```rust
fn batched(weight: &DeviceMatrix, batch: usize) -> Self {
    // Marlin paths only when batch is large enough for tensor-core
    // throughput to dominate per-call launch overhead.
    if batch > MARLIN_DECODE_BATCH_THRESHOLD {
        if let Some(plan) = marlin_dispatch(weight) {  // W4A16 or W4A8
            return plan;
        }
    }
    // Small-batch fallbacks (BF16-native, 1-launch path):
    match (batch, weight.weight_format()) {
        (1, _) => Self::decode(weight),
        (_, WeightFormat::W4A16) => Self::W4A16BatchGemv,
        (_, WeightFormat::W4A8) => Self::<TBD W4A8 BatchGemv? or BF16 fallback?>,
        ...
    }
}
```

**Open question for codex**: does W4A8 have a BF16-native batched path
analogous to `w4a16_gemv_batch_cuda`? If not, small-batch W4A8 decode
must fall back to either (a) W4A16BatchGemv (loses W4A8 weight memory
benefit if Marlin-packed buffer covers both formats, **unlikely**), or
(b) BF16 GEMM (loses W4A8 entirely → only weight-VRAM saving from
W4-packed storage). Plan ahead before R4 #6 sweep, or accept that W4A8
small-batch path is "Marlin+conversion" today and improve in R4 #7+.

## Cross-references

- R1-3 baseline correction: [`2026-05-08-marlin-r1-baseline-correction.md`](../experience/errors/2026-05-08-marlin-r1-baseline-correction.md) (`2853551`)
- R4 #6 plan: [`M_quant-marlin-round4-hybrid-dispatch.md`](../plans/M_quant-marlin-round4-hybrid-dispatch.md) (`6781f46`)
- R4 prep survey (W4A16BatchGemv BF16-native): [`2026-05-08-marlin-w4a16-bench-implementation-gap.md`](../experience/errors/2026-05-08-marlin-w4a16-bench-implementation-gap.md) §"Round 4 prep"
- KV W4A8 plan (orthogonal): [`M_quant-kv-w4a8.md`](../plans/M_quant-kv-w4a8.md) (`1e713de`)
- Skill v1.2.0: [`.claude/skills/kernel-optimization/SKILL.md`](../../.claude/skills/kernel-optimization/SKILL.md) (`4add8d7`) — anti-patterns 11-13, isolation-motive callout
- M_quant master plan: [`M_quant-fp8-w4-magnitude-path.md`](../plans/M_quant-fp8-w4-magnitude-path.md)
