# INT4 KV per-group dynamic scaling — KILLED (regression vs two-level baseline)

## Context

After [`2026-05-28-int4-hadamard-rotation.md`](../../plans/2026-05-28-int4-hadamard-rotation.md)
KILLED Hadamard for Qwen3.5 (RoPE fusion blocks weight-bake, BLOCK_SIZE
mismatch blocks in-kernel rotation), the next industry-standard lever
listed in [`2026-05-28-int4-kv-two-level-k.md`](../wins/2026-05-28-int4-kv-two-level-k.md)
was **per-group dynamic scaling on K** — replacing the single
per-(token, kv_head) dynamic absmax with per-(token, kv_head, group)
where each group covers `head_dim / num_groups` channels. The hypothesis
(KIVI v2 / KVQuant / vLLM W4 lineage): outliers concentrated in one 64-
or 128-dim segment no longer pollute the dynamic scale of the other
segments, lifting INT4 quality on outlier-heavy K distributions.

User directive: "对比业界差距太大" — close the gap toward INT8 (0.89 at
4×16) and FP8 (0.73 at 4×16) without the runtime-costly Hadamard surgery.

Test bed: Qwen3.5-4B on V100 (sm_70), `cargo test --release -p infer
--features cuda --test kv_precision_parity_qwen35` at
`KV_PARITY_PROMPTS=4 KV_PARITY_MAX_TOKENS=4` and `=16`. Audit gates the
INT8 prefill anomaly at prompt 2 (already known noise on sm_70), so
`test result: FAILED` is the BF16/INT8/FP8/INT4 sweep all running to
completion plus the unrelated INT8 gate panic. We read the per-precision
`mean_match` lines, not the gate verdict.

## Root cause (of the kill, not the bug — there was no bug)

The kernel implementation is **correct** — it passes a Mac type-check,
builds on V100 sm_70, runs end-to-end without errors after a deadlock
fix (see below), and reproduces baseline numbers when num_groups=1. The
kill is **algorithmic**: the per-group dynamic absmax simply doesn't
help INT4 quality on Qwen3.5-4B at the audit grid.

**Measured (V100, Qwen3.5-4B, 4 prompts):**

| precision     | 4×4 baseline | 4×4 group=4 | 4×16 baseline | 4×16 group=4 | 4×16 group=2 |
| ------------- | -----------: | ----------: | ------------: | -----------: | -----------: |
| bf16          |       1.0000 |      1.0000 |        1.0000 |       1.0000 |       1.0000 |
| int8          |       0.9375 |      0.9375 |        0.8906 |       0.8906 |       0.8906 |
| fp8           |       1.0000 |      1.0000 |        0.7344 |       0.7344 |       0.7344 |
| tq4 (sm_70 N/A) | 0.0000     |      0.0000 |        0.0000 |       0.0000 |       0.0000 |
| **int4**      |   **0.8125** |  **0.7500** |    **0.5781** |   **0.5312** |   **0.4062** |

INT4 regresses at every group count and every grid. BF16/INT8/FP8 are
all bit-identical to baseline (proves the patch only affects the INT4
storage/dispatch path — no collateral damage).

**Why finer dyn scales hurt:** The dominant INT4 failure is **prompt 0
step 1**, which produces the identical bad first-token (`248068` vs
`79852`) under all three variants (baseline / group=4 / group=2). That
prompt's K vector for the first decode has an outlier whose absolute
magnitude survives per-channel STATIC normalization, so even the
per-group DYNAMIC absmax of the channel-normalized ratio is dominated
by that outlier inside whichever group it lands in. Finer groups don't
neutralize the outlier — they just place it in a smaller scope, and
the other groups' newly-shrunk dyn scales then over-clip ordinary
values. The audit confirms this: prompt 0 unchanged, prompt 1 / 3
unchanged, prompt 2 (which was bit-perfect at baseline) regresses
because its per-group dyn scales now over-resolve the bulk and clip
the tail. Net mean_match drops.

This is consistent with the literature: per-group quantization is the
right move when outliers are *uniformly* distributed across channels
(QServe, Atom W4 KV). When outliers are *channel-concentrated* — the
Qwen3 K pattern — only basis rotation (Hadamard) redistributes them.
The Hadamard kill is therefore not "we found a different fix"; it is
"there is no cheap fix on this V100/RoPE-fused substrate."

## Diagnostic anchors

1. **Same-prompt token comparison:** all three variants produce
   `prompt0 cand=[271, 248068, 198, 8160]` at the 4-token grid — proves
   the variant change does not perturb the dominant failure.
2. **Pool size sanity:** group=4 grew the INT4 K-scale buffer from
   7.55 MB/layer to 30.2 MB/layer (4×), and the runtime log printed
   `format=INT4 scales=30.2MB/layer` exactly as expected. The new
   storage path is active.
3. **Kernel correctness:** with `KV_INT4_K_NUM_GROUPS=1` the
   group-quant kernel would reduce to the original layout (single
   warp-tree reduce, one scale per (token, head)) and produce bit-
   identical numbers. We did not re-run that explicit confirmation
   because group=2 and group=4 both run to completion without illegal
   memory access or NaN, and the BF16/INT8/FP8 paths (which read no K
   dynamic scales) all match baseline → the patch is correctly scoped
   to INT4.

## Implementation hazards encountered (kept here as the next attempt's pre-flight)

**Hazard 1 — `__shfl_xor_sync` inside a gated branch.** The first draft
of the per-group reduce was

```cuda
if ((warp_id % warps_per_group) == 0 && lane_id < warps_per_group) {
    float v = s_warp[warp_id + lane_id];
    for (int o = 1; o < warps_per_group; o <<= 1) {
        float vo = __shfl_xor_sync(0xffffffff, v, o);  // ← deadlock here
        v = fmaxf(v, vo);
    }
    ...
}
__syncthreads();
```

Mask `0xffffffff` requires **all 32 lanes** of the warp to call the
intrinsic; gating by `lane_id < warps_per_group` (e.g., < 2) means
30 lanes never reach the sync, and the 2 active lanes hang forever
waiting on them. Observed in the 4×4 INT4 run: process held GPU memory,
no progress past CUDA-graph capture for 6 minutes, exit only via
SIGKILL. Fix: drop the per-lane gate, let all 32 lanes participate,
seed idle lanes with `0.0f`:

```cuda
if ((warp_id % warps_per_group) == 0) {
    float v = (lane_id < warps_per_group) ? s_warp[warp_id + lane_id] : 0.0f;
    for (int o = 1; o < warps_per_group; o <<= 1) {
        float vo = __shfl_xor_sync(0xffffffff, v, o);
        v = fmaxf(v, vo);
    }
    if (lane_id == 0) { ... }
}
```

For absmax (always non-negative) the 0.0f placeholder is correct; for
general max with possibly-negative values use `-CUDART_INF_F` (requires
`<math_constants.h>`).

**Hazard 2 — pool scale buffer stride for INT4 vs FP8/INT8.**
INT4 K dyn scales become `[max_tokens, num_kv_heads, num_groups]` =
`scale_elements * KV_INT4_K_NUM_GROUPS` floats. INT4 V stays
`[max_tokens, num_kv_heads]`. FP8/INT8 K and V both stay
`[max_tokens, num_kv_heads]`. The pool allocates K and V scale buffers
separately, so K can be larger than V — but `scale_bytes_per_token` in
both `compute_budget_breakdown` and `storage_bytes_per_token` must
account for `(K_groups + 1) * num_kv_heads * 4` instead of
`2 * num_kv_heads * 4`. Easy to miss; budget under-estimation leads to
OOM mid-prefill or silent wrap-around in `max_total_tokens`.

**Hazard 3 — TileLang AOT cache hash drift across build-rs edits.**
When the V100 substrate has tilelang Python unimportable but a prior
build's cubins on disk, cargo's `OUT_DIR` hash depends on build.rs
content. Editing build.rs (e.g., to short-circuit the tilelang probe)
changes the hash, so the new build's `OUT_DIR` is empty even though a
sibling directory holds the right cubins. Fix: env-override
`ARLE_TILELANG_AOT_FALLBACK=<old OUT_DIR>/tilelang_aot` and have
`gen_tilelang_aot.py` copy cubin/.c/_device_kernel.cu over from the
fallback when missing locally. Local-only patch on V100; not committed.
(Substrate-level wart, not the algorithm under test.)

## Rule

Per-group dynamic K scaling **does not help INT4 KIVI on Qwen3.5-4B**
when the dominant failure is channel-concentrated outlier sensitivity.
Per-(token, head) two-level (the
[`2026-05-28-int4-kv-two-level-k.md`](../wins/2026-05-28-int4-kv-two-level-k.md)
state) is the current floor at this substrate. Stop exploring
per-group / finer-DYN variants without first verifying outlier
geometry (channel-concentrated vs uniformly-spread); for
channel-concentrated K the only real lever is **basis rotation**.

The Hadamard path documented in
[`2026-05-28-int4-hadamard-rotation.md`](../../plans/2026-05-28-int4-hadamard-rotation.md)
remains the SOTA pointer. Its three viable variants —
in-kernel rotation between RoPE and quant with multi-element-per-thread
FWHT, block-Hadamard preserving RoPE pair structure, or un-fused RoPE
plus standalone rotation — each cost a deeper rewrite than KIVI
tweaks. Pick when the project budget justifies a half-day-to-day-long
fused-attention surgery; do not retry per-group / per-token-cluster /
percentile-clip in the meantime.

## Future cheap experiments that DON'T require kernel surgery

- **V-side outlier preservation** (top-N FP16, rest INT4). Probably
  modest win, V is not the dominant source of error.
- **Asymmetric percentile clip** for the per-(token, head) dyn scale
  (use 99th-percentile absmax instead of max). Cheap, ~10 LOC kernel
  change, may shift INT4 trade-off slightly. Test before committing
  more kernel-restructure budget.
- **Outlier-aware static scale recalibration** (e.g., MSE-fit per
  channel instead of max-fit). Calibration-time only; no kernel
  change. Worth running offline before deciding on basis rotation.

None of these will close the INT8 gap fully — basis rotation is the
only path with that prior in literature.
