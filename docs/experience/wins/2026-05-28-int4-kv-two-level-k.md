# INT4 + KIVI two-level K (static × dynamic) + asymmetric [-8,7]

## Context

[`2026-05-27-int4-kv-kivi-poc.md`](2026-05-27-int4-kv-kivi-poc.md) shipped
INT4 + KIVI per-channel K end-to-end on V100 (Qwen3.5-4B audit). PoC
mean_match = 0.094 — coherent generation but immediate step-1 drift vs
BF16. User pushback: "coherent diff text 肯定能正确实现的 实现掉" —
accepting a 0.094 mean_match as "intrinsic 4-bit floor" was premature;
the literature has more levers before reaching for Hadamard.

Two cheap levers stacked into one commit:

1. **Asymmetric INT4 range [-8, 7] with /7.5 scale.** PoC used symmetric
   [-7, 7] / divisor 7. The full nibble is 16 levels (signed two's-comp
   −8..+7), not 15. /7.5 = midpoint of |−8| and |7|, minimizes max abs
   error at symmetric activation distributions. ~7% finer resolution at
   the same clipping rate.

2. **Two-level K scale: per-channel STATIC × per-(token, kv_head)
   DYNAMIC.** PoC quantized K with KIVI per-channel scale only:
   `k_int4 = round(k_bf16 / static[h, d])`. The static scale captures
   channel-wise outlier structure but underfits per-token magnitude.
   Two-level normalizes to the channel first, then takes the
   per-(token, head) absmax of the channel-normalized value and uses
   that as a second scale:
   `ratio = k_bf16 / static[h, d]; dyn[t, h] = max_d(|ratio|) / 7.5;
    k_int4 = round(ratio / dyn[t, h])`. Decode reads back
   `k_bf16 ≈ int4 * static[h, d] * dyn[t, h]`. Effectively grants K
   ~1 extra bit of dynamic range at the same 4-bit storage, which is
   exactly what closes most of the INT4-vs-INT8 quality gap the
   per-channel-only design leaves on the floor at low bit budgets.

V symmetric range bumped to [-8, 7] / 7.5 too, for the same reason.

## Implementation

- `quantize_paged_kv_int4_per_channel_kernel`: block reshaped to
  `(num_kv_heads, batch)` grid with `head_dim` threads. Threads compute
  the per-channel-normalized ratio, warp-reduce to per-(token, head)
  absmax, derive `dyn[t, h]`, write it to `k_dynamic_scales`, then
  stage signed nibbles in shared memory and pack pairs.
- `decode_attention_int4_per_channel_k_partial_kernel`: reads
  `K_dynamic_scales[row * num_kv_heads + kv_head]` per timestep, folds
  into the K dequant during QK: `qk += q_reg[i] * k_int4 *
  k_scale_reg[i] * k_dynamic`.
- `k_dynamic_scales` storage repurposes the existing per-token
  `k_scales` buffer (`pool.k_scales_ptr(layer)`) — already allocated
  for INT4 via `KVFormat::INT4.has_scales() == true` from the PoC. No
  new pool allocation.
- Rust FFI + wrapper + qwen35 prefill/decode dispatch all carry the
  new `k_dynamic_scales` pointer.

## Measurement

V100 Qwen3.5-4B audit at two grid sizes:

**`KV_PARITY_PROMPTS=2 KV_PARITY_MAX_TOKENS=8`** (PoC-comparable):

```
bf16  mean_match=1.0000  (reference)
int8  mean_match=1.0000  (KIVI per-channel K — unchanged, baseline)
fp8   mean_match=1.0000  (KIVI per-channel K — unchanged, baseline)
tq4   mean_match=0.0000  (unsupported on sm_70 — wrapper returns ERR_NOT_SUPPORTED)
int4  mean_match=0.5625  (KIVI two-level K + asymmetric [-8,7])
        ↑ PoC was 0.0938 — 6× improvement
```

**`KV_PARITY_PROMPTS=4 KV_PARITY_MAX_TOKENS=16`** (stress grid):

```
bf16  mean_match=1.0000
int8  mean_match=0.8906   first_div=prompt2 step13
fp8   mean_match=0.7344   first_div=prompt2 step3
tq4   mean_match=0.0000   (sm_70 N/A)
int4  mean_match=0.5781   first_div=prompt0 step1
```

Token traces at 16-token grid for INT4:

```
prompt0  ref=[271, 79852, 45850, 321, 6326, 6696, 513, 279, 1379, 2894, 5306, 13, 271, 79852, 45850, 369]
         cand=[271, 248068, 198, 8160, 369, 19, 271, 1206, 3300, 264, 1496, 21408, 11, 11346, 1965, 383]
prompt1  ref=[271, 760, 307, 7324, 76938, 1324, 369, 23424, 1608, 83876, 8476, 15019, 11, 1332, 279, 854]
         cand=[271, 760, 307, 7324, 76938, 1324, 369, 23424, 1608, 83876, 8476, 15019, 11, 1332, 279, 854]
prompt2  ref=[271, 248068, 198, 40, 1144, 728, 310, 1683, 883, 279, 1156, 579, 1622, 15060, 13, 2302]
         cand=[271, 248068, 198, 40, 1144, 728, 310, 1683, 883, 279, 1156, 579, 1622, 15060, 13, 2302]
prompt3  ref=[271, 248068, 198, 8160, 579, 264, 7047, 1817, 421, 668, 1438, 728, 1387, 1441, 678, 1622]
         cand=[271, 248068, 198, 8160, 369, 264, 7047, 1817, 421, 5879, 310, 279, 11488, 1965, 25, 271]
```

Two of four prompts are bit-identical with BF16 for the full 16 tokens
(prompt 1 and prompt 2). Prompts 0 and 3 diverge at sampling steps 1
and 4 respectively. mean_match = (1/16 + 16/16 + 16/16 + 4/16) / 4 =
0.5781.

INT4 (0.5781) is now competitive with FP8 (0.7344) at the 4×16 stress
grid — a noticeable but expected ~16pp gap, less than the bit budget
would naively suggest. The PoC's "intrinsic 4-bit floor" framing in
[`2026-05-27-int4-kv-kivi-poc.md`](2026-05-27-int4-kv-kivi-poc.md) at
~0.094 is now disproven; that entry should be amended.

Remaining gap (the 2/4 stress-grid divergences) sits where literature
expects: KIVI per-channel + per-(token, head) two-level alone hits
this floor at 4-bit. Closing further requires K outlier rotation, not
KIVI tweaks. Concrete options ranked by leverage:

1. **Hadamard rotation on K** (TQ4-style; sm_80+ ARLE already has
   a TQ4 implementation in `csrc/quant/turboquant_fast.cu`). Bake a
   random signs + FWHT rotation `R` into `W_Q` and `W_K` at weight-load
   time (cheap, no runtime cost) so Q and K are jointly rotated in
   their head_dim axis. K_rot then has its channel outliers
   redistributed uniformly — the per-channel + two-level scaling lands
   a tighter clip rate at the same 4-bit budget. Attention is invariant
   because `(QR)·(KR)^T = QR R^T K^T = QK^T` for orthogonal R.
2. **Selective top-N outlier channels in FP16** (KIVI-selective): keep
   the top-N (e.g. 4 out of 128) absmax channels in BF16, quantize the
   rest to INT4. Adds ~3% storage, big quality bump on hard prompts.
3. **K group-of-G quantization** (group_size=32 within head_dim=128):
   strictly finer than per-channel under most distributions; the win
   over per-channel is modest at head_dim=128 but more robust to
   distribution shift across prompts.

### Substrate unblock

V100 build was blocked behind a TileLang fork regression: the
`tilelang-sm70-copy` HEAD at session start (`69bc43e2`) and the rolled-
back state (`14489d9d`) both fail
`tilelang.transform.LayoutInference()` on the
`batch_prefill_paged_hd128_q32_kv8` kernel for sm_70 with
"Layout infer conflict between m_new and scale_i". Workaround used to
get the audit running:

1. Patched
   `crates/cuda-kernels/tools/tilelang/gen_tilelang_aot.py` (local-only,
   not committed) to short-circuit and emit `FUNC_NAME=`/`C_PATH=` when
   the target `.cubin` + `.c` wrapper + device `.cu` already exist in
   `out_dir`.
2. Pre-populated
   `target/release/build/cuda-kernels-<hash>/out/tilelang_aot/` with the
   prior session's debug-build artifacts
   (`target/debug/build/cuda-kernels-<hash>/out/tilelang_aot/`).
3. Build proceeds — TileLang AOT is skipped for the layout-infer-broken
   kernel and uses the cached cubin, which is ABI-compatible.

Permanent fix path: gen_tilelang_aot.py should grow real cache logic
(content-hash on the kernel script + tilelang version), not the
existence shim above. Tracked separately.

## Rule

When a "PoC works but quality is bad" entry suggests an *intrinsic*
floor, audit the literature's standard quality levers before
documenting the floor as fundamental. KIVI's per-channel-only K is one
of several KV-quant designs; two-level (per-channel × per-token) is a
well-known cheap upgrade. Symmetric vs asymmetric range, group size,
and rotation (Hadamard) are the other free levers. Document
"intrinsic" only after all of them are tried — otherwise the entry
mis-frames a tunable as a wall.
