# W4A8 100% garbage output — Claude investigation, hypothesis space narrowed

> Companion to errors entry [`81b6481`](../experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md).
> Read the W4A8 code paths to narrow root-cause search before codex investigation.
> Findings to feed codex: what's checked, what's still suspect, suggested debug
> probe.

## Code paths read

1. `/tmp/quantize_qwen3_w4a8.py` (171 LOC) — codex's quantize packer
2. `infer/src/ops/linear.rs::run_marlin_w4a8_linear` (lines 777-870) — Rust dispatch wrapper
3. `infer/src/weight_loader.rs::663-715` — W4A8 scale tensor loading
4. (NOT read: `marlin_w4a8_kernel.cu` — too long for this tick; codex own)

## Quantize-script logic trace

```python
# Per-channel scale (mapped to INT8 range 127):
s_channel = max_per_channel(weight) / 127.0  # float32 [1, n]

# Per-group weight scale (mapped to INT4 range 7):
s_per_group = max_per_group(weight) / 7.0  # float16

# Stored ratio (so kernel reconstructs s_per_group via multiply):
s_group_stored = s_per_group / s_channel  # float16

# Quantize weights (after divide by per-group scale):
w_int4 = round(weight / s_per_group) + 8  # in [0, 15]
```

Reconstruction at runtime (kernel-side, hypothetical):

```
weight_fp = (int4_value - 8) * s_group_stored * s_channel
         = (int4_value - 8) * (s_per_group / s_channel) * s_channel
         = (int4_value - 8) * s_per_group  ✓ algebraically correct
```

Algebra OK if kernel multiplies BOTH s_group AND s_channel.

## Dispatch wrapper trace (`run_marlin_w4a8_linear`)

```rust
ffi::gemm_w4a8_marlin_cuda(
    xq_ptr      as *const i8,      // INT8 activation
    mp_ptr      as *const u8,      // W4 packed
    reduce_ptr  as *mut i32,       // INT32 intermediate
    yf_ptr      as *mut ffi::Half, // BF16 output
    s1_ptr      as *const f32,     // s1 = activation scale (runtime)
    s2_ptr      as *const f32,     // s2 = s_channel (loaded f32)
    s3_ptr      as *const ffi::Half, // s3 = s_group (loaded raw u16)
    ...
)
```

**Three scales all passed.** s1 (activation) computed by `quantize_bf16_rows_to_int8_cuda`
at runtime per-row. s2 (channel) and s3 (group) are loaded from the W4A8 checkpoint.

## Hypothesis 1 — RULED OUT (probably): scale plumbing

All 3 scales are passed to kernel. Order matches the FFI signature
(`s1, s2, s3`). Channel f32 and group as raw u16 bytes.

## Hypothesis 2 — STILL SUSPECT: FP16 vs BF16 mismatch on s_group

Quantize script at line 88 + 97:
```python
s_per_group = ... .to(torch.float16)   # IEEE FP16
s_group_stored = (... / s_channel).to(torch.float16)  # IEEE FP16
```

Stored as IEEE FP16 binary (1+5+10 sign/exp/mantissa).

Loader reads as `&[u16]` (raw bytes, no reinterpretation). DeviceMatrix
stores as raw u16. Kernel reads through `*const ffi::Half`.

**`ffi::Half` is `__nv_bfloat16` in ARLE convention** (verified at
[`b3f22ea`](../experience/errors/2026-05-08-marlin-w4a16-bench-implementation-gap.md)
Round 4 prep — `turboquant_weight_gemv.cu:82-83` uses `__nv_bfloat16`).

If `marlin_w4a8_kernel.cu` casts `s3` to `__nv_bfloat16` internally, it reads
FP16 bytes as BF16 → wildly wrong values (different exponent encoding).
If it casts to `__half` (FP16) internally, the bytes match the script and OK.

**Diagnostic for codex**: grep `marlin_w4a8_kernel.cu` for how `s3` parameter
is dereferenced:
- If `__half *s3` or `half *s3` or `*reinterpret_cast<__half*>(s3)` → OK (matches FP16 script)
- If `__nv_bfloat16 *s3` or `*reinterpret_cast<__nv_bfloat16*>(s3)` → BUG (mismatch with script)

Probability: medium. Original Marlin uses `__half`; codex likely kept that
convention. But ARLE's broader `ffi::Half` aliasing risks confusion.

## Hypothesis 3 — STILL SUSPECT: get_perms / scale_perm mismatch

The quantize script's `get_perms()` and `scale_perm` permutations are
load-bearing — they re-arrange weight bytes to match Marlin's tensor-core
fragment layout. These permutations are kernel-specific.

If the script's permutations don't match codex's W4A8 kernel layout (e.g.,
copied from a different Marlin variant), packed weights would be in wrong
order → all 252 layers produce wrong output.

**Diagnostic for codex**: compare `get_perms()` in script (lines 33-61) with
the load layout assumed in `marlin_w4a8_kernel.cu`'s tile loading code.
If they target different mma fragment shapes (`16×8×8` vs `16×8×16` vs
others), this is the bug.

Probability: medium-high. Marlin permutations are notoriously version-specific.

## Hypothesis 4 — UNLIKELY: weight unpack + 8 offset

`w += 8; w = clamp(0, 15)` — INT4 stored as unsigned [0, 15]. Kernel
should subtract 8 during dequant.

If kernel forgets to subtract 8, weights are systematically biased by +8.
Every weight is 8 too large → outputs scaled wildly.

**Diagnostic for codex**: grep for `int4 - 8` or `& 0x0F` decoding pattern
in kernel. Marlin kernels typically do this in `dequantize` step.

Probability: low. This is a 1-line bug commonly caught by smoke tests; PR
#31 cherry-pick likely already had it.

## Hypothesis 5 — UNLIKELY: activation INT8 scale runtime computation

The `quantize_bf16_rows_to_int8_cuda` kernel computes s1 (activation
per-row scale). If it produces wrong scale (e.g. uses INT4 range 7 instead
of INT8 range 127), activations are wrong → garbage output.

**Diagnostic for codex**: read `w4a8_activation_quant.cu` (59 LOC). Should
compute `s_act = max(|x|) / 127.0` per row.

Probability: low (small focused kernel, easy to verify).

## Recommended codex investigation order

1. **First**: `marlin_w4a8_kernel.cu` s3 dtype handling (Hypothesis 2 — quick
   grep). 5 minutes.
2. **Second**: layout-permutation match (Hypothesis 3). Compare script
   `get_perms()` with kernel's load pattern. 15-30 minutes.
3. **Third** (if 1+2 OK): unit-test single linear layer end-to-end,
   compare output magnitude to BF16 reference (catches Hypothesis 4 + 5).
   30-60 minutes.

If none of those pin it: deeper kernel debug with a known-input scenario.

## Output pattern analysis

W4A8 output: `".........11.1.11111111 baudaskan1 baud111askan11"`

Suggests:
- High-frequency tokens (`.`, `1`, ` `) dominating → logits magnitude collapsed
  near zero, softmax bias-dominated
- Strange tokens (`baudaskan`) appearing → these are likely high-token-id
  bytes that get hit when uniform softmax samples randomly

Consistent with **weights significantly off-magnitude** (probably too small
by ~127× — exactly matching s_channel divisor 127.0). If kernel ignores
s_channel multiply, this is the EXACT regression we'd see.

⇒ **Strongly favors Hypothesis 2 (FP16/BF16 mismatch on s_group, kernel reads garbage scale → effectively no scale applied → weights too small by s_per_group×s_channel factor)**.

## Cross-references

- W4A8 errors entry: [`docs/experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md`](../experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md) (`81b6481`)
- W4A8 substrate LAND (reframed): [`docs/experience/wins/2026-05-08-w4a8-marlin-prod-bench-mixed-outcome.md`](../experience/wins/2026-05-08-w4a8-marlin-prod-bench-mixed-outcome.md) (`e61d26e`)
- Codex W4A8 kernel: `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (987 LOC, `a019a0e`)
- Codex W4A8 activation quant: `crates/cuda-kernels/csrc/gemm/w4a8_activation_quant.cu` (59 LOC, `a019a0e`)
- Quantize script: `/tmp/quantize_qwen3_w4a8.py` (codex authored, uncommitted)
- Dispatch wrapper: `infer/src/ops/linear.rs::run_marlin_w4a8_linear` (lines 777-870)
- Loader: `infer/src/weight_loader.rs:663-715`
- ffi::Half convention: `crates/cuda-kernels/csrc/gemm/turboquant_weight_gemv.cu:82-83` (`__nv_bfloat16`)

## Rule

When integrating a third-party Marlin kernel (or any tensor-core GEMM kernel)
into a BF16 stack, **`ffi::Half` typedef ambiguity is anti-pattern #11** (skill v1.3.0).
W4A8 kernel inherits PR #31 source which likely uses `__half`/FP16; ARLE's
convention is `__nv_bfloat16`/BF16 for the same typedef. **Always grep the
new kernel's `.cu` for `__half` vs `__nv_bfloat16` literals before licensing
the integration.** This was a flagged risk in `b3f22ea` Round 4 prep — and
W4A8 may have realized it.

If codex confirms Hypothesis 2, skill v1.3.0 anti-pattern #11 documentation
should be updated to reference this real-world catch.
