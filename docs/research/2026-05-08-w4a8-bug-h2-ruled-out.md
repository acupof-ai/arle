# W4A8 bug — Hypothesis 2 RULED OUT (s3 dtype consistent)

> Follow-up to [`e20f24c`](2026-05-08-w4a8-bug-claude-investigation.md)
> Hypothesis 2 (FP16/BF16 mismatch on s_group)。Quick code read confirms
> bytes consistent end-to-end。

## Code evidence

`crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`:

- Line 54:`using FragS_GROUP = Vec<half2, 1>; // weight per-group quantization scales`
  → kernel uses **half2 (IEEE FP16)** internally
- Line 262:`const int4* __restrict__ s3, // fp16 weight per-group quantization scales`
  → param is opaque `int4*` byte container,not typed Half pointer
- Line 523:`reinterpret_cast<int4*>(&frag_s3[k % 2])[0] = sh_s3_stage[s3_sh_rd];`
  → byte-copy 4×u16 chunk into half2 frag

`infer/src/ops/linear.rs::run_marlin_w4a8_linear` passes `s3_ptr as *const ffi::Half`
(BF16 alias),but kernel param `const int4*` 是 byte container,**FFI alias mismatch
不会触发 reinterpret bug** — 只是 4×u16 byte transfer。

`/tmp/quantize_qwen3_w4a8.py` writes `.to(torch.float16)` → bytes layout matches
kernel's `half2` interpretation。

## Conclusion

**Bytes consistent FP16 end-to-end**(script → loader → FFI int4 byte copy → kernel half2 reinterpret)。Hypothesis 2 RULED OUT。

## Remaining ranked candidates(per `e20f24c`)

| H | Hypothesis | Probability | Codex investigation needed |
|---|---|---|---|
| **3** | get_perms / scale_perm mismatch | **medium-high** | compare quantize script `get_perms()` lines 33-61 vs kernel mma fragment shape (16×8×8 / 16×8×16) |
| 4 | int4 - 8 offset(unpack)| low | grep kernel for `int4_value - 8` or `& 0x0F` pattern |
| 5 | activation INT8 scale wrong range | low | read `w4a8_activation_quant.cu` 59 LOC,verify `s_act = max/127` |

## Recommended codex next

**H3 (perms layout mismatch) is now the prime suspect**。

Cheap diagnostic (15-30 min):
1. Read script `get_perms()` lines 33-61(target mma fragment layout)
2. Grep kernel's tile loading code `cp_async4_stream` + `frag_b1.x` 等 reference
3. If permutations target different mma shapes → BUG。Marlin permutations are notoriously version-specific between W4A16 / W4A8 / Hopper variants。

If H3 also rules out → run microscale debug:single layer Wq @ x with known input,compare BF16 vs W4A8 output element-wise。
