# 2026-04-09 · W4A16 GPTQ Quality Issue

## Context

Attempting to support GPTQ-Int4 pre-quantized models. Both naive symmetric INT4 (our quantizer) and GPTQ calibrated INT4 produce garbage output.

## Symptoms

- Naive W4A16 (our symmetric quantizer, group_size=32 or 128): "isometric projection" / unrelated text
- GPTQ-Int4 converted model (JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4): "too too too" repetition
- W8A16 (our symmetric quantizer): works perfectly, 53.6 tok/s
- CPU simulation of W4 GEMV kernel: produces correct results

## Investigation

### Bug 1: GPTQ qzeros unpack interleaving (FIXED)
- `zeros_unpacked` reshape was `[G, 8, N//8] → [G, N]` with wrong dimension order
- Fixed by using `qzeros.unsqueeze(2) >> shifts.view(1, 1, -1)` → `[G, N//8, 8]` → reshape `[G, N]`
- Same issue may exist in `weight_unpacked`: needs `[K//8, 8, N]` → `[K, N]` with correct interleave

### Bug 2: W4A16 GEMV kernel - CPU correct, GPU wrong
- Python CPU simulation matches reference perfectly
- GPU kernel produces wrong output for all test cases
- Not yet isolated: could be CUDA compiler optimization, alignment, or subtle unpack mismatch

### Bug 3 (possible): weight_unpacked interleaving
- `qweight [K//8, N]` → expand to `[K//8, 8, N]` → reshape `[K, N]`
- C-order reshape gives `weight_unpacked[pack*8 + shift, n]` — may or may not be correct
- GPTQ packs element `k` into `qweight[k//8, n]` at bit position `(k%8)*4`
- So `pack=k//8`, `shift=k%8`. C-order: `pack*8+shift = (k//8)*8+(k%8) = k`. Correct!

### Open Questions

1. Why does the GPU W4 GEMV kernel produce wrong output when CPU simulation is correct?
2. Is the GPTQ `qweight` unpacking correct? The C-order reshape should be right but needs GPU-level verification.
3. Should we just use Marlin (proven kernel) + BF16↔FP16 conversion instead of debugging our W4 kernel?

## Current Status

- W8A16: **works, 53.6 tok/s (+79% vs BF16)**
- W4A16 naive: broken output
- W4A16 GPTQ: broken output (likely conversion + kernel issues)
- Marlin kernel: compiled and integrated, but requires FP16 (we use BF16)

## Next Steps

1. Add BF16↔FP16 conversion kernels (done: dtype_convert.cu)
2. Wire Marlin through with conversion — bypass our W4 GEMV entirely
3. Or: write a GPU unit test for the W4 GEMV kernel with known data
4. Long term: investigate why CPU-correct kernel produces wrong GPU output

## Rule

When debugging quantization quality: always compare against a known-good dequantized BF16 model first. If dequanted BF16 also produces garbage, the conversion is broken, not the quantization kernel.
