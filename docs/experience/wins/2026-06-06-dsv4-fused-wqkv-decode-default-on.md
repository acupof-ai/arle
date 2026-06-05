# DSv4 fused-wqkv decode default-on — adopt DeepGEMM for the MLA-LoRA projection (+18.4%, the #1 decode kernel)

**Date:** 2026-06-06. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** flipped default-on (opt-out `ARLE_DSV4_FUSED_WQKV_DECODE=0`),
**licensed by a same-binary env A/B** (no rebuild). Found by a clean decode profile.

## Context — found by killing a profile confounder first

The first "decode" profile blamed a 9.46ms scalar `dsv4_hybrid_attention_kernel`. A
**path probe** (`ARLE_DSV4_FLASHMLA_PROBE`, scoped to real decode steps) proved that
was a **prefill confounder** — decode is FlashMLA-clean on every layer (no scalar
fallback). A **clean decode-only profile** (8-tok prompt + 64 decode steps, prefill
diluted to ~1/64) then gave the true decode kernel order:

| decode kernel | % GPU |
|---|---|
| **`dsv4_fp8_gemv_batch_kernel`** (scalar FP8 GEMV, MLA-LoRA proj) | **16.9%** |
| `ncclAllReduce` + `ncclAllGather` (TP comm) | 18.4% |
| `dsv4_mhc_params` (HC) | 8.9% |

So the #1 real decode *kernel* is the scalar FP8 GEMV — not attention.

## What Worked

- **Flip `dsv4_fused_wqkv_decode_enabled` default-on** (opt-out `=0`). It fuses
  `wq_a`+`wkv_a` into one **FP8 DeepGEMM** (tensor-core) instead of two scalar
  `dsv4_fp8_gemv_batch_kernel` GEMVs — the SGLang `fused_qkv_a_proj` approach. Already
  implemented behind the flag; this is the licensed default-flip. `_alloc_` falls
  through, so the fused scratch allocates under the default.
- **Same-binary env A/B (no rebuild), 64-tok decode, TP=8/EP=8 pod:**
  off **31.774 tok/s → on 37.633 tok/s = +18.4%**, **token-exact** (`[344, 34837, …]`
  both). Decode ~31.5ms → **~26.6ms/token**.

## Rule

A decode "profile" over a run that includes the prefill forward is **confounded** —
the prefill kernels (here CSA/HCA scalar attention, since FlashMLA-prefill is off)
land in the per-token sum and mis-rank the levers. Dilute prefill (short prompt +
many decode steps) or capture-range the decode window before reading the kernel
table. Once clean, the biggest decode *kernel* (FP8 GEMV) had a vendored fused
DeepGEMM path already built behind a flag — license with a same-binary env A/B (no
rebuild) and flip. (And: verify each gate's wall win on a same-harness A/B, not the
muddy default-vs-default tok/s that masked #4's real D2D win.)
