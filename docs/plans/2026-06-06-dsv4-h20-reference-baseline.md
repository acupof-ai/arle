# DSv4-Flash on H20 — reference baseline + re-anchored targets (the "该多快" ground truth)

**Date:** 2026-06-06. **Why:** we were optimizing without a reference — "感觉不对劲" (ckl). This
establishes what DSv4-Flash P/D *should* be on H20 (from official + SGLang sources), the gap to
ARLE, and the corrected target. Searched the web (ckl: 直接搜索 dsv4 flash 性能数据).

## DeepSeek-V4-Flash — what it is (HF deepseek-ai/DeepSeek-V4-Flash)

- **284B total / 13B active per token** (small active!), 1M context.
- MoE experts **FP4** + rest **FP8** mixed; **CSA + HCA** hybrid attention (= ARLE's DSv4 path).
- 1M-ctx: **27% of single-token FLOPs + 10% KV cache** vs V3.2.

## H20 hardware

96GB HBM3, **4.0 TB/s**, **148 BF16 / 296 FP8 TFLOPS**, **78 SMs**, 350W. Bandwidth ≈ H100, but
compute is ~15% of H100 → **H20 is compute-starved**: prefill (compute-bound) is the weak axis;
decode (bandwidth-bound) is relatively fine. (SGLang reference uses H20-**3e** = 141GB HBM3e
variant; the pod is H20-96G — same 78-SM compute profile.)

## Reference P/D on H20 (SGLang issue #23896, DeepSeek API, Artificial Analysis)

| metric | reference | config |
|---|---|---|
| Decode TPOT | **79.3 ms/token** | SGLang H20-3e, TP4, FP8, **EAGLE 3-step/4-draft**, 100 concurrency, 1024/1024 |
| Decode (single-stream, optimized) | **~7-12 ms/token** (83-143 tok/s) | DeepSeek API (full stack + MTP/EAGLE) |
| Decode (single-stream W4A16+FP8+MTP) | **~9 ms/token** (111 tok/s) @128k | with self-speculation |
| Prefill TTFT | **5.3 s** (1024 in) | SGLang H20-3e @100 concurrency (queue-dominated) |
| Prefill TTFT | **~1.1-1.4 s** | DeepSeek API single-stream |
| Theoretical decode floor | **~0.3 ms/GPU** | 13B active @ FP4/FP8 / 8 / 4TB/s — so decode is NOT weight-bandwidth-bound; it's comm + sparse-prepare + launch bound |

## ARLE current vs reference (measured, variable-shape suite)

| | reference (H20, should-be) | ARLE official DSA | ARLE legacy | verdict |
|---|---|---|---|---|
| Decode @4096 | base ~20-35ms; spec'd ~7-9ms | **26ms (flat, correctness-pending)** | 124ms | official = AT base reference; legacy 4.8× off |
| Prefill @1024 (warm) | ~1.1-1.5s | ~1.5s | ~1.4s | ~at reference |
| Prefill @4096 (warm) | ~2-4s (compute-starved H20) | 7.2s | 9.5s | ~2-3× off (official −25%) |

## ⭐ The re-anchoring (corrects 3 wrong assumptions)

1. **6ms decode is REAL and achievable — but REQUIRES MTP/EAGLE speculation.** DeepSeek's own API
   hits ~7ms; single-stream W4A16+MTP ~9ms. **Base (no-spec) decode is ~20-35ms**, not 6ms. ARLE's
   official DSA already hits **26ms base = at reference**. The 26→~6ms step IS spec decode.
2. **MTP/EAGLE is INDUSTRY-STANDARD for DSv4, not optional.** SGLang runs EAGLE-3step/4draft on
   H20; DeepSeek API uses it. **Parking MTP + the earlier "EAGLE killed on DSv4" verdict was the
   wrong call** — the kill was an ARLE-impl artifact; the industry runs EAGLE on DSv4. MTP/EAGLE
   goes back on the mainline (pixel-level copy SGLang's nextn/EAGLE + frozen-KV verify).
3. **The "14s prefill / 7-10× off" was a COLD-START artifact** (model load + DeepGEMM JIT in the
   first run). Warm prefill is ~1.5s @1024 (at reference), ~7.2s @4096 (~2-3× off — H20
   compute-starved, needs official FlashMLA sparse_fwd).

## Re-anchored targets + path

- **Decode base → ~20-26ms** (✅ official DSA already there, once correct) — fix DSA correctness, flip default-on.
- **Decode ~6-9ms** → **MTP/EAGLE** on the correct 26ms base (mainline, not parked; pixel-level SGLang).
- **Prefill @4096 → ~2-4s** → official FlashMLA `sparse_fwd` (H20 compute-bound; cold-start load/JIT also worth a warmup).
- **Honest:** 6ms single-token on H20 = base official kernels + spec decode; both are SGLang-standard, neither is hand-rolled.

Sources: HF deepseek-ai/DeepSeek-V4-Flash · SGLang issue #23896 · DeepSeek API (Artificial Analysis) · LMSYS H20 serving blog.
