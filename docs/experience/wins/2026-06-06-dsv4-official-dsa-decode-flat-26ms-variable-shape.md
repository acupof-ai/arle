# DSv4 official DSA indexer — decode flat ~26ms vs legacy's context-scaling 124ms (4.8× @4096); PERF confirmed, correctness PENDING

**Date:** 2026-06-06. **Backend:** CUDA, DSv4-Flash FP8 TP-? (parity harness), 8×H20.
**Status:** opt-in `ARLE_DSV4_DSA_INDEXER`, **default-OFF** — perf potential confirmed,
**correctness NOT yet passing** (degenerate output), so NOT flipped default-on. This is a
sanity read from the variable-shape suite (`scripts/dsv4_variable_shape_dsa_gate.py`,
committed `567d3f25`), not a licensed default flip.

## The data (variable-shape suite — legacy hand-rolled csa_select vs official DSA, no spec)

| ctx | legacy prefill | official prefill | **legacy decode ms/tok** | **official decode ms/tok** | Δ decode |
|---|---|---|---|---|---|
| 64¹ | 4527 | 9229 | 27.45 | 28.15 | +0.70 |
| 256 | 469 | 479 | 28.16 | 26.02 | −2.14 |
| 512 | 705 | 704 | 37.11 | 26.06 | −11.05 |
| 1024 | 1447 | 1452 | 25.77 | 26.05 | +0.29 |
| 2048 | 3206 | 3227 | 48.03 | 26.11 | −21.92 |
| 4096 | 9525 | 7189 | **124.37** | **26.15** | **−98.22** |

¹ cold-start (first prompt in the sequential session — model load + DeepGEMM JIT; official higher
because it JITs the extra logits+topk kernels). Exclude from steady analysis.

## What it shows

- **Official DSA decode is FLAT ~26ms/token across all context lengths** (multi-SM
  `fp8_paged_mqa_logits`); **legacy csa_select scales with context** (single CTA / 1-SM:
  27→37→48→**124ms** as kv_len grows). At 4096: **124 → 26ms = 4.8× faster.** This confirms the
  root cause (csa_select is the context-scaling decode bottleneck) and the fix (official kernel).
- **26ms lands in the H20 base reference** for DSv4-Flash (no-spec single-stream ~20-35ms — see
  the reference baseline below). So official-kernel adoption pulls decode to the industry base.
- **Prefill: warm ~1.5ms/token** (256→479, 1024→1452, 2048→3227, 4096→7189), official ≈ legacy
  except **−25% @4096** (9525→7189). **Correction:** the earlier "14s prefill / 7-10× off" was the
  COLD-start number (load+JIT); warm 1024 prefill (~1.5s) is close to the H20 reference (~1.1-1.5s).

## ⚠️ Correctness PENDING — this is fast-but-not-yet-correct

Official DSA output is **degenerate (`' ` ` ` ` `'`)** on most shapes → Codex did NOT flip
default-on. Logits parity already PASSED (scorer correct, max_abs 0.062), so the bug is DOWNSTREAM:
suspect (1) `topk_transform_512` output format → FlashMLA index contract, (2) FP8 index-K cache
store/read layout mismatch, (3) prefill/decode posture inconsistency. Root-cause in progress.
**The 26ms is real only once the output is coherent.**

## Rule

A flat-vs-scaling decode curve across context lengths is the signature of an under-parallelized
(1-SM) selector replaced by a proper multi-SM kernel — but a perf delta on a correctness-failing
path is not a win until coherent ([[feedback_correct_inference_not_baseline_identity]]). Variable-
shape multi-prompt testing (not one fixed shape) is what exposed both the scaling curve AND the
shape-dependent degeneracy. Perf A/B deferred per ckl; correctness (needle + coherence) is the gate.
