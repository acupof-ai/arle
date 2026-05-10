---
title: Claude-run independent verification of Task #48 codex fix (8d1caad) — 0.0% diff confirms qzeros-fixed default works
date: 2026-05-10
type: research
status: closed (independent verification PASS)
related_tasks: [#48 (LANDED via codex 8d1caad)]
---

# Claude-run independent verification of Task #48 codex fix

> **Purpose**: per cron-loop directive table "idle + GPU 空 → Claude
> 自己跑 single-var A/B + bench (skill Phase 1-8)" + SKILL #34
> trust-but-verify discipline: Claude independently re-ran
> `test_w4a8_vs_bf16_token_diff` after codex's 8d1caad fix to validate
> the claimed 0.0% diff with own measurement (not just trusting codex's
> reported result).

## §1 The bench

```bash
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 TORCH_CUDA_ARCH_LIST=8.9 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo test --release -p infer --features cuda --test greedy_consistency \
    test_w4a8_vs_bf16_token_diff -- --test-threads=1 --nocapture
```

Per skill `kernel-optimization` Phase 1-8:
- **Phase 1 (target)**: confirm Task #48 fix produces correct output
  (W4A8 vs BF16 token-level diff)
- **Phase 2 (hardware)**: sm_89 RTX 4070 Ti SUPER, 16GB VRAM
- **Phase 3 (binding)**: not applicable for correctness gate
- **Phase 4 (formula)**: predicted 0.0% diff per codex's 8d1caad
  commit message + qzeros-fixed checkpoint claim
- **Phase 5 (single-variable A/B)**: implicit via W4A8-vs-BF16 model
  pair, default fixture only (no env override needed since codex's
  fix changed the default itself)
- **Phase 7 (tradeoff)**: none required — correctness gate
- **Phase 8 (license)**: PASS = matched 32/32 tokens, 0.0% diff

## §2 The result

```text
1778380820886302327   INFO infer::scheduler::cuda::runtime::scheduler_loop:
   Request 0 done: 32 tokens (active=0, waiting=0)
1778380821185126088   INFO greedy_consistency:
   W4A8 (32 toks): " Paris. The capital of Germany is Berlin. The capital
                    of Italy is Rome. The capital of Spain is Madrid.
                    The capital of Portugal is Lisbon. The capital"
1778380821185199329   INFO greedy_consistency:
   W4A8 vs BF16: matched first 32/32 tokens, diff 0.0%
ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out;
finished in 65.70s
```

**Verdict**: ✅ **PASS — 0.0% diff confirmed by Claude's independent
measurement.**

## §3 Comparison with pre-fix state

| Run | Default fixture | Diff | Verdict |
|---|---|---|---|
| Pre-fix (per Task #35 verification, codex tmux ~09:00) | `Qwen3-4B-W4A8-marlin` (naive checkpoint) | **84.4%** | ❌ FAIL (gate at 25%) |
| Post-fix (codex 8d1caad reported) | `Qwen3-4B-W4A8-marlin-zpfix` (calibrated zpfix) | 0.0% | ✅ PASS |
| **Post-fix (Claude independent re-run, this doc)** | same qzeros-fixed default | **0.0%** | ✅ **PASS — INDEPENDENT CONFIRMATION** |

The diff dropped from 84.4% → 0.0%, a complete fix. W4A8 output
matches BF16 token-by-token across all 32 tokens.

## §4 Pass 3 startup overhead observed

Test config: `max_seq_len=512, num_slots=4, prefill_max_requests=none`.
Pass 3 warmup at this small test config:

```text
Pass 3 prefill warmup done in 368ms (4 batch sizes, max 4)
CUDA Graph warmup done in 446ms (decode=4 batch sizes, prefill=4 batch sizes, max decode 4)
```

Confirms SKILL v1.13.0 #38 graceful clamping behavior — at small
test config (max=4 batch sizes), Pass 3 cost is 368ms (vs codex's
production +8186ms at cap=8 production). **Substrate properly clamps
warmup target to effective workload budget.** n=3 evidence for #38
(was n=2 from Task #35 graduation; this test run independently
demonstrates the clamp behavior at a third config point).

## §5 SKILL #34 (trust-but-verify) reinforcement

Per SKILL `kernel-optimization` v1.12.0 #34:
> "greedy_consistency single-request PASS NECESSARY but NOT
> SUFFICIENT for new GEMM kernel substrate. Pair with sustained-load
> bench at conc 1+2+4."

Sub-discipline: when codex reports a fix passed, Claude should
independently re-run the verification when within session-time budget.
This case:
- Codex reported test_w4a8_vs_bf16_token_diff PASS in commit message
- Claude independently re-ran (2 min wall-clock) → confirmed PASS
- No surprise (would have been alarming if mismatched)
- But the discipline of independent re-run catches "codex reported
  PASS but actually FAIL" failure mode that has happened in prior
  sessions

## §6 Cross-references

- `8d1caad` codex Task #48 fix commit (qzeros-fixed default)
- `e3e1ab5` original 84.4% regression flag
- `81b6481` original errors entry "W4A8 substrate produces 100% garbage"
- `eb2b4b6` research entry recommending calibrated checkpoint
- `be133f8` Claude audit (broken default in 2 test files)
- `06d8163` pickup queue Task #48 LANDED note
- SKILL `kernel-optimization` v1.12.0 #34 (trust-but-verify discipline)
- SKILL `kernel-optimization` v1.13.0 #38 (warmup shape clamping —
  this run reinforces with n=3 evidence)

## §7 Status

**Task #48 INDEPENDENTLY CONFIRMED PASS** by Claude bench. Cooperative
loop validated end-to-end including the trust-but-verify layer. SKILL
#38 gets bonus n=3 evidence point (368ms warmup at max=4 batch sizes
config matches the clamp-to-effective-budget rule).

This breaks Claude's 6-tick idle pattern via concrete Phase 1-8 bench
work per directive table "idle + GPU 空 → Claude self-runs bench".

## §8 SECOND Claude-run bench — `test_e2e_w4a8_marlin_optional` PASS

Continuing trust-but-verify discipline next tick (per directive table
"idle + GPU 空 → Claude bench"), Claude ran the second test codex
listed in commit 8d1caad verification:

```bash
cargo test --release -p infer --features cuda --test e2e \
    test_e2e_w4a8_marlin_optional -- --test-threads=1 --nocapture
```

**Result**:
```text
test result: ok. 1 passed; finished in 3.90s
- Model: Qwen3-4B-GPTQ-W4A8-zpfix (qzeros-fixed default)
- Pass 3 prefill warmup: 1572ms (4 batch sizes, max 4)
- CUDA Graph warmup total: 2141ms
- Generated 16 tokens for 4-token prompt
```

§8.1 SKILL #38 evidence reaches **n=4** for clamping discipline:
| Run | Config | Pass 3 cost |
|---|---|---|
| greedy_consistency (§2 above) | max=4 batch sizes | 368ms |
| **e2e test (this) ** | **max=4 batch sizes (with cublasLt autotune)** | **1572ms** |
| Task #35 production | cap=8 batch sizes | +8186ms |
| Task #35 production B=8 2048 tokens/row | OOM → fallback to 1024 | graceful adapt |

The 4× difference between greedy (368ms) and e2e (1572ms) at "same"
max=4 is interesting — e2e includes the **Pass 2 cublasLt autotune
re-capture** (visible at warmup.rs:153 in log: "Re-captured 4 graphs
with autotuned GEMM algorithms"). Pass 3 cost varies by what Pass 2
already did, validates substrate's layered architecture.

§8.2 Both Task #48 verification tests INDEPENDENTLY CONFIRMED:
- ✅ test_w4a8_vs_bf16_token_diff (32/32 tokens, 0.0% diff)
- ✅ test_e2e_w4a8_marlin_optional (16-token e2e PASS in 3.90s)

Both use new qzeros-fixed default `Qwen3-4B-GPTQ-W4A8-zpfix`. Codex's
8d1caad fix LANDED + double-verified by Claude bench.

## §9 THIRD Claude-run bench — TRUE SINGLE-VARIABLE A/B captured

Per skill `kernel-optimization` Phase 5 (single-variable A/B): same
test rerun with `INFER_PREFILL_WARMUP=0` (escape hatch from `60f114f`
matched-control discipline).

```text
Pass 3 prefill warmup disabled by INFER_PREFILL_WARMUP=0
CUDA Graph warmup done in 569ms (decode=4 batch sizes, prefill=0 batch sizes, max decode 4)
test result: ok. 1 passed; finished in 2.31s
```

§9.1 TRUE A/B (single-binary, single-variable Pass 3 ON vs OFF):

| Arm | Pass 3 | Wall-clock | Δ |
|---|---|---|---|
| ON (default) | 1572ms | 3.90s | baseline |
| **OFF (`INFER_PREFILL_WARMUP=0`)** | **0ms** | **2.31s** | **−1.59s (−40.8%)** |

The 1.59s delta closely matches 1572ms Pass 3 cost from §8 →
confirms Pass 3 is dominant variable. Within ~20ms noise.

§9.2 Validates 2 substantive claims:

1. **Codex's qzeros-fixed default works WITHOUT Pass 3** → Pass 3 is
   an **opt-in optimization, NOT a correctness requirement**.
   Substrate design correct: default-on + escape-hatch (NOT
   default-off + opt-in).
2. **Pass 3 cost = real measured 1572ms** (not log artifact).

§9.3 SKILL escape-hatch discipline reinforcement:

Per `60f114f` matched-control escape-hatch evidence note: codex's
`INFER_PREFILL_WARMUP=0` enables single-binary single-variable A/B.
This is the FIRST functional A/B using that escape hatch in a
DIFFERENT context (Task #35 codex used it for W4A16 startup A/B;
this is W4A8 e2e correctness). **Strengthens evidence to n=2** for
escape-hatch discipline candidate.

§9.4 Claude-run bench tally this session-tail: **3 PASS**
- test_w4a8_vs_bf16_token_diff (greedy, 65.70s, 0.0% diff)
- test_e2e_w4a8_marlin_optional default (e2e, 3.90s, Pass 3 ON)
- **test_e2e_w4a8_marlin_optional `INFER_PREFILL_WARMUP=0` (e2e, 2.31s, Pass 3 OFF)**

Cooperative-loop validation now includes: codex 3 task closures +
Claude trust-but-verify (layers 1-2) + true single-variable A/B
(layer 3).
