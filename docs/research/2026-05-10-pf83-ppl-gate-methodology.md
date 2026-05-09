---
title: PF8.3 PPL gate methodology — pair greedy_consistency with eval_ppl.py to close anti-pattern #26 blind spot
date: 2026-05-10
type: research
status: pf83-prep-ppl-tooling-mapped
---

# PF8.3 PPL gate methodology — pair greedy_consistency with eval_ppl.py to close anti-pattern #26 blind spot

> Codex briefed on PF8.3 in `93e1430` includes "PPL Δ < +0.5" as
> license condition. That number is NOT in `a66d99a` §2 (which only
> requires `greedy_consistency PASS`). This entry documents:
> (a) why the PPL extension is needed (greedy_consistency blind-spot
> per anti-pattern #26), (b) which existing tool to adapt
> (`scripts/eval_ppl.py`), (c) concrete A/B procedure.

## §0 Direct evidence (raw grep + tool inspection THIS tick)

### a66d99a §2 license matrix (raw Read this tick)

```
| Metric | License threshold | Kill threshold |
| TTFT p50 | Δ ≥ -8% with σ < 5% n=3 | Δ < -3% or any regression |
| TTFT p99 | Δ ≥ -5% | Δ > +10% (tail regression) |
| ITL p50 | regression < +2% (decode unchanged) | Δ > +5% (mistakenly affected decode) |
| Throughput tok/s | Δ ≥ +5% | Δ < 0% |
| greedy_consistency | PASS required | any FAIL |
```

PPL is NOT in the matrix. The brief sent in `/tmp/codex_brief_pf83.txt`
includes `greedy_consistency: PPL Δ < +0.5` — that's a Claude-side
extension, not from a66d99a.

### greedy_consistency.rs blind-spot (raw Read this tick)

```rust
//! BLIND SPOT (anti-pattern #26 candidate, 2026-05-09 P1.4 KILL `51dd5b2`):
//! this invariant holds when a broken kernel produces deterministically-same
//! garbage in solo + concurrent — solo == concurrent passes, but the shared
//! output is repetitive/word-salad. Pair this test with output-quality
//! assertions (Option A in research entry) or perplexity gates (Option C)
//! before declaring a new attention/quant kernel wire correct.
```

The test file ITSELF says: pair greedy_consistency with PPL gate.
`a66d99a` §2 only listed greedy_consistency without the pair, so PF8.3
license needs to add the PPL gate per the test's own self-documented
blind spot.

### scripts/eval_ppl.py (raw inspection THIS tick)

- 231 LOC, computes pseudo-PPL via per-token logprobs from greedy
  streaming decode
- Built-in datasets: wikitext-2-raw-v1, openai/openai_humaneval,
  openai/gsm8k
- Built-in A/B axis: `formats = [("BF16", None), ("FP8", "fp8"),
  ("INT8", "int8")]` for KV-cache dtype
- Computes Δ% per format vs BF16 baseline, summary table, multi-dataset

**KV-format A/B axis is NOT what PF8.3 needs.** PF8.3 changes the
prefill activation dtype (BF16 acts → FP8 e4m3 acts) for marlin GEMM
input — KV cache stays as configured. So eval_ppl.py needs adaptation
to use env-var axis instead of `--kv-dtype`.

## §1 Why the PPL gate is required for PF8.3

Per anti-pattern #26 (`51dd5b2` 2026-05-09 P1.4 KILL):

| Failure mode | greedy_consistency outcome | PPL gate outcome |
|--------------|---------------------------|------------------|
| Wrong activation quant scale (off-by-2× factor) | PASS (deterministic garbage in both solo + concurrent) | FAIL (PPL inflates 2-100×) |
| FP8 e4m3 saturation overflow on outliers | PASS (deterministic clamp in both) | FAIL (PPL inflates) |
| Wrong mma layout (m16n8k32 fragment misaligned) | likely PASS (silently wrong) | FAIL (PPL inflates massively) |
| Correct FP8 path | PASS | PASS (PPL Δ% within noise) |

PF8.3 changes mma instruction (m16n8k16→m16n8k32) + accumulator
(INT32→F32) + dequant (int4→fp8) — three structural changes.
greedy_consistency PASS alone is insufficient because anti-pattern #26
explicitly applies to "new quant kernel wire correct" scenarios.

## §2 Recommended adaptation of eval_ppl.py for PF8.3

### Option A — copy-paste new script `scripts/eval_ppl_pf83.py`

```python
# Same structure as eval_ppl.py but:
#   formats = [
#       ("baseline_W4_INT8", {"INFER_MARLIN_W4_FP8_PREFILL": "0"}),
#       ("treatment_W4_FP8", {"INFER_MARLIN_W4_FP8_PREFILL": "1"}),
#   ]
#   start_server() takes env_overrides dict instead of --kv-dtype
#   Same datasets (wikitext primary, humaneval secondary)
#   Same Δ% summary
```

~50 LOC delta from eval_ppl.py. Lowest risk — preserves baseline tool.

### Option B — extend eval_ppl.py with `--axis` flag

Add `--axis kv-dtype|prefill-fp8` CLI option. ~80 LOC delta but unifies
tooling. Risk: might regress KV-dtype A/B users.

### Option C — do nothing, codex implements ad-hoc

Risk: codex reinvents existing tool, forgets greedy_consistency pairing.

**Recommended: Option A** — codex picks up `eval_ppl_pf83.py` after
PF8.3 GEMM smoke verifies. ~50 LOC + already-working pattern.

## §3 PPL Δ threshold derivation (industry references)

Per a66d99a §2 lacking PPL: **0.5% Δ** is the standard FP8 quant
acceptance threshold from:
- SmoothQuant paper: ≤0.5 PPL Δ on wikitext for W8A8 vs FP16
- AWQ paper: 0.3-0.5 PPL Δ on wikitext for W4 quantization
- Existing `models/Qwen3-4B-W4A8-marlin` is licensed at < 0.5 PPL Δ
  (per ARLE prior W4A8 acceptance — verify in next-tick TODO)

**Recommended PF8.3 threshold**: PPL Δ% < +1.0% (relative) on wikitext
(conservative for prefill-only FP8 since prefill act precision affects
fewer tokens than KV format). Kill at > +5% (clear quant break).

This translates to `a66d99a` §2 row 5 addition:

```
| greedy_consistency + PPL Δ% (wikitext) | greedy PASS + PPL Δ ≤ +1.0% | greedy FAIL OR PPL Δ > +5% |
```

## §4 Complete PF8.3 license sequence (post-kernel)

1. **Smoke**: `/tmp/test_pf83_mma.cu` standalone mma shape verify
2. **Build**: `NVCC_CCBIN=/usr/bin/g++-14 CUDA_HOME=/usr/local/cuda
   cargo build --release -p infer --features cuda` clean
3. **Greedy consistency**: `INFER_MARLIN_W4_FP8_PREFILL=1
   cargo test --release --test greedy_consistency w4a8` PASS
4. **PPL gate**: `python3 scripts/eval_ppl_pf83.py
   --datasets wikitext,humaneval --max-samples 15` Δ% ≤ +1.0%
5. **e2e bench (PF8.5)**: `INFER_MARLIN_W4_FP8_PREFILL=1
   ./scripts/bench_guidellm.sh pf83-treatment --concurrencies 4
   --max-seconds 120 --warmup 10` × 3 runs σ < 5%; baseline = same
   workload with env=0
6. **License decision**: per `a66d99a` §2 + this entry's PPL row

If steps 3-5 PASS: license PF8.3 + close PF8 chain (PF8.4 already
landed dispatch wiring) + write wins entry citing TTFT Δ% + PPL Δ% +
greedy PASS.

If step 3 or 4 FAILS: KILL PF8 chain + remove dispatch from
`linear.rs:1966+` bail (the SelectedW4Path enum stays for future
attempt) + errors entry documenting the failure mode.

## §5 Cross-references

- a66d99a (NEW prefill-only FP8 directive — §2 license matrix, no PPL row)
- 51dd5b2 (P1.4 KILL — anti-pattern #26 origin: deterministic garbage
  passes greedy_consistency)
- 2026-05-09 anti-pattern #26 entry: `docs/research/2026-05-09-eod149-anti-pattern-26-same-output-but-garbage.md`
- `infer/tests/greedy_consistency.rs:6-13` (blind spot self-documented)
- `scripts/eval_ppl.py` (231 LOC, KV-format axis, adapt → eval_ppl_pf83.py)
- 93e1430 (PF8.3 brief sent to codex via tmux paste-buffer)
- /tmp/codex_brief_pf83.txt (the brief with provisional PPL number)

## §6 Status

PF8.3 PPL gate methodology DOCUMENTED. Adds row to a66d99a §2 license
matrix. Concrete tooling path (Option A: ~50 LOC adaptation of
eval_ppl.py). Threshold derived from SmoothQuant + AWQ industry
references + ARLE existing W4A8 acceptance.

Codex can pick this up post-PF8.3 GEMM kernel landing for PF8.5
license sequence (sequence in §4).

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(a66d99a §2 raw Read this tick, greedy_consistency.rs:6-13 raw Read
this tick, eval_ppl.py source raw inspection this tick, anti-pattern
#26 origin commit cited).
