# KV-precision-parity gate re-port (#58) — DSv4 lever gate live; three gated levers run license-or-kill

**Date:** 2026-06-10. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Harness:** `scripts/dsv4_needle_gate.py` + `scripts/dsv4_lever_gate.sh`
(committed `f239e31f`). **Binary:** main `ee4b8eae`+`42e7e039` build.
**Serve:** allreduce default lane, 16K max_seq, port 18189; one env-flip per
lever, fresh serve per gate run. Raw logs `/data01/build/needle_gate_*.log`,
`ab_*_decode.log`.

## Goal

Issue #58: re-port the monolith's KV-precision-parity audit (died with
`infer/` at `e81b98fb`; original at `git show e81b98fb^:infer/tests/
kv_precision_parity.rs`) onto the rewrite stack, with gate semantics per
strategy v2 §2.6 — **correct inference** (needle retrieval + same-config-repeat
non-determinism floor + self-consistency), NOT byte-identity-vs-baseline
(invalid under DSv4 MoE run-to-run non-determinism). Run it on the three
gated levers; defaults flip only on PASS.

## Gate semantics

- Matrix: needle `738291` at depth 0, lengths {115, 300, 446, 2000, 8000},
  ×3 same-config repeats, greedy, raw `/v1/completions` (#56-validated
  harness, non-degenerate filler).
- Baseline envelope (same binary, default lane): exact counts
  `3(DET)/0/2/2/3`, decode smoke 39.38 tok/s, plus the #56 distribution as
  the second same-config sample.
- **PASS** = per-length exact count within ±1 of baseline, zero garbage-class
  outputs (tag salad / mojibake / token loops), coherent misses only, decode
  smoke within noise of baseline with identical greedy opening.
- **KILL** = any garbage-class output, exact-count collapse beyond the
  envelope, or decode-path divergence (incoherent continuation).

## Results

| Lever (env) | exact @115/300/446/2000/8000 | decode smoke | Verdict |
|---|---|---|---|
| baseline (none) | 3(DET)/0/2/2/3 | 39.38 tok/s | — (envelope) |
| FlashMLA sparse decode (`ARLE_DSV4_FLASHMLA_DECODE=1`) | 3/1/1/3/3 | 38.67 tok/s (−1.8%), opening byte-identical | **PASS — licensed** (all lengths within ±1, no garbage) |
| Fused WQKV decode (`ARLE_DSV4_FUSED_WQKV_DECODE=1`) | 3(DET)/1/3/3/2 | 38.13 tok/s (−3.2%, single-run), opening byte-identical | **PASS — licensed** (8000 single miss is the known coherent filler-quote confabulation class, within ±1) |
| Pooled/contig MoE (`ARLE_DSV4_GPU_ROUTER=1`) | 3(DET)/1/3/3/3 | **29.95 tok/s (−24%)**, greedy opening DIVERGES from baseline (coherent; pooled reduction order ≠ masked) | **Correctness PASS / default-flip KILL** — needle distribution within ±1 (the 300-len misses are the baseline's own degenerate-loop class, one numeric-loop variant noted as caveat), but −24% decode reconfirms the standing pooled-vs-masked regression; not a flip candidate |

## Scope notes

- **Correctness license ≠ default flip.** These gates license the levers'
  *correctness*; flipping a default additionally needs the perf side per the
  bench spec (and the pooled-MoE lever is already known −20% at B=1 per
  `reference_dsv4_pooled_decode_slower_than_masked_b1` — its gate run is for
  the record, not a flip candidate).
- **Qwen3.5 4-precision matrix (BF16/INT8/FP8/TQ4): BLOCKED, explicitly.**
  The rewrite's Qwen paged-KV path has no INT8/FP8/TQ4 KV dispatch yet (the
  weight/KV quant Rust dispatch is still pending re-port — support-matrix §0).
  The old harness structure is recoverable from git history when that lands;
  the gate semantics + driver shape from this re-port carry over directly.
- **DSv4 has no BF16-KV serving arm** (the arena is FP8-packed by design),
  so the DSv4 "reference axis" is the default lane's same-config envelope +
  self-consistency, not a dtype A/B. The FP8-vs-BF16 fidelity question for
  the #56 trailing-digit residual therefore needs either the (broken, #67-era)
  SGLang reference lane or a BF16 arena build — recorded as the residual's
  open discriminator.

## Problems

- ROUTER's 300-len run-1 miss is a numeric loop (`## 2.2.2.2…`) — same
  degenerate-loop class the baseline itself produces at 300, but a louder
  surface form. If a future lever shows loops at lengths where the baseline
  does NOT loop, that is a KILL, not an envelope match.
- ROUTER's greedy opening diverges from the other three configs (which are
  byte-identical to each other). Expected (pooled MoE changes float reduction
  order) and explicitly NOT the gate, but worth remembering when eyeballing
  smoke outputs: divergent-but-coherent ≠ broken.
- Decode smokes are single-run (±3% noise band); the FlashMLA/WQKV −2~3%
  deltas are not perf claims. Perf licenses need the bench-spec A/B protocol.

## Rule

- A kernel-path lever flips its default ONLY through this gate (correctness)
  plus a wall-clock perf license — `plan_label`/reachability is not a license
  (per the distilled lessons), and byte-identity is not the gate (MoE
  non-determinism).
- Gate runs require a fresh serve per lever (env is read at boot) and the
  baseline envelope re-measured on the SAME binary as the levers.
