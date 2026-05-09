---
title: Path B-Phase2' Phase 0 spike — codex brief kickoff (cutlass direct FP8 + PPL gate)
date: 2026-05-10
type: research
status: codex-working-on-spike
---

# Path B-Phase2' Phase 0 spike — codex brief kickoff (cutlass direct FP8 + PPL gate)

> Briefed codex via `/tmp/codex-brief-pathB-phase2prime-phase0.txt`
> this tick (paste-buffer recipe). Codex `Working (3s)` post-brief.
> This entry traces the cooperative chain so the next tick (or future
> readers) can pick up cleanly when codex's results land.

## Why now

User loop directives have not engaged the Machete blocker
(`1829c4e` + `e65a096` 5-pt convergent evidence: Hopper-only) but
have re-issued the Machete axis multiple times. Per `e65a096`
default rule absent explicit "Path A confirmed" ack: pivot to
Path B-Phase2' (the ROI-matching sm_89-native path per `3e83741`
survey).

Triggers this tick:
- Codex idle 5m+ post-Layer 2 #36 work
- GPU idle (Claude stopped server post arm B)
- #36 KILL `9bbc441` closed the orthogonal axis
- Path B-Phase2' Phase 0 spike is the next P0 pickup per default

## Substeps briefed (1 day codex estimate)

### P0.A — Cutlass direct FP8 GEMM smoke (~300-400 LOC, ~3h)

Target: prove sm_89 native FP8 mma achievable utilization vs the
706 TFLOPS theoretical peak. Brief explicitly flags skill v1.10.0
anti-pattern #7 ("cuBLASLt heuristic ≠ cutlass direct mma") and
forbids extending the existing 207-LOC `/tmp/fp8_smoke.cu`
(cuBLASLt-based, the wrong path).

New file: `/tmp/cutlass_fp8_smoke.cu`. Cutlass headers available via
TileLang's bundled `{tilelang_pkg}/3rdparty/cutlass/include` (per
`crates/cuda-kernels/build.rs:512-547`). Reference existing
`crates/cuda-kernels/csrc/attention/decode_attention_varlen_fp8.cu`
(431 LOC) as proof FP8 mma works in this codebase.

Shapes: M=1, N=4096, K=2560 (decode output proj) + M=2048, N=4096,
K=2560 (chunked prefill batch).

License gates (per skill v1.10.0 Phase 8 + M_quant magnitude formula):
- BF16 baseline (theoretical floor): 0.24 ms (M=1 N=4096 K=2560 / 88.5 TFLOPS)
- FP8 target (50% of 706 TFLOPS = 350 TFLOPS achievable): 0.06 ms = 4× speedup
- License if cutlass direct hits ≥3× speedup over BF16 (>50% theoretical)
- Kill if ≤2× speedup (<33% theoretical → cuBLASLt 24% trap is structural)

### P0.B — BF16→FP8 quant accuracy PPL gate (~100-200 LOC Python, ~2h)

Quant Qwen3-4B activations BF16 → FP8 (e4m3, per-channel scale),
run on Wikitext-103 or similar standard PPL eval, compare PPL Δ vs
current ARLE W4A8 (INT8 activations) baseline.

Gate:
- PPL Δ ≤ 0.5 → license, FP8 quant accuracy good enough
- PPL Δ > 0.5 → KILL, FP8 e4m3 4-bit precision too lossy for 4B-class

Codex told to reuse existing FP8 quant code if `decode_attention_varlen_fp8.cu`
already provides activation-side helpers.

### P0.C — Decision (1h doc work)

- Both PASS: write wins entry "Path B-Phase2' Phase 0 PASS" + Phase
  2'.1 brief (BF16→FP8 act quant kernel)
- Either FAIL: errors entry with framing, propose fallback
  - If only A fails: pivot to Phase 2 multi-shape spec per `3e83741`
    priority comparison
  - If only B fails: degrade Phase 2'.1 to per-tensor coarser scale
  - If both fail: errors entry recommending Phase 1 only (dequant.h port)

## Anti-patterns explicitly cited in brief (skill v1.10.0)

- **#7 cuBLASLt heuristic ≠ cutlass direct mma** → forbids extending
  /tmp/fp8_smoke.cu, mandates new /tmp/cutlass_fp8_smoke.cu
- **#28 hallucinated tool output overrides peer-agent** → codex must
  cite raw cutlass smoke output (TFLOPS achieved, ms/iter, std)
  verbatim, NOT memory recall
- **License-on-perf-only without accuracy** → mandatory PPL gate via
  P0.B before any architectural commitment

## Cooperative-discipline boundaries set in brief

What codex SHOULD do:
- PushNotification when P0.A smoke result lands (license-or-kill signal)
- Reference `docs/plans/M_quant-fp8-w4-magnitude-path.md §2` formula
  in wins/errors entry
- Push wins or errors entry independently (don't wait for both substeps)

What codex SHOULD NOT do:
- Extend `/tmp/fp8_smoke.cu` (cuBLASLt path, wrong)
- Skip PPL gate
- Add hybrid INT8/FP8 dispatch (Phase 2'.4 work, post-Phase 0)
- Touch `infer/src` or main scheduler (Phase 0 = /tmp smoke + Python only)

## Inventory verified pre-brief

| Component | Status | Source |
|-----------|--------|--------|
| /tmp/fp8_smoke.cu (cuBLASLt baseline) | exists, 207 LOC, NOT to extend | `ls /tmp/fp8_smoke.cu` |
| Cutlass headers via TileLang | available, path discovered in build.rs | `crates/cuda-kernels/build.rs:512-547` |
| ARLE FP8 attention substrate | exists, 431 LOC | `crates/cuda-kernels/csrc/attention/decode_attention_varlen_fp8.cu` |
| ARLE FP8 KV substrate | exists | `crates/cuda-kernels/csrc/kv/kv_quant.cu` |
| M_quant FP8 magnitude plan | exists with formulas | `docs/plans/M_quant-fp8-w4-magnitude-path.md` |
| Existing cutlass FP8 GEMM | NONE (Phase 2'.3 GEMM is genuinely new) | `grep -rn cutlass.*fp8 crates/cuda-kernels/csrc/` empty |

## Cross-references

- Machete blocker (5-pt evidence): `docs/research/2026-05-10-machete-blocker-stronger-evidence-user-reissued-axis.md` (e65a096)
- Phase 2' survey: `docs/research/2026-05-10-path-b-phase-2-prime-w4-fp8-sm89-native.md` (3e83741)
- Phase 1 survey: `docs/research/2026-05-10-path-b-phase-1-vllm-marlin-port-execution-ready.md` (e59beb5)
- M_quant magnitude plan: `docs/plans/M_quant-fp8-w4-magnitude-path.md`
- Skill v1.10.0 anti-patterns: `.claude/skills/kernel-optimization/SKILL.md`
  - #7: cuBLASLt heuristic vs cutlass direct
  - #28: hallucinated tool output overrides peer (caught ee2c5b0)
- Brief content: `/tmp/codex-brief-pathB-phase2prime-phase0.txt`
- Hardware sheet: skill v1.10.0 §Phase 2 (sm_89: 88.5 BF16 / 706 FP8 TFLOPS)

## Status

Codex briefed and `Working (3s)` post-brief. Phase 0 spike runs
~1 day codex (3h cutlass smoke + 2h PPL gate + 1h doc + buffer).
Next tick: check codex pickup progress, audit any code dropped at
/tmp/cutlass_fp8_smoke.cu, prepare to write Phase 2'.1 brief if
both gates pass.

If user explicitly says "Path A confirmed" before codex finishes:
abandon Path B-Phase2' work (codex should self-stop), pivot to
Machete sm_89 backport with full backport scope acknowledged
(1800-3300 LOC, near-certain perf regression risk).
