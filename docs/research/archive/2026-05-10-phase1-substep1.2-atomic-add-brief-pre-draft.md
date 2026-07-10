---
title: Phase 1 Substep 1.2 atomic_add — pre-drafted codex brief (upstream pattern verified)
date: 2026-05-10
type: research
status: ready-for-codex-pickup-post-substep1.1-commit
---

# Phase 1 Substep 1.2 atomic_add — pre-drafted codex brief (upstream pattern verified)

> Codex Working (17m+) on Substep 1.1 cargo fmt + clippy gate. After
> 1.1 commits separately (per codex's stated discipline), Substep 1.2
> (atomic_add reduce opt-in) is the natural next pickup. Pre-drafting
> the brief now using verified upstream pattern so codex can pick up
> with zero context-switch.

## §0 Direct evidence (raw `gh api` this tick, NOT memory recall per skill v1.10.0 #28)

### Upstream atomic_add template signature (marlin_template.h:268-269)

```bash
$ gh api repos/vllm-project/vllm/contents/csrc/quantization/marlin/marlin_template.h \
    | base64 -d | sed -n '263,275p'
```

```cpp
int prob_n,      // output dimension n
int prob_k,      // reduction dimension k
int lda,         // A.stride(0), equal to prob_k is A is contiguous
int* locks,      // extra global storage for barrier synchronization
bool has_bias,
bool use_atomic_add,   // whether to use atomic add to reduce       ← Substep 1.2
bool use_fp32_reduce,  // whether to use fp32 global reduce         ← bonus opt-in
int max_shared_mem) {
```

Two independent opt-in flags. Substep 1.2 per e59beb5 covers
`use_atomic_add` only; `use_fp32_reduce` is bonus (could be Substep
1.3 if license).

### Atomic_add reduce path (marlin_template.h:1725-1745)

```cpp
#pragma unroll
for (int i = 0;
     i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
     i++) {
  if (c_gl_wr < c_gl_wr_end) {
    if (use_atomic_add && slice_count > 1) {                        ← gate condition
      c_scalar_t2* C_half2 = reinterpret_cast<c_scalar_t2*>(&C[c_gl_wr]);
      c_scalar_t2* sh_red_half2 = reinterpret_cast<c_scalar_t2*>(&sh_red[c_sh_rd]);
      #pragma unroll
      for (int a = 0; a < 4; a++) {
        atomicAdd(&C_half2[a], sh_red_half2[a]);                    ← atomic add
      }
    } else {
      C[c_gl_wr] = sh_red[c_sh_rd];                                  ← existing direct-write
    }
    c_gl_wr += c_gl_wr_delta;
    c_sh_rd += c_sh_rd_delta;
  }
}
__syncthreads();
```

**Key insight**: atomic_add path only taken when `slice_count > 1`
(multi-slice GEMM where multiple SMs contribute to the same output
column). Single-slice case still uses direct write — no atomic
overhead for non-split workloads.

### ARLE marlin_kernel.cu existing reduce path

ARLE's `marlin_cuda(...)` at `marlin_kernel.cu:731` allocates the
`max_par × 64 × n` FP32 reduce buffer pre-call (per e59beb5 survey).
Eliminating this buffer is the main TTFT win mechanism for Substep 1.2
(saves the `alloc_zeros(...)` per call).

## §1 Pickup brief for codex (post Substep 1.1 commit)

```
Substep 1.2 — atomic_add reduce opt-in (~100 LOC delta)

Per Phase 1 e59beb5 substep breakdown + upstream evidence verified
this tick (raw gh api).

Pull origin/main first:
  git pull --ff-only origin main

Phase 1.1 SHOULD BE LANDED post your fmt + clippy + greedy_consistency
verification. Substep 1.2 is the next tranche.

Substep scope:

1. Add `use_atomic_add` template parameter to ARLE Marlin kernel
   (mirror marlin_template.h:268 signature pattern). Default: false
   to preserve numerical baseline.

2. Add the atomic_add reduce path in the reduce loop (mirror
   marlin_template.h:1725-1745 verbatim with namespace adjustment).
   Gate: `use_atomic_add && slice_count > 1` so single-slice
   workloads keep the existing direct-write path.

3. Host-side wiring in `marlin_cuda(...)`:
   - Read env var `INFER_MARLIN_ATOMIC_REDUCE` to determine flag value
   - When flag=true, SKIP the alloc_zeros(max_par × 64 × n) reduce
     buffer allocation
   - Pass flag through to kernel template instantiation

4. Bench A — baseline (existing direct-write path):
   PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
     scripts/bench_guidellm.sh path-b-p1-2-baseline \
       --concurrencies 4 --max-seconds 120 --warmup 10 \
       --data 'prompt_tokens=4096,...,output_tokens=256,...'

5. Bench B — treatment (atomic_add):
   INFER_MARLIN_ATOMIC_REDUCE=1 \
     scripts/bench_guidellm.sh path-b-p1-2-atomic ...

License gates per e59beb5 + skill v1.10.0 Phase 8:
- TTFT Δ ≥ -2% with σ < 5% n=3 → license atomic_add opt-in
- Greedy_consistency PASS required (atomic_add can introduce non-
  determinism in floating-point, but for FP16/BF16 single-output-col
  case the result should be deterministic since each SM writes
  to disjoint output column ranges — only `slice_count > 1` triggers
  cross-SM atomic, and even then bit-determinism depends on
  add ordering)
- Any TTFT regression > +2% → KILL specific change, keep direct-write
  default

Conservative gain estimate (per e59beb5):
- TTFT additional -2-5% (saves alloc_zeros)

What you should NOT do:
- Touch dequant.h port (Substep 1.1 territory, already landed)
- Make atomic_add the default (opt-in via env var only this tranche;
  default flip is Substep 1.3 pending license)
- Add use_fp32_reduce in same tranche (separate concern, defer to
  Substep 1.3 if 1.2 licenses)

What you SHOULD do:
- PushNotification when bench A → bench B Δ% lands
- Cite raw bench output verbatim (TTFT/ITL p50/p95 per arm) per
  skill v1.10.0 #28
- Push wins or errors entry per gate matrix
- If LICENSE: open Substep 1.3 brief (use_fp32_reduce + flip
  use_atomic_add to default-true if greedy stable)
- If KILL: errors entry citing which gate failed; Phase 1 closes at
  Substep 1.1 only

Wall time: 0.5-1 day codex.
```

## §2 Decision matrix (post Substep 1.1 outcome)

| Substep 1.1 outcome | Recommended Substep 1.2 action |
|---|---|
| LICENSE (ITL Δ ≥ -3%) | Run Substep 1.2 atomic_add per pre-draft brief above |
| Marginal (-1 to -3%) | Still run Substep 1.2 — small-win compound |
| KILL (regression > +2%) | SKIP Substep 1.2; Phase 1 closes at 1.1 KILL; pivot to #34 unblock OR P3 prefill-only FP8 |
| greedy_consistency FAIL on W4A16 (not the W4A8 known-broken default) | KILL Substep 1.1 + 1.2 entirely; revert dequant.h port |

## §3 Cross-references

- Phase 1 substep breakdown: `docs/research/2026-05-10-path-b-phase-1-vllm-marlin-port-execution-ready.md` (e59beb5)
- Substep 1.1 wins skeleton: `docs/experience/wins/SKELETON-2026-05-10-path-b-phase1-substep1.1-dequant-port.md` (48c6e49)
- Substep 1.1 audit (CLEAN): `docs/research/2026-05-10-phase1-substep1.1-codex-impl-audit-clean.md` (70b4d7b)
- W4A8 default broken model audit: `docs/research/2026-05-10-w4a8-test-model-default-broken-codex-correct-call.md` (eb2b4b6)
- #34 rescope (next-pickup queue): `docs/research/2026-05-10-task-34-rescope-substrate-exists-only-cli-surface-missing.md` (7feb260)
- Upstream marlin_template.h: https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/marlin_template.h
- Skill v1.10.0 anti-patterns + Phase 8 license matrix: `.claude/skills/kernel-optimization/SKILL.md`

## §4 Status

Substep 1.2 brief pre-drafted with verified upstream pattern. Codex
can pick up post Substep 1.1 commit with zero context-switch. Decision
matrix in §2 covers all 1.1 outcomes (license / marginal / kill /
greedy fail). Per skill v1.10.0 #28: every claim verified by raw `gh
api` output, NOT memory recall.
