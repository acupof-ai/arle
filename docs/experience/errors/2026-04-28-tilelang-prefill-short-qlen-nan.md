# TileLang prefill HD128 emits NaN logits for short prompts (qlen < BLOCK_M)

## Context

The TileLang prefill HD128 kernel
(`crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`) was
made the default Qwen3 prefill path on 2026-04-28 via
`docs/experience/wins/2026-04-28-bench-guidellm-cuda-l4-tilelang-prefill-causal-bound.md`.
The wins entry validated correctness only at `qlen=4096` (the canonical
guidellm bench prompt). The e2e regression test
(`cargo test --release -p infer --test e2e --features cuda`) uses
4-token prompts ("Tell me a story", "My name is", etc.) and started
emitting `"!!!!!!!!!!"` — token id 0 — for every output position.

## Root cause

For partial Q-tiles where `qlen < BLOCK_M=64` (every prompt under 64
tokens, including chat / few-token user inputs), invalid rows
(`row0+i >= qlen`) accumulate `m_i = -inf` initially and `scores = -inf`
after the bounds-mask. The running-softmax delta then collapses to
`(-inf) - (-inf) = NaN` at line 195 (`p[i,j] = exp2((scores - m_new) *
log2e)`), and `T.GemmWarpPolicy.FullRow` propagates that NaN through
the `p_bf16 @ v_tile` mma into VALID rows' `acc_o` in the same warp
tile (BLOCK_M=64 rows / 4 warps × 16 rows per warp; tensor-core mma
processes a 16x8 or 16x16 tile at a time, so NaN in one row pollutes
the entire output tile). The final RMSNorm + LM-head GEMV then
produces uniform-NaN logits → softmax uniform → argmax = token 0 = `"!"`.

The kernel was bench-only validated at `qlen=4096` so the partial-tile
regime was never exercised before e2e short-prompt verification.

## Fix attempted (failed) and resolution

**Attempted kernel fixes (none of which produced different logprobs)**:

1. Guard the final divide for invalid rows
   (`safe_l = if row<qlen then l_i else 1.0`).
2. Initialize `m_i` for invalid rows to a finite value (0 or -1e9)
   instead of `-T.infinity()`.
3. Replace the `-T.infinity()` score-mask sentinel with finite `-1.0e9`.

After hard-rebuilding cubins (`rm -rf target/release/.../tilelang_aot/...`
+ `touch crates/cuda-kernels/build.rs`) and verifying the cubin file
mtime advanced, the running model still emitted *identical*
deterministic logprobs (`-2.262328, -0.830096, -0.814637`) — strongly
suggesting either cargo's incremental build was caching the cubin under
a content hash that didn't change, or the AOT generator was rebuilding
from a stale source path. Three good-faith attempts produced no
behavioral change, hitting the **2-strike-then-revert** rule from
`CLAUDE.md`.

**Resolution (2026-04-28)**: REVERTED `tilelang-attn` from the default
`cuda` feature. FlashInfer prefill HD128 is the default again.
Re-enable explicitly via `--features cuda,tilelang-attn` only when
prompts are guaranteed `qlen >= 64` (e.g. the canonical
`scripts/bench_guidellm.sh --fast` 4096-in shape). The proper kernel
fix is filed as task #26.

## Rule

- **A bench is not a correctness test.** Performance numbers can look
  perfect while output is gibberish. Greedy-text comparison against an
  HF reference (or even just a "first non-special token must match a
  known good list") MUST gate every kernel change before promoting to
  default. Add this gate to wins-entry CI / verification checklist.
- **Validate kernel correctness across `qlen` regimes.** Partial Q-tile
  paths (`qlen < BLOCK_M`) and full-tile paths (`qlen >= BLOCK_M`)
  exercise different reduction code in TileLang. A kernel passing the
  4096-in bench may still NaN at qlen=1, 4, 16, 32, or 63. Always
  bench at `qlen ∈ {1, 4, BLOCK_M-1, BLOCK_M, BLOCK_M+1, 4096}` before
  promotion.
- **AOT cubin rebuild is opaque.** When iterating on a TileLang kernel,
  do NOT trust `mtime` of the cubin file as evidence the kernel
  changed. Either disassemble the cubin (`cuobjdump --dump-sass`) and
  diff, or insert a marker constant the kernel reads/writes that's
  observable from the host side. Otherwise iteration fails silently.
- **Two-strike rule.** When two good-faith kernel patches don't move
  the observable, switch from "patch the kernel" to "ship the
  workaround now, debug the kernel offline". Don't keep stacking
  attempts — the user is blocked.
