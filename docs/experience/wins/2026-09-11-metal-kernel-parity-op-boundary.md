# GDR recurrent kernel op-boundary extraction — Metal, 2026-09-11

> Status: Shipped on the 4B hybrid with a wash A/B; the canonical 35B run is
> deferred (shared-box swap pressure, see Problems). PR #332.

## Goal

Confirm that extracting the compiled-model GDR recurrent dispatch
(`gated_delta_step` / `gated_delta_step_tape` in
`crates/mlx-sys/src/mlx_qwen35_model.cpp`) into a shared
`run_gated_delta_step()` launcher, to expose an op-boundary extern for parity
tests, does not change c=1 decode or prefill latency and does not change
generated tokens.

## Hypothesis

The refactor moves template-list and output-shape construction into one
helper and adds an extern entry point; the Metal kernel source, threadgroup
geometry, dtypes, and input materialization are unchanged. The scalar
step-count array is created once per forward and passed into every GDR
layer on the main path; the shared helper must take that same array rather
than build one per layer. Expected: decode TPOT and prefill TTFT within the
matched-A/B noise envelope (±10%), needle ladder identical.

## Parameters

Two release builds of `arle` (`--features metal,no-cuda,cli`), alternating
single-user serves (one model loaded at a time), same prompt set
(`scripts/bench_local_metal.py`, temp=0, ignore_eos, unique nonce prefixes):

- Baseline: `c8ce2ca5b`, `/tmp/arle-s30-baseline`
  (sha256 `26aa7175…`)
- Treatment: fixed lane HEAD, `/tmp/arle-s30-lane2`
  (sha256 `abf7120f…`)
- Model: `mlx-community/Qwen3.5-4B-MLX-4bit` — same hybrid family as the
  canonical 35B (24 linear_attention + 8 full_attention layers, GDR heads
  Hk=16/Hv=32/D=128). Substituted for the 35B because the shared box had
  18.9/20 GiB swap in use by other tenants; the 35B resource guard rejects
  startup (weights 19 GiB + headroom exceed ~14 GiB free). Forcing
  `--allow-swap` on 19 GiB of weights was not attempted (SSD-thrash freeze
  hazard documented for this box).
- Serves: `--max-running-requests 1 --system-reserve-bytes 1GiB
  --memory-budget-bytes 8GiB --allow-swap` (c=1 configuration; default
  static GDR state of 12.3 GiB is the default slot count and the fixed floor
  would not fit — 1 slot is the measured shape anyway).
- Shape: P=543/N=128, K=6 repeats, ordering B,L,L,B.

## Environment

- Host: Apple Silicon, 48 GiB RAM, macOS 25.3
- Backend: Metal, 14 GiB wired-limit clamp under residual swap
- Model: Qwen3.5-4B-MLX-4bit (2 GiB weights), c=1
- Background: 18.9 GiB swap in use by CorpLink (root) and other Claude
  sessions — swap-contended.

## Results

c=1 decode (P=543, N=128, median of 6, alternating B,L,L,B):

| arm | TPOT ms | decode tok/s | TTFT ms |
|---|---:|---:|---:|
| baseline | 43.74 | 22.9 | 807.5 |
| lane | 43.66 | 22.9 | 809.7 |

TPOT −0.17% (baseline's own between-serve spread 1.9%, lane 0.4%);
TTFT +0.27%. A wash.

Correctness — needle ladder (`scripts/lever_gate.sh`,
`mlx-community/Qwen3.5-0.8B-MLX-4bit`, lengths 115/300/446, 2 runs each):

| arm | len=115 | len=300 | len=446 |
|---|---|---|---|
| baseline | 2/2 exact | 2/2 exact | 2/2 exact |
| lane | 2/2 exact | 2/2 exact | 2/2 exact |

Both arms pass with zero misses. The 4B produces identical fixed output in
both arms but is not a needle-capable model — the gate uses the 0.8B hybrid
(18 GDR layers).

Raw artifacts: `/tmp/s30-ab-fix-results.jsonl`,
`/tmp/s30-needle-{baseline,lane}-08.log`.

## Problems

- The first A/B of the refactor showed +3.1% TPOT, larger than either arm's
  serve spread. Cause: the extracted helper built `array(T)` inside itself,
  adding one scalar array per GDR layer per step (24 on the 4B) instead of
  reusing the single per-forward `gdr_t_arr`. The op-boundary extern also
  hardcoded `threadgroup_y=(Dv+31)/32` instead of the production
  `qwen35_cpp_gdr_threadgroup_y(S)` (they agree at Dv=128 only by
  coincidence). Both were fixed: the helper takes the shared T array, and
  the extern calls the production threadgroup-y helper. After the fix TPOT
  is −0.17%. Rule retained: an op-boundary extraction must reuse the
  forward's hoisted scalar inputs, not recreate them per layer.
- The canonical Qwen3.6-35B-A3B-4bit could not be loaded on this box: 18.9
  GiB swap was already in use by a root CorpLink process and other user
  sessions, leaving ~14 GiB free against a 38 GiB fixed requirement. No
  killable process was mine to free. The A/B used the 4B model from the
  same qwen3_5 hybrid family, which runs the identical custom GDR kernels
  at the same head geometry.
- Needle on 4B is all-miss for both binaries (fixed template output), so
  forward correctness used the 0.8B CI gate model instead.

## Learnings

PASS on the 4B: c=1 TPOT −0.17% and TTFT +0.27%, needle 2/2 exact at every
length on both arms — correct inference and latency are preserved once the
shared per-forward T array is threaded through. The kernel numeric gate is
`crates/mlx-sys/tests/kernel_parity.rs` (7 tests, f64 oracles). A 35B
wall-clock confirmation remains deferred to a box without swap pressure; it
is not blocked by this change, and the 35B forward path is also exercised by
the needle gate in CI's Metal lane when a runner with headroom is available.
