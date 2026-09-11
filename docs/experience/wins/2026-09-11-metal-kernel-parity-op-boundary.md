# GDR recurrent kernel op-boundary extraction — Metal, 2026-09-11

> Status: Shipped (S30, PR #332)

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
geometry, dtypes, and input materialization are unchanged. Expected:
decode TPOT and prefill TTFT within the matched-A/B noise envelope
(±10%), needle ladder identical.

## Parameters

Two release builds of `arle` (`--features metal,no-cuda,cli`), alternating
single-user serves (one model loaded at a time), same prompt set
(`scripts/bench_local_metal.py`, temp=0, ignore_eos, unique nonce prefixes):

- Baseline: `c8ce2ca5b`, `/tmp/arle-s30-baseline`
  (sha256 `26aa7175…`)
- Treatment: lane HEAD, `/tmp/arle-s30-lane`
  (sha256 `02297429…`)
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
- Shapes: prefill P=543/N=128 K=6 repeats (decode TPOT), two matched
  prefill pairs at P=4127 in both orderings.
- Ordering B,L,L,B at P=543; reversed L,B at P=4127 to cancel time drift.

## Environment

- Host: Apple Silicon, 48 GiB RAM, macOS 25.3
- Backend: Metal, 14 GiB wired-limit clamp under residual swap
- Model: Qwen3.5-4B-MLX-4bit (2 GiB weights), c=1
- Background: 18.9 GiB swap in use by CorpLink (root) and other Claude
  sessions — swap-contended; adds noise to prefill.

## Results

c=1 decode (P=543, N=128, median of 6, alternating pairs):

| arm | TPOT ms | decode tok/s | TTFT ms |
|---|---:|---:|---:|
| baseline | 46.58 | 21.5 | 807 |
| lane | 48.05 | 20.8 | 805 |

Decode TPOT +3.1% (within the ±10% matched-A/B bound; baseline's own
between-serve spread is 1.8%, lane 1.7%); TTFT −0.2%.

Prefill at P=4127, one pair in each order: lane −2.9% when run after
baseline, +8.8% when run first — the spread is time drift on the
swap-contended box, not a systematic delta. Decode tok/s derived from the
P=543 pairs only (the 1–7 token differences at P=4127 are dominated by
post-prefill scheduling).

Correctness — needle ladder (`scripts/lever_gate.sh`,
`mlx-community/Qwen3.5-0.8B-MLX-4bit`, lengths 115/300/446, 2 runs each):

| arm | len=115 | len=300 | len=446 |
|---|---|---|---|
| baseline | 2/2 exact | 2/2 exact | 2/2 exact |
| lane | 2/2 exact | 2/2 exact | 2/2 exact |

Both arms pass with zero misses. The 4B produces identical fixed output in
both arms (byte-identical 6/6) but is not a needle-capable model — the
gate uses the 0.8B hybrid (18 GDR layers).

Raw artifacts: `/tmp/s30-ab-results.jsonl`, `/tmp/s30-needle-{baseline,lane}-08.log`.

## Problems

- The canonical Qwen3.6-35B-A3B-4bit could not be loaded on this box: 18.9
  GiB swap was already in use by a root CorpLink process and other user
  sessions, leaving ~14 GiB free against a 38 GiB fixed requirement. No
  killable process was mine to free. The A/B therefore used the 4B model
  from the same qwen3_5 hybrid family, which runs the identical custom GDR
  kernels at the same head geometry.
- Prefill latencies at P=4127 varied by ~9% with serve ordering; the
  matched ordering shows it is drift rather than a treatment effect.
- Needle on 4B is all-miss for both binaries (fixed template output), so
  forward correctness used the 0.8B CI gate model instead.

## Learnings

PASS. The op-boundary extraction is a wash on c=1 decode (+3.1% TPOT,
−0.2% TTFT, inside the matched-A/B envelope) and the needle ladder is
2/2 exact at every length on both arms — correct inference is preserved.
The numeric parity for the kernels themselves is the op-boundary gate in
`crates/mlx-sys/tests/kernel_parity.rs` (7 tests, f64 oracles). A 35B
wall-clock confirmation remains blocked by shared-box swap pressure, not
by this change; the 35B forward path is already covered by the needle gate
in CI's Metal lane when a runner with headroom is available.
