# DSv4 issue #104+ cleanup: delete tree verify and debug-only paths

## Goal

Close the 2026-06-16 DSv4 issue cluster (#104, #105, #107, #110-#114) with a
deletion-first cleanup: make `--spec-type mtp` self-sufficient, remove the
unused strict tree-verify/debug plumbing, keep verify acceptance matrix-based,
and harden the bench/eval scripts that produced noisy or incomplete reports.

## Hypothesis

The current MTP implementation is a chain, not a branching tree. The old
ancestor-mask/SW-ring tree attention and one-off dump/probe env gates are
dead complexity: deleting them should leave the canonical chain verify path
clearer, with the target logits matrix as the verify source of truth.

## Params

- Local gate: `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`
- Script gate: `python3 -m py_compile scripts/bench_dsv4_trace_http.py scripts/dsv4_c_sweep.py scripts/dsv4_sparse_adversarial_probe.py`
- Remote perf/correctness: pending H20 run on the gcc-13 MTP lane; the current
  Mac host cannot execute DSv4 CUDA kernels. `~/bin/pod` was checked, but the
  visible checkout was old/dirty (`2f021c0`) and an OPD CUDA run/build was active,
  so it was not a clean DSv4 gate.

## Env

- Host: macOS no-nvcc typecheck path.
- CUDA feature shape: `cuda,no-cuda`, `infer-api` lib only.
- H20 pod verification: pending-remote; current pod was not clean/idle enough
  for an attributable DSv4 result.

## Results

- Deleted `Dsv4TreeAttnMeta`, the tree FlashMLA index builder FFI/kernel, and
  tree/ancestor schedule fields. Verify is chain-only and per-row causal.
- Deleted DSv4 dump/probe-only code paths for attention/CSA/KV/tail/MTP rollback.
- MTP CLI default draft depth is d2; explicit `--mtp-draft-tokens` still wins.
- Verify now returns a `SpecVerifyResult` with the full target logits matrix,
  the greedy top-1 view, and per-row hidden states. Non-greedy requests fall
  back to baseline decode instead of erroring through greedy-only MTP.
- `scripts/bench_dsv4_trace_http.py` now targets text `/v1/completions` SSE.
- `scripts/dsv4_c_sweep.py` supports repeats and median/spread summaries.
- Added `scripts/dsv4_sparse_adversarial_probe.py` with forced `FINAL:` answers
  and robust extraction for CSA/HCA adversarial probes.

## Problems

- H20 runtime verification is still required for DSv4 correctness/perf:
  needle/self-consistency for the deleted tree path, d2 default, batched MTP
  engagement at c>=4, and the 3-repeat compressor-batch sweep.
- The cleanup intentionally did not default-flip the remaining proven-on env
  gates; that needs separate same-binary A/B evidence per gate.

## Learnings

Tree-width had already been deleted; keeping strict tree-attention plumbing
around a chain made the code harder to audit without adding acceptance power.
The canonical path is now: draft a chain, run target verify, keep the logits
matrix, accept from its greedy view for greedy decode, and fall back for
sampling until rejection sampling is implemented deliberately.
