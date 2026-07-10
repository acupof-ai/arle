# DSv4 CUDA SW Parity Phase 1 Stopped

## Context

Phase 1 tried to validate the Phase 0 CUDA DSv4 small-window one-token decode
surface against the CPU reference oracle for the same local
`infer/models/dsv4-mini-1B-init` checkpoint.

The intended tolerance policy was:

- Absolute tolerance: `5e-2`
- Relative tolerance: `5e-3`
- Top-1 argmax: exact match

The diagnostic command was run with the same local CUDA env required by the
Phase 0 entry:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
cargo test -p infer --features cuda dsv4_cuda_sw_one_token_parity -- --ignored --nocapture
```

The parity instrumentation was temporary and was removed before commit. It was
not left in-tree because it is a known-failing exact test and would also be
picked up by the broader Phase 0 substring filter
`dsv4_cuda_sw_one_token -- --ignored`.

## Root Cause

Runtime evidence for token `1`:

| Metric | Value |
| --- | --- |
| CPU top-1 | `73126`, logit `2.894220352` |
| CUDA top-1 | zero-logit tie, `argmax` returned `129279` |
| Top-1 match | `false` |
| Max abs diff | `2.894220352` at index `73126` |
| Max rel diff | `1.000000000` |
| CPU top-10 | `73126`, `3041`, `40195`, `38536`, `101739`, `55231`, `16113`, `55385`, `104148`, `60125` |
| CUDA top-10 | all zero logits, first indices `0..9` by sorted display |

Source evidence shows the CUDA path is still a Phase 0 shape/finite shell:

- `infer/src/model/deepseek/weights.rs:117-149` constructs the CUDA model with
  `embed_tokens=None`, `lm_head=None`, `norm=None`, `head_hc=None`, and
  `layers=Vec::new()`.
- `infer/src/model/deepseek/forward.rs:43-54` allocates
  `decode_logits` as a zero-filled device vector of `vocab_size`.
- `infer/src/model/deepseek/forward.rs:79-93` validates token range, clears
  prefill logits, advances sequence length, and returns without executing
  embedding, attention, FFN, final norm, or head projection.
- `infer/src/model/deepseek/state.rs:37-41` returns `decode_logits` when no
  prefill logits are present.

Therefore the parity failure is not a BF16 tolerance issue, host-side reduction
bug, dtype conversion bug, or SW attention numerical drift. The CUDA decode path
does not compute logits yet.

## Fix

Stopped per the Phase 1 license-or-kill gate. Do not loosen tolerance and do
not expand into prefill, CSA/HCA, MLA, MoE routing, MTP, or batch decode.

Candidate repair tranche:

1. Load the minimal tensors needed for one-token decode into CUDA memory:
   embedding, layer norms, hyper-connection weights, SW attention projections,
   shared-expert/selected MoE policy, final norm, and head.
2. Decide a licensed minimal FFN policy before implementation. CPU reference
   uses routed experts plus shared expert, so a shared-expert-only shortcut is
   not parity-equivalent.
3. Implement one layer/op class at a time with CPU-reference comparison after
   each boundary. Keep Phase 1.A separate from prefill and multi-token decode.
4. Reintroduce the ignored parity test only once the path is expected to pass
   or as an explicitly accepted failing diagnostic with a non-overlapping test
   filter.

## Rule

Shape/finite smoke is not a numerical substrate. The next licensed unit must
make CUDA decode compute real logits before any Phase 0.5 prefill, MLA, MoE, or
MTP work builds on top of it.
