# M_b.1 — BF16 TileLang Split-KV Substrate

## Context

M_b.1 extends the existing HD128 BF16 TileLang decode path with a
FlashDecoding-style split-KV substrate. The user guard rail for this tranche:
ship runtime dispatch only if BF16 split-KV ties or beats the current BF16 path
at longctx 4k/c=4; otherwise land the substrate but keep the runtime path-cut
gated.

## What Worked

- Added TileLang partial + merge phase kernels for HD128 BF16 decode,
  `kMaxSplits=16`, for the existing `(q_heads, kv_heads)` configs.
- Expanded `TileLangWorkspace` from ZST to optional real split buffers, while
  keeping default metadata/workspace allocations zero-sized.
- Wired FFI/build generation in lockstep.
- Runtime dispatch is opt-in via `INFER_TILELANG_BF16_SPLIT_KV=1` and still
  falls back to the single-kernel path below the split threshold.
- Default serving remains on the existing BF16 path, so this lands as substrate
  without a production path-cut.

## Bench Status

Status: landed-substrate, runtime default off.

Baseline entries already recorded:
- `2026-05-07-m_b22-vllm-longctx-baseline.md`
- `2026-05-07-m_b1-phaseB-no-highconc-delta.md`

No default-path macro bench is claimed here because default runtime behavior is
unchanged. The opt-in split-KV path still needs the canonical BF16 longctx
4k/c=4 license run before it can become default-on.

## Verification

- `cargo check --release -p infer --features cuda`
- `cargo clippy --release -p infer --features cuda -- -D warnings`
- `cargo test --release -p infer --features cuda --test e2e`
- `cargo test --release -p infer --features cuda --test greedy_consistency`
- `cargo check --release -p infer --no-default-features --features cuda,no-cuda`

## Rule

For kernel substrate with uncertain macro-bench value, keep the generated
kernels and workspace plumbing mergeable, but make the runtime path-cut explicit
and opt-in until the shape-specific bench earns default-on.
