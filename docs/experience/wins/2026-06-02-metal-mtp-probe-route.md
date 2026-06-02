# Metal MTP Probe Route

## Context

MTP support needs a local Metal route that can be verified before any draft or
verify loop lands. The online implementations converge on the same shape: MTP is
a target-model-attached draft head, not a generic separate draft model. It
combines the target hidden state with the current token embedding, runs MTP
decoder layer(s), then verifies with the target model.

Primary source survey:

- vLLM exposes `speculative_config={"method":"mtp","num_speculative_tokens":1}`
  and implements Qwen MTP with `mtp.fc`, pre-FC RMSNorms, MTP decoder layers,
  and target logits verification.
- SGLang's Qwen3.5/Frozen-KV MTP path feeds `forward_batch.spec_info.hidden_states`
  into the MTP module and disables overlap scheduling / mixed chunked prefill.
- llama.cpp's `draft-mtp` route asks the target context for pre-norm embeddings
  and advances draft-side state only according to accepted tokens.

## What Worked

Implemented a Metal-only probe route:

- `metal_serve --spec-type auto`
- `metal_serve --spec-type mtp`
- `arle serve --backend metal --spec-type mtp`

The route scans safetensors index/header tensor names for `mtp.*`, `.mtp.`,
`nextn`, or `next_n` and reports the result during Metal backend load. It does
not change decode behavior and falls back to standard Metal decode until the
real MTP draft/verify implementation exists.

## Verification

Unit tests:

```text
cargo test -p infer --no-default-features --features metal mtp -- --nocapture
  4 passed

cargo test -p cli spec_type -- --nocapture
  2 passed

cargo build -p infer --no-default-features --features metal --bin metal_serve
  passed

cargo clippy -p infer --no-default-features --features metal --bin metal_serve -- -D warnings
  passed

cargo clippy -p cli -- -D warnings
  passed
```

Real canonical Metal model probe:

```text
RUST_LOG=info timeout 30s target/debug/metal_serve \
  --model-path /Users/bytedance/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --spec-type auto \
  --warmup 0 \
  --port 8123

Metal MTP auto: no mtp/nextn tensors detected; using standard Metal decode
```

The process was timeout-scoped and no `metal_serve` process remained afterward.

## Rule

Do not land fake MTP acceptance. The local Metal route may expose and verify
weight readiness first, but real speedup requires MTP-preserving weights,
target-hidden-state plumbing, draft-side state commit/rollback, and target
verification before any benchmark claim.

No guidellm run was attached because this tranche is a probe/CLI route and does
not alter generation behavior or claim a performance win. Commit under test:
`a3818be7` plus this working-tree change.
