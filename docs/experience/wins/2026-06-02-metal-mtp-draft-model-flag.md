# Metal MTP Draft Model Flag

## Context

`mlx-community/Qwen3.6-35B-A3B-MTP-4bit` is not a replacement target model.
It is a split MTP drafter checkpoint:

- `config.json` declares `model_type = qwen3_5_mtp`
- `block_size = 3`
- `model.safetensors.index.json` contains 46 drafter tensors such as
  `fc.weight`, `layers.0.*`, `pre_fc_norm_embedding.weight`, and
  `pre_fc_norm_hidden.weight`

Directly passing it as `--model-path` fails correctly because it does not
contain the target Qwen3.6 text weights.

## What Worked

Added an explicit Metal MTP drafter flag:

```text
--mtp-draft-model mlx-community/Qwen3.6-35B-A3B-MTP-4bit
```

The flag implies the Metal MTP route and is mutually exclusive with DFlash.
`metal_serve` now resolves/downloads the drafter, validates the split MTP
shape, and logs the resolved local path. `arle serve --backend metal` forwards
the flag to `metal_serve`; non-Metal backends reject it.

Native MTP draft/verify is still not implemented in this tranche, so the
runtime intentionally falls back to standard Metal decode after validating the
drafter.

## Verification

Unit and lint:

```text
cargo test -p infer --no-default-features --features metal mtp -- mtp --nocapture
  7 passed

cargo test -p cli mtp -- --nocapture
  2 passed

cargo clippy -p infer --no-default-features --features metal --bin metal_serve -- -D warnings
  passed

cargo clippy -p cli -- -D warnings
  passed

cargo check -p infer --no-default-features --features no-cuda -q
  passed
```

Real local startup:

```text
RUST_LOG=info timeout 60s target/debug/metal_serve \
  --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
  --mtp-draft-model mlx-community/Qwen3.6-35B-A3B-MTP-4bit \
  --warmup 0 \
  --port 8127

Metal MTP external draft model resolved: requested=mlx-community/Qwen3.6-35B-A3B-MTP-4bit path=/Users/bytedance/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-MTP-4bit/snapshots/0295b81421bf4d0fccca9a7c0fcfb1418dda3516 model_type=qwen3_5_mtp block_size=3 tensors=46 [...]
Metal server listening on 127.0.0.1:8127 (mlx-community/Qwen3.6-35B-A3B-4bit)
```

Installed CLI forwarding:

```text
RUST_LOG=info timeout 45s arle serve \
  --backend metal \
  --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
  --mtp-draft-model mlx-community/Qwen3.6-35B-A3B-MTP-4bit \
  --port 8128 \
  -- --warmup 0

[ARLE serve] launching metal backend via /Users/bytedance/.local/bin/metal_serve
Metal MTP external draft model resolved: requested=mlx-community/Qwen3.6-35B-A3B-MTP-4bit [...]
Metal server listening on 127.0.0.1:8128 (mlx-community/Qwen3.6-35B-A3B-4bit)
```

Both processes were killed after startup verification; no `metal_serve` process
remained.

No guidellm run was attached because generation behavior is unchanged.

## Rule

Treat split MTP checkpoints as draft models, never as target replacements.
Startup support is useful readiness evidence, but performance claims require
real target-hidden-state plumbing, drafter forward, target verify, and
accept/rollback accounting.
