# Pretrain Alias Retired

## Context

`crates/train` still carried both `pretrain` and `pretrain_qwen3` as Cargo
bins pointing at the same source file. The runtime itself was already generic
Qwen-family code, but the extra alias kept every `cargo` invocation noisy and
left active docs half in the old naming.

## What Worked

- Renamed the canonical source file to `crates/train/src/bin/pretrain.rs`.
- Removed the `pretrain_qwen3` Cargo bin alias instead of keeping two targets
  for the same implementation.
- Refreshed active architecture/test/codebase docs so they now point at the
  single `pretrain` entrypoint.
- Re-ran the real chain with the canonical name:
  `pretrain -> eval_lm -> cpu_serve`.

## Rule

Once a train entrypoint is the only supported surface, keep exactly one Cargo
bin and one source file name for it. Do not keep compatibility aliases that
only create warning noise.
