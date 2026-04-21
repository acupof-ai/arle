# CPU backend now honors tokenizer-aware `max_tokens` budgets

## Context

While re-running the current train → eval → serve smoke on 2026-04-21, the
CPU dev backend violated the OpenAI-compatible completion contract.

Request:

```json
{"prompt":"hello","max_tokens":8,"temperature":0.0}
```

Observed before the fix:

- response `finish_reason = "length"`
- `usage.completion_tokens = 16`

The bug only showed up when a tokenizer was loaded. The CPU backend clipped
the synthetic response by whitespace words, then counted completion tokens
with the tokenizer, so the text budget and token budget drifted apart.

## What Worked

- Changed `infer/src/backend/cpu.rs` to enforce the completion budget with the
  loaded tokenizer when available.
- Kept the old whitespace clipping as a fallback for tokenizer-less CPU smoke
  models.
- Added a regression test that builds a tiny local word-level tokenizer and
  asserts `completion_tokens == max_new_tokens`.
- Re-ran the full local smoke on current code:
  - `pretrain`
  - `eval_lm`
  - `cpu_serve`
  - `/v1/completions`
  - `/v1/chat/completions`

## Verification

- `cargo test -p infer --release --no-default-features --features cpu,no-cuda --lib backend::cpu -- --nocapture`
- `cargo clippy -p infer --no-default-features --features cpu,no-cuda --lib -- -D warnings`
- `cargo test -p infer --release --no-default-features --features cpu,no-cuda --lib`
- `cargo fmt --all --check`
- `git diff --check`
- `cargo run -p train --release --bin pretrain -- --model-family qwen35 --corpus crates/train/data/sample.txt --tokenizer models/Qwen3-0.6B/tokenizer.json --out /tmp/agent_infer_pretrain_e2e_cpu_fix --steps 2 --batch 1 --seq 16 --lr 1e-4 --log-every 1 --save-every 2 --hidden 16 --layers 2 --heads 2 --kv-heads 1 --head-dim 8 --intermediate 32 --max-pos 32 --seed 123`
- `cargo run -p train --release --bin eval_lm -- --model-family qwen35 --model-path /tmp/agent_infer_pretrain_e2e_cpu_fix/latest --data /tmp/agent_infer_pretrain_eval.jsonl --seq-len 32`
- `cargo build -p infer --release --no-default-features --features cpu,no-cuda --bin cpu_serve`
- `curl -s http://127.0.0.1:18092/v1/completions ...` → `completion_tokens = 8`
- `curl -s http://127.0.0.1:18092/v1/chat/completions ...` → `completion_tokens = 8`

## Rule

If a backend reports tokenizer-based usage, its generation budget must also be
tokenizer-based. Mixing word clipping with tokenizer accounting makes
OpenAI-compatible `max_tokens` look correct at the HTTP layer while breaking
the contract at the backend boundary.
