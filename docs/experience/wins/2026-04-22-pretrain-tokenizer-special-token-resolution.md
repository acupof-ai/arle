# `pretrain` now resolves BOS/EOS from the tokenizer instead of hardcoding Qwen IDs

## Context

The scratch `pretrain` entrypoint used Qwen BOS/EOS defaults (`151643/151645`)
even when the run pointed at a non-Qwen tokenizer. That made external
tokenizer experiments look cleaner than they really were: training ran, but
the synthesized `config.json` / `generation_config.json` carried the wrong
special-token semantics.

This surfaced immediately on the GPT-Neo tokenizer line used for the aligned
QA experiments. The run should have been using `<|endoftext|>` for both BOS
and EOS (`50256`), but the old code still wrote Qwen IDs unless the caller
manually overrode them.

## What Worked

- Added generic special-token lookup helpers to
  `crates/train/src/tokenizer.rs`:
  - token/id lookup
  - explicit token or id override validation
  - inference from a short list of common BOS/EOS token strings
- Changed `crates/train/src/bin/pretrain.rs` so BOS/EOS are no longer
  hardcoded defaults. The precedence is now:
  - explicit `--bos-token-id` / `--eos-token-id`
  - explicit `--bos-token` / `--eos-token`
  - tokenizer-based inference
  - BOS falls back to EOS when the tokenizer only has one shared stop token
    (for example GPT-Neo / GPT-2 style `<|endoftext|>`)
- Fixed the qwen3.5 pretrain checkpoint save path to synthesize
  `generation_config.json` from the resolved BOS/EOS pair directly instead of
  the old `eos_token_id.max(cfg.eos_token_id)` style fallback.
- Emitted the resolved special tokens into the `run_start` metrics and the
  startup log so future training runs expose the actual BOS/EOS pair.

## Verification

- `cargo test -p train --release --lib`
- `cargo test -p train --release --bin pretrain`
- `cargo clippy -p train --release --bin pretrain --lib -- -D warnings`

GPT-Neo tokenizer smoke:

- Command:
  `cargo run -p train --release --features metal --bin pretrain -- --model-family qwen35 --corpus /tmp/dolly_chat_exp_mid/plain_qa_corpus.txt --tokenizer /tmp/gptneo_tokenizer/tokenizer.json --out /tmp/chat_qa_gptneo23m_400 --steps 400 --batch 1 --seq 128 --lr 1e-3 --grad-accum-steps 4 --log-every 50 --save-every 100 --backend metal --hidden 320 --layers 4 --heads 5 --kv-heads 5 --head-dim 64 --intermediate 1280 --max-pos 512 --metrics-jsonl /tmp/chat_qa_gptneo23m_400/metrics.jsonl`
- Startup now logs the correct GPT-Neo special-token pair:
  - `bos=50256 (Some("<|endoftext|>"))`
  - `eos=50256 (Some("<|endoftext|>"))`
- The run stayed internally consistent through checkpoint save/load and
  `metal_request` inference on the generated checkpoint dirs.

Aligned QA experiments after the fix:

- `23.05M` params, `400` steps, Metal:
  - stable special-token resolution
  - `loss 10.91 -> 3.31 @ step 350 -> 4.28 @ step 400`
  - steady throughput about `328 tok/s`
- `3.50M` params, `800` steps, Metal:
  - same tokenizer resolution
  - `loss 10.79 -> 3.78 @ step 700 -> 3.83 @ step 800`
  - steady throughput about `1376 tok/s`

The important part is not that either QA run became a usable chat model; they
did not. The important part is that these experiments are now measuring the
training recipe itself rather than a hidden Qwen-tokenizer mismatch.

## Rule

When a train entrypoint accepts an arbitrary tokenizer, special tokens must be
resolved from that tokenizer or from explicit CLI overrides. Never smuggle in
family-specific BOS/EOS defaults and then treat the resulting experiment as a
clean tokenizer comparison.
