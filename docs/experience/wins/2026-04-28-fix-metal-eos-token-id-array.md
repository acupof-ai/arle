# Mac Metal inference: HF-spec eos_token_id resolver fixes Qwen3.6 multimodal MoE chat-stop

## Context

User reported `arle` REPL on macOS produced clean output for turn 1 but
degenerate output thereafter when running `mlx-community/Qwen3.6-35B-A3B-4bit`
(MoE multimodal, `Qwen3_5MoeForConditionalGeneration`):

- Turn 1 (`你好`): 83 tok / 8.1 tok/s, clean reply.
- Turn 2 (`你好`): 92 tok / 62.8 tok/s, role markers (`user`, `assistant`)
  leaking as plain text between repeated copies of the same greeting.
- Turn 3 (`测试一下你的所有工具`): identical garbage — model ignored the
  new prompt entirely.

## What Worked

Diagnosis: the model's `config.json` has `eos_token_id` as the array
`[248046, 248044]` at the root (chat `<|im_end|>` + `<|endoftext|>`) but a
scalar `248044` inside `text_config`. The Metal config loader fell back to
`text_config` (multimodal nesting), then `get_eos` extracted only the first
element — picking `248044`. The C++ generate path (`qwen35.rs:2387`,
`generate.rs:104`) passed only `config.eos_token_id` (a single `u32`) into
the C++ stop check, so the model never stopped at `<|im_end|>` (248046) and
walked past it generating fake new turns. With the tokenizer decoding via
`skip_special_tokens=true`, `<|im_start|>` / `<|im_end|>` decoded to empty
strings — leaving the literal "user" / "assistant" role names visible.
Generation eventually halted only when the model emitted `<|endoftext|>`
(248044), which IS in the (single-element) stop list.

Fix (universal, not model-specific):

1. **`infer/src/backend/metal/config.rs`** — replace `get_eos` +
   `load_stop_token_ids` with a generic HuggingFace-precedence resolver:
   - Read `eos_token_id` from `generation_config.json` (HF inference-time
     authority), then `config.json` root, then `text_config`.
   - Accept either scalar OR array uniformly.
   - Merge sources, dedup, fall back to 151645 if all empty.
   - `MetalModelConfig.eos_token_id: u32` is now derived as the first id
     from the resolved array — `stop_token_ids` is the authoritative list.
2. **`infer/src/backend/metal/qwen35.rs`** + **`generate.rs`** — the C++
   generate paths now pass `config.stop_token_ids` (full list) into the C++
   stop check, not just `config.eos_token_id`. Sort + dedup after merge.

Verification (same machine: Apple M4 Max, Metal):

```
printf "你好\n你好\n测试一下你的所有工具\n/quit\n" | arle ... --max-tokens 512
```

| Turn | Input | Output | tok | tok/s |
|---|---|---|---|---|
| 1 | 你好 | 你好！有什么我可以帮你的吗？ | 81 | 11.2 |
| 2 | 你好 | 你好！有什么我可以帮你的吗？ | 13 | 24.1 |
| 3 | 测试一下你的所有工具 | 工具测试完成。Python 工具运行正常。 | 267 | 64.6 |

All three turns now produce on-topic, EOS-terminated replies. No role-marker
leakage. Tool execution works on turn 3.

Unit tests added covering the HF-spec resolver:

- `resolves_stop_tokens_following_hf_precedence` — generation_config wins,
  union with root + text_config, dedup preserves first-seen order.
- `resolves_stop_tokens_handles_scalar_and_missing_generation_config`.
- `resolves_stop_tokens_falls_back_when_nothing_specified` (151645).

`cargo test --release --no-default-features --features metal -p infer` →
481 passed. `cargo clippy ... -D warnings` clean.

## Rule

When a HuggingFace model's config triggers an inference bug, fix the
**generic HF spec** (precedence: `generation_config.json` → `config.json`
root → `text_config`; scalar-or-array fields normalized to lists), not the
specific model's symptom. Single-element fields like `eos_token_id` that
flow through to stop checks must always be backed by the full resolved
array — never pass a scalar where a list of stop ids is required.

Bench scope: this is a correctness fix on the Metal C++ generate path. No
throughput impact expected; the same path is now correctly terminated on
chat-end tokens. Throughput regression check on the affected backend is
covered by the existing Qwen3.5/3.6 baselines under `wins/2026-04-27-*.md`
— rerun if any throughput regression is suspected.
