# ChatML Base + SFT Probe

## Context

After the plain `User:/Assistant:` toy runs stayed template-heavy, the training
line was moved onto the same ChatML convention already used by the repo's SFT
and serving path. The goal here was not a final quality claim; it was to verify
that the mainstream small-model route in this tree is coherent:

1. build a reusable byte-level BPE tokenizer
2. pretrain a small base model on ChatML-formatted text
3. run assistant-only SFT on top of that base
4. validate the saved checkpoints through `infer`

## What Worked

- Built a ChatML-aligned Dolly corpus under `/tmp/dolly_chatml/` and a `4k`
  byte-level BPE tokenizer with explicit ChatML special tokens.
- Pretrained a tiny `qwen35` base model on Metal:
  - model: `hidden=128, layers=4, heads=4, kv_heads=4, head_dim=32, ffn=512`
  - params: `1.64M`
  - corpus: `2,838,212` tokens
  - run: `/tmp/dolly_chatml_base_1p64m`
  - result: `3000` steps, final logged loss `4.544847`, steady throughput
    around `4.2k tok/s`
- Verified `train_sft` on top of that base works on Metal with the repo's
  existing ChatML masking path and `cosine-with-warmup` schedule.
- Verified SFT checkpoints are directly loadable by `infer`:
  `target/release/metal_request --model /tmp/dolly_chatml_sft_probe2/step_000001 ...`
  succeeded and loaded the merged base + adapter checkpoint layout.
- Ran a longer SFT probe under `/tmp/dolly_chatml_sft_1000` and confirmed the
  `step_000500` checkpoint saves cleanly and answers in assistant-style text
  instead of raw continuation noise.

## Rule

- Keep tokenizer, base-pretrain corpus, SFT template, and serving prompt format
  on the same ChatML convention. Do not mix plain role-prefix corpora with a
  ChatML serving path and expect stable chat behavior.
- For small-model chat work in this repo, use the same staged path strong
  training teams use in practice: `base pretrain -> assistant-only SFT ->
  held-out prompt checks at intermediate checkpoints`.
