# DSv4 @ --max-total-tokens 16384: solo needle truncation (`738.`) — pre-existing, NOT Phase 3b

## Context

The Phase 3b gate ran the first-ever needle correctness lanes at
`--max-total-tokens 16384` (the config was previously uncoverable: the slot
cliff left 1-3 slots and the only 16384 evidence was guidellm throughput —
no needle checks). Multi-shape sweep found deterministic solo misses:
`s1-2000` 0/3 and `s1-8000` 0/3 exact on the 3b binary, always the
`"The secret access code is 738."` early-truncation signature (needle
738291). Same prompts at `--max-total-tokens 2048` pass 59/60 (E6) and
15/15 (E1).

## Attribution (same-day, same-salt, pre-3b control)

fc850c7c6 (pre-3b, whole-band identity allocation, 3 slots at 16384) on the
same box/GPUs/salts: `s1-2000` **0/3 exact**, `s1-8000` **1/3 exact** —
same signature, same determinism. n=4 arms at 16384 mostly pass (3/4-4/4).

Verdict: **pre-existing 16384-config correctness bug**, independent of the
3b demand-paging change. Suspect class: the deep-position content bug
(#146, `>2048 garble = deep-pos CONTENT`) or a max_seq-scaled state
(indexer/DSA sizing) interacting with the needle position — NOT root-caused
here; the 3b gate only needed the 3b-vs-pre-3b attribution.

## Rule

- A "new" failure on a config the gate exercises for the first time needs a
  baseline-control run on that config before it can indict the change under
  test — the pre-3b control cost one serve boot and settled it.
- Throughput-only coverage (guidellm) of a config is NOT correctness
  coverage; the first needle lane at 16384 found a deterministic
  truncation that months of benches never saw.
