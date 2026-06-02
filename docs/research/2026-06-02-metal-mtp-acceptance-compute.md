# Metal MTP Acceptance Compute Notes

Date: 2026-06-02
Scope: Metal Qwen3.6 Frozen-KV MTP

## Bottom-level compute

ARLE's MTP draft path builds one speculative block as:

1. Start with the current committed token and the target verifier seed hidden.
2. For each draft suffix step:
   - embed the current token with the target embedding table;
   - reshape both token embedding and seed hidden to `[1, hidden]`;
   - RMSNorm both inputs with MTP-specific pre-FC norms;
   - concatenate to `[1, 2 * hidden]`;
   - project through MTP `fc` back to hidden size;
   - run one full-attention Qwen3.6 MoE draft layer;
   - final RMSNorm, target `lm_head`, greedy sample;
   - feed sampled token and draft hidden to the next draft suffix step.

The draft attention is Frozen-KV:

- it reads target full-attention KV pair 0 up to `target_cache_len`;
- it computes draft K/V for the current draft token inside the transient graph;
- it concatenates target prefix KV with the transient draft K/V for attention;
- it does not commit draft K/V to target KV.

The RoPE phase is `target_cache_len - 1`, matching the last committed target
slot. That is the contract copied from SGLang Frozen-KV MTP.

Target verify then forwards the whole `[current, draft...]` block through the
target C++ model at `cache_pos = target_cache_len`. The C++ verifier samples
target logits for every row and computes:

```text
matched_prefix = count_prefix_i(target_sampled[i] == block_tokens[i + 1])
accepted_inputs = matched_prefix + 1
next_token = target_sampled[matched_prefix]
```

So `accepted_inputs == 1` means the current input was target-verified and the
posterior target token is emitted, but the draft suffix saved zero target rows.

## Acceptance implications

MTP only helps when suffix acceptance is high enough to amortize the extra
draft-layer work and longer target verify block. Low acceptance has two costs:

- draft work is paid but discarded;
- target verify still runs a multi-token block, then rolls back rejected GDR
  state.

This is why packed verify alone is not a complete fix. Packing lowers
multi-request verifier overhead, but it does not make a low-quality draft suffix
useful.

## Stabilization policy

The first safe stabilization is per-request adaptive fallback:

- track consecutive blocks where `accepted_inputs == 1`;
- after a threshold, pause MTP for a cooldown window;
- run standard target decode during cooldown;
- keep capturing MTP seed hidden during fallback so MTP can resume with a fresh
  target hidden state.

Default local policy:

```text
ARLE_METAL_MTP_ZERO_ACCEPT_LIMIT=4
ARLE_METAL_MTP_COOLDOWN_TOKENS=16
ARLE_METAL_MTP_ADAPTIVE=1
```

Set `ARLE_METAL_MTP_ADAPTIVE=0` to disable it.

## Ngram fallback

SGLang's NGRAM path is not a simple MTP parameter. It is a separate speculative
worker with a CPU corpus/trie, draft-token retrieval, tree mask construction,
target tree verify, accepted-token fill, KV slot free/move, and corpus updates.

Ngram is useful for repeated text, code patterns, JSON, copied context, and
external-corpus matches. It is not a universal replacement for a model drafter.
For ARLE, the clean design is a second draft source:

```text
MTP active if recent suffix acceptance is good.
Ngram active if the local trie has a long enough match.
Standard target decode if neither source is licensed for the current request.
```

Do not mix NGRAM into the MTP state machine as a hidden fallback. It needs its
own metrics, draft candidates, verify input, and cache rollback contract.

## Landed local ngram prototype

The first ARLE implementation is deliberately narrower than SGLang's NGRAM
worker:

- draft source: request-local token history only;
- match: linear suffix scan, longest non-overlapping prior suffix first;
- verify: existing Qwen3.6 C++ target `verify_block_summary` on
  `[current_token, ngram_suffix...]`;
- commit: target verifier remains the only owner of KV/GDR advancement;
- rollback: rejected GDR suffix is replayed with
  `qwen35_rollback_to_accepted_varlen`; extra KV columns are ignored because
  `cache_len` advances only by accepted inputs;
- MTP coexistence: when MTP state exists, ngram verify captures target final
  hidden for the accepted row and refreshes the MTP seed hidden.

Runtime knobs:

```text
ARLE_METAL_NGRAM_SPEC=1
ARLE_METAL_NGRAM_MAX_DRAFT_TOKENS=4
ARLE_METAL_NGRAM_MIN_MATCH=3
ARLE_METAL_NGRAM_MAX_CONTEXT=4096
ARLE_METAL_NGRAM_MAX_MISSES=4
```

Default remains off. If enabled but no candidate is found for
`ARLE_METAL_NGRAM_MAX_MISSES` consecutive decode steps, the request disables
ngram and the standard decode path can resume its double-buffer prequeue.

Local Qwen3.6 warm-pair evidence on the repeated `metal_bench` prompt:

| route | gen tok/s | ngram acceptance | notes |
|---|---:|---:|---|
| standard step-driver | 86.80 | n/a | warmup=1, runs=3 |
| ngram max_draft=8 | 185.20 | 100% | 12 blocks, 96/96 accepted draft tokens |
| no-candidate forced (`min_match=64`) | 84.14 | n/a | auto-disabled after misses |

This is a licensed local-repeat win, not a default flip. The workload is highly
repetitive and c=1. The default-worthy route needs natural prompt/code/JSON
coverage plus a router that enables ngram only when the request has a real
candidate signal.

## c=1 benchmarking discipline

For c=1, multiple concurrent requests are not evidence. They change the workload
into a c>1 scheduler/batching test. Multiple cases are valid only as sequential
single-request runs (`max_requests=1`, one active request at a time).

The only valid c=1 MTP evidence in this pass is sequential `metal_bench`:

```text
./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --use-step-driver \
  --prompt 'Write a compact Rust function that reverses a string and explain it briefly.' \
  --generation-tokens 64 \
  --warmup 1 \
  --runs 3 \
  --ignore-eos
```

Matched sequential A/B:

| route | gen tok/s mean | repo-e2e tok/s mean | TTFT mean | total mean | MTP acceptance |
|---|---:|---:|---:|---:|---:|
| baseline step-driver | 79.83 | 74.75 | 54.3 ms | 861.6 ms | n/a |
| MTP split draft | 98.66 | 91.27 | 52.5 ms | 701.3 ms | 68.5% |

Delta for this warm code prompt: +23.6% generation tok/s, +22.1% repo-e2e
tok/s, and -18.6% total time. This licenses "MTP can win on a warm c=1 code
shape", not "MTP is default-worthy". GuideLLM/HTTP c=1 natural cases still need
sequential max_requests=1 replication.

## Rule

Speculative decode must be adaptive per request, and the fallback must preserve
target-verifier ownership of committed tokens. Low acceptance is a draft-quality
problem first; optimize verifier packing only after token parity and acceptance
are controlled.
