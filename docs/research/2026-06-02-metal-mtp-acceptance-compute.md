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

## Rule

Speculative decode must be adaptive per request, and the fallback must preserve
target-verifier ownership of committed tokens. Low acceptance is a draft-quality
problem first; optimize verifier packing only after token parity and acceptance
are controlled.
