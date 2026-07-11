# DSpark draft-KV: cap full-layer at per-request ceiling — 544→64 MB/slot, 8× slots, lossless

## Context

The DFlash draft full-attention layer sized its per-slot KV cache from
`max_seq_len` (= `total_pages × 16` = the 128K KV-pool floor), not the
per-request token ceiling. Measured 512 MB/slot = 94% of the 544 MB draft-KV;
`--max-total-tokens 8192` did NOT shrink it (`max_seq_len` is pool-derived,
independent of the flag), clamping slots 256→84 (→32 on a busier GPU) and the
per-request arena to 4096 (a 13K prompt didn't fit one slot). Commit `1ee72d809`.

Root cause was a config-plumbing gap, NOT a full-attention-drafting cost — the
initial "window the full layer (lossy)" hypothesis was **corrected by the
measurement** (§0: the load-bearing assumption is the one to measure). The
scheduler admits nothing past `max_total_tokens`, so the full draft layer never
caches more than that: `ctx_cap = min(max_seq_len, max_total_tokens)`. Threaded
`max_total_tokens` through `from_qwen35_safetensors` (4 sigs).

- H20 GPU 3, TP=1, base `Qwen3.6-27B-FP8` + `Qwen3.6-27B-DFlash`, binary sha
  `37b8c516` (≠ prior `a4c17d89` — fix in the tested binary). B=1 greedy.

## Result — memory shrinks, slots recover, NO accept/tok-s regression

| config | draft/slot | num_slots (req→clamp) | full-layer share |
|---|---|---|---|
| `--max-total-tokens 8192` | **64 MB** (was 544) | **256, no clamp** (was 32) | 32 MB (was 512) |
| default (65536) | 288 MB | 135 | 288 branch |
| `--max-total-tokens 16384` | 96 MB | 241 | — |

`max_seq_len` stays 131072 (trunk pool unchanged); only the draft cache resizes.
Compute-check matches the logged bytes exactly (`2×(4·2064 + (min(131072,MTT)+16))·1024·2`).

Lossless — the win is preserved/higher (MTT 8192, no-prefix):

| shape | dspark Δ vs no-spec | accept_rate | P1 anchor |
|---|---|---|---|
| short ~256 tok | **2.49×** | 0.204 | 2.39× / 0.199 |
| ~3K ctx | **3.76×** | 0.273 | 3.14× / 0.228 |

Needle 7391 byte-identical across two greedy runs (self-consistent). **Long-ctx
unblocked**: a 12681-token prompt now fits one dspark slot at MTT 16384 (accept
0.393, coherent) — the 544 MB/slot clamp (arena 4096) previously blocked it.

## Rule

Size a per-slot speculative cache from the per-request admission ceiling
(`max_total_tokens`), never the whole-pool positional budget (`max_seq_len`).
The scheduler already bounds request length, so the tighter cap is lossless —
no accept-rate tradeoff, no windowing. A "full-attention layer is inherently
expensive" hypothesis is worth one measurement before accepting a lossy fix:
here it was a config-plumbing gap, and the lossless cap gave 8× the slots for
free.
