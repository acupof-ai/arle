# Cross-cp flash-decoding decode merge — CUDA, 2026-08-17

> Status: pending-remote

## Goal

Correct decode under 2D (attn_tp × cp) KV sequence-sharding (T3.2b Part E).
Each cp rank's pool holds only its own shard's pages (block-cyclic: logical
page `i` on shard `i % cp`), so a decode step's KV set is split across cp
ranks. The full-attention output must equal the unsharded output within the
correctness-parity envelope (needle ladder ×3, same config), with no
per-step host readback/sync.

## Hypothesis

Flash-decoding merge: each shard runs FA3 paged decode over its local page
subset, producing a NORMALIZED partial `(out_c, lse_c)` (lse = logsumexp of
that shard's attention weights). The merged output is the weighted average

```
out = sum_c w_c * out_c / sum_c w_c,  w_c = exp(lse_c - max_c lse_c)
```

accumulated in f32 (SGLang #21637 "separate local combine kernel" shape).
The ratio is invariant to the max choice, so no global pre-reduce is needed.
lse rides the bf16 all-gather as f32 pairs (NCCL moves bytes, so it stays
f32 end to end); out gathers as bf16. The new-token KV write is
owner-conditional: only the shard with `page_idx % cp == c` writes, so the
sharded pool stays exactly-once without a location table.

## Parameters

```bash
# Correctness gate (world=4, attn_tp=2, cp=2), needle ladder ×3 same config:
python3 scripts/needle_gate.py --url <url> --model <model> --runs 3
# CP_DECODE_MIN_KV_TOKENS re-tune: under 2D the GDN decode pair now engages
# from the first decode token (was 8192, a B2-length-dependent floor that
# does not transfer). c-sweep of decode tok/s at kv_seq_len 128..131072:
#   pair-on (this commit) vs pair-off (env/flag) at matched concurrency.
```

- Baseline: parent of the Part E commit (Part B pool sharding + Part D ring
  prefill; decode under 2D not yet correct)
- Treatment: Part E commit (sharded decode meta + owner-conditional write +
  cross-cp merge + FA3-only 2D lane)
- Trials: 3 (needle ladder ×3, same config)

## Environment

- Host / GPU: 8×H20 pod (sm_90), world=4 (attn_tp=2, cp=2)
- Driver / CUDA: TBD
- Model / dtype: Qwen3.5/3.6 hybrid, BF16 KV pool (2D decode is BF16-only;
  quant pools bail at the sharded-meta guard)
- TP / EP / slots / KV: attn_tp=2, cp=2, prefix cache disabled under 2D
- Server flags: 2D engaged (world ≥ 4, attn_tp ≥ 2, cp ≥ 2), DSpark off
  (taps need the full head set)

## Results

| arm | needle ladder | errors | garble | decode tok/s c=1 | delta |
|---|---|---:|---:|---:|---|
| baseline | | | | n/a (incorrect) | — |
| treatment | | | | | |

Raw artifacts: TBD.

## Problems

None yet. Known v1 limitations (by design):

- **FA3 decode graph capture bakes `num_pages` as a host arg.** Graphs are
  inert under CP today; the 2D lane errors if FA3 is disabled (TileLang
  produces no lse, so no merge is possible).
- **Quantized KV pools unsupported under 2D decode.** The sharded meta
  guards BF16-only; the quant FA3/varlen lanes produce no lse and their
  split-KV workspaces are sized for the unsharded head count.
- **CP_DECODE_MIN_KV_TOKENS re-tune is analytical, not measured.** Under 2D
  the GDN decode pair's trade is length-independent (recurrent state is
  O(1) per step; the pool is already cp-sharded), so the floor is 0 under
  2D and B2 keeps its measured 8192. The c-sweep above is the
  pending-remote confirmation.
- **Empty-shard steps need FA3's zero-KV guard.** For the first
  `page_size * cp` tokens a non-owner shard holds no pages; its FA3 partial
  must come back `lse = -inf, out = 0` so the merge weights it zero. A NaN
  from an unguarded empty-KV row would poison the merge. Verify on the pod
  at `kv_seq_len < page_size * cp` (decode steps 1..31 at cp=2).

## Learnings

pending-remote. Design points that held up:

- **The merge is topology-gated, not floor-gated.** Under 2D the pool is
  always sharded (Part B sets the shard spec at construction), so every
  decode step must merge; there is no unsharded fast path to fall back to.
  The CP_DECODE_MIN_KV_TOKENS floor only gates the GDN decode pair, which is
  an independent efficiency axis.
- **Normalized partials keep the merge stateless.** FA3 already outputs
  `(lse, out)` normalized per shard; the merge needs no running `(m, l,
  oaccum)` triple across steps. The raw-triple fallback stays available if
  a future lane exposes only unnormalized accumulators.
- **Exactly-once ownership is one predicate.** The new token's owner is
  `floor(kv_seq_len / page_size) % cp` — the same block-cyclic predicate as
  the pool's alloc filter and the radix location. The prep kernel still
  norms+ropes K/V on every shard (cheap); only the pool write is skipped on
  non-owners.
- **The reduce scope splits by attention family.** Full-attn reduces over
  attn_tp only (the cp merge makes cp ranks identical); GDN with the decode
  pair reduces over world (1/cp v-heads per rank). Both are correct
  independently because they are separate functions with separate partials.
