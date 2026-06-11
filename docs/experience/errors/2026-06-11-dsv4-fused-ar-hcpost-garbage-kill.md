# Fused AR+hc_post kernel produces garbage — correctness KILL (reverted), reduction-visibility the prime suspect

**Date:** 2026-06-11. 8×H20, `--comm-backend auto` (one-shot active, 7-rank
ack), gated `ARLE_DSV4_FUSED_AR`. Matched back-to-back pair.

## What happened

| arm (same auto-comm serve, one-shot active) | B=1 p50 | output |
|---|---|---|
| D2: pair (AR kernel + hc_post kernel) | 41.27 | correct (" Paris. The capital of the United Kingdom is London") |
| E: fused AR+hc_post (attn site) | 42.76 | **GARBAGE** (`nahil#r#t#r#t#r#…` token-repeat) |

The fused kernel is +3.6% faster AND wrong — the classic dangerous shape
(a partial no-op runs fast). Speed is meaningless until correctness passes;
this never cleared the gate. **Reverted the whole Rung-3 commit**
(`a72ad950`) so main is byte-clean; the kernel/entry/wrapper live in git
history for a focused debug session.

## Design (for the v2 debug)

`dsv4_fused_ar_hc_post_kernel<ngpus>`: start barrier → grid-stride over
hidden columns, each thread reduces one column across all ranks
(`dp.ptrs[r][col]`, fixed order) → hc_post mix from registers → end
barrier. Mirrors the working `cross_device_allgather_1stage` (same
RankData/Signal/barrier framework, `buffers_.find(input_ptrs[rank])`).

## Ruled out (by source re-read, not device)

- Output layout: `out[dst*hidden+col]` matches the unfused hc_post idx
  decomposition for token=0. ✓
- residual layout `[src*hidden+col]` matches. ✓
- `dp.ptrs` is rank-indexed (the AG kernel uses `dp.ptrs[src]` and produces
  correct rank-major output). ✓
- scratch_ptr == input_ptrs[rank] (the unfused AR, which reads
  input_ptrs[rank] internally, produces correct output on the same serve). ✓

## Prime suspect (next diagnostic)

The repeating-token degeneracy = attention contributing ~constant/garbage,
i.e. `new_x` (the in-kernel reduction) is wrong. The unfused AR writes the
sum back to `buf` then hc_post reads it; the fused path reads peer scratch
DIRECTLY inside the same kernel. Candidate: the start barrier guarantees
the peer COUNTER is visible, but the peer's staged DATA may not be visible
across the P2P link at the point my threads read it — the bare 1stage gets
away with it because its read pattern / fence placement differs, or because
`packed_reduce` touches memory differently. **Next:** device-printf
`new_x[0]` for token 0 vs a reference (run unfused AR into a scratch, D2H
both, compare) — one eprintln settles it, the way the FP8-KV
"catastrophe" was settled. Likely fix: `need_fence=true` on the start
barrier (release/acquire instead of volatile) so the staged data is ordered
before peers read.

## Rules

- **Correctness gate is decode-the-actual-tokens, every time.** A fused
  collective that "wins" 3.6% with `text=''`/repeat-garbage is a no-op, not
  a win (same trap as the masked-MoE-in-graph +24% degenerate).
- **A gated-but-broken kernel is a half-state — revert it.** Keep the design
  in git history, not a flippable footgun in the tree.
- Serve scripts hardcode `--comm-backend nccl`; the one-shot/fused path
  needs `--comm-backend auto` (derive from arle_serve_allreduce.sh + flag,
  not env). Verify "one-shot AR/AG active" in the serve log BEFORE trusting
  a fused-comm A/B — else try_fused silently falls back and you bench the
  pair twice.

## State

Rung 3 = KILLED pending the visibility-fence debug (cheap, bounded). Rung 2
(multi-block hc_enter segment) is independent and unaffected. Rung-1 stands
(+9.3% matched, campaign 39.51→~44).
