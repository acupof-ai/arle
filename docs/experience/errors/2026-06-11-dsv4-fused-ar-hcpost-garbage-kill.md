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

## UPDATE — device-printf diagnostic ran: kernel is correct at layer 0, bug is deeper

Ran the prescribed diagnostic (guarded `printf` in both the fused kernel and
the unfused `dsv4_mhc_post_kernel`, `NVCC_PREPEND_FLAGS=-DARLE_FUSED_AR_DEBUG`,
same prompt). Layer-0 token-0 column-0:

| | out[0] | new_x | post0 | comb0 | res0 |
|---|---|---|---|---|---|
| unfused (correct " Paris…") | 0.024204 | 1.726562 | 0.025342 | 0.888959 | -0.020630 |
| fused (empty/garbage e2e) | 0.024216 | 1.727051 | 0.025342 | 0.888959 | -0.020630 |

The 8 per-rank partials summed correctly to new_x (e.g. -1.469+0.594+…=1.727),
and the fused `out[0]` matches the unfused to **bf16 precision**. So BOTH the
cross-rank reduction AND the hc_post mix are correct — the visibility-fence
hypothesis is REFUTED (the reduction reads valid peer data). Yet e2e still
generates empty/garbage. The bug is therefore NOT in the kernel's layer-0
value; it is a deeper-layer, sync, or buffer-lifetime interaction the
layer-0-col-0 probe cannot see. Candidates for the next session:
deeper-layer residual aliasing (does the fused branch leave `attn_out`
unreduced where a later consumer needs it?), the end-barrier vs the
subsequent moe AR staging into the same scratch, or a per-layer
buffer-recycle race. Next diagnostic: print out[0] at the LAST layer + dump
the full attn_stream row hash fused-vs-unfused per layer to find the
divergence layer.

## Prime suspect (original — now partly refuted)

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

## FINAL — exhaustive static audit closes correctness; perf premise is the real kill

Rather than burn more pod cycles chasing the e2e race, did a full free static
audit of the reverted kernel (`git show 2d60fb1d`) against the live unfused
path. Ruled out, line by line:

- **Striding** — hidden=4096, threads=256, blocks=min(36,16)=16 → exactly 4096
  threads, one col/thread, no stride iteration; each writes 4 dst lanes.
  `out[0]` correct ⇒ all cols correct (per-column independent, identical code).
- **comb/post/residual indexing** — byte-identical to `dsv4_mhc_post_kernel`
  (`comb[dst*hc_mult+src]`, `residual[src*hidden+col]`, `post[dst]`, out lane
  `dst*hidden+col`). NOT transposed; dst lanes 1-3 correct too.
- **Second consumer of `attn_out`** — NONE. At both fused sites
  (`forward_tokens_stream_impl` ~1897, `forward_decode_batch_stream_impl` ~1440)
  `attn_out` is consumed only by hc_post, then `stream = attn_stream`; `attn_out`
  drops out of scope. The "unreduced second consumer" hypothesis is refuted.
- **Staging / visibility** — the fused entry stages via `memcpy_dtod_async →
  scratch_ptr` then `multi_gpu_barrier<8,true>`, **identical** to the working
  one-shot `all_reduce_sum_inplace` (tp.rs:1004) + `cross_device_reduce_1stage`.
  Same fence mechanism; if it raced, the licensed AR would too.
- **Premature free** — mhc.post/comb (dsv4.rs:1850-1852), attn_out (1896),
  stream (1841), attn_stream all keepalive'd. Not freed before the async kernel.
- **Barrier desync** — `self_counter[block][peer]+=1` with `val%2` ping-pong is
  self-consistent under lockstep regardless of block-count mismatch vs the moe AR.

The kernel is mathematically **byte-identical** to the licensed AR+hc_post for
all cols and all lanes. The residual e2e race (printf-on correct, printf-off
garbage = Heisenbug) is real but **not worth chasing**, because:

**The perf premise is dead.** Fusing [AR + hc_post] removes one launch boundary
(~5µs) + an 8 KB intermediate round-trip (~5ns) per attn site × 60 layers ≈
300µs/token ≈ **~1%** at 44 tok/s — below the ±6% drift floor. B=1 decode is
GPU-bound (`feedback_b1_decode_gpu_bound_overhead_removal_wash`); this is pure
overhead-removal → wash even when correct. And the real headroom is elsewhere:
the same-day nsys (`nsys_b1_allreduce`) shows the AR+AG **collectives are 34%**
of GPU-busy time — fuse/overlap THOSE, not the hc_post boundary. See
[`2026-06-11-dsv4-rung2-kill-comm-is-the-real-b1-lever.md`].

## State

Rung 3 = **KILLED, terminal.** Kernel proven math-correct by static audit; perf
premise dead (≤1% overhead-removal on a GPU-bound path). Not reopening the race
debug — even a fixed Rung 3 is wash. Rung 2 = KILLED before build (same
comm-is-the-real-lever evidence). Rung-1 stands (+9.3% matched, campaign
39.51→~44). Ladder exhausted; next lever is TP/EP comm (34%).
