# DSv4 MTP tree fast path — end-to-end design (no more half-steps)

Status: DESIGN. One coherent build per this doc, then ONE license-or-kill.
Supersedes the piecemeal sequence (chain → per-token → ring-replay tree), each
of which fixed one cost center and got killed by the next. This doc closes the
whole budget first.

## Evidence base (same-binary b6e8d9d7, 2026-06-12, 8×H20)

| config | tok/s | A | step | structure |
|---|---|---|---|---|
| no-spec | 33.07 | 1.0 | 30 ms | 1 decode forward |
| chain d1 | 16.68 | 1.36 | 82 ms | 1 mtp + 2-row verify + ~1.4-row re-forward, all per-row attention |
| tree topk2 d2 | 11.49 | 1.81 | 158 ms | 3 mtp + 7-row verify + re-forward + ring fix-ups |

Differencing: **verify row ≈ 10 ms** (43 layers of per-row decode-attention
launches + row copies), **mtp expansion ≈ 10 ms** (1-layer forward + 129k-vocab
lm_head GEMV + full-logits D2H host top-k). Tree needle exact ×6 — the accept
machinery (DraftTree, longest-path, frozen rules) is correct and stays.

Break-even: `step_ms < A × 30.2`.

## Why every per-step forward must be batched AND counted

The step is `draft + verify + commit`. With per-row attention each "forward"
costs `~10ms × rows`; batched, a whole multi-row forward costs ≈ one decode
forward (~30 ms — weight-read-bound holds once attention batches too).
Counting forwards is the whole game:

| design | forwards/step | step @ A=1.81 | tok/s | verdict |
|---|---|---|---|---|
| today (per-row everything) | ~5.2 equiv | 158 ms | 11.5 | shipped, KILL |
| batched verify, keep re-forward | draft 2×~10 + verify 32 + re-fwd 32 | ~84 ms | 21.5 | still loses — **do not stop here** |
| + re-forward eliminated (self-heal verify) | draft 2×~10 + verify 32 | ~52 ms | 34.8 | +5% — marginal |
| + accept-rate fixed (see P0) → A≈2.2 | same | ~52 ms | 42 | +27% — the actual prize |

Conclusion: **all three pillars are load-bearing**. Building any one alone
produces another kill — that is the "half-step" trap this doc forbids.

## P0 — accept-rate root cause (the multiplier, probe FIRST)

Measured first-draft accept: 36% on the ab_decode prompt (a trivially
predictable list continuation), 60% on the needle window. A healthy nextn MTP
head sits 60–80%+. If 36% is a defect (position off-by-one in `mtp_forward`,
h_prev mismatch, digit tokenization), fixing it moves A more than any kernel
work; if it is genuinely the head's quality on that text, the budget rows
above stand.

Probe (cheap, read-only): log `(draft, target_argmax)` pairs per step for one
ab_decode run (the old `ARLE_DSV4_MTP_DRAFT_DUMP` env is dead code — re-add a
guarded eprintln in `spec_step`), eyeball 50 pairs: are rejects near-misses
(quality) or nonsense (bug)? Decide before building P2 depth.

## P1 — batched verify on the EXISTING sparse kernel

`try_flashmla_prefill_attention` (attention.rs:3905) already does everything
the tree verify needs, for 41/43 layers (21 CSA cr=4 + 20 HCA cr=128):
batched QKV, one `arle_flashmla_csa_pack_kv` into a unified bf16 pool
`[SW cache rebased | chunk K | compressed pool]`, per-query
`indices/topk_length`, ONE `arle_flashmla_sm90_sparse_prefill_fwd` per layer,
TP gather/repack/slice. The per-token crutch exists only because the batched
verify once fell into the host-start_pos prefill path un-frozen and garbled
(wins/2026-06-11-dsv4-mtp-dsa-rollback-selfheal-fix.md UPDATE) — the fix is to
drive THIS path with tree-correct inputs, not to serialize rows.

Integration deltas (all in our code, no vendored changes):

1. **Per-row positions into RoPE**: `dsv4_prepare_qk*` assumes consecutive
   positions from one `start_pos`. Add a positions-array variant
   (`positions: *const i32`, one per row) — csrc/misc, trivial kernel edit.
   Tree rows sit at `start_pos + depth(r)` (repeats allowed).
2. **Tree indices**: per-row = committed-window rows + ancestor chunk rows +
   self, padded to `topk_unified % 128 == 0` (window 128 + index_topk 512 +
   128 tree tail = 768 ✓). v1 builds them on HOST (n ≤ 64 rows × ~768 i32 →
   one ≤200 KB H2D), replacing `csa/hca_build_indices`' causal-contiguous
   assumption. Device kernel only if the H2D shows up in nsys.
3. **Frozen pins on the batched lane**: `csa_select`'s P1-A `key_count` pin
   currently triggers only via `start_pos_device.is_some()` (decode); apply
   the same pin when `dsv4_verify_frozen()` on the batched lane.
   `compressor_forward`'s P1-1 gates are call-level — already shape-agnostic.
4. **Verify becomes PURE**: indices point into `kv_unified` (chunk K rows),
   so the verify needs **zero ring writes** → no capture/restore, no node
   scratch, no fix-ups on this lane. The ring-replay machinery (b6e8d9d7)
   stays as the validated fallback lane (`is_chain` fallback + per-row tree),
   selected when FlashMLA prefill is unavailable.
5. **SW-only layers (0, 1)**: same sparse fwd, `kv_unified = [SW cache |
   chunk K]`, indices = window + ancestors + self. d_qk = head_dim = 512 ✓.
   (Or per-row swa ×2 layers ≈ 0.5 ms — decide by simplicity, it is off the
   critical path either way.)
6. **MoE/pointwise half**: already batched (`forward_tokens_stream_impl`
   token-parallel) — untouched.

## P2 — eliminate the commit re-forward (self-heal verify, SGLang-style)

The frozen verify forces a second full forward to commit the accepted prefix.
SGLang does NOT freeze the verify: the verify writes compressed state at
deterministic boundary-floored positions, rollback shrinks `seq_len` and lets
the next step self-heal overwrites (our own record:
wins/2026-06-11-dsv4-mtp-dsa-rollback-selfheal-fix.md — "SGLang
mutates-then-self-heals, doesn't freeze the verify"). With a batched
non-frozen verify, accepted rows' compressor/ring/DSA writes ARE the commit;
only the rejected tail needs the (already-built, §0.1-enumerated) restore:
ring tail restore + `packed_rows`/bootstrapped + `dsa_official.packed_rows`
clamp + compressed `seq_len` truncate. The P1-era long-context bugs came from
an INCOMPLETE enumeration; the enumeration is now complete and needle-gated.

Risk ranking: P2 re-opens the verify-mutation surface — so P1 (frozen,
re-forward kept) lands first and gates green; P2 flips frozen→self-heal as one
reviewed diff against the green gate. If P2's gate fails, P1+deeper trees
(depth 3–4 once verify is row-cheap) is the fallback to clear break-even.

## P3 — level-batched draft + device top-k

- Expand a whole frontier level in ONE `mtp_forward` batch: tokens `[m]`,
  h_prev `[m, stream]`; linears/lm_head batch (129k×4096 GEMV → GEMM over m).
  MTP layer is SW-only ⇒ siblings attend committed window + ancestors: per-row
  attention at ONE layer (~0.2 ms × m) is fine — the lm_head was the cost.
- Device top-k: k ≤ 4 → k× (argmax + mask) launches on the [129280] logits,
  D2H of k token ids only. Kills the 258 KB logits D2H per expansion.
- Draft ring writes on target layer 0 stay (chain-validated frozen-draft
  semantics unchanged); with P1 the verify no longer cares about them, and
  `capture_spec_rings` shrinks to the draft's layer-0 writes only.

## Validation ladder (one pass at the end, plus one early gate)

1. After P1: needle 3000,6000 ×3 @0.5 — chain (fallback lane untouched) AND
   tree topk2 d2 on the batched lane; same-binary ab_decode triplet.
   Early kill: if batched verify > 45 ms/step at n=7, profile before P2.
2. After P2: same gate + reject-heavy long-context (the P1-era failure shapes:
   SW-wrap partial-accept, compression-boundary crossing) ×3.
3. After P3 + P0 outcome: depth/topk sweep (d2k2, d3k2, d2k3) → pick by
   measured A vs step; license-or-kill vs same-day no-spec.

Kill criteria: best tree config < no-spec +10% wall-clock → MTP stays opt-in,
document and stop (the A machinery remains correct for future heads with
higher accept rates).
