# DSv4 depth-2 MTP probe: sequential chaining is ~33% (kill); EAGLE-tree is the 6ms path

## Context

After MTP single-draft landed (+71%, decode ~15ms), the decode-6ms question was whether
**multi-token MTP** (draft >1 token/step) closes the gap. I'd been *assuming* the depth-1
head wouldn't chain — §0 says test, don't assume. So I measured it: added an isolated,
opt-in depth-2 probe in `verify_forward_selftest` (`ARLE_DSV4_MTP_DEPTH2_PROBE=1`).
`mtp_forward` now returns its wide `ffn_stream` so a 2nd draft can chain from the 1st.

## What the probe measured (8×H20 TP=8, 3 prompts, first decode step)

| prompt | depth-1 (d1 vs real@L+1) | depth-2 chain (d2 vs real@L+2) |
|---|---|---|
| needle | ✓ | ✗ (1613 vs 929) |
| capital | ✓ | ✓ (455) |
| canon | ✓ | ✗ (223 vs 455) |

- **depth-1 accept = 3/3 (100%)** — the head is excellent at 1-ahead (matches the +71%).
- **depth-2-top1 accept = 1/3 (~33%)** — chained from the 1st draft's (unverified) stream,
  the head is off-distribution; the 2nd sequential draft lands ~⅓ of the time.

Expected gain: ~0.85 × 0.33 ≈ +0.28 tok/step → depth-2 ≈ 2.1 tok/step → **~13ms (~+15%)**.

## Rule / verdict

- **Depth-2 *sequential* MTP is KILLED for the 6ms target** — ~+15% (~13ms) doesn't justify
  the multi-token-rollback complexity, and it's nowhere near 6ms. This is now EVIDENCE
  (33% measured), not the assumption I'd been asserting — the §0 license-or-kill fix.
- **EAGLE-tree spec is the licensed 6ms path**: draft top-K candidates per depth so the
  right depth-2 token is in the set (lifting the measured 33% top-1), + a tree-verify
  mask. DSv4-Flash has only `num_nextn_predict_layers=1`, so there's no trained
  multi-layer draft to exploit — the tree (multiple candidates from the one good depth-1
  head) is how to convert its 100% depth-1 / 33% depth-2-top1 into >2 accepted tok/step.
- Combined decode-6ms map (all evidence-backed this session): single-draft MTP ✅ (~15ms);
  decode graph ❌ wash; mHC fuse ❌ launch-bound; depth-2 sequential ❌ ~13ms; **EAGLE-tree
  = the remaining lever**, now licensed by the depth-2-top1 data.
- Infra landed: `mtp_forward` → `(token, wide_stream)` (reusable for the tree); opt-in
  depth-2 probe in the selftest.
