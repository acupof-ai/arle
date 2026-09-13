# Parity bound provenance

Date: 2026-09-13 · Owner: f4 · Status: ongoing, card-free.
Tracks the agenda row `parity-bound-provenance`: every positive-arm bound
states both sides — the clean floor it must not fire on and the smallest
target defect it must fire on — and carries one of three labels:
**derived** (both sides from arithmetic), **partially derived** (one side
derived or measured, the other not), **measured two-sided** (both sides
measured over a set equal to the run set).

## FlashMLA decode pack arms — separation unknown

Recorded at session wrap-up 2026-09-13 so it survives in the repo, not in
chat. In both `flashmla_sparse_decode_parity.rs` and
`flashmla_hca_decode_parity.rs`:

- `PACK_NOPE_REL = 0.07` has a derived **clean-side** supremum (pure e4m3 RN
  repack: worst bf16 value rounding down a bin over the smallest shared e4m3
  value is 16/272 ≈ 0.0588; f32-scale bind 17/273 ≈ 0.0623; correct packs
  measure ≤0.044). It has **no measured defect side at all** — no corrupted
  pack run records how far a real pack fault moves the tile ratio, so the
  bound's power to separate a fault is unknown. The arm is not dead; its
  reach is unmeasured.
- `PACK_ROPE_ABS = 0.01` has **neither** side: the rope channel is a bf16 copy
  with no e4m3 step, so it wants a tight bf16-round derivation, and no
  defective run exists. Clean-run-set, underived.

Next card window should measure both against the smallest target pack defect
(the first-written token's tile-0 e8m0 scale byte, already built by the
negative arm) before either label changes. Do not derive the rope bound from
theory only when the run is available.
