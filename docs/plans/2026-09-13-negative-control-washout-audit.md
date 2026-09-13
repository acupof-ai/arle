# Negative-control washout audit

Date: 2026-09-13 · Status: read-only paper audit; no gate edited, no card run.
Paths are repo-relative.

**Every ratio below is a worst-case, on-paper estimate derived from the
constants and comparator in each source file. Nothing here was measured on a
GPU. Do not cite these numbers as measurements; they predict whether a tooth
can fire, they do not report that it did.**

**What "Fires" / SAFE means in this table — correction after card
measurement (2026-09-13).** A SAFE verdict here means the **metric does not
dilute** the corruption (exact compare, per-element `viol==0`, or a
row/tile-local denominator). It does **not** mean the corruption's resulting
effect exceeds the bound. Those are two separate quantities; this audit often
established only the first. A non-diluted tooth whose effect sits below the
cap still reads green, and two rows rated SAFE on the dominant/localized
construction are exactly that when measured (see the card-measured corrections
on the two decode rows below). Trust a "Fires" verdict without reservation
only where the row gives an explicit effect-to-bound ratio I computed; rows
marked **[construction only]** prove the effect is not averaged away but did
not compare its magnitude against the cap on paper. The bound itself —
clean-run-set with no supremum — is tracked separately under
parity-bound-provenance (f4), not here.

A negative control corrupts one side and the gate must go red. It goes red
only if the corruption's effect, in the metric the pass predicate actually
uses, clears the bound. Two card runs today showed teeth that could not fire:

- `flashmla_prefill` (#439): a realistic swap washed out and needed a
  dominant-key construction.
- `marlin_fp8`: flipping one weight byte moved rel-L2 1.7e-3 → 1.8e-3 against
  an 8e-3 bound; nothing tripped.

## Design axis

Dilution-proof teeth are backed by one of:

- **E** — an exact / ordered discrete compare (index, token, set multiset).
- **V** — a per-element clause requiring `violators == 0`; a single forced
  violator trips regardless of vector size.
- **W** — a worst-element metric whose denominator is local to the corrupted
  region (a row, a tile, the worst row), so adding clean elements cannot
  shrink it.

The risky shape is **N** — a global relative-L2 over a slab or the whole
output — paired only with a *whole-reference* sabotage (e.g. ×3 everything).
That tooth saturates, but the metric cannot see a localized corruption (one
byte, one column), whose energy is divided by the full reference norm. A
`max_viol_frac` bound is in this category unless the tooth is measured on the
corrupted row itself.

Two recurring caveats, stated once:

- **Dead-block edge.** A scale-flip tooth that corrupts one quantization block
  cannot fire when that block has no live contribution to the compared output
  (block outputs all near zero / not routed). Same class as a washout: a tooth
  that is structurally silent for some inputs. Applies to every per-block
  scale-flip tooth (`dsv4_tp_oproj`, the FlashMLA pack teeth); the gates avoid
  it by corrupting a first-written/live block, which is a construction
  property, not a metric guarantee.
- **Comparator visibility (equality families).** Equality families have no
  numeric margin; their equivalent failure is a comparator that structurally
  cannot see the corruption (a set compare versus a positional swap — the #436
  defect). Each equality row below states whether the comparator can see it.

Fix rule (for the gates given to f4): a washed tooth is fixed with a bigger or
better-placed corruption or a dominant construction, never by loosening the
bound or scoping the comparator to where the sabotage landed (that couples the
comparator to its own negative control).

## Flagged gates (fix input for the G1 conversion)

| Gate | Metric | Tooth | Problem |
|---|---|---|---|
| `marlin_fp8_parity.rs` | global rel-L2 over a 1024-column slab, bound `MAX_REL_L2=8e-3`, plus marlin/gemv ratio ≤4 | ×3 whole slab reference (rel≈2, fires) | **The metric already computes a worst-element quantity and discards it.** `lane_stats` returns `max_over_rms` (marlin_fp8_parity.rs:294) and it is printed at :560/:563/:622, but no pass predicate uses it (predicate :544-547 is `rel_l2` + ratio only). `REF_SLAB=512`, `reference_columns` covers 1024 columns, so one corrupted column carries ~2/1024 of the error energy → rel-L2 ≈ sqrt(2/1024)=4.4e-2 vs 8e-3, ~5.5× under, independent of n and M. This is the confirmed marlin_fp8 card result. |
| `marlin_w8a16_parity.rs` | global rel-L2 over the **full** `m*n` output (no slab, no worst-element term), bound `MAX_REL_L2=0.20`, plus ratio ≤ fallback | ×3 full reference → rel≈0.67, ~3.3× over, so the shipped tooth passes | Same metric shape as fp8: a localized column/byte corruption at large `m*n` washes exactly as there (denominator is every element). Not broken for the shipped ×3 tooth, but it has <1× headroom for a localized real failure and converts into the fp8 bug under a local corruption. Give it a localized tooth or a worst-element clause. |

## Template gates (copy these anti-washout patterns)

| Gate | Pattern to copy |
|---|---|
| `flashmla_prefill_parity.rs` (#439) | **Dominant construction.** Overwrites one compressed row with the last query's direction at 20× latent (score ~7.6 vs background ~0.4) so it wins the softmax outright; the strong tooth drops that key. The realistic non-dominant weak swap is asserted ONLY on the exact index family and explicitly not on the washed numeric metric. Also runs a positive-control tamper proving the index comparator itself fires. |
| `fa3_hd256_shim_parity.rs` | **Measure the tooth on the corrupted row, not the global fraction.** Mirror and anchor teeth are evaluated per corrupted row (the comment notes the global `max_viol_frac` would be diluted by more checked rows); two teeth per family; untargeted families are asserted still clean. |
| `paged_quant_attn_parity.rs` | **Two-sided oracle, full-attended sabotage.** Corrupts every attended token's full-D V on row 0 to dtype max (order-one across the row), then asserts the sabotaged output both (a) mismatches the CLEAN oracle and (b) matches an oracle rebuilt from the sabotaged pool — proves the kernel consumed the buffer and cannot pass by agreeing with clean. |
| `moe_routing_parity.rs` | **Specificity.** Seven discrete sabotages, each against an independent comparator, with an assertion that the tooth trips ONLY its family and every sibling stays green — catches both a dead comparator and a non-independent one. |
| `flashmla_sparse_decode_parity.rs` / `flashmla_hca_decode_parity.rs` | **Dominant pinned slot + localized metric.** Dominant key pinned at a known slot (`DOM_SLOT=0`, quant-exact dominant channel); pack error is a per-tile ratio (`tile_max_err/tile_amax`), not a global norm; exact indices and worst-abs LSE backstop the numeric output. |

## Per-gate rows

Metric legend: **E** exact/ordered/discrete compare · **V** per-element
`violators==0` · **W** worst-element with a local denominator · **N** global
rel-L2. "Fires" is the worst-case on-paper ratio of effect to bound; ≫1 fires.

| Gate (file) | Metric · bound | What the tooth corrupts | Predicted effect vs bound | Verdict |
|---|---|---|---|---|
| argmax (`argmax_parity.rs`) | E exact id per row/lane | flip expected index 0↔1 (batch + singular) | guaranteed mismatch | Fires |
| deepgemm_grouped_prefill (`deepgemm_grouped_prefill_parity.rs`) | V `viol==0`, floor 2e-2/slope .1, rel cap 8e-2 | single cell row0,col0 += 5·max(\|w\|,1) | forced violation ~5 vs floor .02 (both masked and unmasked comparators) | Fires |
| dspark_draft_attn (`dspark_draft_attn_parity.rs`) | Single/Batched V (floor+slope, `viol==0`); Cross E `maxdiff==0` | +1.0 over the whole Single or Batched world vector; +1.0 cross-maxdiff | uniform +1 forces violations; cross exact | Fires |
| dspark_drafter_batch_invariance (`qwen35/dspark_batch_invariance.rs`) | rel-L2 vs a 2e-2 FLOOR the same single-vs-batched identity passes at ~0 | swap two slots' start positions (incl. non-block-aligned 512+5 and ring-wrap) and exchange K/V base rings | position/rope/window change at ctx≈32K is orders above the 2e-2 identity floor | Fires — **[construction only]** ("orders above" is qualitative; no computed effect/bound ratio) |
| dspark_dsa (`dspark_dsa_parity.rs`) | attn V (live floor) cap 4e-2; indexer W worst-row rel `SCALE_REL_MAX=1e-2` + E index | attn want[0]+=1.0; `want_scales[0]*=2`; replace top selection with the row's argmin | attn +1 forces a violator; ×2 on a scale row maxes the worst-row ratio; argmin is outside top-k | Fires |
| dspark_sampler (`dspark_sampler_parity.rs`) | filter/q W per-element max-excess (floor 2e-4, rel 2e-2); token E exact id (small-vocab arm) | prob[0]+=0.1; expected token id +1 | +0.1 ≫ 2e-4 floor; token id differs | Fires |
| dsv4_decode_moe (`dsv4_decode_moe_parity.rs`) | V (floor 2e-2/slope .1, `viol==0`) and N `MAX_REL_L2=0.06` | uniform +1.0 bias to swiglu or down over all elements | nearly every element violates the floor | Fires |
| dsv4_tp_oproj (`dsv4_tp_oproj_parity.rs`) | V `viol==0` floors 2.5e-2/3.5e-2 | Slice shifts a whole head; Gather/CacheMap shift a whole group; wb-scale flips one e8m0 byte (×8) on a 128-col block | slice/gather catastrophic; ×8 over the block × rows forces many violators (~1e4× typical, estimate not derived) — **dead-block edge applies** | Fires — **[construction only]** for slice/gather and the scale tooth's ~1e4× estimate |
| elementwise (`elementwise_parity.rs`) | V common_compare floor 3e-3/slope 3e-2, rel 2e-2; split2 bit-exact E | rms/silu want[0]+=0.1; split2 first/second want[0]+=0.5 | +0.1/+0.5 force per-element violations / bit differs | Fires |
| fa2_sm70 (`fa2_sm70_parity.rs`) | V floor 2e-2/slope 6e-2 | key0 of kv head 0 spiked (4.0 on d0); every query attends key 0 | catastrophic across outputs | Fires when run on target — **[construction only]**, not derived vs bound; **see sm_70 note** |
| fa3_hd256_shim (`fa3_hd256_shim_parity.rs`) | row-scoped mirror + anchor teeth (see templates) | one family's mirror expectation and its anchor, per corrupted row | row-scoped so undiluted, but no effect-vs-floor ratio computed | Fires (non-diluted) — **[construction only]** |
| flashmla_hca_decode (`flashmla_hca_decode_parity.rs`) | W per-tile pack ratio; E indices; W lse abs; output worst/total-RMS backstopped by E | start shift + first-written token's tile-0 e8m0 scale byte | pack per-tile ratio ~1; start shift changes exact indices | **CARD-CORRECTED — OUTPUT TOOTH DEAD:** pack and indices fire, LSE moves 1.1676, but per-head output move is **0.04477 vs the 0.05 cap** (0.90×); the output negative arm reads green. Dominant/local construction prevents dilution, it does not clear the cap. **[construction only]** for the output tooth. |
| flashmla_prefill (`flashmla_prefill_parity.rs`) | dominant strong tooth on out/lse/maxlogit; weak swap on E index only; pack bit-exact | dominant compressed row 20× latent; weak non-dominant swap; one unified SW bf16 bit | strong tooth wins softmax; weak numeric washout is documented and not asserted | Fires (strong) / index-only (weak, by design). Strong-tooth effect-vs-bound was not recomputed here — **[construction only]** for the numeric strong arm; the #439 dominant build is the reason it exists. |
| flashmla_sparse_decode (`flashmla_sparse_decode_parity.rs`) | W pack per-tile `PACK_NOPE_REL=0.07`, `PACK_ROPE_ABS=0.01`; E indices; dominant pinned | mask DOM_SLOT selected entry; first-written token e8m0 scale byte | dominant key quant-exact and dropped → output moves; tile ratio localized | **CARD-CORRECTED — OUTPUT TOOTH DEAD:** pack and indices fire, LSE moves 1.5476, but per-head output move is **0.04518 vs the 0.05 cap** (0.90×); the output negative arm reads green. The cap is clean-run-set with no supremum (gate comment :108), so it sits just above the largest effect a correct dominant removal produces — bound-provenance issue, f4. **[construction only]** for the output tooth. |
| gdr_decode (`gdr_decode_parity.rs`) | V floors ≤1e-2 (state 2e-3), rel caps 2e-2/2e-3 | conv_out/conv_state/gdr_out want[0]+=1.0; state[0]+=1.0 | +1 forces a violator at all four floors | Fires |
| gdr_varlen (`gdr_varlen_parity.rs`) | V floors 1e-2..2e-2 (`viol==0`) | conv/ring/gdr +1.0, state +0.5 at slot 0 | +1 forces per-element violations | Fires |
| marlin_fp4 (`marlin_fp4_correctness.rs`) | W worst-element/row-max <1e-2 | ×3 whole CPU reference over all checked rows | max_rel ≈ 2 | Fires (~200×) |
| marlin_fp8 (`marlin_fp8_parity.rs`) | N slab rel-L2 8e-3 (worst-el computed and unused) | ×3 whole slab (fires); localized byte is invisible | ×3 ≈2 fires; one column ~4.4e-2 = 0.55× bound | **Flagged** |
| marlin_w8a16 (`marlin_w8a16_parity.rs`) | N full-output rel-L2 0.20, no worst-el | ×3 full reference (fires); localized corruption washes | ×3 ≈0.67 fires (~3.3×); localized <1× at size | **Flagged (shape)** |
| moe_routing (`moe_routing_parity.rs`) | E per family (index/count/offset/m-indices/combine); weights W rel 2e-5; pack **set compare per span**; totals E | seven discrete sabotages incl. route-value flips; specificity asserts each trips only its family | each changes the value/multiset its comparator reads | Fires |
| paged_quant_attn (`paged_quant_attn_parity.rs`) | V/order-one + clean-oracle mismatch (see templates) | all attended V bytes of row 0 pinned to dtype max | order-one magnitude argument, not a computed ratio vs the bound | Fires via the structural clean-oracle-mismatch clause — **[construction only]** for the numeric magnitude |

## moe_routing pack comparator — post-#436 note

As of dbfc04faa (#436) the pack family compares each expert span as a
**multiset**: both spans are sorted (`got_span.sort_unstable()`,
`want_span.sort_unstable()`) before `==`, because within-span slot order is
not a kernel contract. A positional swap therefore does NOT trip it — the
comparator structurally cannot see one, which is the #436 defect. The negative
tooth was changed in the same PR to a route-value flip
(`expected.orc.packed_route_slot[0] ^= 1`) that changes the span's multiset,
so the tooth fires against the set compare. Do not reintroduce a positional
swap tooth for this family without making the comparator ordered.

## fa2_sm70 — gate uninformative on the box that runs it

`fa2_sm70_parity.rs` targets compute capability sm_70. The parity batch runs
on H20 (sm_90). If the sm_70 kernel cannot launch on sm_90, both its red and
green arms are uninformative on every batch run it takes part in — it neither
passes meaningfully nor exercises its negative control on that hardware. This
is independent of washout arithmetic; it needs a separate determination of
whether the binary launches on sm_90 (or the gate must run on an sm_70 card).
Out of scope for this audit beyond noting it.
