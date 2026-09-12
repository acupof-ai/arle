# Parity gates passed negative controls that never perturbed a kernel input

Date: 2026-09-13. Found in the P1 registry parity batch: eleven of 22 gates
reported green with a negative control that cannot detect a wrong kernel.

## Context

`scripts/parity_gpu_batch.sh` runs each `*_parity.rs` example once positive
and once with `--negative-control`, and keys the negative verdict on the
literal marker `NEGATIVE CONTROL OK`. A device-input audit of all 22 gates
(file:line evidence per row below) found the negative flag perturbs a real
device buffer on only six gates. On fifteen gates it changes a host-side
value — the reference/expectation, or the captured output after
device-to-host — and on one gate there is no negative arm by design.

Cross-referencing the six device teeth against the P1 results
(the batch's `results.tsv`, status column: 11 PASS / 9 FAIL / 2 SKIP):
none of the six is among the eleven PASS rows. Every gate that reported green
in that batch did so on a host-side tooth. All six device teeth sit on gates
that failed, panicked, or skipped. The marker text is identical in both
shapes, so the batch output could not distinguish them.

Three shapes exist, not two:

1. **Device-input sabotage.** A byte, scale, index, start position, or
   pointer sent to the kernel is corrupted; the reference is rebuilt from
   the same corrupted inputs. A kernel that ignores that input or computes
   the wrong result fails.
2. **Oracle/reference corruption.** Only the host expectation changes
   (`want[0] += 0.5`, `reference * 3.0`). The kernel runs on identical
   inputs in both arms.
3. **Post-dtoh result corruption.** The device output is copied to host and
   then edited before comparison (`moe_routing` clones the completed
   `Bundle` and edits host `Vec`s; `dspark_dsa` overwrites
   `got_raw[0] = argmin` after `clone_dtoh`). The kernel never sees a
   changed input.

Shapes 2 and 3 have the same blindness: the kernel's output need not reach
the comparator, and the kernel need not launch at all, for the tooth to
fire.

## Evidence

| gate | shape | evidence (`crates/infer-cuda/examples/`) |
|---|---|---|
| fa2_sm70_parity | device | `fa2_sm70_parity.rs:336-341` spikes K (`k[0]=4.0`) before htod; oracle stays clean |
| dsv4_tp_oproj_parity | device | `:467-476` `Sabotage::Slice` mutates the device slice buffer; `:495` DgCacheMap; green-baseline guards `:746`,`:840` |
| flashmla_prefill_parity | device | `:501-517` bad selected-index buffer htod → `flashmla_csa_build_indices_raw`; oracle rebuilt from it `:535` |
| flashmla_hca_decode_parity | device | `:383-391` bad `start_pos` htod → forward. Separate defect: OUT tooth washed out by a dominant-key fixture |
| flashmla_sparse_decode_parity | device | `:397-405` `bad_sel[slot]=-1` htod → forward; same washout defect |
| dspark_drafter_batch_invariance | device | `dspark_batch_invariance.rs:243-256` swaps launch starts and layer-0 K/V ring DeviceVecs (real pointers; honored at `dspark.rs:1296-1328`); kv tooth weak, 1.54e-2 vs the 2e-2 floor |
| paged_quant_attn_parity | oracle | `paged_quant_attn_parity.rs:453` reaches only `support/attn_common.rs:171` expectation `w += 0.5`; `build_case:204` takes no corrupt flag |
| fa3_hd256_shim_parity | oracle | same `compare_rows` expectation-only path |
| argmax_parity | oracle | `:165-167`,`:205-207` flip the expected index |
| marlin_fp8_parity | oracle | `:488-489` `bad_reference = 3.0 * reference`; neither htod site is reachable from the negative flag |
| marlin_w8a16_parity | oracle | `:338-340` `bad_ref = 3.0 * ref_out` post-dtoh |
| marlin_fp4_correctness | oracle | `:198-201` CPU expectation `*= 3.0` |
| elementwise_parity | oracle | eight branches, all host-side: rms `:201-202`,`:219-220`; silu `:250-251`; split `:276-277`,`:327-329`; qkv `:370-372`; embedding `:459-460`; uploaded buffers untouched |
| dspark_sampler_parity | oracle | `:380-381`,`:415-416`,`:431`,`:450`, chain `:598-602`; all expectation/verdict edits |
| dsv4_decode_moe_parity | oracle | `:435-442` adds the family bias to that family's oracle only |
| gdr_decode_parity | oracle | `:419-444` and singular `:562-590` mutate `want_go`/`reference.state` around the check; device inputs identical |
| gdr_varlen_parity | oracle | one `DeviceRun`; `compare()` `:497-515` adds bias in the host comparison; `bound()` `:532-536` |
| dspark_draft_attn_parity | oracle | `bound()` `:532-536` adds bias to expected `w`; `evaluate()` `:616-633`; device rings untouched |
| deepgemm_grouped_prefill_parity | oracle | `:346-348` and grouped `:598-600` add `5|w|` to the reference element; both arms currently panic at `:399` (separate harness bug) before reaching it |
| dspark_dsa_parity | oracle + post-dtoh | `want[0] += 1` `:300`; `want_scales[0] *= 2` `:447`; indexer overwrites `got_raw[0] = argmin` `:574-581` after dtoh |
| moe_routing_parity | post-dtoh | `verify_negative_teeth` `:782` clones the finished `Bundle`, edits host Vecs (`:800`,`:801`,`:803-805`,`:807-810`,`:820`,`:828`,`:831`) after the pipeline ran; no edit reaches a device buffer |
| dsv4_parity | none | allowlisted, no negative arm; multi-rank model gate, SKIP without `INFER_DSV4_MODEL_PATH` |

Tally: 6 device, 14 host-side (13 oracle-only plus dsa's mixed path),
1 post-dtoh (moe), 1 no arm. 22 rows.

## Root Cause

The gate convention specified only that a negative arm must fail; it did not
specify what the negative flag must perturb. Corrupting the expectation is
the cheapest implementation and it exercises one real property: the
comparator registers a deviation of that magnitude, so it cannot be
comparing a buffer against itself and its tolerance band cannot be
infinitely wide. Both defects have occurred in this repository. The
convention stopped there and never required the second property: that a
fault in the thing the gate names — the kernel — propagates to the
comparator. A self-comparing comparator and a dead kernel are two different
failure modes, and one tooth per gate was treated as covering both.

The identical marker sealed the ambiguity: both shapes print
`NEGATIVE CONTROL OK`, so a batch run aggregates them as one population and
the distinction is invisible without reading every negative branch. It was
invisible until this audit.

## Fix

Additive. The host-side teeth stay: comparator liveness is a real guarantee
and they carry it cheaply. Each gate gets a second tooth with the second
job — corrupt a device input that the compared output causally depends on,
rebuild the reference from the corrupted input, and require the comparison
to fail. Two teeth, two jobs.

Pilot is paged_quant_attn: its diagnosed fix is exactly the missing device
tooth (flip a real int8 K/V byte or per-row scale on a row the
`num_splits=1` comparator reads), and the pilot reports diff size and
gate-specific vs shared cost before the remaining gates are scoped. The
drafter and flashmla hca/sparse fixes already diagnosed follow: strengthen
the drafter kv-base perturbation (floor fixed at 2e-2), and de-dominate the
attention fixture so the index→output chain is tested while keeping the
dominant-key case as an additional positive. Every device tooth is shown
red with the corruption in place and green with it removed; red must be
demonstrably caused by the kernel seeing bad data, not by the edit itself.

## Rule

A negative control has to name what it perturbs. A tooth that changes the
reference or the captured result proves the comparator is alive and
nothing else; a gate about a kernel needs an additional tooth that changes
an input the kernel reads, and that tooth's red must be shown to come from
the kernel's output. The batch marker cannot encode the distinction — the
gate code must, and a gate audit classifies the negative branch by where
its edit lands (device input / host reference / post-dtoh result), never by
the marker text.
