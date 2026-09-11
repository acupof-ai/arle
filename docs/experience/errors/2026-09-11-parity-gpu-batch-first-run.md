# First registry parity GPU batch: 7 PASS / 12 FAIL / 2 SKIP (all FAIL undetermined)

## Context

`scripts/parity_gpu_batch.sh` (#325) runs every example listed in
`operators/registry.toml` `correctness_gate` — 21 gates — once positive and
once under `--negative-control`, claiming one free GPU through
`scripts/pick-gpu.sh`. This was the first end-to-end execution of the whole
registry on H20 hardware; until now the gates were merged code with zero GPU
runs.

- Source commit: `16e2ad5bb` (origin/main at run time)
- Kernel bundle: `bundle:1a11735f68f986f1cb0d9d5d51b2e0f539f8d0dfbc220df2f105e17e12482632`
- GPU: 0 (H20, `cc=9.0 sms=78`, not currently claimed by another process)
- Window: 2026-09-11 08:00:03Z → 08:21:33Z (~21 min)
- prereg row: `parity-gpu-batch-20260911080003`
  (`docs/experience/prereg.jsonl`, status `rejected`, result `pass=7 fail=19
  skip=2` — the fail count double-counts a positive failure with its missing
  negative control; the gate count is 12 FAIL)
- Built release `--features cuda,nccl` in the pod build tree, 6m15s, no
  compile errors (21/21 examples built).

The batch completed on its own and wrote results before the 09:15Z V4.1 gate
run reserved GPUs 0–5; no gate was interrupted, so every row below is a
complete run. "no negative control" means the batch skipped the negative arm
because the positive arm failed first; it is not a separate gate.

### Results (all 21 rows)

PASS means positive printed `ALL PASS` (rc=0) and the negative arm printed
`NEGATIVE CONTROL OK` (rc=0).

| Gate | Result | First failing line / note |
|---|---|---|
| argmax_parity | **FAIL** | `Error: Sync failed: DriverError(CUDA_ERROR_MISALIGNED_ADDRESS, "misaligned address")` — the two checked cases themselves passed (`got=3 want=3`); negative arm not reached. |
| deepgemm_grouped_prefill_parity | **FAIL (SM90)** | panic at `crates/infer-cuda/examples/deepgemm_grouped_prefill_parity.rs:280:16`: `index out of bounds: the len is 16 but the index is 16`. Negative arm not reached. |
| dspark_draft_attn_parity | PASS | `[dspark-draft-attn] ALL PASS` / `NEGATIVE CONTROL OK` |
| dspark_dsa_parity | **FAIL** | `[indexer fused_q rows=256] scale_rel_worst=1.00e0 violators=16381 FAIL … Error: dspark-dsa-parity FAILED family: indexer` (topk cases PASS). Negative control OK. |
| dspark_sampler_parity | **FAIL** | `Error: dspark-parity FAILED families: filter probs, draft q row, draft token id`; B=8 combined rel-L2 up to 1.44, maxExcess up to 3.9e-2 across lanes 1–7. Negative control OK. |
| dsv4_decode_moe_parity | PASS | `[dsv4-decode-parity] ALL PASS` / `NEGATIVE CONTROL OK` |
| dsv4_parity | SKIP | `needs multi-rank model launcher + INFER_DSV4_MODEL_PATH` (both arms; allowlisted to have no negative mode) |
| dsv4_tp_oproj_parity | **FAIL (SM90)** | head o-slice exact and `[tp=1 s=1 gemv] PASS latent+partial-sum in bound (l2=0.000e0)`, then `[tp=1 s=1 deepgemm] rank=0 latent l2=1.000e0 / Error: tp=1 s=1 deepgemm: wo_a latent FAILED`. Negative arm not reached. l2=1.0 indicates a wrong layout/construction, not precision. |
| elementwise_parity | **FAIL** | `Error: elementwise-parity FAILED family: split2 first half`; `[split2-first qwen36-27b d=17408 B=8] n=139264 mismatches=121856`, `[split2-first dsv4 d=2048 B=8] n=16384 mismatches=14336`. Negative control OK. |
| fa2_sm70_parity | SKIP | `requires sm_70, device is sm_90 … run on the V100 box` (both arms) |
| fa3_hd256_shim_parity | PASS (SM90) | `[fa3-hd256-shim-parity] ALL PASS` / `NEGATIVE CONTROL OK` |
| flashmla_hca_decode_parity | **FAIL (SM90)** | `Error: pack family: decoded pool differs from BF16 sources (nope 1, rope 0.00012206659)` at b=8. Negative arm fails identically. |
| flashmla_prefill_parity | **FAIL (SM90)** | CSA s_q=128 numeric lines in bound (`out=0.00080 lse=0.00001 maxlogit=0.00011`) then `Error: CSA s_q=128 pack mismatch`. Negative arm not reached. |
| flashmla_sparse_decode_parity | **FAIL (SM90)** | `Error: pack family: decoded pool differs from BF16 sources (nope 1, rope 0.00012206659)` at b=8. Negative arm fails identically. |
| gdr_decode_parity | PASS | `[gdr-parity] ALL PASS` / `NEGATIVE CONTROL OK` |
| gdr_varlen_parity | PASS | `[gdr-varlen-parity] ALL PASS` / `NEGATIVE CONTROL OK` (the varlen kernels it exercises are the ones #300 deletes) |
| marlin_fp4_correctness | PASS | `[marlin-fp4-parity] ALL PASS` / `NEGATIVE CONTROL OK (the GEMM family failed as required)` |
| marlin_fp8_parity | **FAIL** | numerics pass (`[lm_head …] m:PASS g:PASS`), then `[declined n%64 n=96 k=5120 …] expected repack decline, got a Marlin layout — shape is no longer a boundary / Error: marlin-fp8-parity FAILED family: gemv lane`. Negative control OK. |
| marlin_w8a16_parity | **FAIL** | same shape: `[attn_sq …] … ratio=1.00 PASS`, then `[declined n%64 n=96 k=5120 …] expected repack decline, got a Marlin layout — shape is no longer a boundary / Error: marlin-parity FAILED family: declined fallback`. Negative control OK. |
| moe_routing_parity | **FAIL** | `[mode=ScoreDecides tokens=8 ep=32] FAIL m_indices`, route-set token gaps ≥1e-5 against the expected ids, `weights[6] got=2.469071e-1 want=2.446012e-1`, then `Error: Sync failed: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered")`. Negative arm not reached. |
| paged_quant_attn_parity | PASS | `[paged-quant-attn-parity] ALL PASS` / `NEGATIVE CONTROL OK` |

Summary: **7 PASS, 12 FAIL, 2 SKIP** (dsv4_parity needs the model launcher;
fa2_sm70 needs the V100 box). The 7 PASS gates cover GDR decode/varlen,
DSv4 decode MoE, FA3 paged shim, paged quantized attention, DSpark draft
attention, and Marlin FP4 correctness — both arms each.

## Root cause

Cause unknown for all 12 FAILs; triage in progress (owners in the
`kernel-parity-gates` agenda note, 2026-09-11).

A failing parity gate has two possible causes that the batch cannot
distinguish:

1. **Gate bug** — out-of-bounds harness indexing, a mismatched layout
   convention between the example and production, or a stale boundary premise
   (the two Marlin `declined n%64` failures explicitly report the shape "is no
   longer a boundary").
2. **Kernel bug** — the production kernel or its glue returns wrong bytes.

Several failures point structurally at a layout/pack mismatch rather than
numerics (`l2=1.000e0`, `pack mismatch`, `nope 1`, or OOB/illegal-address
driver errors), which is more often a harness-construction error, but that is
not established without reading the code.

## Fix

Pending triage. Gate bugs get fixed directly in a lane; suspected kernel bugs
are reported with evidence before any production change. Fixed gates rerun
through `parity_gpu_batch.sh` on the pod build tree's release examples once
GPUs are released by the V4.1 run. `dsv4_parity` needs
`INFER_DSV4_MODEL_PATH` + the multi-rank launcher; `fa2_sm70_parity` runs on
the V100 host.

## Rule

A parity gate counts as coverage only after it has run positive and negative
on its target GPU. The first run of 21 merged gates had 12 fail, so
"code-complete, GPU pending" must not be reported as covered — the registry
rows before this run described executable gates with zero measured evidence,
and the coverage language now reflects what has actually run.
