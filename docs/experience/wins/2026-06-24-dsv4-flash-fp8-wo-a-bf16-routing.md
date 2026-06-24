# DSv4-Flash-FP8 dense-BF16 `wo_a` routing (C2 follow-up)

**Pod-verified 2026-06-24** on 8×H20 TP=8/EP=8 (commit `91822235`).

## Context

The F32-block-scale fix ([`a8ca816d`](../../../)) got `/host/DeepSeek-V4-Flash-FP8`
through every FP8 projection, but that checkpoint serializes the tiny low-rank
attention output down-projection `wo_a` as **dense BF16** (no `.scale` sibling),
unlike native DSv4 (FP8 block-scaled). The DSv4 loader hard-required `wo_a.scale`,
so the checkpoint failed to load on the last weight. `wo_b` and every other
projection are FP8 and already passed.

## What Worked

Commit `91822235`. Dtype-route `wo_a` in `load_dsv4_attention`:

- `F8_E4M3`/`I8` → unchanged block-scaled sharded path (grouped route-GEMV +
  DeepGEMM levers).
- `BF16`/`F32` → new `load_dsv4_bf16_sharded` (dense twin of the block-scaled
  sharded loader; `Shard::Column{0}`/`Replicated` — the only shards `wo_a` uses).
- `build_dsv4_wo_a_group_tables` grows a `DenseBf16` branch carrying only the
  group **shape** (the grouped route-GEMV kernel is FP8/FP4-only, so its
  pointer/scale tables are dead weight for dense).
- DeepGEMM-cache gate splits: `wo_b` stays FP8-cached; `wo_a` caches build only
  when `wo_a` is itself block-scaled.

**No new kernel.** DSv4-Flash `o_groups=8` → on the production TP=8 pod each rank
owns exactly one o-group (`groups==1`), so `mla_oproj` already runs the dense
weight through `dsv4_linear`→`gemm_batch`. `groups>1` BF16 (TP1/2/4) errors clearly
in `dsv4_wo_a_grouped_linear` (upgrade path: a bf16 grouped route-GEMV kernel).

native FP8 / oniond `wo_a` are byte-identical — the FP8 arm is untouched, no
regression.

Local verify: `cargo check` + `cargo clippy` `cuda,no-cuda` clean.

## Pod gate (passed)

`/host/DeepSeek-V4-Flash-FP8` (274 GB FP8, 46 shards) on TP=8/EP=8, built in an
isolated clean tree at HEAD `d965f093` (`--features deepep` +
`ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`, BUILD_EXIT=0 in 6m15s):

1. **Load → engine-ready ✓.** All 8 ranks pass the attention weights with **zero**
   `wo_a.scale` / block-scale / missing-tensor errors (`grep -c` = 0); uniform
   41577 MiB/GPU (predicted 41570). `GET /v1/models` → 200. This is the exact step
   that hard-failed before the fix.
2. **Needle DET ✓** (`needle_gate.py`, `RAW=1 TEMPLATE=dsv4`, needle `738291`):
   len 115/446/2000 → exact=3 DET; len 8000 depth=0.5 → exact=2, the one NONDET a
   benign markdown diff (`738291.` vs `**738291**.`, both correct — expected MoE
   non-determinism, not a miss). Free-form output coherent + factual, no garbage.
3. **Symbol-check ✓.** `strings arle | grep` hit all 4 fix markers
   (`DSv4 dense sharded load supports Column{dim:0}/Replicated`, the three
   `DSv4 dense wo_a …` messages) — the binary built the right tree.

## Rule

An FP8 re-serialization may leave individual low-rank weights dense BF16 even when
every block-scaled projection passes — dtype-route per weight, don't assume a
checkpoint is uniformly quantized. Scope the cheap fix to the production TP shape
(`groups==1` rides the existing dense GEMM) and error clearly on the unbuilt path
rather than writing a kernel no production config exercises.
