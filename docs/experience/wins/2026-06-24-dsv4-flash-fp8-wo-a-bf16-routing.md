# DSv4-Flash-FP8 dense-BF16 `wo_a` routing (C2 follow-up)

`pending-remote` — CUDA-only, pod needle-gate running on the 8×H20 box.

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

## Pod gate (pending)

`/host/DeepSeek-V4-Flash-FP8` on TP=8/EP=8: (1) full load → engine-ready,
(2) `needle_gate.py` same-config-twice within the non-determinism floor,
(3) `strings target/release/arle | grep "DSv4 dense sharded"` symbol-check.
Results fold in when the devops run reports.

## Rule

An FP8 re-serialization may leave individual low-rank weights dense BF16 even when
every block-scaled projection passes — dtype-route per weight, don't assume a
checkpoint is uniformly quantized. Scope the cheap fix to the production TP shape
(`groups==1` rides the existing dense GEMM) and error clearly on the unbuilt path
rather than writing a kernel no production config exercises.
