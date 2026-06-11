# DSv4 DeepGEMM contiguous grouped GEMM: unaligned segments cross BLOCK_M tiles → boundary rows use the wrong expert

## Context

Found 2026-06-11 while porting the Qwen3.5/3.6 BF16 m-grouped path
(`95058ceb`): DeepGEMM's `MGroupedContiguous` scheduler resolves the B
(expert) group **once per BLOCK_M=128 output tile** —
`m_indices[m_block_idx * BLOCK_M]` (`vendor/deepgemm/.../scheduler/gemm.cuh
get_global_idx`; BLOCK_M is pinned to 128 by
`get_mk_alignment_for_contiguous_layout`). The contract is therefore: every
128-row tile of the A matrix must belong to ONE expert, i.e. per-group row
counts must be 128-aligned (pad rows between groups).

The **DSv4 prefill contiguous path** (`deepgemm_grouped_experts` in
`crates/cuda-kernels/src/moe.rs` dsv4 region) builds compact UNALIGNED
segments from the plain exclusive scan: group boundaries fall mid-tile, so
the rows of a tile that spans two experts are all computed against the
FIRST expert's weights — silent wrong-expert output for boundary rows.

## Root Cause

`m_indices` is per-row metadata but the kernel samples it per-tile;
compact packing violates the per-tile single-group precondition. Pinned by
CPU tests in `crates/infer-cuda/src/moe.rs`:
`compact_layout_violates_per_tile_group_contract` (negative) and
`aligned_layout_satisfies_per_tile_group_contract` (positive).

## Fix

Not yet applied to DSv4 (left byte-identical in `95058ceb` — single-variable
discipline). The new BF16 path uses `moe_exclusive_scan_aligned_i32` +
padded pack and never copies the broken pattern. DSv4 follow-up: switch its
prefill contiguous call to the aligned scan + padded pack (the decode
contiguous variant — per-route 128 tiles — is NOT affected), then re-run the
DSv4 needle gate; impact magnitude depends on how often prefill routed
counts are non-multiples of 128 (almost always at real batch shapes).

## Rule

- A grouped-GEMM layout contract lives in the SCHEDULER's index math, not
  the API signature — read `get_global_idx` (or its equivalent) before
  building the host-side layout, and pin the contract with a CPU test that
  would fail under compact packing.
