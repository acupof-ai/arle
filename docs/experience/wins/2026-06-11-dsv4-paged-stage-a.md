# DSv4 Paged Conversion Stage A — Table-Routed Identity Layout (#85 P2) — pending-remote

## Goal

Stage A of [`docs/plans/2026-06-11-dsv4-paged-kv-conversion.md`](../../plans/2026-06-11-dsv4-paged-kv-conversion.md):
replace DSv4's raw FP8 band arena with the shared
`TokenKVPool(PackedBytes 584B, page=64)` and route every read/write/reset/
swap path through per-slot page tables — while the physical layout stays
**byte-identical** to the band arena (identity tables, verified at
construction), so behavior is unchanged by construction and the
table-vs-identity same-binary A/B lever exists for the P3 pod gate.

## Naming rule (ckl, 2026-06-11)

The 64-token unit is a **page** everywhere in this codebase; the per-slot
mapping is a **page table**. The word "block" appears ONLY at the FlashMLA
FFI boundary, whose API calls our page a block (`block_table`,
`page_block_size`). A mid-flight `dsv4_block_table.rs` was renamed to
`dsv4_page_table.rs` with `physical_page` / `contiguous_page_table_byte_range`
and all prose unified.

## Params

- `Dsv4LayerKvLayout`: `flashmla_fp8_kv_pool: CudaSlice<u8>` band →
  `Option<TokenKVPool>`; `flashmla_slot_range` (band arithmetic) **deleted**
  — the compiler forced every caller through the new accessors
  (`flashmla_page_table` = the pool's own `page_indices`, no parallel
  table state; `flashmla_pages_byte_range` = identity-guarded byte range).
- Construction allocates every slot's pages up-front in slot order and
  `ensure!`s the identity run per slot; TP-lockstep invariant documented
  (tables derive from construction constants, rank-identical).
- Two conventions, both documented at the call sites:
  - host-built bulk packs (`flashmla_pack_sw_ring`) hand the kernel
    **physical page ids + the whole-pool base** — already valid for
    fragmented tables (Stage-B-ready);
  - device-computed single-token/compressed packs and the decode consumer
    still use band-base views, **licensed by the identity guard** — a
    fragmented table fails loudly instead of aliasing another slot
    (`contiguous_page_table_byte_range`). Stage B hands these kernels a
    device-resident page table.
- Whole-slot swap images now move via the pool's `copy_pages_to_host/
  from_host` — the same transport the #82/#83 page tier uses, so the
  serializer survives table fragmentation unchanged.
- `TokenKVPool::k_data_slice_mut` added (packed pools are single-plane;
  memset/D2D restore need a mutable slice view).

## Env / Results

Implemented ~80% by a delegated agent (stopped mid-run and taken over;
its incremental-compile discipline meant the tree was green at takeover),
finished + renamed + cast-cleanups by hand. Local: infer-cuda 65 tests
(page-table identity properties + gapped-table rejection among 6 new),
cuda-kernels 7, infer-core 50, `cuda,no-cuda` typecheck clean, clippy
17→14 warnings (3 fixed in touched hunks; rest pre-existing elsewhere).

**pending-remote** (P3 pod gate): needle ×3 @ 4K/32K/128K +
same-config-twice on Stage A (expect byte-identical numerics), then the
c-sweep license. Stage A changes no default behavior.

## Stage B remainder (next commit, precise)

1. `ARLE_DSV4_PAGED=1` gate: DSv4 switches from the dummy host pool to
   `HostPagedKvPool(page_size=64)`; executor mirrors host tables per row
   (Qwen `mirror_slot` pattern) instead of the construction-time identity
   loop.
2. Admission flip rides the same gate: `cuda_admission_total_pages` DSv4
   max-reservation arm → paged budget; `kv_budget_num_slots` drops the
   arena term (sidecars/scratch stay max-reserved per the plan's v1 scope).
3. Device-resident page table for the device-computed pack/index kernels
   (removes the identity-guard restriction).
4. MTP draft appends allocate pages (Qwen spec-token precedent).

## Learnings

Deleting the band-arithmetic entry point (`flashmla_slot_range`) instead of
deprecating it made the compiler enumerate every conversion site — the same
exhaustive-match trick as P1, applied to a function. The two-convention
split (physical+pool-base where the table is host-built, identity-guarded
band views where ids are device-computed) kept Stage A honest without
pretending the device kernels are table-aware before they are.
