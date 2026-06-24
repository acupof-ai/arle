# DSv4 Stage-B — dynamic shared KV pool (kernel spec)

Status: spec, ready to execute. The load-bearing remainder of the unified KV-memory
architecture ([design](2026-06-24-unified-kv-memory-architecture.md) Δ-B). Engineering-
bound (no algorithmic novelty), ~9-14 days, bit-identity-gated. Grounded in the
`Map DSv4 Stage-B kernel work` survey (2026-06-24).

## Goal
Drop DSv4's Stage-A per-slot identity pre-alloc (`total = tokens_per_slot × num_slots`,
`attention.rs:745-808`); make the MLA latent KV + the persistent DSA caches a dynamic
shared pool drawn per-request from free-VRAM (like dense Qwen3 / SGLang). This is the
only thing that lifts DSv4's num_slots ceiling — there is no contained shortcut.

## What's reusable (the leverage)
- **FlashMLA decoder ALREADY accepts a device `block_table`** (read-side,
  `vendor/flashmla/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`) — **no decoder change.**
- **Dense Qwen3 `physical_token_rows(table, page_size, start, count)`**
  (`paged_kv_table.rs:52-75`) flattens logical→physical for identity-assuming kernels —
  DSv4 reuses this exact pattern.
- **`HostPagedKvPool`** (shared LIFO free-stack, dynamic `alloc`) — no change.
- The band-addressing gate `contiguous_page_table_byte_range` (`paged_kv_table.rs:90-117`)
  is what Stage-B retires for the dynamic path.

## The change-list (file:line)
1. **Pack kernel — `cuda-kernels/csrc/attention/dsv4_fp8_kv_pack.cu:152-307`** (~2-3d).
   Today: `block_base = packed_kv + block_id × (page_block_size × 584)` with `block_id =
   token_block_id[t]` (host-populated identity). Change: accept a device page table
   `[num_logical_pages]→physical`; `logical = token_idx / page_block_size`;
   `block_id = table[logical]`. Quantization body unchanged. **GATE: identity table
   (`table[i]=first+i`) must produce BYTE-IDENTICAL output to Stage-A** — unit test +
   needle.
2. **DSA index kernels — `cuda-kernels/csrc/misc/dsv4_dsa_official.cu`** + the DSA
   caches (`attention.rs:818-829`, budget `dsv4.rs:1574-1646`) (~3-5d). The rotated_keys
   + compressor/indexer compressed caches are PERSISTENT per-request state → move into
   the paged pool (Option A: extend `TokenKVPool` with DSA records; alloc per-request on
   admission), pass the device page table to the index kernels (same lookup as #1).
3. **Rust dispatch — `attention.rs:745-808`** (~2-3d). Delete the Stage-A pre-alloc loop
   (`alloc_tokens(slot, tokens_per_slot)` ×N, lines 784-796) + the `×num_slots` total.
   Size the pool from profiled free-VRAM (the dense path); `alloc(slot,tokens)` on
   admission / `free_slot` on release; extract the device page table via the existing
   `mirror_slot` pattern (Qwen3) and pass to pack/index. Then re-enable high num_slots.
4. **Budget — `dsv4.rs:1574-1646`**: drop the per-slot×max_seq DSA terms from
   `kv_budget_num_slots` once they live in the shared pool (they stop multiplying slots).

## Phasing (each pod-verified, bit-identity first)
1. Pack kernel page-table lookup + a host bit-identity unit test (identity table ==
   Stage-A). Land behind a flag, default Stage-A.
2. Dispatch: dynamic alloc + page-table extraction; flip the pack kernel to the table.
   Pod: needle exact ×3 DET (correctness) at the OLD num_slots (no regression).
3. DSA caches → paged pool + index-kernel page table. Pod: needle exact.
4. Drop the budget per-slot DSA terms; pod: **num_slots=32 boots + c跑 scales past the
   slot-4 ceiling** (the payoff), VRAM fits.

## Gate
- Bit-identity: identity page table → byte-match Stage-A pack output (offline + unit).
- Correctness: `needle_gate.py` exact ×3 DET (the MoE-non-det floor) at each phase.
- Payoff: DSv4 EP num_slots=32 boots (was OOM) + concurrent c跑 throughput scales.

## Risks
- Off-by-one page-table math fails silently → the bit-identity gate is mandatory per phase.
- DSA caches are stateful (grow with seq) → allocate per-request upfront, never mid-seq.
- FlashMLA pack-output byte layout must match the decoder's block_table expectations
  exactly → offline bit-match before landing.
