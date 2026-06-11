# DSv4 Paged-KV Conversion — #85 Route A as Mainline

Approved direction (ckl, 2026-06-11): convert DSv4's per-slot contiguous MLA
arena to paged KV management. Whole-slot swap (`6ec31f53`) stays as the
fallback tier route until parity, then demotes to backstop.

## Why (capacity arithmetic, verified against `kv_budget_num_slots`)

Every slot reserves at `max_seq_len`: FP8 arena (`max_seq_len × 584 B ×
num_layers`) ×2 overhead multiplier, plus DSA selector/compressor caches all
scaling with `max_seq/cr` (`crates/infer-cuda/src/dsv4.rs:974-1050`). At a
128K config that is ≈7–8 GB/slot/rank (MLA latent is `kv_heads=1`,
replicated per rank — TP does not shard it), so a ~30 GB KV budget serves
**~4 slots**; 256K → 2 (the #57/#67 pain). Agent-workload requests average
4–16K → reservation waste 16–60×. Paged allocation bounds slots by actual
tokens: **c=32+ in the same HBM**, multiplying directly into the #60 batched
lane (+57% @ c=8 already measured). Deletes the #57 band-aid and the #67
budget gymnastics.

## The cost-collapsing fact

FlashMLA `sparse_decode` is already block-table native: KV tensor
`[num_blocks, 64, h_k, d]`, per-token top-k `indices` ARE block selections.
Today's arena is a degenerate identity block table (contiguous per-slot
bands). **The attention kernel needs zero changes.** The conversion surface
is host-side: allocator, block-table construction, the pack paths that
assume band contiguity, and the indexer key-cache scan.

## Decision points (resolve in order; each is a tranche gate)

1. **Page size = 64** (FlashMLA block). Host `HostPagedKvPool` already takes
   arbitrary `page_size`; engine radix block size follows `kv.page_size()`.
2. **What gets paged in v1: the FP8 MLA arena only** (the dominant ×2'd
   term). DSA key cache + compressor compressed caches stay per-slot
   contiguous in v1 but sized by the REQUEST's `max_total_tokens` at
   admission instead of global `max_seq_len` (intermediate reservation win
   without paging their scan kernels). Their paging is v2.
3. **SWA ring stays a per-slot ring** (2 layers, fixed window, small).
4. **Sidecar boundary state (pending/prev_overlap)**: unchanged in v1 —
   prefix cache stays DISABLED for DSv4 until the snapshot protocol (#85
   original scope) lands on top of paging. v1's win is capacity, not
   sharing.
5. **Lockstep**: page tables must be rank-identical. They derive from the
   host pool, which is plan-driven and deterministic — same invariant as
   today's slot accounting (NCCL min-reduce stays for the budget).

## Unified-abstraction rule (ckl, 2026-06-11 — second enforcement)

Paged KV is ONE abstraction with model adapters, never a per-model
implementation (`feedback_unified_abstraction_not_per_model`). The host side
already is one (`HostPagedKvPool` + engine + radix, any `page_size`). The
device side must be too: **no `Dsv4PageTable` / DSv4-private pool** — the
page-table mirror, occupant epochs, retain/release, `copy_pages_to_host`/
`from_host` (which is what the T1/T2 tier consumes!), and the block-table
metadata builders already exist once in `cuda-kernels::TokenKVPool`. The
ONLY genuinely model-specific pieces are (a) the record layout and (b) which
sidecar caches exist — those are the adapter.

## Tranches

- **P1 (cuda-kernels, host-logic CPU-testable)**: generalize `TokenKVPool`
  with a packed-record format — `KVFormat::PackedBytes { bytes_per_token }`
  (or equivalent): `kv_heads=1`, single plane (no V buffers — MLA latent is
  one packed record: 448B FP8 NoPE + 64×2B BF16 RoPE + 8B e8m0 = 584 B), no
  separate scales/norms (embedded in the record), `page_size=64`. Everything
  else — `mirror_slot`, `page_indices`, `slot_epochs`, `attach_pages`,
  `retain/release_pages`, `copy_pages_to_host/from_host`,
  `build_paged_kv_metadata` — is inherited untouched. Unit tests on the new
  format mirror the BF16 ones.
- **P2 (infer-cuda, typecheck on Mac, pod-verified)**: DSv4's
  `flashmla_fp8_kv_pool` band arena is REPLACED by a
  `TokenKVPool(PackedBytes(584), page=64)`; DSv4 switches from the dummy
  host pool to a real `HostPagedKvPool(page_size=64)` and the executor
  mirrors host tables per row exactly like the Qwen arm. Prefill writes,
  `flashmla_pack_sw_ring`, compressed-delta pack, and the top-k index
  translation route through the pool's block table; decode graph capture
  keyed on table length (Qwen `GraphBucket` precedent). Admission accounting
  moves off `cuda_admission_total_pages`'s DSv4 max-reservation arm in the
  SAME tranche (host-paged budget must flip together with device paging or
  admission over-promises arenas the device still max-reserves).
  `kv_budget_num_slots` shrinks to the non-paged terms (sidecars + scratch).
  Bonus that falls out for free: the page-granular T1/T2 tier (#82/#83)
  covers DSv4 through the SAME `copy_pages_*` calls the Qwen arm uses.
- **P3 (pod)**: needle gate ×3 + same-config-twice at 4K/32K/128K, c-sweep
  vs the slot-arena baseline (same binary, table-vs-identity A/B), TTFT/ITL
  license per bench spec. KILL: any needle regression, or c-sweep showing
  the indirection costs >2% ITL with no capacity gain at the SLO shape.
- **P4**: prefix-sharing on top (original #85 sidecar snapshot protocol);
  whole-slot swap demotes to backstop; page-tier (#82/#83) covers DSv4.

## Interactions

- #60 batched lane: P1's admission change is the same accounting the batched
  lowering touches — coordinate to avoid double-churn.
- #62/#70 MTP/graphs: P2's graph re-capture keying must cover MTP's extra
  appends (draft tokens allocate pages like spec tokens do on the Qwen path).
- Whole-slot swap images carry the band today; after P2 they carry
  page lists — the serializer's full-band copy becomes per-page copies
  (mechanical change, the image abstraction already hides it).
