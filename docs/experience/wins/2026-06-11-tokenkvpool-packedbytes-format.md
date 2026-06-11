# TokenKVPool PackedBytes Format — Paged Conversion P1 (#85) — pending-remote

## Goal

P1 of [`docs/plans/2026-06-11-dsv4-paged-kv-conversion.md`](../../plans/2026-06-11-dsv4-paged-kv-conversion.md):
generalize the ONE device paged pool with an MLA-style packed-record format
instead of building a DSv4-private pool — the unified-abstraction rule's
enforcement (a `Dsv4PageTable` draft was reverted on the spot when ckl
called it).

## Hypothesis

`KVFormat::PackedBytes { bytes_per_token }` (canonical 584 B latent record,
`kv_heads=1`, single plane in `k_data`, no scales/norms, page=64) can ride
every existing `TokenKVPool` mechanism — mirror/epochs, retain/release,
attach, `copy_pages_to_host/from_host` (the #82/#83 tier transport),
metadata builders — with only sizing and plane-count touched.

## Params

- `kv_types.rs`: the variant + `packed_record_bytes_per_token()` as the
  single packed-sizing routing point; `bytes_per_element` panics as a
  tripwire (returning a fake 1 would silently mis-size 584 B records);
  `stable_tag` 13 for the canonical 584 B shape (TurboQuant per-shape
  policy); page default 64; no work buffer.
- `paged_kv.rs`: `validate_format_shape` (kv_heads=1, free fn for no-GPU
  tests); single-plane construction (`v_data` empty); budget/storage =
  layers × tokens × bytes_per_token (head_dim proven inert by test); page
  copies move K-plane only; **bf16/int8/fp8/tq migration kernels
  `ensure!`-bail for packed records** — un-guarded, the bf16 path would have
  launched a `kv_dim×2`-bytes kernel into a 584 B/token buffer (device OOB).
- Implemented by a delegated agent against the tranche spec; full KVFormat
  match-site audit (sized-through vs bails-until-P2) reviewed.

## Env / Results

Local Apple Silicon. cuda-kernels 7 tests (3 new), infer-cuda 59,
infer-core 50, `cuda,no-cuda` typecheck clean, zero new clippy warnings.
**pending-remote**: the device consumer is P2 (DSv4 band arena → shared
pool); pod gate per the plan (needle ×3 @ 4K/32K/128K, table-vs-identity
same-binary A/B, c-sweep license). P1 alone changes no default behavior —
no format flips, no serving path touched.

## Learnings

The exhaustive-match compile errors ARE the audit: adding the variant
surfaced every sizing/dispatch site, and the per-site verdict (route through
`bytes_per_token` vs bail-until-P2) is exactly the §0.1 enumeration applied
to an enum. The tripwire-panic choice over fake values matters for packed
records — a silent `bytes_per_element=1` would have produced plausible but
~4.5× undersized pools.
