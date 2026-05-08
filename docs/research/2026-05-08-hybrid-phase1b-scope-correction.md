# Hybrid Phase 1b §2.1 scope correction — 25-35 LOC across 2 files

> Per `6be30ce` Phase 1b directive originally estimated §2.1 detection
> patch at "5 LOC"。Direct code-grep of `weight_loader.rs:440-525` shows
> actual scope is **25-35 LOC across 2 files**(`weight_loader.rs` +
> `quant.rs`)。
>
> Updates pickup queue effort estimate but does not change recommendation。

## Actual code structure discovered

`infer/src/weight_loader.rs:441-449` `QuantLoadConfig` struct:
```rust
pub(crate) struct QuantLoadConfig {
    pub(crate) group_size: Option<usize>,
    pub(crate) bits: Option<u8>,
    pub(crate) tq_bits: Option<u8>,
    pub(crate) marlin_w4a8: bool,
    pub(crate) unsupported_reason: Option<&'static str>,
}
```

`from_meta` has **7 arms**(lines 451-509)each explicitly initializing
`marlin_w4a8`:
1. `QuantMeta::Gptq(config) if !config.sym` — unsupported
2. `QuantMeta::Gptq(config) if config.group_size > 0` — line 463
3. `QuantMeta::Gptq(config)` — line 470
4. `QuantMeta::Awq(config) if config.zero_point` — unsupported
5. `QuantMeta::Awq(config)` — line 483
6. `QuantMeta::Int8(_)` — line 490
7. `QuantMeta::MarlinW4A8(config)` — line 497(only sets `marlin_w4a8: true`)
8. `QuantMeta::TurboQuant(config)` — line 504

## Updated §2.1 scope

For Phase 1b loader to support `marlin_w4_hybrid`:

1. Add `QuantMeta::MarlinW4Hybrid(QuantMetaConfig)` variant in `infer/src/quant.rs` ~5 LOC
2. Add `marlin_w4_hybrid: bool` field to `QuantLoadConfig` — 1 LOC
3. Add new `from_meta` arm for `MarlinW4Hybrid` — ~5-10 LOC
4. Update all 7 existing arms with `marlin_w4_hybrid: false` — 7 LOC
5. Add `"marlin_w4_hybrid"` to `INFER_QUANT_FORMAT_OVERRIDE` line 514 — 1 LOC
6. Update `enabled()` predicate — 1 LOC
7. `quant.rs` config detection from safetensors metadata — ~10 LOC

**Total §2.1**:**25-35 LOC** across `weight_loader.rs` + `quant.rs`。

§2.2 tensor reading at `:663-715` is additional scope(reading BOTH
`marlin_*` and `marlin_w4a8_*` per Linear when hybrid flag set)。

§2.3 DeviceMatrix Option A extension(adding `hybrid_w4a8_qweight: Option<...>`
fields)is additional scope。

## Updated effort estimate

| Phase | Original | Corrected |
|-------|---------:|----------:|
| Phase 1b §2.1 detection | 5 LOC | **25-35 LOC** |
| Phase 1b §2.2 tensor read | 50 LOC | ~50 LOC |
| Phase 1b §2.3 DeviceMatrix | 50 LOC | ~50 LOC |
| Phase 1b §2.4 constructor | 30 LOC | ~30 LOC |
| **Total Phase 1b** | **~135 LOC** | **~155-175 LOC** |
| Wall time(codex)| 0.5 day | **0.75-1 day** |

Still well within "P0 small task" bracket。Doesn't change recommendation
to keep Phase 1b as P0 alongside #33 KV W4A8。

## Why I didn't apply the patch this tick

Originally planned to start with §2.1 detection patch as smallest
verification step。But after reading code structure,realized:
- 25+ LOC across 2 files
- 7 explicit init sites
- Needs `QuantMeta` enum extension that has its own hierarchy
- Without compile + test verify(GPU build),risk of unfinished half-state

Per CLAUDE.md "no half-states" rule,half-applied detection without
matching tensor reading path would leave dead code。Better to bundle
§2.1 + §2.2 + §2.3 + §2.4 as single codex pickup。

## Cross-references

- Phase 1b directive: `6be30ce`
- Hybrid plan main: `9754aca`
- Phase 0 reconnaissance: `1959a21`
- Phase 1a checkpoint merge tool: `b6502f7`

## Status

Phase 1b stays P0 alongside #33 KV W4A8 per `9596566` priority bump
recommendation。Effort estimate updated to 0.75-1 day codex(was 0.5d)。

This brief itself is the Claude tick deliverable — not blocking,
just sets accurate expectation for codex pickup。
