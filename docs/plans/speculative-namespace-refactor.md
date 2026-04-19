# Speculative Decoding Namespace Refactor

_Date: 2026-04-19_
_Status: deferred — trigger when second spec-decode method (standard-draft / EAGLE) is about to land_
_Owner: TBD_

## Problem

Today the only shipped speculative-decoding path is **DFlash** on Metal. Its
naming has leaked into user-visible surfaces:

- CLI flag `--dflash-draft-model` (3 binaries: `metal_serve`, `infer_cli`, etc.)
- Prometheus metrics: `dflash_acceptance_rate`, `dflash_*_total`
- Config key: `MetalSchedulerConfig.metal_dflash_concurrency_off`
- Public types: `MetalDflashOptions`

DFlash is a *concrete method* (Apple's draft-model + speculative variant
shipped via MLX), at the same level as **EAGLE2/3**, **Medusa**, and **standard
draft model**. The roadmap (`docs/plans/speculative-decoding-impl.md`,
`docs/research/speculative-decoding-feasibility.md`) explicitly plans for these
to coexist. Once the second method lands, `dflash_*` user-facing names become
misleading — they look like the category but are actually one implementation.

## Why not now

- Survey 2026-04-19 found ~150–200 identifier sites and ~40 files. Renaming
  today is pure churn — no second implementation exists to share an abstraction
  with.
- See `feedback_no_speculative_interface_shaping.md`: traits/handles trail real
  callers. The umbrella shape is unknowable until method #2 has a working diff.
- Internal Metal naming (`metal::dflash` module, FFI `dflash_draft_*`) is
  *correctly scoped* to the implementation and should stay regardless.

## Trigger

Land this refactor in the **same commit window** that introduces method #2
(per current roadmap, standard-draft Qwen3-0.5B on CUDA). Doing it earlier =
churn; doing it later = two methods both wearing the wrong name.

## Scope (when triggered)

### User-facing renames

| Old | New | Notes |
|---|---|---|
| `--dflash-draft-model <path>` | `--speculative-method dflash --speculative-draft-model <path>` | Keep `--dflash-draft-model` as deprecated alias for one release with stderr warning |
| metrics `dflash_acceptance_rate` | `speculative_acceptance_rate{method="dflash"}` | Add `method` label; old metric kept emitting for one release |
| metrics `dflash_*_total` | `speculative_*_total{method="dflash"}` | Same |
| `MetalDflashOptions` (if exposed in public config) | `SpeculativeOptions { method: SpeculativeMethod, ... }` | Internal `MetalDflashOptions` can stay as the dflash-specific impl detail |

### Untouched (stays `dflash`)

- `infer/src/backend/metal/dflash.rs` module path
- `crates/mlx-sys/src/mlx_dflash_draft_model.cpp` and FFI symbols
- Internal Metal types: `MetalDflashOptions` (as the per-method options struct,
  one variant inside `SpeculativeOptions`)
- `metal_dflash_concurrency_off` scheduler flag (Metal-internal tuning knob,
  not user-visible)

### New umbrella shape (sketch — do not implement until trigger)

```rust
pub enum SpeculativeMethod {
    Dflash,        // Metal only, ships today
    StandardDraft, // CUDA first (Qwen3-0.5B), per spec-decode plan
    Eagle,         // future
    Medusa,        // future
}

pub struct SpeculativeOptions {
    pub method: SpeculativeMethod,
    pub draft_model: Option<PathBuf>,
    pub k: usize,
    pub method_specific: SpeculativeMethodOptions,  // enum dispatch
}
```

CLI:
```
--speculative-method {dflash,standard-draft,eagle}
--speculative-draft-model <path>
--speculative-k <N>
```

## Acceptance

- Method #2 lands behind `--speculative-method <name>` in same commit window.
- `--dflash-draft-model` still works (deprecated, prints stderr warning).
- Both old and new metrics emit for one release; old removed in the release
  after.
- `docs/experience/wins/` entry shows method-#2 numbers + dflash regression
  check (no behavior change for dflash users).

## Cross-refs

- `docs/plans/speculative-decoding-impl.md` — the standard-draft impl plan
  (likely method #2)
- `docs/research/speculative-decoding-feasibility.md` — methods coexistence
  rationale
- `docs/experience/wins/2026-04-19-metal-qwen35-concurrent-dflash-default-on.md`
  — current dflash production state
- `memory/feedback_no_speculative_interface_shaping.md` — why we wait for #2
  before designing the abstraction
