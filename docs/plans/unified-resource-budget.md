# Unified resource budget — VRAM / RAM / SSD convergence

Status: in-progress (2026-06-12). Owner: ckl. Tracks the systematic follow-up
to #60: budget logic must be globally managed and converged, not per-backend
ad-hoc.

## Audit (evidence, file:line)

#60 fixed one *phantom* (per-slot×per-layer scratch with one live copy). The
systematic sweep for siblings found **no second phantom** — every other
per-slot allocation is a legitimate stateful cache:

- Qwen3.5 CUDA `k_caches`/`v_caches` (qwen35.rs:148-149) — per-full-layer KV,
  legitimate. `gdr_states`/`conv_states` (151,153) — per-linear-layer recurrent
  state, legitimate.
- DSv4 per-slot selector/compressor caches (dsv4.rs:1081-1113) — stateful,
  cross-step, legitimate; already itemized in the budget.

The real systematic gap is **budget fragmentation + hardcoded host tiers**:

| Surface | Where | Behavior | Gap |
|---------|-------|----------|-----|
| DSv4 CUDA VRAM | `Dsv4Model::kv_budget_num_slots` dsv4.rs:1024-1158 | bottom-up: `mem_get_info().free × 0.9 − dsa_shared − moe_decode_shared`, `/ per_slot`, **NCCL min-reduce** (1144) | reject-below-fixed guard absent (saturating_sub to 0 → max(1)) |
| Qwen3 + Qwen3.5 CUDA VRAM | executor.rs:89,99 | `num_slots` passed straight through — **NO clamp** (doc confirms at 259-262 only DSv4 clamps) | **can OOM at large max_seq_len — the #60 failure class, unfixed** |
| Metal unified VRAM | `plan_resource_budget` resource.rs:156-302 | top-down: `resolve_memory_limit − (weights + headroom + static_state)`, page-capacity floor, full guard suite + paging pressure | mature; the template |
| Host RAM tier (T1) | kv_tier.rs:19 `DEFAULT_KV_TIER_BUDGET_BYTES = 4<<30` | **hardcoded 4 GiB** | not system-derived, not coordinated with VRAM/disk |
| Host SSD tier (T2) | kv_tier.rs:23 `DEFAULT_KV_SSD_BUDGET_BYTES = 20<<30` | **hardcoded 20 GiB** | not free-disk-derived |

Three backends each roll their own VRAM arithmetic; two of three CUDA models
have no clamp at all; the host tiers are constants disconnected from the
machine. That is the "统一管理统一收敛" target.

## Design — neutral policy kernel in `infer-seam`, backends keep probing

`infer-seam` is the only crate **both** `infer-cuda` and `infer-metal` already
depend on (`infer-metal` does not depend on `infer-util`). The neutral kernel
lives there, as `infer-seam/src/resource.rs`.

**Boundary (SOLID-honest): unify what is genuinely identical, keep
backend-specific what isn't.**

Neutral (pure arithmetic, no I/O — keeps infer-seam's "host-only, no backend
types" character):

- `budget_after_fixed(memory_limit, fixed_bytes) -> Result<usize>` — the
  reject-below-fixed guard + subtraction, ONE definition (Metal has it; CUDA
  lacked it).
- `affordable_units(budget_bytes, per_unit_bytes) -> usize` — single-floor
  divide (DSv4/Qwen3.5 slots).
- `clamp_to_affordable(requested, affordable) -> (planned, clamped)`.
- `split_host_tiers(system_ram, free_disk, opts) -> HostTierBudget` — derive
  T1 (RAM) + T2 (SSD) from machine numbers, replacing the two constants.

Backend-specific (stays put — genuinely not shared):

- Memory probing: CUDA `mem_get_info`; Metal `sysctl hw.memsize` + `vm_stat` +
  MLX recommended working set; host RAM/disk probe for tiers.
- NCCL min-reduce across TP ranks (DSv4).
- Byte sizing of weights / per-slot KV / scratch (model + dtype specific).
- Metal's macOS guard suite (paging pressure, wired-limit, anti-swap reserve,
  nested page-capacity floor) — unified-memory-specific.

## Phased landing (small committed tranches, bench/wins each)

- **Phase B (this tranche):** add `infer-seam/src/resource.rs` (pure kernel +
  unit tests). Refactor Metal `plan_resource_budget` to route its `−fixed` and
  `clamp` through the kernel. Gate = **byte-identical plan numbers** (A/B same
  `describe()` output). Smallest, safest, has a clean gate.
- **Phase C1:** DSv4 `kv_budget_num_slots` routes through the kernel; add the
  reject-below-fixed guard; NCCL min-reduce stays CUDA-side. Gate = same
  affordable count on the pod (budget log unchanged).
- **Phase C2:** Qwen3.5 (+ Qwen3 dense) CUDA gains the clamp via the kernel —
  the real bug fix. Gate = boots at a max_seq_len that previously OOM'd.
- **Phase C3:** kv_tier.rs derives T1/T2 from system RAM + free disk via
  `split_host_tiers`; CLI opt-out preserved. Gate = default budget on a known
  machine matches/improves the old constants, documented.

Each phase: `cargo test --workspace` + clippy clean + wins/ entry (or
`pending-remote` for pod-only). No default-flip without a wall-clock license
per the bench spec.
