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

- **Phase B ✅ (`15ec44cd`):** added `infer-seam/src/resource.rs` (pure kernel +
  10 unit tests). Shipped behavior-neutral (no consumers wired yet). The Metal
  `plan_resource_budget` re-route is split out as **B′ (optional, deferred)** —
  Metal is already top-down-correct; routing it through the kernel is pure
  convergence with a byte-identical gate, not a fix, so it follows the bug-fix
  tranches.
- **Phase C1 ✅ (`a23336eb`):** DSv4 `kv_budget_num_slots` routes through the
  kernel. **Byte-identical** (`from_free` reproduces `floor(free×0.9)−Σfixed`;
  the two saturating_subs fold, proven in a unit test). NCCL min-reduce stays
  CUDA-side. Budget log unchanged. Pod re-confirm = pending-remote.
- **Phase C2 ✅ (`c7fe1aea`):** Qwen3.5/3.6 CUDA gains the clamp it lacked — the
  real bug fix — via `Qwen35Model::kv_budget_num_slots` (per-slot bytes mirror
  `new_slot_state`) + the same kernel + NCCL min-reduce, wired in the executor
  constructor. Strictly safer (no-op when `requested ≤ affordable`). Dense
  Qwen3 out of scope (global `PagedKVPool`, no slot-multiplied OOM). GPU
  boot-at-previously-OOMing-shape gate = pending-remote.
- **Phase C3 ✅ (`5353f17f`):** `kv_tier.rs` derives T1/T2 from system RAM
  (`/proc/meminfo`, dep-free) + free disk (`statvfs`, libc/unix) via
  `split_host_tiers`; CLI opt-out preserved. Caps == old constants, so an ample
  host (the pod) is byte-identical and a constrained one scales down; probe miss
  → cap. 7/7 kv_tier tests (incl. new probe tests).

**Convergence reached:** VRAM (C1+C2) and host RAM/SSD (C3) all flow through the
one neutral infer-seam kernel. Only B′ (Metal re-route) remains, and it is
optional polish, not a correctness gap.

Each phase: `cargo test --workspace` + clippy clean + wins/ entry (or
`pending-remote` for pod-only). No default-flip without a wall-clock license
per the bench spec.
