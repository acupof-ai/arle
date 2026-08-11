//! Backend-neutral resource-budget policy kernel.
//!
//! The arithmetic of "how much memory is left for KV slots, and how many fit"
//! is identical across backends:
//!
//! ```text
//! budget = floor(memory × fraction) − fixed        (weights + headroom + shared scratch)
//! affordable = budget / per_unit                    (slots, or KV-bytes-per-token)
//! planned = min(requested, affordable)              (clamp)
//! ```
//!
//! What differs — how to *probe* device/host memory (`mem_get_info` vs
//! `sysctl`/`vm_stat`), cross-rank min-reduce (NCCL), the byte sizes of
//! weights / per-slot KV / scratch (model + dtype specific), and unified-memory
//! paging guards — stays backend-side. This module is the shared core: pure
//! functions over numbers, no I/O, no backend types, so it keeps infer-seam's
//! host-only character and is exhaustively unit-testable.
//!
//! Consumers:
//! - DSv4 CUDA `kv_budget_num_slots` (slot budget, then NCCL min-reduce)
//! - Qwen3/Qwen3.5 CUDA (slot budget — the clamp they previously lacked)
//! - Metal `plan_resource_budget` (page clamp + below-fixed guard)
//! - CUDA KV tier store (host RAM/SSD tier split)

const GIB: usize = 1 << 30;

/// The reserve (OS/FS headroom) may never exceed half the total resource.
/// On a small box where `reserve_floor_bytes > total/2`, the cap degrades
/// the reserve to `total/2` so the tier still gets a non-zero budget
/// instead of collapsing to 0.
const RESERVE_CAP_FRACTION: f64 = 0.5;

/// Memory left for slots/pages after the count-independent (fixed) reservation,
/// and the per-unit cost — the shared top-of-stack of every backend's budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotBudget {
    pub budget_bytes: usize,
    /// Bytes consumed per slot (or per KV token, for page-framed backends).
    pub per_unit_bytes: usize,
}

impl SlotBudget {
    /// Budget from a *free*-memory reading: `floor(free × fraction) − fixed`.
    ///
    /// Mirrors the DSv4 CUDA path (`free × 0.9 − dsa_shared − moe_decode_shared`).
    /// `fixed_bytes` is the sum of every count-independent term; subtraction
    /// saturates (a fixed reservation larger than the fractioned free memory
    /// yields a zero budget → zero affordable units, the caller's reject/floor
    /// policy decides what that means).
    pub fn from_free(
        free_bytes: usize,
        fraction: f64,
        fixed_bytes: usize,
        per_unit_bytes: usize,
    ) -> Self {
        let scaled = (free_bytes as f64 * fraction) as usize;
        Self {
            budget_bytes: scaled.saturating_sub(fixed_bytes),
            per_unit_bytes,
        }
    }

    /// Budget from an absolute *limit*: `limit − fixed` (saturating).
    ///
    /// Mirrors the Metal path, where `memory_limit` is already the
    /// fraction/reserve-resolved cap and `fixed = weights + headroom +
    /// static_state`. Use [`Self::fits_fixed`] first for the reject guard.
    pub fn from_limit(
        memory_limit_bytes: usize,
        fixed_bytes: usize,
        per_unit_bytes: usize,
    ) -> Self {
        Self {
            budget_bytes: memory_limit_bytes.saturating_sub(fixed_bytes),
            per_unit_bytes,
        }
    }

    /// The reject-below-fixed guard: is there *any* room after the fixed
    /// reservation? `false` means the fixed terms alone exceed the limit — the
    /// caller should fail closed with a backend-specific message rather than
    /// admit a startup that cannot hold a single slot.
    pub fn fits_fixed(memory_limit_bytes: usize, fixed_bytes: usize) -> bool {
        memory_limit_bytes > fixed_bytes
    }

    /// Whole units that fit (single floor). `None` when `per_unit_bytes == 0`
    /// (the caller decides whether a zero per-unit cost is a bug or means
    /// "unbounded").
    pub fn affordable(&self) -> Option<usize> {
        self.budget_bytes.checked_div(self.per_unit_bytes)
    }
}

/// Clamp a requested unit count to what is affordable.
///
/// Returns `(planned, clamped)` where `planned = min(requested, affordable)`
/// and `clamped` flags that the request was reduced. Shared by every backend so
/// the clamp semantics (and the `clamped` reporting) are defined once.
pub fn clamp_to_affordable(requested: usize, affordable: usize) -> (usize, bool) {
    let planned = requested.min(affordable);
    (planned, planned < requested)
}

/// Bounds for `mem_fraction_static`: a backend never claims more than ~all of HBM
/// for KV (leave room for activations/scratch even at the top). The lower bound is
/// small (not 0.5) so an explicitly co-resident caller — OPD's rollout engine
/// sharing VRAM with a trainable student — can deliberately shrink its KV pool;
/// `PROFILE_KV_TOKENS_FLOOR` still guarantees the pool never collapses to zero, so
/// the floor (not a 0.5 clamp) is what protects admission at the bottom.
const MEM_FRACTION_STATIC_MIN: f64 = 0.05;
const MEM_FRACTION_STATIC_MAX: f64 = 0.97;

/// Floor on the profiled token pool: a transient tiny free-VRAM reading (another
/// process spiking, a fragmented allocator) must not size the KV pool to ~zero
/// and wedge admission. 4096 tokens is far below any real serving budget yet
/// keeps a short request alive; callers floor again at `page_size`.
pub const PROFILE_KV_TOKENS_FLOOR: u64 = 4096;

/// Clamp a requested static-memory fraction into the safe operating band
/// `[0.05, 0.97]`. Exposed so callers can report the clamp; the profiler applies
/// it internally regardless. A `NaN` fraction (a parse/compute bug) resolves to
/// the conservative max rather than propagating `NaN` through the budget.
#[must_use]
pub fn clamp_mem_fraction_static(mem_fraction_static: f64) -> f64 {
    if mem_fraction_static.is_nan() {
        return MEM_FRACTION_STATIC_MAX;
    }
    mem_fraction_static.clamp(MEM_FRACTION_STATIC_MIN, MEM_FRACTION_STATIC_MAX)
}

/// SGLang-style KV pool sizing from a *measured* free/total VRAM reading taken
/// AFTER weights are resident — the model-agnostic foundation. Knows nothing
/// about any model: it takes bytes and a per-token cell cost and returns how
/// many KV tokens fit.
///
/// ```text
/// reserve = total_bytes × (1 − mem_fraction_static)   (headroom for activations/scratch/fragmentation)
/// rest    = free_bytes − reserve                        (saturating)
/// tokens  = rest / cell_bytes_per_token                 (floored at PROFILE_KV_TOKENS_FLOOR)
/// ```
///
/// `cell_bytes_per_token = num_kv_heads · head_dim · num_layers · 2(K+V) ·
/// sizeof(kv_dtype)` (plus any per-token scale/norm/work bytes the backend's
/// pool charges) — computed backend-side and passed in, so this stays device-
/// and model-neutral.
///
/// `mem_fraction_static` is clamped to `[0.05, 0.97]` ([`clamp_mem_fraction_static`]).
/// A `cell_bytes_per_token` of 0 (a bug — an empty pool shape) returns the floor
/// rather than dividing by zero. The result is floored at
/// [`PROFILE_KV_TOKENS_FLOOR`] so a transient tiny `free_bytes` can't wedge
/// admission at ~0 tokens.
#[must_use]
pub fn profile_kv_pool_tokens(
    free_bytes: u64,
    total_bytes: u64,
    cell_bytes_per_token: u64,
    mem_fraction_static: f64,
) -> u64 {
    if cell_bytes_per_token == 0 {
        return PROFILE_KV_TOKENS_FLOOR;
    }
    let frac = clamp_mem_fraction_static(mem_fraction_static);
    // reserve = total × (1 − frac); subtract from free, saturating to 0.
    let reserve = (total_bytes as f64 * (1.0 - frac)) as u64;
    let rest = free_bytes.saturating_sub(reserve);
    let tokens = rest / cell_bytes_per_token;
    tokens.max(PROFILE_KV_TOKENS_FLOOR)
}

/// Inputs/policy for the unified L2 (host DRAM) tier budget ([`dram_l2_budget`]).
///
/// The L2 store is *pageable* host memory holding demoted KV pages; the box is
/// shared, so being greedy swaps or OOM-kills co-tenants. The budget is
/// `clamp(fraction × MemAvailable, [floor, MemAvailable − reserve])` with a
/// reserve that scales with total RAM (the OS + service host: weight staging,
/// pinned ring, page cache).
#[derive(Debug, Clone, Copy)]
pub struct DramTierPolicy {
    /// Fraction of *available* DRAM the L2 tier may claim. Default 0.5 (NOT 0.8):
    /// the store is pageable host memory on a shared box.
    pub fraction: f64,
    /// Floor on the L2 budget when DRAM is ample (so a tiny `MemAvailable`
    /// reading never collapses the tier, but the reserve still wins when DRAM
    /// is genuinely scarce — see [`dram_l2_budget`]).
    pub floor_bytes: usize,
    /// Absolute reserve floor: never claim DRAM that would leave less than this
    /// for the OS + co-tenants.
    pub reserve_floor_bytes: usize,
    /// Fractional reserve: `reserve = max(reserve_floor_bytes, reserve_fraction
    /// × total_ram)`. Scales the headroom up on a big box.
    pub reserve_fraction: f64,
}

impl Default for DramTierPolicy {
    fn default() -> Self {
        Self {
            fraction: 0.5,
            floor_bytes: 4 * GIB,
            reserve_floor_bytes: 64 * GIB,
            reserve_fraction: 0.15,
        }
    }
}

/// Inputs/policy for the unified L3 (NVMe) tier budget ([`nvme_l3_budget`]).
///
/// `clamp(fraction × free_disk, [floor, free_disk − reserve])` with a reserve
/// that scales with total disk (the FS itself + other tenants).
#[derive(Debug, Clone, Copy)]
pub struct NvmeTierPolicy {
    /// Fraction of *free* disk the L3 tier may claim. Default 0.5.
    pub fraction: f64,
    /// Floor on the L3 budget when free disk is ample.
    pub floor_bytes: usize,
    /// Absolute reserve floor for the FS + other tenants.
    pub reserve_floor_bytes: usize,
    /// Fractional reserve: `reserve = max(reserve_floor_bytes, reserve_fraction
    /// × total_disk)`.
    pub reserve_fraction: f64,
}

impl Default for NvmeTierPolicy {
    fn default() -> Self {
        Self {
            fraction: 0.5,
            floor_bytes: 8 * GIB,
            // L3 is a write-through spill layer, not a budgeted cache: no
            // reserve (neither absolute nor fractional). The budget caps how
            // much disk the spill may consume; disk is consumed only on actual
            // spill, so a reserve would just silently shrink the usable tier.
            reserve_floor_bytes: 0,
            reserve_fraction: 0.0,
        }
    }
}

/// Clamp `fraction × resource` into `[floor, resource − reserve]`, where
/// `reserve = max(reserve_floor, reserve_fraction × total)`.
///
/// The clamp order is deliberate: the upper bound (`resource − reserve`,
/// saturating) is applied LAST so the reserve always wins on a scarce box — a
/// `floor` larger than `resource − reserve` collapses to `resource − reserve`
/// (which may be 0), never claiming into the reserve. `total >= resource` is
/// the expected relationship (free/available ≤ total) but is not required.
fn tier_budget_with_reserve(
    resource_bytes: usize,
    total_bytes: usize,
    fraction: f64,
    floor_bytes: usize,
    reserve_floor_bytes: usize,
    reserve_fraction: f64,
) -> usize {
    // Cap the reserve at RESERVE_CAP_FRACTION of total so a small box
    // (total < reserve_floor / RESERVE_CAP_FRACTION) still gets a non-zero
    // budget instead of collapsing to 0.
    let reserve = reserve_floor_bytes
        .max((total_bytes as f64 * reserve_fraction) as usize)
        .min((total_bytes as f64 * RESERVE_CAP_FRACTION) as usize);
    let ceiling = resource_bytes.saturating_sub(reserve);
    let scaled = (resource_bytes as f64 * fraction) as usize;
    // Lower-bound by the floor, then upper-bound by the reserve ceiling (the
    // reserve must win — `.min` last so a floor above the ceiling cannot claim
    // into the reserve).
    scaled.max(floor_bytes).min(ceiling)
}

/// Unified **L2 (host DRAM)** KV-tier budget — measured-hardware, leave-a-reserve
/// (the box is shared; the store is pageable host memory).
///
/// `budget = clamp(fraction × available_ram, [floor, available_ram − reserve])`,
/// `reserve = max(reserve_floor, reserve_fraction × total_ram)`.
///
/// `available_ram_bytes`/`total_ram_bytes` of `None` (a probe miss off Linux)
/// fall back to `floor_bytes` so a `/proc`-less typecheck build never
/// over-claims. When both are present but `available ≤ reserve`, the budget
/// floors at `0` — the box has no spare DRAM for an L2 tier.
#[must_use]
pub fn dram_l2_budget(
    available_ram_bytes: Option<usize>,
    total_ram_bytes: Option<usize>,
    policy: DramTierPolicy,
) -> usize {
    let (Some(avail), Some(total)) = (available_ram_bytes, total_ram_bytes) else {
        return policy.floor_bytes;
    };
    tier_budget_with_reserve(
        avail,
        total,
        policy.fraction,
        policy.floor_bytes,
        policy.reserve_floor_bytes,
        policy.reserve_fraction,
    )
}

/// Unified **L3 (NVMe)** KV-tier budget — measured free disk, leave a reserve for
/// the FS + other tenants.
///
/// `budget = clamp(fraction × free_disk, [floor, free_disk − reserve])`,
/// `reserve = max(reserve_floor, reserve_fraction × total_disk)`.
///
/// `free_disk_bytes`/`total_disk_bytes` of `None` (a probe miss) fall back to
/// `floor_bytes`. When `free ≤ reserve` the budget floors at `0`.
#[must_use]
pub fn nvme_l3_budget(
    free_disk_bytes: Option<usize>,
    total_disk_bytes: Option<usize>,
    policy: NvmeTierPolicy,
) -> usize {
    let (Some(free), Some(total)) = (free_disk_bytes, total_disk_bytes) else {
        return policy.floor_bytes;
    };
    tier_budget_with_reserve(
        free,
        total,
        policy.fraction,
        policy.floor_bytes,
        policy.reserve_floor_bytes,
        policy.reserve_fraction,
    )
}
