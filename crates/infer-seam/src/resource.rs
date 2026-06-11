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
//! Consumers (see `docs/plans/unified-resource-budget.md`):
//! - DSv4 CUDA `kv_budget_num_slots` (slot budget, then NCCL min-reduce)
//! - Qwen3/Qwen3.5 CUDA (slot budget — the clamp they previously lacked)
//! - Metal `plan_resource_budget` (page clamp + below-fixed guard)
//! - CUDA KV tier store (host RAM/SSD tier split)

const GIB: usize = 1 << 30;

/// Memory left for slots/pages after the count-independent (fixed) reservation,
/// and the per-unit cost — the shared top-of-stack of every backend's budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotBudget {
    /// Bytes available for slots/pages after the fixed reservation.
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

/// Host RAM (T1) + SSD (T2) tier budgets, derived from machine state instead of
/// hardcoded constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostTierBudget {
    /// In-RAM (T1) budget, bytes.
    pub ram_bytes: usize,
    /// On-disk (T2) budget, bytes (0 when no disk path is configured).
    pub ssd_bytes: usize,
}

/// Inputs/policy for [`split_host_tiers`].
#[derive(Debug, Clone, Copy)]
pub struct HostTierPolicy {
    /// Fraction of *available* system RAM the T1 prefix tier may claim.
    pub ram_fraction: f64,
    /// Hard cap on the T1 RAM budget (the proven default, never exceeded so we
    /// cannot regress an ample machine into over-claiming RAM).
    pub ram_cap_bytes: usize,
    /// Floor on the T1 RAM budget when *some* RAM is available, so tiny
    /// available-RAM readings don't collapse the tier to nothing.
    pub ram_floor_bytes: usize,
    /// Fraction of *free* disk the T2 spill tier may claim.
    pub ssd_fraction: f64,
    /// Hard cap on the T2 SSD budget (the proven default).
    pub ssd_cap_bytes: usize,
}

impl Default for HostTierPolicy {
    fn default() -> Self {
        // Caps are the monolith-era proven constants (4 GiB RAM / 20 GiB SSD);
        // the fractions shrink the budget on small machines without ever
        // exceeding what historically shipped.
        Self {
            ram_fraction: 0.25,
            ram_cap_bytes: 4 * GIB,
            ram_floor_bytes: GIB,
            ssd_fraction: 0.5,
            ssd_cap_bytes: 20 * GIB,
        }
    }
}

/// Derive T1 (RAM) and T2 (SSD) tier budgets from system memory and free disk.
///
/// Replaces the hardcoded `DEFAULT_KV_TIER_BUDGET_BYTES` / `DEFAULT_KV_SSD_BUDGET_BYTES`:
/// each tier is `clamp(resource × fraction, floor, cap)`, capped at the proven
/// constant so an ample machine matches the old default and a constrained one
/// scales down. `available_ram_bytes`/`free_disk_bytes` of `None` fall back to
/// the cap (preserve the historical constant when the probe is unavailable).
/// `disk_configured == false` zeroes the SSD tier (opt-in path unchanged).
pub fn split_host_tiers(
    available_ram_bytes: Option<usize>,
    free_disk_bytes: Option<usize>,
    disk_configured: bool,
    policy: HostTierPolicy,
) -> HostTierBudget {
    let ram_bytes = match available_ram_bytes {
        Some(avail) => {
            let scaled = (avail as f64 * policy.ram_fraction) as usize;
            scaled
                .min(policy.ram_cap_bytes)
                .max(policy.ram_floor_bytes.min(policy.ram_cap_bytes))
        }
        None => policy.ram_cap_bytes,
    };
    let ssd_bytes = if disk_configured {
        match free_disk_bytes {
            Some(free) => ((free as f64 * policy.ssd_fraction) as usize).min(policy.ssd_cap_bytes),
            None => policy.ssd_cap_bytes,
        }
    } else {
        0
    };
    HostTierBudget {
        ram_bytes,
        ssd_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;

    #[test]
    fn from_free_matches_dsv4_arithmetic() {
        // DSv4: floor(free × 0.9) − (dsa + moe), then / per_slot.
        let free = 57_035 * MIB; // ~ the pod budget log
        let dsa = 36 * MIB;
        let moe = 114 * MIB;
        let per_slot = 924 * MIB;
        let b = SlotBudget::from_free(free, 0.9, dsa + moe, per_slot);
        // Two saturating_subs fold into one: x.sat_sub(a).sat_sub(b) == x.sat_sub(a+b).
        let manual = ((free as f64 * 0.9) as usize)
            .saturating_sub(dsa)
            .saturating_sub(moe);
        assert_eq!(b.budget_bytes, manual);
        assert_eq!(b.affordable(), manual.checked_div(per_slot));
    }

    #[test]
    fn saturating_sub_fold_is_identical() {
        // Prove the two-term fold the DSv4 refactor relies on, across regimes.
        for &(x, a, c) in &[
            (100usize, 30usize, 40usize),
            (100, 80, 40),
            (100, 120, 40),
            (50, 20, 20),
        ] {
            assert_eq!(
                x.saturating_sub(a).saturating_sub(c),
                x.saturating_sub(a + c)
            );
        }
    }

    #[test]
    fn from_limit_matches_metal_kv_budget() {
        let limit = 80 * GIB;
        let fixed = 30 * GIB;
        let per_token = 4096;
        let b = SlotBudget::from_limit(limit, fixed, per_token);
        assert_eq!(b.budget_bytes, 50 * GIB);
        assert_eq!(b.affordable(), Some((50 * GIB) / per_token));
    }

    #[test]
    fn fits_fixed_guards_below_fixed() {
        assert!(SlotBudget::fits_fixed(80 * GIB, 30 * GIB));
        assert!(!SlotBudget::fits_fixed(30 * GIB, 30 * GIB)); // strict >
        assert!(!SlotBudget::fits_fixed(20 * GIB, 30 * GIB));
    }

    #[test]
    fn affordable_zero_per_unit_is_none() {
        assert_eq!(
            SlotBudget {
                budget_bytes: 100,
                per_unit_bytes: 0
            }
            .affordable(),
            None
        );
    }

    #[test]
    fn clamp_reduces_and_flags() {
        assert_eq!(clamp_to_affordable(8, 32), (8, false));
        assert_eq!(clamp_to_affordable(32, 8), (8, true));
        assert_eq!(clamp_to_affordable(8, 8), (8, false));
    }

    #[test]
    fn host_tiers_cap_at_proven_constants_when_ample() {
        let p = HostTierPolicy::default();
        // Ample RAM (256 GiB) and disk (1 TiB): each tier caps at the constant.
        let b = split_host_tiers(Some(256 * GIB), Some(1024 * GIB), true, p);
        assert_eq!(b.ram_bytes, 4 * GIB);
        assert_eq!(b.ssd_bytes, 20 * GIB);
    }

    #[test]
    fn host_tiers_scale_down_on_small_machine() {
        let p = HostTierPolicy::default();
        // 8 GiB available RAM × 0.25 = 2 GiB (< 4 GiB cap); 8 GiB free disk × 0.5 = 4 GiB.
        let b = split_host_tiers(Some(8 * GIB), Some(8 * GIB), true, p);
        assert_eq!(b.ram_bytes, 2 * GIB);
        assert_eq!(b.ssd_bytes, 4 * GIB);
    }

    #[test]
    fn host_tiers_floor_protects_tiny_available() {
        let p = HostTierPolicy::default();
        // 1 GiB available × 0.25 = 256 MiB < 1 GiB floor → floored to 1 GiB.
        let b = split_host_tiers(Some(GIB), None, false, p);
        assert_eq!(b.ram_bytes, GIB);
        assert_eq!(b.ssd_bytes, 0); // disk not configured
    }

    #[test]
    fn host_tiers_unknown_probe_falls_back_to_constants() {
        let p = HostTierPolicy::default();
        let b = split_host_tiers(None, None, true, p);
        assert_eq!(b.ram_bytes, 4 * GIB); // cap = old constant
        assert_eq!(b.ssd_bytes, 20 * GIB);
    }

    #[test]
    fn host_tiers_disk_off_zeroes_ssd() {
        let p = HostTierPolicy::default();
        let b = split_host_tiers(Some(64 * GIB), Some(512 * GIB), false, p);
        assert_eq!(b.ssd_bytes, 0);
    }
}
