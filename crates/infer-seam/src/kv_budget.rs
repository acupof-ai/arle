//! Requested KV tier budget — the backend-neutral seam type.
//!
//! Like [`crate::KvCacheDtype`], this is the *request*, not the resolution:
//! the CLI/`EngineLoadConfig` carries it to the backend builder, which
//! resolves `Fraction` against the measured resource at construction.

/// KV tier budget request: absolute bytes or a fraction of the measured
/// resource (L2: available DRAM; L3: free disk at the spill root). Budgets
/// are DEPLOYMENT-TOTAL — the engine constructor divides by the TP world
/// size — and are caps, not reservations.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum KvTierBudget {
    Fraction(f64),
    Bytes(usize),
    Off,
}

impl Default for KvTierBudget {
    fn default() -> Self {
        Self::Fraction(0.5)
    }
}
