//! Requested KV-cache storage dtype — the backend-neutral seam type.
//!
//! This is the *request*, not the resolution: the engine carries it from the
//! CLI/`EngineLoadConfig` down to a backend builder, which resolves it against
//! its own support matrix. Keeping the request enum at the seam is what makes
//! KV-quant an *engine capability* rather than a per-model fork: a backend
//! that cannot honor a requested dtype fails loud at construction, it does not
//! silently downgrade.

/// Requested KV-cache storage dtype, resolved per-backend at construction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheDtype {
    /// Backend default. Metal resolves this to INT8 after the Metal int8 gate;
    /// other backends keep their established default (CUDA: BF16).
    #[default]
    Auto,
    Bf16,
    /// INT8 KV cache. Metal uses MLX affine 8-bit groups; CUDA support is a
    /// separate backend implementation detail and must not be silently assumed.
    Int8,
    /// FP8 (E4M3) KV cache — CUDA paged-KV quant path only (#68 T3).
    Fp8,
    /// Trellis 4-bit KV cache — CUDA paged-KV quant path only (#68 T3).
    Tq4,
}

impl KvCacheDtype {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Bf16 => "bf16",
            Self::Int8 => "int8",
            Self::Fp8 => "fp8",
            Self::Tq4 => "tq4",
        }
    }
}
