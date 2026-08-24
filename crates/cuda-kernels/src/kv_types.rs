#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KVCacheDtype {
    #[default]
    BF16,
    INT8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KVFormat {
    #[default]
    BF16,
    FP8E4M3,
    INT8,
    TurboQuant {
        key_bits: u8,
        val_bits: u8,
    },
    /// Opaque packed single-plane record format (MLA-style latent KV): one
    /// `bytes_per_token` record per token, stored in the K plane only — no V
    /// plane and no separate scale/norm buffers (everything is embedded in
    /// the record). Canonical DSv4 use: 584 B = 448 B FP8 NoPE + 64×2 B BF16
    /// RoPE + 8 B e8m0 scale. `num_kv_heads` must be 1 (the latent record is
    /// head-less) and sizing routes through `bytes_per_token`, never
    /// `bytes_per_element`. Per-head attention/quant kernels do not consume
    /// it; its consumers (FlashMLA block-table paths) arrive in P2.
    PackedBytes {
        bytes_per_token: usize,
    },
}

impl KVFormat {
    pub fn default_page_size(self) -> usize {
        match self {
            Self::BF16 => 16,
            Self::FP8E4M3 | Self::INT8 => 16,
            Self::TurboQuant { .. } => 1,
            // FlashMLA block size — block-table entries are 64-token pages.
            Self::PackedBytes { .. } => 64,
        }
    }

    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::FP8E4M3 | Self::INT8 => 1,
            Self::TurboQuant { key_bits, .. } => {
                let effective = if key_bits == 3 { 4 } else { key_bits as usize };
                effective.div_ceil(8)
            }
            // Per-(kv_dim)-element sizing is meaningless for an opaque packed
            // record; every sizing path must route through
            // `packed_record_bytes_per_token` instead.
            Self::PackedBytes { .. } => {
                panic!("KVFormat::PackedBytes sizes via bytes_per_token, not bytes_per_element")
            }
        }
    }

    /// `Some(record bytes)` for the packed single-plane record format,
    /// `None` for every per-head format. Sizing paths use this to bypass
    /// `kv_dim`-based math for packed records.
    pub fn packed_record_bytes_per_token(self) -> Option<usize> {
        match self {
            Self::PackedBytes { bytes_per_token } => Some(bytes_per_token),
            _ => None,
        }
    }

    pub fn has_scales(self) -> bool {
        matches!(self, Self::FP8E4M3 | Self::INT8)
    }

    pub fn has_norms(self) -> bool {
        matches!(self, Self::TurboQuant { .. })
    }

    pub fn needs_work_buffer(self) -> bool {
        // PackedBytes records are written directly by their producer — no
        // bf16 staging buffer (the P2 FlashMLA pack path writes the packed
        // record in one shot).
        !matches!(self, Self::BF16 | Self::PackedBytes { .. })
    }
}
