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
}

impl KVFormat {
    pub fn default_page_size(self) -> usize {
        match self {
            Self::BF16 => 16,
            Self::FP8E4M3 | Self::INT8 | Self::TurboQuant { .. } => 1,
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
        }
    }

    pub fn has_scales(self) -> bool {
        matches!(self, Self::INT8)
    }

    pub fn has_norms(self) -> bool {
        matches!(self, Self::TurboQuant { .. })
    }

    pub fn needs_work_buffer(self) -> bool {
        !matches!(self, Self::BF16)
    }

    pub fn is_turboquant(self) -> bool {
        matches!(self, Self::TurboQuant { .. })
    }

    #[cfg(feature = "cuda")]
    pub fn pool_bytes_per_kv_head(self, head_dim: usize) -> usize {
        match self {
            Self::BF16 => head_dim * 2,
            Self::FP8E4M3 => head_dim,
            Self::INT8 => head_dim + 4,
            Self::TurboQuant { key_bits, .. } => {
                let packed = crate::turboquant_state::packed_bytes_per_head(head_dim, key_bits);
                packed + 2
            }
        }
    }
}
