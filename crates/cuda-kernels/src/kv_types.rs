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
    /// Plain INT4 packed (2 nibbles per byte) with KIVI asymmetric scales:
    /// K per-(kv_head, head_dim) static, V per-(row, kv_head). Symmetric
    /// range [-7, 7]. ~1/4 the bytes of BF16 for K/V data. PoC parallel to
    /// TurboQuant's 4-bit path; the two differ on outlier-channel handling
    /// — per-channel scaling (this) vs Hadamard rotation (TurboQuant).
    INT4,
    TurboQuant {
        key_bits: u8,
        val_bits: u8,
    },
}

impl KVFormat {
    /// Stable wire-level discriminant used in persisted KV fingerprints.
    /// **These values must not change once written to disk.** Adding a
    /// new variant without assigning a unique tag here is a
    /// compile-time-adjacent error: the match must stay exhaustive and
    /// every `TurboQuant` bit-pair combination that ships to production
    /// gets its own explicit arm. Unknown bit-pairs return `None` so
    /// callers can fail fast instead of stamping a collision-prone
    /// fallback onto the disk format (M4 review finding A4 — the old
    /// `saturating_add` fallback could collapse two distinct bit-pairs
    /// onto the same tag).
    pub fn stable_tag(&self) -> Option<u8> {
        let tag = match *self {
            Self::BF16 => 1,
            Self::INT8 => 3,
            Self::FP8E4M3 => 5,
            Self::INT4 => 7,
            Self::TurboQuant {
                key_bits: 2,
                val_bits: 2,
            } => 10,
            Self::TurboQuant {
                key_bits: 3,
                val_bits: 3,
            } => 11,
            Self::TurboQuant {
                key_bits: 4,
                val_bits: 4,
            } => 12,
            Self::TurboQuant { .. } => return None,
        };
        Some(tag)
    }

    pub fn default_page_size(self) -> usize {
        match self {
            Self::BF16 => 16,
            Self::FP8E4M3 | Self::INT8 | Self::INT4 => 16,
            Self::TurboQuant { .. } => 1,
        }
    }

    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::FP8E4M3 | Self::INT8 => 1,
            // 4 bits = 0.5 byte; rounded up to 1 (the actual packed storage
            // uses kv_dim/2 bytes per token — handled at pool allocation, not
            // here, since this fn returns whole bytes for typing).
            Self::INT4 => 1,
            Self::TurboQuant { key_bits, .. } => {
                let effective = if key_bits == 3 { 4 } else { key_bits as usize };
                effective.div_ceil(8)
            }
        }
    }

    pub fn has_scales(self) -> bool {
        matches!(self, Self::FP8E4M3 | Self::INT8 | Self::INT4)
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
            Self::FP8E4M3 => head_dim + 4,
            Self::INT8 => head_dim + 4,
            // INT4: 2 nibbles per byte → head_dim/2 packed bytes, plus
            // f32 V scale (KIVI's K static scale is layer-shared, not
            // per-token, so it doesn't count toward per-token cost).
            Self::INT4 => head_dim.div_ceil(2) + 4,
            Self::TurboQuant { key_bits, .. } => {
                let packed = crate::turboquant_state::packed_bytes_per_head(head_dim, key_bits);
                packed + 2
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::KVFormat;

    #[test]
    fn stable_tags_are_fixed() {
        assert_eq!(KVFormat::BF16.stable_tag(), Some(1));
        assert_eq!(KVFormat::INT8.stable_tag(), Some(3));
        assert_eq!(KVFormat::FP8E4M3.stable_tag(), Some(5));
        assert_eq!(
            KVFormat::TurboQuant {
                key_bits: 2,
                val_bits: 2,
            }
            .stable_tag(),
            Some(10),
        );
        assert_eq!(
            KVFormat::TurboQuant {
                key_bits: 3,
                val_bits: 3,
            }
            .stable_tag(),
            Some(11),
        );
        assert_eq!(
            KVFormat::TurboQuant {
                key_bits: 4,
                val_bits: 4,
            }
            .stable_tag(),
            Some(12),
        );
    }

    #[test]
    fn stable_tag_returns_none_for_unassigned_turboquant_shape() {
        // Bit-pair combinations that have no explicit tag assignment
        // must return None so callers fail fast instead of writing a
        // collision-prone fallback onto disk (M4 review finding A4).
        assert_eq!(
            KVFormat::TurboQuant {
                key_bits: 2,
                val_bits: 3,
            }
            .stable_tag(),
            None,
        );
        assert_eq!(
            KVFormat::TurboQuant {
                key_bits: 14,
                val_bits: 15,
            }
            .stable_tag(),
            None,
        );
        assert_eq!(
            KVFormat::TurboQuant {
                key_bits: 15,
                val_bits: 15,
            }
            .stable_tag(),
            None,
        );
    }
}
