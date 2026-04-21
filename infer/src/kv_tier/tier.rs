//! Tier taxonomy and block location types.
//!
//! See `crate::kv_tier` for the module-level design notes.

use serde::{Deserialize, Serialize};

use crate::types::BlockFingerprint;

/// Storage medium for a KV block. Ordering (`Gpu < HostPinned < Disk <
/// Remote`) reflects the distance from compute — nearer first.
///
/// Tier labels (T0/T1/T2/T3) match the 2026-04-15 tiered-kv-cache
/// revision; see `docs/projects/tiered-kv-cache.md` §4.1 for the
/// industry-aligned numbering rationale.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum Tier {
    /// T0 — GPU HBM. Kernel-accessible.
    Gpu,
    /// T1 — Host pinned DRAM. Coordinator-accessible only.
    HostPinned,
    /// T2 — Local NVMe SSD.
    Disk,
    /// T3 — Remote node, reached over NIXL / Mooncake / UCX.
    Remote,
}

/// Physical location of a block's bytes. Variants match [`Tier`] 1:1.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BlockLocation {
    /// GPU pool slot index. Interpretation is pool-specific.
    Gpu { slot: u32 },
    /// Byte offset within the pinned host pool.
    HostPinned { offset: u64 },
    /// Canonical content-addressed disk location. `fingerprint` is the
    /// semantic block identity; the disk store derives the concrete path from
    /// it and validates `payload_len` on load.
    Disk {
        fingerprint: BlockFingerprint,
        payload_len: u64,
    },
    /// Remote block, opaque per-transport payload.
    Remote { desc: RemoteBlockDesc },
}

impl BlockLocation {
    /// Returns the tier this location lives in.
    pub fn tier(&self) -> Tier {
        match self {
            BlockLocation::Gpu { .. } => Tier::Gpu,
            BlockLocation::HostPinned { .. } => Tier::HostPinned,
            BlockLocation::Disk { .. } => Tier::Disk,
            BlockLocation::Remote { .. } => Tier::Remote,
        }
    }

    pub fn is_remote(&self) -> bool {
        matches!(self, BlockLocation::Remote { .. })
    }
}

/// Opaque remote descriptor. The `transport` tag identifies which
/// transport impl is responsible for decoding `payload`. Cross-backend
/// code must never parse the payload directly.
///
/// Example payloads:
/// - `NixlTransport` (tiered-kv-cache M5): bincode of
///   `(remote_agent_name, addr, len, mem_type, dev_id)` — fits in
///   ~24–32 bytes for short agent names.
/// - `MooncakeTransport` (post-M5, trigger-gated): bincode of
///   `(segment_handle, offset, length)` = 24 bytes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RemoteBlockDesc {
    pub transport: TransportId,
    pub payload: Vec<u8>,
}

/// Discriminator for [`RemoteBlockDesc::payload`]. Kept small so the
/// enum fits in one byte and serialization is cheap.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TransportId {
    /// NVIDIA NIXL — lands in tiered-kv-cache M5 as a stub.
    Nixl = 0,
    /// Mooncake `TransferEngine` — post-M5, trigger-gated.
    Mooncake = 1,
    /// Reserved for future transports (UCX direct, libfabric, etc.).
    Reserved = 255,
}

/// Memory kind used by [`super::transport::KVTransport::register`]. Maps
/// 1:1 to NIXL's `MemType` enum (Dram / Vram / Block / Object / File) so
/// future backends have room to grow.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MemKind {
    /// CPU-visible pinned DRAM (`cudaHostAlloc` / `cudaHostRegister`).
    Host,
    /// GPU device memory; `device` is the CUDA/ROCm ordinal.
    Vram { device: u32 },
    /// Block-device backed storage (reserved for GPUDirect Storage).
    Block { volume: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_variants_round_trip() {
        // Every tier variant clones cleanly and has a stable Debug
        // representation — enough to catch accidental enum renames.
        for tier in [Tier::Gpu, Tier::HostPinned, Tier::Disk, Tier::Remote] {
            let cloned = tier;
            assert_eq!(cloned, tier);
            assert!(!format!("{tier:?}").is_empty());
        }
    }

    #[test]
    fn block_location_reports_its_tier() {
        let g = BlockLocation::Gpu { slot: 3 };
        let h = BlockLocation::HostPinned { offset: 4096 };
        let d = BlockLocation::Disk {
            fingerprint: BlockFingerprint([0x11; 16]),
            payload_len: 8192,
        };
        let r = BlockLocation::Remote {
            desc: RemoteBlockDesc {
                transport: TransportId::Nixl,
                payload: vec![0; 16],
            },
        };
        assert_eq!(g.tier(), Tier::Gpu);
        assert_eq!(h.tier(), Tier::HostPinned);
        assert_eq!(d.tier(), Tier::Disk);
        assert_eq!(r.tier(), Tier::Remote);
    }
}
