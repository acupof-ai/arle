//! Wire protocol mirror of `crates/cuda-kernels/csrc/deepep_sidecar/protocol.hpp`.
//!
//! All structs use `#[repr(C)]` to match the C++ layout. Byte serialization
//! is hand-rolled (little-endian) — no third-party dep needed and easy to
//! audit against the C++ side.

use anyhow::{Result, anyhow, bail};

pub const PROTOCOL_VERSION: u32 = 1;
pub const KMAX_NVL_PEERS: usize = 8;
pub const CHILD_P2C_FD: i32 = 10;
pub const CHILD_C2P_FD: i32 = 11;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    Ok = 0,
    ProtocolMismatch = 1,
    CudaError = 2,
    KernelTimeout = 3,
    BadArgs = 4,
    Internal = 5,
}

impl Status {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Status::Ok,
            1 => Status::ProtocolMismatch,
            2 => Status::CudaError,
            3 => Status::KernelTimeout,
            4 => Status::BadArgs,
            5 => Status::Internal,
            other => bail!("unknown sidecar status code {other}"),
        })
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandId {
    Boot = 0x01,
    Sync = 0x02,
    RoundTrip = 0x10,
    Dispatch = 0x20,
    Combine = 0x21,
    Shutdown = 0x7f,
}

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct MessageHeader {
    pub cmd_or_status: u32,
    pub payload_bytes: u32,
}

impl MessageHeader {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn to_le_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.cmd_or_status.to_le_bytes());
        buf[4..8].copy_from_slice(&self.payload_bytes.to_le_bytes());
        buf
    }

    pub fn from_le_bytes(buf: &[u8; Self::SIZE]) -> Self {
        Self {
            cmd_or_status: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            payload_bytes: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
        }
    }
}

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct BootRequest {
    pub protocol_version: u32,
    pub rank: u32,
    pub world_size: u32,
    pub reserved: u32,
}

impl BootRequest {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn to_le_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.protocol_version.to_le_bytes());
        buf[4..8].copy_from_slice(&self.rank.to_le_bytes());
        buf[8..12].copy_from_slice(&self.world_size.to_le_bytes());
        buf[12..16].copy_from_slice(&self.reserved.to_le_bytes());
        buf
    }
}

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct BootResponse {
    pub device_id: u32,
    pub reserved: u32,
    pub ipc_handle: [u8; 64],
}

impl BootResponse {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn from_le_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() != Self::SIZE {
            bail!(
                "BootResponse expected {} bytes, got {}",
                Self::SIZE,
                buf.len()
            );
        }
        let device_id = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let reserved = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let mut ipc_handle = [0u8; 64];
        ipc_handle.copy_from_slice(&buf[8..72]);
        Ok(Self {
            device_id,
            reserved,
            ipc_handle,
        })
    }

    pub fn to_le_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.device_id.to_le_bytes());
        buf[4..8].copy_from_slice(&self.reserved.to_le_bytes());
        buf[8..72].copy_from_slice(&self.ipc_handle);
        buf
    }
}

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct RoundTripRequest {
    pub num_tokens: u32,
    pub hidden: u32,
    pub num_topk: u32,
    pub num_experts: u32,
    pub num_sms: u32,
    pub nvl_chunked_send: u32,
    pub nvl_chunked_recv: u32,
    pub reserved: u32,
}

impl RoundTripRequest {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn to_le_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.num_tokens.to_le_bytes());
        buf[4..8].copy_from_slice(&self.hidden.to_le_bytes());
        buf[8..12].copy_from_slice(&self.num_topk.to_le_bytes());
        buf[12..16].copy_from_slice(&self.num_experts.to_le_bytes());
        buf[16..20].copy_from_slice(&self.num_sms.to_le_bytes());
        buf[20..24].copy_from_slice(&self.nvl_chunked_send.to_le_bytes());
        buf[24..28].copy_from_slice(&self.nvl_chunked_recv.to_le_bytes());
        buf[28..32].copy_from_slice(&self.reserved.to_le_bytes());
        buf
    }
}

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct RoundTripResponse {
    pub num_recv_tokens: u32,
    pub reserved: u32,
    pub sha256: [u8; 32],
    pub preview: [f32; 8],
}

impl RoundTripResponse {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn from_le_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() != Self::SIZE {
            bail!(
                "RoundTripResponse expected {} bytes, got {}",
                Self::SIZE,
                buf.len()
            );
        }
        let num_recv_tokens = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let reserved = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let mut sha256 = [0u8; 32];
        sha256.copy_from_slice(&buf[8..40]);
        let mut preview = [0.0f32; 8];
        for i in 0..8 {
            preview[i] = f32::from_le_bytes(buf[40 + i * 4..44 + i * 4].try_into().unwrap());
        }
        Ok(Self {
            num_recv_tokens,
            reserved,
            sha256,
            preview,
        })
    }
}

/// Serialize the post-boot SYNC payload — an array of N peer entries
/// (device_id + reserved + IPC handle), padded to KMAX_NVL_PEERS. Slots
/// past `world_size` are zero-filled.
pub fn encode_sync_payload(peers: &[BootResponse]) -> Result<Vec<u8>> {
    if peers.len() > KMAX_NVL_PEERS {
        return Err(anyhow!(
            "world_size {} exceeds KMAX_NVL_PEERS {}",
            peers.len(),
            KMAX_NVL_PEERS
        ));
    }
    let mut out = vec![0u8; BootResponse::SIZE * KMAX_NVL_PEERS];
    for (i, p) in peers.iter().enumerate() {
        let start = i * BootResponse::SIZE;
        out[start..start + BootResponse::SIZE].copy_from_slice(&p.to_le_bytes());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let h = MessageHeader {
            cmd_or_status: 0x10,
            payload_bytes: 32,
        };
        let bytes = h.to_le_bytes();
        let parsed = MessageHeader::from_le_bytes(&bytes);
        assert_eq!(parsed.cmd_or_status, 0x10);
        assert_eq!(parsed.payload_bytes, 32);
    }

    #[test]
    fn boot_request_size() {
        assert_eq!(BootRequest::SIZE, 16);
    }

    #[test]
    fn boot_response_size() {
        assert_eq!(BootResponse::SIZE, 72);
    }

    #[test]
    fn round_trip_request_size() {
        assert_eq!(RoundTripRequest::SIZE, 32);
    }

    #[test]
    fn round_trip_response_size() {
        assert_eq!(RoundTripResponse::SIZE, 72);
    }

    #[test]
    fn sync_payload_size() {
        let peers = vec![
            BootResponse {
                device_id: 0,
                reserved: 0,
                ipc_handle: [0; 64],
            };
            KMAX_NVL_PEERS
        ];
        let payload = encode_sync_payload(&peers).unwrap();
        assert_eq!(payload.len(), 72 * KMAX_NVL_PEERS);
    }
}
