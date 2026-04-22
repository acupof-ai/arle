//! Control-plane readmission plans for staged prefix reuse.
//!
//! A request-facing scheduler should only know:
//! - which radix-backed blocks are already runnable in T0
//! - which blocks need a fetch into host-pinned memory first
//! - how to dedupe an in-flight staged fetch

use crate::types::{BlockFingerprint, BlockId};

use super::{
    BlockLocation, FetchRequest, HostPinnedRegion, RemoteBlockDesc, RequestChunkState,
    SharedHostPinnedPool,
};

/// Concrete source for one staged block that must be promoted back into T0.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReadmissionSource {
    HostPinned {
        region: HostPinnedRegion,
    },
    Disk {
        fingerprint: BlockFingerprint,
        payload_len: u64,
    },
    Remote {
        desc: RemoteBlockDesc,
        payload_len: u64,
    },
}

impl ReadmissionSource {
    fn fetch_request(
        &self,
        block_id: BlockId,
        host_pool: SharedHostPinnedPool,
    ) -> Option<FetchRequest> {
        match self {
            Self::HostPinned { region } => Some(FetchRequest {
                block_id,
                source: BlockLocation::HostPinned {
                    offset: region.offset,
                },
                byte_len: region.len,
                host_pool,
            }),
            Self::Disk {
                fingerprint,
                payload_len,
            } => usize::try_from(*payload_len)
                .ok()
                .map(|byte_len| FetchRequest {
                    block_id,
                    source: BlockLocation::Disk {
                        fingerprint: *fingerprint,
                        payload_len: *payload_len,
                    },
                    byte_len,
                    host_pool,
                }),
            Self::Remote { desc, payload_len } => {
                usize::try_from(*payload_len)
                    .ok()
                    .map(|byte_len| FetchRequest {
                        block_id,
                        source: BlockLocation::Remote { desc: desc.clone() },
                        byte_len,
                        host_pool,
                    })
            }
        }
    }
}

/// One matched radix block in a staged readmission plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadmissionBlock {
    pub block_id: BlockId,
    pub fingerprint: BlockFingerprint,
    pub source: Option<ReadmissionSource>,
}

/// Dedupe key for an in-flight staged fetch batch.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ReadmissionKey {
    staged_fingerprints: Vec<BlockFingerprint>,
}

impl ReadmissionKey {
    pub fn fingerprints(&self) -> &[BlockFingerprint] {
        &self.staged_fingerprints
    }
}

/// Request-local staged prefix plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadmissionPlan {
    pub matched_len: usize,
    pub blocks: Vec<ReadmissionBlock>,
    pub state: RequestChunkState,
}

impl ReadmissionPlan {
    pub fn new(matched_len: usize, blocks: Vec<ReadmissionBlock>) -> Self {
        Self {
            matched_len,
            blocks,
            state: RequestChunkState::Planned,
        }
    }

    pub fn block_ids(&self) -> Vec<BlockId> {
        self.blocks.iter().map(|block| block.block_id).collect()
    }

    pub fn fetch_key(&self) -> Option<ReadmissionKey> {
        let staged_fingerprints = self
            .blocks
            .iter()
            .filter(|block| block.source.is_some())
            .map(|block| block.fingerprint)
            .collect::<Vec<_>>();
        (!staged_fingerprints.is_empty()).then_some(ReadmissionKey {
            staged_fingerprints,
        })
    }

    pub fn fetch_requests(&self, host_pool: &SharedHostPinnedPool) -> Option<Vec<FetchRequest>> {
        let requests = self
            .blocks
            .iter()
            .filter_map(|block| {
                block
                    .source
                    .as_ref()
                    .and_then(|source| source.fetch_request(block.block_id, host_pool.clone()))
            })
            .collect::<Vec<_>>();
        if requests.len()
            == self
                .blocks
                .iter()
                .filter(|block| block.source.is_some())
                .count()
        {
            Some(requests)
        } else {
            None
        }
    }

    pub fn mark_fetching(&mut self) {
        self.state = RequestChunkState::Fetching;
    }

    pub fn mark_ready(&mut self) {
        self.state = RequestChunkState::Ready;
    }

    pub fn mark_consumed(&mut self) {
        self.state = RequestChunkState::Consumed;
    }

    pub fn source_counts(&self) -> (usize, usize, usize) {
        let mut host = 0usize;
        let mut disk = 0usize;
        let mut remote = 0usize;
        for block in &self.blocks {
            match block.source {
                Some(ReadmissionSource::HostPinned { .. }) => host += 1,
                Some(ReadmissionSource::Disk { .. }) => disk += 1,
                Some(ReadmissionSource::Remote { .. }) => remote += 1,
                None => {}
            }
        }
        (host, disk, remote)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_tier::HostPinnedPool;

    #[test]
    fn fetch_key_only_tracks_staged_blocks() {
        let plan = ReadmissionPlan {
            matched_len: 32,
            blocks: vec![
                ReadmissionBlock {
                    block_id: BlockId(1),
                    fingerprint: BlockFingerprint([0x11; 16]),
                    source: None,
                },
                ReadmissionBlock {
                    block_id: BlockId(2),
                    fingerprint: BlockFingerprint([0x22; 16]),
                    source: Some(ReadmissionSource::HostPinned {
                        region: HostPinnedRegion {
                            offset: 0,
                            len: 4096,
                        },
                    }),
                },
                ReadmissionBlock {
                    block_id: BlockId(3),
                    fingerprint: BlockFingerprint([0x33; 16]),
                    source: Some(ReadmissionSource::Disk {
                        fingerprint: BlockFingerprint([0x33; 16]),
                        payload_len: 8192,
                    }),
                },
                ReadmissionBlock {
                    block_id: BlockId(4),
                    fingerprint: BlockFingerprint([0x44; 16]),
                    source: Some(ReadmissionSource::Remote {
                        desc: RemoteBlockDesc {
                            transport: crate::kv_tier::TransportId::SharedFilesystem,
                            payload: vec![9, 8, 7],
                        },
                        payload_len: 4096,
                    }),
                },
            ],
            state: RequestChunkState::Planned,
        };

        let key = plan.fetch_key().expect("staged key");
        assert_eq!(
            key.fingerprints(),
            &[
                BlockFingerprint([0x22; 16]),
                BlockFingerprint([0x33; 16]),
                BlockFingerprint([0x44; 16])
            ]
        );
    }

    #[test]
    fn fetch_requests_preserve_staged_source_locations() {
        let pool = SharedHostPinnedPool::new(HostPinnedPool::new(1 << 20).unwrap());
        let region = HostPinnedRegion {
            offset: 4096,
            len: 2048,
        };
        let plan = ReadmissionPlan {
            matched_len: 32,
            blocks: vec![
                ReadmissionBlock {
                    block_id: BlockId(9),
                    fingerprint: BlockFingerprint([0x44; 16]),
                    source: Some(ReadmissionSource::HostPinned { region }),
                },
                ReadmissionBlock {
                    block_id: BlockId(10),
                    fingerprint: BlockFingerprint([0x55; 16]),
                    source: Some(ReadmissionSource::Disk {
                        fingerprint: BlockFingerprint([0x55; 16]),
                        payload_len: 4096,
                    }),
                },
                ReadmissionBlock {
                    block_id: BlockId(11),
                    fingerprint: BlockFingerprint([0x66; 16]),
                    source: Some(ReadmissionSource::Remote {
                        desc: RemoteBlockDesc {
                            transport: crate::kv_tier::TransportId::SharedFilesystem,
                            payload: vec![1, 2, 3],
                        },
                        payload_len: 2048,
                    }),
                },
            ],
            state: RequestChunkState::Planned,
        };
        let requests = plan.fetch_requests(&pool).unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(
            requests[0].source,
            BlockLocation::HostPinned {
                offset: region.offset,
            }
        );
        assert_eq!(requests[0].byte_len, region.len);
        assert_eq!(
            requests[1].source,
            BlockLocation::Disk {
                fingerprint: BlockFingerprint([0x55; 16]),
                payload_len: 4096,
            }
        );
        assert_eq!(requests[1].byte_len, 4096);
        assert_eq!(
            requests[2].source,
            BlockLocation::Remote {
                desc: RemoteBlockDesc {
                    transport: crate::kv_tier::TransportId::SharedFilesystem,
                    payload: vec![1, 2, 3],
                },
            }
        );
        assert_eq!(requests[2].byte_len, 2048);
    }

    #[test]
    fn staged_plan_state_moves_forward() {
        let mut plan = ReadmissionPlan::new(16, Vec::new());
        assert_eq!(plan.state, RequestChunkState::Planned);
        plan.mark_fetching();
        assert_eq!(plan.state, RequestChunkState::Fetching);
        plan.mark_ready();
        assert_eq!(plan.state, RequestChunkState::Ready);
        plan.mark_consumed();
        assert_eq!(plan.state, RequestChunkState::Consumed);
    }
}
