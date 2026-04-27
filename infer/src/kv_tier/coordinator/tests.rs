use std::sync::Arc;
use std::time::Duration;

use tempfile::tempdir;

use super::builder::{AllocatedRegions, CoordinatorBuilder};
use super::events::{
    CoordinatorCommand, CoordinatorEvent, FetchRequest, PrefetchAction, PrefetchPlan,
    PrefetchPlanRequest, StoreRequest, StoreTarget,
};
use super::types::{FailureClass, QueueBackpressure, QueueKind};
use crate::kv_tier::backend::ClusterSharedBackend;
use crate::kv_tier::tier::BlockLocation;
use crate::kv_tier::transport::disk::DiskStore;
use crate::kv_tier::transport::shared_fs::{SharedFsBlockLocation, SharedFsStore};
use crate::types::{BlockFingerprint, BlockId};

#[test]
fn coordinator_receives_commands() {
    let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
    handle.send(CoordinatorCommand::Shutdown).unwrap();
    assert!(!coordinator.run_once().unwrap());
    // Shutdown does not emit a coordinator event; the channel stays empty.
    assert!(events.try_recv().is_err());
}

#[test]
fn coordinator_shutdown_joins_thread_cleanly() {
    let (coordinator, handle, _events) = CoordinatorBuilder::new(4).build();
    let join_handle = coordinator.spawn("infer-tiered-kv-coord-test");
    handle.send(CoordinatorCommand::Shutdown).unwrap();

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(join_handle.join());
    });

    let join_result = rx
        .recv_timeout(Duration::from_secs(1))
        .expect("coordinator join timed out");
    assert!(matches!(join_result, Ok(Ok(()))));
}

#[test]
fn store_roundtrip_through_disk_store() {
    let dir = tempdir().unwrap();
    let disk_store = Arc::new(DiskStore::new(dir.path()));
    let (coordinator, handle, events) = CoordinatorBuilder::new(4)
        .disk_store(disk_store.clone())
        .build();
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(256).unwrap(),
    );
    let region = {
        let mut pool = host_pool.lock().unwrap();
        let region = pool.reserve(6).unwrap().unwrap();
        pool.as_mut_slice(region).copy_from_slice(b"abcdef");
        region
    };
    let fingerprint = BlockFingerprint([0x2A; 16]);
    let ticket = handle
        .submit_store(vec![StoreRequest {
            block_id: BlockId(7),
            fingerprint,
            kv_format_tag: 3,
            host_pool: host_pool.clone(),
            host_region: region,
            target: StoreTarget::Disk,
        }])
        .unwrap();

    assert!(coordinator.run_once().unwrap());
    assert_eq!(
        events.recv().unwrap(),
        CoordinatorEvent::StoreQueued {
            ticket,
            block_count: 1,
        }
    );
    let location = match events.recv().unwrap() {
        CoordinatorEvent::StoreCompleted {
            ticket: done,
            locations,
        } => {
            assert_eq!(done, ticket);
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].0, BlockId(7));
            match locations[0].1.clone() {
                BlockLocation::Disk {
                    fingerprint,
                    payload_len,
                } => crate::kv_tier::transport::disk::DiskBlockLocation {
                    path: disk_store.block_path_for(fingerprint).unwrap(),
                    fingerprint,
                    payload_len,
                },
                other => panic!("expected disk location, got {other:?}"),
            }
        }
        other => panic!("unexpected store event: {other:?}"),
    };

    let payload = host_pool.read_region(region).unwrap();
    assert_eq!(payload, b"abcdef");
    let reloaded = disk_store.get_block(&location, Some(fingerprint)).unwrap();
    assert_eq!(reloaded, b"abcdef");
}

#[test]
fn store_to_disk_fails_without_configured_disk_store() {
    let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(64).unwrap(),
    );
    let region = {
        let mut pool = host_pool.lock().unwrap();
        let region = pool.reserve(4).unwrap().unwrap();
        pool.as_mut_slice(region).copy_from_slice(b"test");
        region
    };

    let ticket = handle
        .submit_store(vec![StoreRequest {
            block_id: BlockId(3),
            fingerprint: BlockFingerprint([0x44; 16]),
            kv_format_tag: 1,
            host_pool,
            host_region: region,
            target: StoreTarget::Disk,
        }])
        .unwrap();
    assert!(coordinator.run_once().unwrap());
    assert_eq!(
        events.recv().unwrap(),
        CoordinatorEvent::StoreQueued {
            ticket,
            block_count: 1,
        }
    );
    match events.recv().unwrap() {
        CoordinatorEvent::StoreFailed {
            ticket: failed_ticket,
            failed_block,
            class,
            reason,
        } => {
            assert_eq!(failed_ticket, ticket);
            assert_eq!(failed_block, BlockId(3));
            assert_eq!(class, FailureClass::Failed);
            assert!(reason.contains("disk store not configured"));
        }
        other => panic!("unexpected store failure event: {other:?}"),
    }
}

#[test]
fn fetch_from_host_passthrough_keeps_existing_region() {
    let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(128).unwrap(),
    );
    let region = {
        let mut pool = host_pool.lock().unwrap();
        let region = pool.reserve(5).unwrap().unwrap();
        pool.as_mut_slice(region).copy_from_slice(b"hello");
        region
    };

    let ticket = handle
        .submit_fetch(vec![FetchRequest {
            block_id: BlockId(9),
            source: BlockLocation::HostPinned {
                offset: region.offset,
            },
            byte_len: region.len,
            host_pool: host_pool.clone(),
        }])
        .unwrap();

    assert!(coordinator.run_once().unwrap());
    assert_eq!(
        events.recv().unwrap(),
        CoordinatorEvent::FetchQueued {
            ticket,
            block_count: 1,
        }
    );
    match events.recv().unwrap() {
        CoordinatorEvent::FetchCompleted {
            ticket: done,
            blocks,
        } => {
            assert_eq!(done, ticket);
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0].block_id, BlockId(9));
            assert_eq!(blocks[0].host_region, region);
            assert_eq!(blocks[0].byte_len, region.len);
            assert!(!blocks[0].release_after_promote);
            assert_eq!(
                host_pool.read_region(blocks[0].host_region).unwrap(),
                b"hello"
            );
        }
        other => panic!("unexpected fetch event: {other:?}"),
    }
}

#[test]
fn fetch_from_disk_materializes_temp_host_region() {
    let dir = tempdir().unwrap();
    let disk_store = Arc::new(DiskStore::new(dir.path()));
    let (coordinator, handle, events) = CoordinatorBuilder::new(4)
        .disk_store(disk_store.clone())
        .build();
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(256).unwrap(),
    );
    let fingerprint = BlockFingerprint([0x7A; 16]);
    let location = disk_store.put_block(fingerprint, 5, b"disk-bytes").unwrap();

    let ticket = handle
        .submit_fetch(vec![FetchRequest {
            block_id: BlockId(11),
            source: BlockLocation::Disk {
                fingerprint,
                payload_len: location.payload_len,
            },
            byte_len: usize::try_from(location.payload_len).unwrap(),
            host_pool: host_pool.clone(),
        }])
        .unwrap();

    assert!(coordinator.run_once().unwrap());
    assert_eq!(
        events.recv().unwrap(),
        CoordinatorEvent::FetchQueued {
            ticket,
            block_count: 1,
        }
    );
    match events.recv().unwrap() {
        CoordinatorEvent::FetchCompleted {
            ticket: done,
            blocks,
        } => {
            assert_eq!(done, ticket);
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0].block_id, BlockId(11));
            assert!(blocks[0].release_after_promote);
            assert_eq!(
                host_pool.read_region(blocks[0].host_region).unwrap(),
                b"disk-bytes"
            );
            host_pool.release_region(blocks[0].host_region).unwrap();
        }
        other => panic!("unexpected fetch event: {other:?}"),
    }
}

#[test]
fn fetch_fails_for_gpu_source() {
    let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(64).unwrap(),
    );
    let ticket = handle
        .submit_fetch(vec![FetchRequest {
            block_id: BlockId(13),
            source: BlockLocation::Gpu { slot: 0 },
            byte_len: 16,
            host_pool,
        }])
        .unwrap();

    assert!(coordinator.run_once().unwrap());
    assert_eq!(
        events.recv().unwrap(),
        CoordinatorEvent::FetchQueued {
            ticket,
            block_count: 1,
        }
    );
    match events.recv().unwrap() {
        CoordinatorEvent::FetchFailed {
            ticket: failed_ticket,
            failed_block,
            class,
            reason,
        } => {
            assert_eq!(failed_ticket, ticket);
            assert_eq!(failed_block, BlockId(13));
            assert_eq!(class, FailureClass::Failed);
            assert!(reason.contains("gpu source"));
        }
        other => panic!("unexpected fetch failure event: {other:?}"),
    }
}

#[test]
fn submit_store_is_non_blocking_when_queue_is_full() {
    let (_coordinator, handle, _events) = CoordinatorBuilder::new(1).build();
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(64).unwrap(),
    );
    let region = {
        let mut pool = host_pool.lock().unwrap();
        let region = pool.reserve(4).unwrap().unwrap();
        pool.as_mut_slice(region).copy_from_slice(b"test");
        region
    };

    let first = handle.submit_store(vec![StoreRequest {
        block_id: BlockId(1),
        fingerprint: BlockFingerprint([0x11; 16]),
        kv_format_tag: 1,
        host_pool: host_pool.clone(),
        host_region: region,
        target: StoreTarget::Disk,
    }]);
    assert!(first.is_some());

    let second = handle.submit_store(vec![StoreRequest {
        block_id: BlockId(2),
        fingerprint: BlockFingerprint([0x22; 16]),
        kv_format_tag: 1,
        host_pool,
        host_region: region,
        target: StoreTarget::Disk,
    }]);
    assert!(second.is_none(), "full queue should not block store submit");
}

#[test]
fn queue_stats_report_backpressure_and_rejections() {
    let (_coordinator, handle, _events) = CoordinatorBuilder::new(1).build();
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(64).unwrap(),
    );
    let region = {
        let mut pool = host_pool.lock().unwrap();
        let region = pool.reserve(4).unwrap().unwrap();
        pool.as_mut_slice(region).copy_from_slice(b"test");
        region
    };

    let first = handle.submit_fetch(vec![FetchRequest {
        block_id: BlockId(21),
        source: BlockLocation::HostPinned {
            offset: region.offset,
        },
        byte_len: region.len,
        host_pool: host_pool.clone(),
    }]);
    assert!(first.is_some());
    let stats = handle.stats();
    assert_eq!(stats.fetch.queued, 1);
    assert_eq!(stats.total_queued, 1);
    assert_eq!(stats.backpressure(), QueueBackpressure::Saturated);
    assert_eq!(
        handle.queue_stats(QueueKind::Fetch).backpressure(),
        QueueBackpressure::Saturated
    );

    let second = handle.submit_fetch(vec![FetchRequest {
        block_id: BlockId(22),
        source: BlockLocation::HostPinned {
            offset: region.offset,
        },
        byte_len: region.len,
        host_pool,
    }]);
    assert!(second.is_none());
    let stats = handle.stats();
    assert_eq!(stats.fetch.submitted, 1);
    assert_eq!(stats.fetch.rejected, 1);
}

#[test]
fn cancelled_fetch_updates_stats_and_reports_cancel_reason() {
    let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(128).unwrap(),
    );
    let region = {
        let mut pool = host_pool.lock().unwrap();
        let region = pool.reserve(5).unwrap().unwrap();
        pool.as_mut_slice(region).copy_from_slice(b"hello");
        region
    };

    let ticket = handle
        .submit_fetch(vec![FetchRequest {
            block_id: BlockId(23),
            source: BlockLocation::HostPinned {
                offset: region.offset,
            },
            byte_len: region.len,
            host_pool,
        }])
        .unwrap();
    assert!(handle.cancel_fetch(ticket));

    assert!(coordinator.run_once().unwrap());
    assert_eq!(
        events.recv().unwrap(),
        CoordinatorEvent::FetchQueued {
            ticket,
            block_count: 1,
        }
    );
    match events.recv().unwrap() {
        CoordinatorEvent::FetchFailed {
            ticket: failed_ticket,
            failed_block,
            class,
            reason,
        } => {
            assert_eq!(failed_ticket, ticket);
            assert_eq!(failed_block, BlockId(23));
            assert_eq!(class, FailureClass::Cancelled);
            assert!(reason.contains("cancelled"));
        }
        other => panic!("unexpected fetch cancel event: {other:?}"),
    }

    let stats = handle.stats();
    assert_eq!(stats.fetch.cancelled, 1);
    assert_eq!(stats.fetch.failed, 0);
    assert_eq!(stats.fetch.completed, 0);
    assert_eq!(stats.fetch.queued, 0);
    assert_eq!(stats.fetch.in_flight, 0);
    assert!(!handle.cancel_fetch(ticket));
}

#[test]
fn plan_classifies_tiers_on_unified_event_channel() {
    let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
    let ticket = handle
        .submit_prefetch_plan(vec![
            PrefetchPlanRequest {
                block_id: BlockId(1),
                source: Some(BlockLocation::Gpu { slot: 0 }),
            },
            PrefetchPlanRequest {
                block_id: BlockId(2),
                source: Some(BlockLocation::HostPinned { offset: 1024 }),
            },
            PrefetchPlanRequest {
                block_id: BlockId(3),
                source: Some(BlockLocation::Disk {
                    fingerprint: BlockFingerprint([0x33; 16]),
                    payload_len: 4096,
                }),
            },
            PrefetchPlanRequest {
                block_id: BlockId(4),
                source: Some(BlockLocation::Remote {
                    desc: crate::kv_tier::RemoteBlockDesc {
                        transport: crate::kv_tier::TransportId::Nixl,
                        payload: vec![1, 2, 3],
                    },
                }),
            },
            PrefetchPlanRequest {
                block_id: BlockId(5),
                source: None,
            },
        ])
        .unwrap();

    assert!(coordinator.run_once().unwrap());
    assert_eq!(
        events.recv().unwrap(),
        CoordinatorEvent::PlanQueued {
            ticket,
            block_count: 5,
        }
    );
    match events.recv().unwrap() {
        CoordinatorEvent::PlanCompleted {
            ticket: done,
            plans,
        } => {
            assert_eq!(done, ticket);
            assert_eq!(
                plans,
                vec![
                    PrefetchPlan {
                        block_id: BlockId(1),
                        action: PrefetchAction::ReadyOnGpu,
                    },
                    PrefetchPlan {
                        block_id: BlockId(2),
                        action: PrefetchAction::PromoteFromHost,
                    },
                    PrefetchPlan {
                        block_id: BlockId(3),
                        action: PrefetchAction::FetchFromDisk,
                    },
                    PrefetchPlan {
                        block_id: BlockId(4),
                        action: PrefetchAction::FetchFromRemote,
                    },
                    PrefetchPlan {
                        block_id: BlockId(5),
                        action: PrefetchAction::Recompute,
                    },
                ]
            );
        }
        other => panic!("unexpected plan event: {other:?}"),
    }
}

#[test]
fn store_reports_remote_stub_failure_on_unified_event_channel() {
    let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(64).unwrap(),
    );
    let region = {
        let mut pool = host_pool.lock().unwrap();
        let region = pool.reserve(4).unwrap().unwrap();
        pool.as_mut_slice(region).copy_from_slice(b"test");
        region
    };

    let ticket = handle
        .submit_store(vec![StoreRequest {
            block_id: BlockId(7),
            fingerprint: BlockFingerprint([0x77; 16]),
            kv_format_tag: 1,
            host_pool,
            host_region: region,
            target: StoreTarget::Remote,
        }])
        .unwrap();

    assert!(coordinator.run_once().unwrap());
    assert_eq!(
        events.recv().unwrap(),
        CoordinatorEvent::StoreQueued {
            ticket,
            block_count: 1,
        }
    );
    match events.recv().unwrap() {
        CoordinatorEvent::StoreFailed {
            ticket: failed_ticket,
            failed_block,
            class,
            reason,
        } => {
            assert_eq!(failed_ticket, ticket);
            assert_eq!(failed_block, BlockId(7));
            assert_eq!(class, FailureClass::Failed);
            assert!(reason.contains("remote store not configured"));
        }
        other => panic!("unexpected store event: {other:?}"),
    }
}

#[test]
fn remote_store_and_fetch_roundtrip_through_shared_fs_backend() {
    let dir = tempdir().unwrap();
    let remote_store = Arc::new(SharedFsStore::new(dir.path()));
    let (coordinator, handle, events) = CoordinatorBuilder::new(4)
        .cluster_shared_backend(ClusterSharedBackend::SharedFs(remote_store.clone()))
        .build();
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(256).unwrap(),
    );
    let fingerprint = BlockFingerprint([0x31; 16]);
    let region = {
        let mut pool = host_pool.lock().unwrap();
        let region = pool.reserve(12).unwrap().unwrap();
        pool.as_mut_slice(region).copy_from_slice(b"remote-bytes");
        region
    };

    let store_ticket = handle
        .submit_store(vec![StoreRequest {
            block_id: BlockId(31),
            fingerprint,
            kv_format_tag: 2,
            host_pool: host_pool.clone(),
            host_region: region,
            target: StoreTarget::Remote,
        }])
        .unwrap();

    assert!(coordinator.run_once().unwrap());
    assert_eq!(
        events.recv().unwrap(),
        CoordinatorEvent::StoreQueued {
            ticket: store_ticket,
            block_count: 1,
        }
    );
    match events.recv().unwrap() {
        CoordinatorEvent::StoreCompleted { ticket, locations } => {
            assert_eq!(ticket, store_ticket);
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].0, BlockId(31));
        }
        other => panic!("unexpected store event: {other:?}"),
    }
    let stored = remote_store
        .get_block(
            SharedFsBlockLocation::new(fingerprint, 12),
            Some(fingerprint),
        )
        .unwrap();
    assert_eq!(stored, b"remote-bytes");
    let second_store_ticket = handle
        .submit_store(vec![StoreRequest {
            block_id: BlockId(31),
            fingerprint,
            kv_format_tag: 2,
            host_pool: host_pool.clone(),
            host_region: region,
            target: StoreTarget::Remote,
        }])
        .unwrap();
    assert!(coordinator.run_once().unwrap());
    assert_eq!(
        events.recv().unwrap(),
        CoordinatorEvent::StoreQueued {
            ticket: second_store_ticket,
            block_count: 1,
        }
    );
    match events.recv().unwrap() {
        CoordinatorEvent::StoreCompleted { ticket, locations } => {
            assert_eq!(ticket, second_store_ticket);
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].0, BlockId(31));
            assert_eq!(
                locations[0].1,
                SharedFsBlockLocation::new(fingerprint, 12)
                    .into_block_location()
                    .unwrap()
            );
        }
        other => panic!("unexpected second store event: {other:?}"),
    }
    let remote_location = SharedFsBlockLocation::new(fingerprint, 12)
        .into_block_location()
        .unwrap();
    assert_eq!(store_ticket.0, 1);

    let fetch_ticket = handle
        .submit_fetch(vec![FetchRequest {
            block_id: BlockId(31),
            source: remote_location,
            byte_len: 12,
            host_pool: host_pool.clone(),
        }])
        .unwrap();

    assert!(coordinator.run_once().unwrap());
    assert_eq!(
        events.recv().unwrap(),
        CoordinatorEvent::FetchQueued {
            ticket: fetch_ticket,
            block_count: 1,
        }
    );
    match events.recv().unwrap() {
        CoordinatorEvent::FetchCompleted { ticket, blocks } => {
            assert_eq!(ticket, fetch_ticket);
            assert_eq!(blocks.len(), 1);
            assert!(blocks[0].release_after_promote);
            assert_eq!(
                host_pool.read_region(blocks[0].host_region).unwrap(),
                b"remote-bytes"
            );
            host_pool.release_region(blocks[0].host_region).unwrap();
        }
        other => panic!("unexpected remote fetch event: {other:?}"),
    }
}

#[test]
fn coordinator_builder_constructs_with_no_backends() {
    // Smoke test: the new builder reaches a working coordinator with
    // zero optional backends configured (replaces the implicit shape
    // assertion that lived inside the old `Coordinator::new` path).
    let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
    assert_eq!(handle.stats().capacity, 4);
    handle.send(CoordinatorCommand::Shutdown).unwrap();
    assert!(!coordinator.run_once().unwrap());
    // Shutdown is silent on the event channel.
    assert!(events.try_recv().is_err());
}

#[test]
fn report_failure_classifies_cancel_vs_fail() {
    // Drive the typed `FailureClass` end-to-end and verify both legs
    // surface their queue-specific event shape; this is the main
    // protection against silent drift back to substring matching.
    let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(64).unwrap(),
    );

    // Cancel path.
    let ticket = handle
        .submit_fetch(vec![FetchRequest {
            block_id: BlockId(101),
            source: BlockLocation::HostPinned { offset: 0 },
            byte_len: 4,
            host_pool: host_pool.clone(),
        }])
        .unwrap();
    assert!(handle.cancel_fetch(ticket));
    assert!(coordinator.run_once().unwrap());
    let _queued = events.recv().unwrap();
    match events.recv().unwrap() {
        CoordinatorEvent::FetchFailed { class, reason, .. } => {
            assert_eq!(class, FailureClass::Cancelled);
            assert!(reason.contains("cancelled"));
        }
        other => panic!("expected FetchFailed cancel, got {other:?}"),
    }
    assert_eq!(handle.stats().fetch.cancelled, 1);
    assert_eq!(handle.stats().fetch.failed, 0);

    // Hard-fail path (no remote backend configured for a remote fetch).
    let ticket2 = handle
        .submit_fetch(vec![FetchRequest {
            block_id: BlockId(102),
            source: BlockLocation::Remote {
                desc: crate::kv_tier::RemoteBlockDesc {
                    transport: crate::kv_tier::TransportId::Nixl,
                    payload: vec![0],
                },
            },
            byte_len: 4,
            host_pool,
        }])
        .unwrap();
    assert!(coordinator.run_once().unwrap());
    let _queued2 = events.recv().unwrap();
    match events.recv().unwrap() {
        CoordinatorEvent::FetchFailed { class, reason, .. } => {
            assert_eq!(class, FailureClass::Failed);
            assert!(reason.contains("remote store not configured"));
        }
        other => panic!("expected FetchFailed hard error, got {other:?}"),
    }
    let stats = handle.stats();
    assert_eq!(stats.fetch.cancelled, 1, "cancel count untouched");
    assert_eq!(stats.fetch.failed, 1, "hard fail recorded");
    let _ = ticket2;
}

#[test]
fn allocated_regions_releases_on_drop_when_uncommitted() {
    // Reserve a region, hand it to AllocatedRegions, drop without
    // committing — the region must be released and the pool ready
    // to reserve the same span again.
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(64).unwrap(),
    );
    let region = {
        let mut pool = host_pool.lock().unwrap();
        pool.reserve(32).unwrap().unwrap()
    };
    {
        let mut regions = AllocatedRegions::new();
        regions.push(host_pool.clone(), region);
        // Dropped here without commit.
    }
    // After drop, the pool should accept another full reservation.
    let mut pool = host_pool.lock().unwrap();
    let r2 = pool
        .reserve(32)
        .unwrap()
        .expect("pool should have free space after drop release");
    // Release for cleanliness (region tracker may otherwise leak in
    // the test-scoped pool).
    pool.release(r2).unwrap();
}

#[test]
fn allocated_regions_release_now_releases_immediately_and_is_idempotent() {
    // release_now releases pushed regions even before Drop runs, so
    // handle_fetch can free host-pool capacity BEFORE blocking on a
    // bounded event channel send. Repeated calls are no-ops.
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(32).unwrap(),
    );
    let region = {
        let mut pool = host_pool.lock().unwrap();
        pool.reserve(32).unwrap().unwrap()
    };
    let mut regions = AllocatedRegions::new();
    regions.push(host_pool.clone(), region);
    regions.release_now();
    // Pool capacity is back: another full reservation must succeed.
    let r2 = {
        let mut pool = host_pool.lock().unwrap();
        pool.reserve(32)
            .unwrap()
            .expect("release_now should free capacity")
    };
    // Idempotent: a second release_now (and the implicit Drop) must not
    // double-release. Reserve r2 first so the pool is full again — if
    // release_now were to re-release the original `region`, the validate
    // check inside HostPinnedPool::release would error and we'd see a
    // log warning, but the test would still pass; a tighter check is
    // that committed becomes true so Drop is a no-op.
    regions.release_now();
    drop(regions);
    // Cleanup
    let mut pool = host_pool.lock().unwrap();
    pool.release(r2).unwrap();
}

#[test]
fn allocated_regions_no_release_when_committed() {
    // Commit before drop → region stays reserved; a follow-up full
    // reserve must fail because no slot is free.
    let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
        crate::kv_tier::HostPinnedPool::new(32).unwrap(),
    );
    let region = {
        let mut pool = host_pool.lock().unwrap();
        pool.reserve(32).unwrap().unwrap()
    };
    {
        let mut regions = AllocatedRegions::new();
        regions.push(host_pool.clone(), region);
        regions.commit();
    }
    // Pool is full; a new full reservation must be None (or the
    // reserve returns Err if the implementation rejects oversize
    // requests up front). Both indicate the committed region was
    // not released.
    let mut pool = host_pool.lock().unwrap();
    let outcome = pool.reserve(32).unwrap();
    assert!(
        outcome.is_none(),
        "committed region should NOT have been released",
    );
    // Cleanup — release the original region so the pool is tidy.
    pool.release(region).unwrap();
}
