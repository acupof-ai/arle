//! A q tile shorter than the local k/v block must still attend the whole block.
//!
//! Every k-side extent in `cp_causal_sdpa` used to be derived from q's row count,
//! which was exact only while q was the whole local shard. Once the caller tiles q
//! the device path fails loudly on a length mismatch, but this host path would
//! silently attend a truncated prefix.

use autograd::{CpuBackend, Tape, Tensor, TensorStore, ops::ring_attention::cp_causal_sdpa};
use std::sync::Arc;

const H: usize = 2;
const D: usize = 4;
const ROWS: usize = 8;
const TILE: usize = 4;

fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// Rows `[TILE, ROWS)` of q against the full k/v, run two ways: as a tile whose
/// absolute positions are threaded in, and as the tail of a full-length call.
#[test]
fn tiled_q_attends_the_whole_k_block() {
    let mut store = TensorStore::with_backend(Arc::new(CpuBackend));
    let mut tape = Tape::new();
    tape.set_enabled(false);

    let kv = seeded(H * ROWS * D, 0x9e37_79b9);
    let vv = seeded(H * ROWS * D, 0x1234_5678);
    let qv = seeded(H * ROWS * D, 0x51ed_2701);
    let shape = vec![1, H, ROWS, D];

    let k = store.alloc(Tensor::new(kv.clone(), shape.clone(), false).unwrap());
    let v = store.alloc(Tensor::new(vv.clone(), shape.clone(), false).unwrap());
    let q_full = store.alloc(Tensor::new(qv.clone(), shape.clone(), false).unwrap());

    let positions: Vec<usize> = (0..ROWS).collect();
    let full = cp_causal_sdpa(
        q_full,
        k,
        v,
        1,
        0,
        Some(&positions),
        None,
        &mut store,
        &mut tape,
    )
    .expect("full-length cp_causal_sdpa");
    let full_out = store.to_host(full).unwrap();

    // The tail tile, head-major like the full tensor.
    let mut tile = Vec::with_capacity(H * TILE * D);
    for h in 0..H {
        let base = h * ROWS * D + TILE * D;
        tile.extend_from_slice(&qv[base..base + TILE * D]);
    }
    let q_tile = store.alloc(Tensor::new(tile, vec![1, H, TILE, D], false).unwrap());

    let tiled = cp_causal_sdpa(
        q_tile,
        k,
        v,
        1,
        0,
        Some(&positions[TILE..]),
        Some(&positions),
        &mut store,
        &mut tape,
    )
    .expect("tiled cp_causal_sdpa");
    let tiled_out = store.to_host(tiled).unwrap();

    assert_eq!(tiled_out.len(), H * TILE * D);
    for h in 0..H {
        for r in 0..TILE {
            for c in 0..D {
                let got = tiled_out[(h * TILE + r) * D + c];
                let want = full_out[(h * ROWS + TILE + r) * D + c];
                assert!(
                    (got - want).abs() < 1e-5,
                    "head {h} row {r} dim {c}: tiled {got} vs full-length {want}"
                );
            }
        }
    }
}

/// k positions that do not cover the local block are rejected, not silently used.
#[test]
fn short_k_positions_are_refused() {
    let mut store = TensorStore::with_backend(Arc::new(CpuBackend));
    let mut tape = Tape::new();
    tape.set_enabled(false);

    let shape = vec![1, H, ROWS, D];
    let k = store.alloc(Tensor::new(seeded(H * ROWS * D, 1), shape.clone(), false).unwrap());
    let v = store.alloc(Tensor::new(seeded(H * ROWS * D, 2), shape.clone(), false).unwrap());
    let q = store.alloc(Tensor::new(seeded(H * TILE * D, 3), vec![1, H, TILE, D], false).unwrap());

    let positions: Vec<usize> = (0..ROWS).collect();
    let err = cp_causal_sdpa(
        q,
        k,
        v,
        1,
        0,
        Some(&positions[TILE..]),
        Some(&positions[..TILE]),
        &mut store,
        &mut tape,
    );
    assert!(err.is_err(), "short k positions must be refused");
}
