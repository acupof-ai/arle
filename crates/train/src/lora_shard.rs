//! WS5 correctness core — pure TP-sharding math for a LoRA A/B update.
//!
//! Slices each projection's LoRA A/B on the SAME axis the base weight loader
//! shards (`qwen35-spec::Shard`, `infer-cuda/src/loader.rs:2958/3110`), so a
//! per-rank `W_rank += scale·(B·A)_rank` reconstructs the full merge:
//!   - column-parallel (base split on out_features): slice B rows, A replicated;
//!     rank outputs concatenate along out.
//!   - row-parallel (base split on in_features): slice A columns, B replicated;
//!     rank partials all-reduce (sum) over the in split.
//!
//! The core ([`LoraShardKind`], [`shard_range`], [`shard_ab`]) plus its
//! reconstruction tests are device-neutral `Vec<f32>` math — no CUDA, MAC-runnable
//! under `cpu,no-cuda`. Only the [`StudentLoraUpdate`] wrapper + the
//! projection→kind map are `cuda`-gated, since those types reach `train` through
//! the `cuda`-only `infer-api` dep.

/// TP shard axis for a LoRA delta, matching the base weight's Megatron split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraShardKind {
    /// Base split on out_features → slice B rows, A replicated.
    Column,
    /// Base split on in_features → slice A columns, B replicated.
    Row,
    /// Base not TP-sliced (router / shared-expert gate / EP-placed experts).
    Replicated,
}

/// Contiguous even partition of `total` for `tp_rank`, mirroring
/// `infer_topo::column_shard` (`sharding.rs:128`): the last rank absorbs the
/// remainder so `sum(sizes) == total`. Same formula base weights use, so the
/// LoRA shard lines up with its base shard.
pub fn shard_range(total: usize, tp_rank: usize, tp_size: usize) -> std::ops::Range<usize> {
    let base = total / tp_size;
    let offset = tp_rank * base;
    let size = if tp_rank == tp_size - 1 {
        base + total % tp_size
    } else {
        base
    };
    offset..offset + size
}

/// Slice a projection's LoRA A `[lora_rank, in_features]` / B `[out_features,
/// lora_rank]` (both row-major) for `tp_rank`. Returns `(a', b', in', out')`.
/// `Replicated` and `tp_size <= 1` clone unchanged.
pub fn shard_ab(
    a: &[f32],
    b: &[f32],
    lora_rank: usize,
    in_features: usize,
    out_features: usize,
    kind: LoraShardKind,
    tp_rank: usize,
    tp_size: usize,
) -> (Vec<f32>, Vec<f32>, usize, usize) {
    if tp_size <= 1 || kind == LoraShardKind::Replicated {
        return (a.to_vec(), b.to_vec(), in_features, out_features);
    }
    match kind {
        // B is [out_features, lora_rank] → keep this rank's out rows; A full.
        LoraShardKind::Column => {
            let r = shard_range(out_features, tp_rank, tp_size);
            let b_shard = b[r.start * lora_rank..r.end * lora_rank].to_vec();
            (a.to_vec(), b_shard, in_features, r.len())
        }
        // A is [lora_rank, in_features] → keep this rank's in columns; B full.
        LoraShardKind::Row => {
            let r = shard_range(in_features, tp_rank, tp_size);
            let mut a_shard = Vec::with_capacity(lora_rank * r.len());
            for row in 0..lora_rank {
                let base = row * in_features;
                a_shard.extend_from_slice(&a[base + r.start..base + r.end]);
            }
            (a_shard, b.to_vec(), r.len(), out_features)
        }
        LoraShardKind::Replicated => unreachable!("handled above"),
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::{LoraShardKind, shard_ab};
    use infer_api::{
        StudentLoraLayer, StudentLoraMatrices, StudentLoraProjection, StudentLoraProjectionUpdate,
        StudentLoraUpdate,
    };

    impl LoraShardKind {
        /// Projection → base shard axis. Evidence: `qwen35-spec/src/lib.rs::shard_for`
        /// — q/k/v → Column{0}, o → Row{1} (:220,:223); mlp gate/up → Column, down →
        /// Row (:205,:208); shared-expert gate/up → Column, down → Row (:178,:181);
        /// router + shared_expert_gate → Replicated (:175); routed experts → None,
        /// i.e. EP-placed whole, never TP-sliced (:183); linear-attn in-projs →
        /// Column, out_proj → Row (:242,:251).
        ///
        /// `LinearQkv` is base-`MergedColumn` (fused): an even out-row slice still
        /// reconstructs the full delta by concat, but only aligns with the base's
        /// per-rank rows when each fused component is independently tp-divisible —
        /// documented, not silently assumed.
        pub fn for_projection(proj: StudentLoraProjection) -> Self {
            use StudentLoraProjection::*;
            match proj {
                FullQ | FullK | FullV | LinearQkv | LinearZ | LinearB | LinearA | MlpGate
                | MlpUp | MoeSharedGate | MoeSharedUp => Self::Column,
                FullO | LinearOut | MlpDown | MoeSharedDown => Self::Row,
                MoeRouter
                | MoeSharedExpertGate
                | MoeExpertGate { .. }
                | MoeExpertUp { .. }
                | MoeExpertDown { .. } => Self::Replicated,
            }
        }
    }

    fn shard_matrices(
        m: &StudentLoraMatrices,
        kind: LoraShardKind,
        tp_rank: usize,
        tp_size: usize,
    ) -> StudentLoraMatrices {
        let (a, b, in_features, out_features) = shard_ab(
            &m.a,
            &m.b,
            m.rank,
            m.in_features,
            m.out_features,
            kind,
            tp_rank,
            tp_size,
        );
        StudentLoraMatrices {
            a,
            b,
            rank: m.rank,
            in_features,
            out_features,
        }
    }

    /// Shard a full LoRA update for one TP rank: each projection's A/B is sliced on
    /// the axis its base weight is sharded (column → B rows, row → A columns),
    /// leaving `rank`/`alpha` untouched. `tp_size == 1` is identity.
    pub fn shard_lora_for_rank(
        update: &StudentLoraUpdate,
        tp_rank: usize,
        tp_size: usize,
    ) -> StudentLoraUpdate {
        let layers = update
            .layers
            .iter()
            .map(|layer| StudentLoraLayer {
                layer_idx: layer.layer_idx,
                projections: layer
                    .projections
                    .iter()
                    .map(|p| StudentLoraProjectionUpdate {
                        projection: p.projection,
                        matrices: shard_matrices(
                            &p.matrices,
                            LoraShardKind::for_projection(p.projection),
                            tp_rank,
                            tp_size,
                        ),
                    })
                    .collect(),
            })
            .collect();
        StudentLoraUpdate {
            layers,
            rank: update.rank,
            alpha: update.alpha,
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::shard_lora_for_rank;

#[cfg(test)]
mod tests {
    use super::*;

    const RANK: usize = 4;
    const IN: usize = 8;
    const OUT: usize = 6;
    const TP: usize = 2;

    /// Deterministic distinct values so a mis-sliced row/col cannot pass by luck.
    fn ab(rank: usize, in_f: usize, out_f: usize) -> (Vec<f32>, Vec<f32>) {
        let a = (0..rank * in_f).map(|i| i as f32 * 0.5 + 1.0).collect();
        let b = (0..out_f * rank).map(|i| i as f32 * -0.3 + 2.0).collect();
        (a, b)
    }

    /// W = B·A, B [out,rank] · A [rank,in] → [out,in] row-major.
    fn ba(b: &[f32], a: &[f32], out: usize, rank: usize, in_f: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; out * in_f];
        for o in 0..out {
            for i in 0..in_f {
                let mut acc = 0.0;
                for r in 0..rank {
                    acc += b[o * rank + r] * a[r * in_f + i];
                }
                w[o * in_f + i] = acc;
            }
        }
        w
    }

    fn assert_close(x: &[f32], y: &[f32]) {
        assert_eq!(x.len(), y.len(), "length mismatch");
        for (a, b) in x.iter().zip(y) {
            assert!((a - b).abs() < 1e-4, "elem mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn column_parallel_concat_reconstructs_full() {
        let (a, b) = ab(RANK, IN, OUT);
        let want = ba(&b, &a, OUT, RANK, IN);
        // Concat each rank's out-slice B_shard·A along the out (row) axis.
        let mut got = Vec::new();
        for rank in 0..TP {
            let (a_s, b_s, in_s, out_s) =
                shard_ab(&a, &b, RANK, IN, OUT, LoraShardKind::Column, rank, TP);
            assert_eq!(in_s, IN, "A replicated → full in");
            got.extend_from_slice(&ba(&b_s, &a_s, out_s, RANK, in_s));
        }
        assert_close(&got, &want);
    }

    #[test]
    fn row_parallel_sum_reconstructs_full() {
        let (a, b) = ab(RANK, IN, OUT);
        let want = ba(&b, &a, OUT, RANK, IN);
        // Each rank's B·A_shard is [out, in/tp] over disjoint in columns; place
        // into a zero [out,in] at the column offset and sum over ranks — the
        // faithful row-parallel all-reduce.
        let mut got = vec![0.0f32; OUT * IN];
        for rank in 0..TP {
            let (a_s, b_s, in_s, out_s) =
                shard_ab(&a, &b, RANK, IN, OUT, LoraShardKind::Row, rank, TP);
            assert_eq!(out_s, OUT, "B replicated → full out");
            let r = shard_range(IN, rank, TP);
            let part = ba(&b_s, &a_s, OUT, RANK, in_s);
            for o in 0..OUT {
                for (j, col) in r.clone().enumerate() {
                    got[o * IN + col] += part[o * in_s + j];
                }
            }
        }
        assert_close(&got, &want);
    }

    #[test]
    fn tp_size_one_is_identity() {
        let (a, b) = ab(RANK, IN, OUT);
        for kind in [LoraShardKind::Column, LoraShardKind::Row] {
            let (a_s, b_s, in_s, out_s) = shard_ab(&a, &b, RANK, IN, OUT, kind, 0, 1);
            assert_eq!(a_s, a);
            assert_eq!(b_s, b);
            assert_eq!((in_s, out_s), (IN, OUT));
        }
    }

    #[test]
    fn replicated_is_not_sliced() {
        let (a, b) = ab(RANK, IN, OUT);
        let (a_s, b_s, in_s, out_s) =
            shard_ab(&a, &b, RANK, IN, OUT, LoraShardKind::Replicated, 1, TP);
        assert_eq!(a_s, a, "replicated A must stay whole");
        assert_eq!(b_s, b, "replicated B must stay whole");
        assert_eq!((in_s, out_s), (IN, OUT));
    }

    #[test]
    fn shard_ranges_partition_axis_exactly() {
        // No lost/dup indices: shards tile [0,total) contiguously, sizes sum to total.
        for (total, tp) in [(6usize, 2usize), (8, 2), (7, 2), (9, 3), (10, 4)] {
            let mut next = 0;
            let mut sum = 0;
            for rank in 0..tp {
                let r = shard_range(total, rank, tp);
                assert_eq!(
                    r.start, next,
                    "gap/overlap at rank {rank} (total {total}, tp {tp})"
                );
                next = r.end;
                sum += r.len();
            }
            assert_eq!(next, total, "last shard must reach total");
            assert_eq!(sum, total, "sizes must sum to total");
        }
    }

    #[test]
    fn column_reconstructs_when_out_not_divisible() {
        // out=7, tp=2 → rows 3+4; concat still equals full (no lost/dup rows).
        let (a, b) = ab(RANK, IN, 7);
        let want = ba(&b, &a, 7, RANK, IN);
        let mut got = Vec::new();
        for rank in 0..TP {
            let (a_s, b_s, in_s, out_s) =
                shard_ab(&a, &b, RANK, IN, 7, LoraShardKind::Column, rank, TP);
            got.extend_from_slice(&ba(&b_s, &a_s, out_s, RANK, in_s));
        }
        assert_close(&got, &want);
    }

    // Exact reconstruction of the COMPOSED MoE/SwiGLU expert under TP — the piece a
    // routed w=2-vs-1 or an fp32-loss finite-diff can't certify cleanly. An expert
    // is `down( silu(gate·x) ⊙ (up·x) )`. TP shards the intermediate dim: gate/up
    // are column-parallel (each rank owns intermediate cols [r*I/N, ..)), the
    // elementwise silu⊙ is INDEPENDENT per intermediate column (so a rank computes
    // its slice with no cross-rank data), and down is row-parallel (input = the
    // same intermediate slice) producing an [hidden] PARTIAL that sums across ranks
    // to the dense output. This test builds a dense expert, runs the sharded form,
    // and asserts sum-of-partials == dense in plain f32 — no GPU, no loss FD noise.
    // A wrong intermediate split (e.g. gate/up and down disagreeing on which cols a
    // rank owns) breaks it; the single-Linear tests above can't catch that coupling.
    #[test]
    fn moe_tp_composed_swiglu_reconstructs_dense() {
        let (hidden, inter, tp) = (4usize, 6usize, 3usize); // inter divisible by tp
        let x: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.3 - 0.5).collect();
        // Dense weights: gate/up are [inter, hidden] (out=inter), down is [hidden, inter].
        let gate = weight(inter, hidden, 1);
        let up = weight(inter, hidden, 2);
        let down = weight(hidden, inter, 3);

        let silu = |v: f32| v / (1.0 + (-v).exp());
        // Dense reference: g = silu(gate·x) ⊙ (up·x); out = down·g.
        let g: Vec<f32> = (0..inter)
            .map(|o| silu(row_dot(&gate, o, hidden, &x)) * row_dot(&up, o, hidden, &x))
            .collect();
        let dense: Vec<f32> = (0..hidden).map(|o| row_dot(&down, o, inter, &g)).collect();

        // Sharded: each rank owns intermediate cols [range), computes its g-slice,
        // and its down PARTIAL over those cols; partials sum to the dense output.
        let mut summed = vec![0.0f32; hidden];
        for rank in 0..tp {
            let r = shard_range(inter, rank, tp);
            let g_slice: Vec<f32> = r
                .clone()
                .map(|o| silu(row_dot(&gate, o, hidden, &x)) * row_dot(&up, o, hidden, &x))
                .collect();
            // down partial: down rows restricted to this rank's intermediate cols.
            for (o, out) in summed.iter_mut().enumerate() {
                for (j, col) in r.clone().enumerate() {
                    *out += down[o * inter + col] * g_slice[j];
                }
            }
        }
        assert_close(&summed, &dense);
    }

    fn weight(rows: usize, cols: usize, seed: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| ((i * 7 + seed * 13) % 11) as f32 * 0.1 - 0.5)
            .collect()
    }

    fn row_dot(w: &[f32], row: usize, cols: usize, x: &[f32]) -> f32 {
        (0..cols).map(|c| w[row * cols + c] * x[c]).sum()
    }
}
