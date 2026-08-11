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
    Column,
    Row,
    Replicated,
}

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
        LoraShardKind::Column => {
            let r = shard_range(out_features, tp_rank, tp_size);
            let b_shard = b[r.start * lora_rank..r.end * lora_rank].to_vec();
            (a.to_vec(), b_shard, in_features, r.len())
        }
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
        /// Projection → base shard axis. q/k/v/gate/up → Column; o/down → Row;
        /// router/experts → Replicated. `LinearQkv` is fused MergedColumn.
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
