use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;

use crate::{
    AutogradError, Result, ops,
    tensor::{Dirty, TensorId, TensorStore},
};

#[derive(Debug, Clone)]
pub enum SavedContext {
    None,
    Tensor(TensorId),
    Tensors(SmallVec<[TensorId; 4]>),
    TensorAndScalar(TensorId, f32),
    Shape(Vec<usize>),
    MatmulCtx {
        a: TensorId,
        b: TensorId,
    },
    SoftmaxCtx {
        y: TensorId,
    },
    LogSoftmaxCtx {
        y: TensorId,
    },
    GatherCtx {
        indices: Vec<usize>,
        src_shape: Vec<usize>,
    },
    MeanCtx {
        input: TensorId,
        numel: usize,
    },
    RMSNormCtx {
        x: TensorId,
        weight: TensorId,
        inv_rms: Vec<f32>,
        eps: f32,
    },
    SiluCtx {
        x: TensorId,
    },
    SigmoidCtx {
        y: TensorId,
    },
    GeluCtx {
        x: TensorId,
    },
    RoPECtx {
        cos: TensorId,
        sin: TensorId,
    },
    ReshapeCtx {
        input_shape: Vec<usize>,
    },
    SliceCtx {
        input_shape: Vec<usize>,
        starts: Vec<usize>,
        ends: Vec<usize>,
    },
    TransposeCtx {
        axis1: usize,
        axis2: usize,
    },
    AddBroadcastCtx {
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    },
    EmbeddingCtx {
        indices: Vec<usize>,
        table_shape: Vec<usize>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackwardOp {
    Add,
    Mul,
    MulScalar,
    Exp,
    Sum,
    Matmul,
    Softmax,
    LogSoftmax,
    Gather,
    Mean,
    RMSNorm,
    Silu,
    Sigmoid,
    Gelu,
    RoPE,
    Reshape,
    Slice,
    Transpose,
    AddBroadcast,
    Embedding,
}

#[derive(Debug, Clone)]
pub struct TapeEntry {
    pub op: BackwardOp,
    pub output_id: TensorId,
    pub input_ids: SmallVec<[TensorId; 2]>,
    pub saved: SavedContext,
}

pub(crate) type GradPairs = SmallVec<[(TensorId, TensorId); 2]>;

#[derive(Debug, Default)]
pub struct Tape {
    pub entries: Vec<TapeEntry>,
    pub enabled: bool,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            enabled: true,
        }
    }

    pub fn record(&mut self, entry: TapeEntry) {
        if self.enabled {
            self.entries.push(entry);
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn backward(
        &mut self,
        loss_id: TensorId,
        store: &mut TensorStore,
    ) -> Result<HashMap<TensorId, TensorId>> {
        let was_enabled = self.enabled;
        self.enabled = false;

        let result = (|| {
            // Batch-flush all Dirty::Device tape outputs in a single
            // `mlx_eval` call before walking the backward graph. The naive
            // per-id `ensure_host` loop would call `eval` once per handle —
            // a regression once M5.3b.1 made `sum` lazy, because both `y`
            // and `loss` end up Dirty::Device and each per-id eval crosses
            // the FFI boundary + grabs `MLX_GUARD`. MLX consumes the batch
            // as one graph realization (terminal handles share upstream
            // nodes), so subsequent per-id `readback`s are O(copy) only.
            let device_ids: Vec<TensorId> = self
                .entries
                .iter()
                .filter(|entry| {
                    store
                        .get(entry.output_id)
                        .is_some_and(|tensor| tensor.dirty == Dirty::Device)
                })
                .map(|entry| entry.output_id)
                .collect();
            store.flush_to_host_batch(&device_ids)?;

            let mut entry_by_output = HashMap::with_capacity(self.entries.len());
            for (index, entry) in self.entries.iter().enumerate() {
                entry_by_output.insert(entry.output_id, index);
            }

            let mut relevant_tensors = HashSet::new();
            let mut visited_outputs = HashSet::new();
            let mut post_order = Vec::new();
            collect_relevant(
                loss_id,
                &entry_by_output,
                &self.entries,
                &mut relevant_tensors,
                &mut visited_outputs,
                &mut post_order,
            );

            let mut grads = HashMap::new();
            let loss_grad_id = store.fill_like(loss_id, 1.0)?;
            grads.insert(loss_id, loss_grad_id);
            if store
                .get(loss_id)
                .is_some_and(|tensor| tensor.requires_grad)
            {
                store.accumulate_grad(loss_id, loss_grad_id)?;
            }

            for &entry_index in post_order.iter().rev() {
                let entry = self.entries[entry_index].clone();
                let output_grad_id = match grads.get(&entry.output_id).copied() {
                    Some(grad_id) => grad_id,
                    None => continue,
                };

                let input_grads = match entry.op {
                    BackwardOp::Add => ops::add_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Mul => ops::mul_backward(&entry, output_grad_id, store)?,
                    BackwardOp::MulScalar => {
                        ops::mul_scalar_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::Exp => ops::exp_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Sum => ops::sum_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Matmul => ops::matmul_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Softmax => ops::softmax_backward(&entry, output_grad_id, store)?,
                    BackwardOp::LogSoftmax => {
                        ops::log_softmax_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::Gather => {
                        ops::gather_last_dim_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::Mean => ops::mean_backward(&entry, output_grad_id, store)?,
                    BackwardOp::RMSNorm => ops::rmsnorm_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Silu => ops::silu_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Sigmoid => ops::sigmoid_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Gelu => ops::gelu_backward(&entry, output_grad_id, store)?,
                    BackwardOp::RoPE => ops::rope_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Reshape => ops::reshape_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Slice => ops::slice_backward(&entry, output_grad_id, store)?,
                    BackwardOp::Transpose => {
                        ops::transpose_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::AddBroadcast => {
                        ops::add_broadcast_backward(&entry, output_grad_id, store)?
                    }
                    BackwardOp::Embedding => {
                        ops::embedding_backward(&entry, output_grad_id, store)?
                    }
                };

                for (input_id, grad_id) in input_grads {
                    merge_grad(&mut grads, input_id, grad_id, store)?;
                }
            }

            Ok(grads)
        })();

        self.enabled = was_enabled;
        result
    }
}

fn collect_relevant(
    tensor_id: TensorId,
    entry_by_output: &HashMap<TensorId, usize>,
    entries: &[TapeEntry],
    relevant_tensors: &mut HashSet<TensorId>,
    visited_outputs: &mut HashSet<TensorId>,
    post_order: &mut Vec<usize>,
) {
    relevant_tensors.insert(tensor_id);
    let Some(&entry_index) = entry_by_output.get(&tensor_id) else {
        return;
    };

    let entry = &entries[entry_index];
    if !visited_outputs.insert(entry.output_id) {
        return;
    }

    for &input_id in &entry.input_ids {
        collect_relevant(
            input_id,
            entry_by_output,
            entries,
            relevant_tensors,
            visited_outputs,
            post_order,
        );
    }

    post_order.push(entry_index);
}

fn merge_grad(
    grads: &mut HashMap<TensorId, TensorId>,
    tensor_id: TensorId,
    new_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<()> {
    if let Some(existing_grad_id) = grads.get(&tensor_id).copied() {
        let expected = store.tensor(existing_grad_id)?.shape.clone();
        let incoming = store.tensor(new_grad_id)?.shape.clone();
        if expected != incoming {
            return Err(AutogradError::GradientShapeMismatch {
                tensor_id,
                expected,
                got: incoming,
            });
        }

        let incoming_data = store.to_host(new_grad_id)?;
        let existing = store.tensor_mut(existing_grad_id)?;
        for (dst, src) in existing.data.iter_mut().zip(incoming_data) {
            *dst += src;
        }
    } else {
        let cloned_grad_id = store.clone_tensor(new_grad_id)?;
        grads.insert(tensor_id, cloned_grad_id);
    }

    if store
        .get(tensor_id)
        .is_some_and(|tensor| tensor.requires_grad)
    {
        store.accumulate_grad(tensor_id, new_grad_id)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn backward_on_empty_tape_does_not_panic() {
        let mut store = TensorStore::default();
        let loss = store.alloc(Tensor::new(vec![5.0], Vec::new(), true).expect("create scalar"));
        let mut tape = Tape::new();

        let grads = tape.backward(loss, &mut store).expect("backward succeeds");

        let grad_id = grads.get(&loss).copied().expect("loss grad exists");
        assert_eq!(store.to_host(grad_id).expect("copy grad"), vec![1.0]);
    }
}
