//! Shape and norm primitives shared by every Qwen3.5/3.6 forward path.

use super::*;

pub(super) fn qwen35_rmsnorm(
    x: TensorId,
    weight: TensorId,
    eps: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let weight_shape = store
        .get(weight)
        .ok_or(AutogradError::InvalidTensorId(weight))?
        .shape
        .clone();
    if weight_shape.len() != 1 {
        return Err(AutogradError::InvalidRank {
            expected: "rank-1 RMSNorm weight",
            got: weight_shape.len(),
        }
        .into());
    }
    let ones = store.alloc(Tensor::new(
        vec![1.0; weight_shape[0]],
        weight_shape,
        false,
    )?);
    let offset_weight = add(weight, ones, store, tape)?;
    Ok(rmsnorm(x, offset_weight, eps, store, tape)?)
}

pub(super) fn validate_sequence_window(
    input_ids: &[u32],
    position_ids: &[u32],
    window: SequenceWindow,
) -> Result<()> {
    if input_ids.len() != position_ids.len() {
        return Err(Qwen35Error::InputLenMismatch {
            input_len: input_ids.len(),
            expected_len: position_ids.len(),
        });
    }
    if window.start >= window.end {
        return Err(Qwen35Error::InvalidConfig(
            "sequence logits window must be non-empty",
        ));
    }
    if window.end > input_ids.len() {
        return Err(Qwen35Error::InputLenMismatch {
            input_len: window.end,
            expected_len: input_ids.len(),
        });
    }
    Ok(())
}

pub(super) fn linear_forward(
    x: TensorId,
    weight: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x_shape = store
        .get(x)
        .ok_or(AutogradError::InvalidTensorId(x))?
        .shape
        .clone();
    let weight_shape = store
        .get(weight)
        .ok_or(AutogradError::InvalidTensorId(weight))?
        .shape
        .clone();
    if weight_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: weight_shape.len(),
        }
        .into());
    }

    let input_dim = *x_shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if input_dim != weight_shape[1] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![weight_shape[1]],
            got: vec![input_dim],
        }
        .into());
    }

    let prefix_elems = x_shape.iter().product::<usize>() / input_dim;
    let flat_x = reshape(x, &[prefix_elems, input_dim], store, tape)?;
    let projected = matmul_bt_with_site(flat_x, weight, store, tape, "lm_head")?;
    let mut output_shape = x_shape[..x_shape.len() - 1].to_vec();
    output_shape.push(weight_shape[0]);
    Ok(reshape(projected, &output_shape, store, tape)?)
}

pub(super) fn broadcast_to_shape(
    x: TensorId,
    shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let zeros = store.alloc(Tensor::new(
        vec![0.0; shape.iter().product()],
        shape.to_vec(),
        false,
    )?);
    Ok(add_broadcast(zeros, x, store, tape)?)
}

pub(crate) fn qwen35_to_autograd(err: Qwen35Error) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(err.to_string().into_boxed_str()))
}

pub(super) fn copy_frozen_tensor_map(
    source: &HashMap<&'static str, TensorId>,
    target: &HashMap<&'static str, TensorId>,
    store: &mut TensorStore,
) {
    let mut names = source.keys().copied().collect::<Vec<_>>();
    names.sort_unstable();
    for name in names {
        copy_frozen_tensor(source[&name], target[&name], store);
    }
}

pub(super) fn copy_frozen_tensor(
    source_id: TensorId,
    target_id: TensorId,
    store: &mut TensorStore,
) {
    let mut replacement = store
        .get(source_id)
        .cloned()
        .expect("source parameter should remain readable");
    replacement.requires_grad = false;
    replacement.grad = None;
    store.tensors[target_id] = Some(replacement);
}

pub(super) fn split_heads(
    x: TensorId,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x = reshape(x, &[batch, seq_len, heads, head_dim], store, tape)?;
    Ok(transpose(x, 1, 2, store, tape)?)
}

pub(super) fn merge_heads(
    x: TensorId,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x = transpose(x, 1, 2, store, tape)?;
    Ok(reshape(
        x,
        &[batch, seq_len, heads * head_dim],
        store,
        tape,
    )?)
}

pub(super) fn build_rope_cache(
    cfg: &Qwen35Config,
    store: &mut TensorStore,
) -> Result<(TensorId, TensorId)> {
    let max_positions = cfg.rope_cache_len_hint.ok_or(Qwen35Error::InvalidConfig(
        "train-side qwen3.5 requires rope_cache_len_hint",
    ))?;
    let half_dim = cfg.rotary_dim / 2;
    let inv_freq = (0..half_dim)
        .map(|index| {
            1.0 / cfg
                .rope_theta
                .powf((2.0 * index as f32) / cfg.rotary_dim as f32)
        })
        .collect::<Vec<_>>();
    let mut cos = vec![0.0; max_positions * half_dim];
    let mut sin = vec![0.0; max_positions * half_dim];

    for position in 0..max_positions {
        let base = position * half_dim;
        for (freq_index, &freq) in inv_freq.iter().enumerate() {
            let angle = position as f32 * freq;
            cos[base + freq_index] = angle.cos();
            sin[base + freq_index] = angle.sin();
        }
    }

    let cos_cache = store.alloc(Tensor::new(cos, vec![max_positions, half_dim], false)?);
    let sin_cache = store.alloc(Tensor::new(sin, vec![max_positions, half_dim], false)?);
    Ok((cos_cache, sin_cache))
}
