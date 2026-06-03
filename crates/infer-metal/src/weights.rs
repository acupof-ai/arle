//! Weight tensor representation for the R3a Qwen35 Metal port.

use anyhow::{Context, Result};

use crate::mlx::{MlxArray, concatenate_axis, eval};

/// A projection weight registered with the Qwen35 compiled model.
#[derive(Clone)]
pub(crate) enum WeightTensor {
    /// Pre-transposed dense weight: shape `[in, out]`.
    Dense(MlxArray),
    /// MLX affine quantized weight.
    Quantized {
        w: MlxArray,
        scales: MlxArray,
        biases: MlxArray,
        group_size: i32,
        bits: i32,
    },
}

impl WeightTensor {
    pub(crate) fn output_dim(&self) -> Result<i32> {
        match self {
            WeightTensor::Dense(w_t) => w_t
                .shape()
                .get(1)
                .copied()
                .context("dense projection missing output dimension"),
            WeightTensor::Quantized { w, .. } => w
                .shape()
                .first()
                .copied()
                .context("quantized projection missing output dimension"),
        }
    }
}

/// MLP input projections for one dense Qwen3.5 layer.
#[allow(dead_code)]
pub(crate) enum MlpInputProjection {
    Split {
        gate_proj: WeightTensor,
        up_proj: WeightTensor,
    },
    MergedQuantized {
        gate_up_proj: WeightTensor,
        gate_dim: i32,
        up_dim: i32,
    },
}

/// Merge affine-quantized projections by output rows when their layouts match.
pub(crate) fn merge_quantized_projection_rows(
    weights: &[&WeightTensor],
) -> Result<Option<WeightTensor>> {
    if weights.is_empty() {
        return Ok(None);
    }

    let mut ws = Vec::with_capacity(weights.len());
    let mut scales = Vec::with_capacity(weights.len());
    let mut biases = Vec::with_capacity(weights.len());
    let mut expected_w_dtype = None;
    let mut expected_scales_dtype = None;
    let mut expected_biases_dtype = None;
    let mut expected_group_size = None;
    let mut expected_bits = None;
    let mut expected_packed_in = None;
    let mut expected_group_cols = None;

    for weight in weights {
        let WeightTensor::Quantized {
            w,
            scales: scale,
            biases: bias,
            group_size,
            bits,
        } = weight
        else {
            return Ok(None);
        };

        let packed_in = w
            .shape()
            .get(1)
            .copied()
            .context("quantized projection missing packed input dimension")?;
        let group_cols = scale
            .shape()
            .get(1)
            .copied()
            .context("quantized projection missing scale group dimension")?;

        if expected_group_size.is_some_and(|expected| *group_size != expected)
            || expected_bits.is_some_and(|expected| *bits != expected)
            || expected_packed_in.is_some_and(|expected| packed_in != expected)
            || expected_group_cols.is_some_and(|expected| group_cols != expected)
            || expected_w_dtype.is_some_and(|expected| w.dtype() != expected)
            || expected_scales_dtype.is_some_and(|expected| scale.dtype() != expected)
            || expected_biases_dtype.is_some_and(|expected| bias.dtype() != expected)
        {
            return Ok(None);
        }

        expected_group_size.get_or_insert(*group_size);
        expected_bits.get_or_insert(*bits);
        expected_packed_in.get_or_insert(packed_in);
        expected_group_cols.get_or_insert(group_cols);
        expected_w_dtype.get_or_insert(w.dtype());
        expected_scales_dtype.get_or_insert(scale.dtype());
        expected_biases_dtype.get_or_insert(bias.dtype());

        ws.push(w.clone());
        scales.push(scale.clone());
        biases.push(bias.clone());
    }

    let merged_w = concatenate_axis(&ws, 0);
    let merged_scales = concatenate_axis(&scales, 0);
    let merged_biases = concatenate_axis(&biases, 0);
    eval(&[&merged_w, &merged_scales, &merged_biases]);

    Ok(Some(WeightTensor::Quantized {
        w: merged_w,
        scales: merged_scales,
        biases: merged_biases,
        group_size: expected_group_size.unwrap_or_default(),
        bits: expected_bits.unwrap_or_default(),
    }))
}

/// Concatenate projection output rows for dense or mergeable quantized weights.
pub(crate) fn concat_weight_rows(lhs: &WeightTensor, rhs: &WeightTensor) -> Result<WeightTensor> {
    if let Some(merged) = merge_quantized_projection_rows(&[lhs, rhs])? {
        return Ok(merged);
    }

    match (lhs, rhs) {
        (WeightTensor::Dense(lhs), WeightTensor::Dense(rhs)) => Ok(WeightTensor::Dense(
            concatenate_axis(&[lhs.clone(), rhs.clone()], 1),
        )),
        _ => anyhow::bail!("cannot row-concatenate mixed weight formats"),
    }
}
