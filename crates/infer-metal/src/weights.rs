//! Weight tensor representation for the Qwen3.5/Qwen3.6 Metal port.

use anyhow::{Context, Result};

use crate::loader::TensorMap;
use crate::mlx::{MlxArray, concatenate_axis, dequantize, eval, transpose_all};

#[derive(Clone)]
pub(crate) enum WeightTensor {
    /// Pre-transposed dense weight: shape `[in, out]`.
    Dense(MlxArray),
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

    /// Quantization bit width, or `None` for a dense weight. Used to detect
    /// mixed-bit gate/up pairs (OptiQ) that cannot row-merge.
    pub(crate) fn quant_bits(&self) -> Option<i32> {
        match self {
            WeightTensor::Dense(_) => None,
            WeightTensor::Quantized { bits, .. } => Some(*bits),
        }
    }

    /// Pre-transposed dense `[in, out]` array for this weight. A quantized
    /// weight is dequantized (raw layout `[out, in]`) then transposed to match
    /// the `Dense` convention. Used to row-concatenate projections whose
    /// quantized layouts cannot be merged in place (mixed-bit per-weight quant).
    fn to_dense_in_out(&self) -> MlxArray {
        match self {
            WeightTensor::Dense(w_t) => w_t.clone(),
            WeightTensor::Quantized {
                w,
                scales,
                biases,
                group_size,
                bits,
            } => {
                let dense = dequantize(w, scales, biases, *group_size, *bits);
                transpose_all(&dense)
            }
        }
    }
}

/// A 3-D stack of affine-quantized expert weights for Qwen3.6 MoE. Group size
/// and bits come from the MoE config, not the tensor, so they are not carried here.
pub(crate) struct StackedQuantized {
    pub(crate) weight: MlxArray,
    pub(crate) scales: MlxArray,
    pub(crate) biases: MlxArray,
}

pub(crate) fn load_quantized_with_bits(
    tensors: &TensorMap,
    base: &str,
    group_size: i32,
    bits: i32,
) -> Result<WeightTensor> {
    let w = tensors
        .get(&format!("{base}.weight"))
        .cloned()
        .with_context(|| format!("missing quantized weight '{base}.weight'"))?;
    let scales = tensors
        .get(&format!("{base}.scales"))
        .cloned()
        .with_context(|| format!("missing quantized scales '{base}.scales'"))?;
    let biases = tensors
        .get(&format!("{base}.biases"))
        .cloned()
        .with_context(|| format!("missing quantized biases '{base}.biases'"))?;
    Ok(WeightTensor::Quantized {
        w,
        scales,
        biases,
        group_size,
        bits,
    })
}

pub(crate) fn load_stacked_quantized(tensors: &TensorMap, base: &str) -> Result<StackedQuantized> {
    let weight = tensors
        .get(&format!("{base}.weight"))
        .cloned()
        .with_context(|| format!("missing stacked quantized weight '{base}.weight'"))?;
    let scales = tensors
        .get(&format!("{base}.scales"))
        .cloned()
        .with_context(|| format!("missing stacked quantized scales '{base}.scales'"))?;
    let biases = tensors
        .get(&format!("{base}.biases"))
        .cloned()
        .with_context(|| format!("missing stacked quantized biases '{base}.biases'"))?;
    anyhow::ensure!(
        weight.shape().len() == 3,
        "stacked quantized weight '{base}.weight' must be 3-D, got shape {:?}",
        weight.shape()
    );
    Ok(StackedQuantized {
        weight,
        scales,
        biases,
    })
}

/// Gate+up input projection for one dense Qwen3.5 layer.
///
/// `MergedQuantized` row-merges gate and up into a single projection (matching
/// quantized layouts merge in place; same-bit dense bf16 concat by output rows).
/// `Separate` keeps gate and up as two distinct quantized weights — used for
/// mixed-bit per-weight quant (e.g. OptiQ gate=4-bit/up=8-bit) which cannot
/// row-merge, so each runs as its own quantized matmul instead of being
/// dequantized to a dense merged projection.
pub(crate) enum MlpInputProjection {
    MergedQuantized {
        gate_up_proj: WeightTensor,
        gate_dim: i32,
    },
    Separate {
        gate_dim: i32,
    },
}

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
///
/// Quantized weights with matching layouts merge in place. Otherwise -- dense
/// inputs, or quantized inputs whose layouts differ (mixed-bit per-weight quant
/// such as OptiQ's gate 4-bit / up 8-bit) -- each operand is dequantized to
/// dense `[in, out]` and concatenated on the output axis, yielding a single
/// dense merged projection the C++ MLP consumes via its dense fallback.
pub(crate) fn concat_weight_rows(lhs: &WeightTensor, rhs: &WeightTensor) -> Result<WeightTensor> {
    if let Some(merged) = merge_quantized_projection_rows(&[lhs, rhs])? {
        return Ok(merged);
    }

    let lhs_dense = lhs.to_dense_in_out();
    let rhs_dense = rhs.to_dense_in_out();
    let merged = concatenate_axis(&[lhs_dense, rhs_dense], 1);
    eval(&[&merged]);
    Ok(WeightTensor::Dense(merged))
}
