//! DeepSeek V4 safetensors loading helpers.
//!
//! The public V4 Flash checkpoint stores most dense weights as block-scaled
//! FP8 and routed experts as packed FP4-in-I8. ARLE's current CUDA linear path
//! consumes BF16 `DeviceMatrix`, so this module implements a correctness-first
//! startup path: shard the tensor for this rank, dequantize only that shard to
//! BF16 on the host, and upload the BF16 shard to the GPU.

use std::collections::HashMap;
use std::ops::Range;

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use safetensors::{Dtype, SafeTensors};

use crate::tp::{TpLoadContext, TpShardAxis};

use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec};

#[derive(Clone, Debug)]
enum MatrixShard {
    Full,
    Rows { range: Range<usize>, total: usize },
    Cols { range: Range<usize>, total: usize },
}

impl MatrixShard {
    fn from_tp(tp: Option<&TpLoadContext>) -> Self {
        let Some(tp) = tp else {
            return Self::Full;
        };
        match tp.axis {
            TpShardAxis::Column => Self::Rows {
                range: tp.sharding.range(),
                total: tp.sharding.total,
            },
            TpShardAxis::Row => Self::Cols {
                range: tp.sharding.range(),
                total: tp.sharding.total,
            },
        }
    }

    fn row_range(&self, rows: usize) -> Result<Range<usize>> {
        match self {
            Self::Full | Self::Cols { .. } => Ok(0..rows),
            Self::Rows { range, total } => {
                ensure!(
                    *total == rows,
                    "row shard total {total} does not match tensor rows {rows}"
                );
                ensure!(
                    range.end <= rows,
                    "row shard {:?} exceeds rows {rows}",
                    range
                );
                Ok(range.clone())
            }
        }
    }

    fn col_range(&self, cols: usize) -> Result<Range<usize>> {
        match self {
            Self::Full | Self::Rows { .. } => Ok(0..cols),
            Self::Cols { range, total } => {
                ensure!(
                    *total == cols,
                    "column shard total {total} does not match tensor cols {cols}"
                );
                ensure!(
                    range.end <= cols,
                    "column shard {:?} exceeds cols {cols}",
                    range
                );
                Ok(range.clone())
            }
        }
    }
}

pub(super) fn load_dsv4_matrix_bf16(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<DeviceMatrix> {
    load_dsv4_matrix_bf16_sharded(ctx, shards, weight_map, name, None)
}

pub(super) fn load_dsv4_matrix_bf16_sharded(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    tp: Option<&TpLoadContext>,
) -> Result<DeviceMatrix> {
    let (host, rows, cols) =
        load_dsv4_matrix_host_bf16(shards, weight_map, name, MatrixShard::from_tp(tp))?;
    DeviceMatrix::from_host(ctx, &host, rows, cols)
        .with_context(|| format!("uploading DeepSeek V4 matrix {name} [{rows}, {cols}]"))
}

pub(super) fn load_dsv4_vec_bf16(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<DeviceVec> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    ensure!(
        shape.len() == 1,
        "{name}: expected 1D tensor, got shape {:?}",
        shape
    );
    let mut out = Vec::with_capacity(shape[0]);
    for idx in 0..shape[0] {
        out.push(bf16::from_f32(scalar_f32(
            tensor.dtype(),
            tensor.data(),
            idx,
        )?));
    }
    DeviceVec::from_host(ctx, &out)
        .map(|v| v.with_label(Box::leak(format!("{name}[{}]", out.len()).into_boxed_str())))
}

pub(super) fn dsv4_matrix_host_bf16_for_test(
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    tp: Option<&TpLoadContext>,
) -> Result<(Vec<bf16>, usize, usize)> {
    load_dsv4_matrix_host_bf16(shards, weight_map, name, MatrixShard::from_tp(tp))
}

fn load_dsv4_matrix_host_bf16(
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    shard: MatrixShard,
) -> Result<(Vec<bf16>, usize, usize)> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    ensure!(
        shape.len() == 2,
        "{name}: expected 2D tensor, got shape {:?}",
        shape
    );
    let rows = shape[0];
    let physical_cols = shape[1];
    let logical_cols = match tensor.dtype() {
        Dtype::I8 => physical_cols * 2,
        _ => physical_cols,
    };
    let row_range = shard.row_range(rows)?;
    let col_range = shard.col_range(logical_cols)?;
    let out_rows = row_range.len();
    let out_cols = col_range.len();
    let mut out = Vec::with_capacity(out_rows * out_cols);

    let scale = if matches!(tensor.dtype(), Dtype::F8_E4M3 | Dtype::I8) {
        let scale_name = name
            .strip_suffix(".weight")
            .map(|prefix| format!("{prefix}.scale"))
            .with_context(|| format!("{name}: quantized DSv4 tensor must end with .weight"))?;
        Some(
            find_tensor(shards, weight_map, &scale_name)
                .with_context(|| format!("{name}: missing block scale tensor {scale_name}"))?,
        )
    } else {
        None
    };

    for row in row_range {
        for col in col_range.clone() {
            let value = matrix_value_f32(&tensor, scale.as_ref(), row, col, rows, logical_cols)
                .with_context(|| format!("dequantizing {name}[{row}, {col}]"))?;
            out.push(bf16::from_f32(value));
        }
    }
    Ok((out, out_rows, out_cols))
}

fn matrix_value_f32(
    tensor: &safetensors::tensor::TensorView<'_>,
    scale: Option<&safetensors::tensor::TensorView<'_>>,
    row: usize,
    col: usize,
    rows: usize,
    logical_cols: usize,
) -> Result<f32> {
    match tensor.dtype() {
        Dtype::BF16 | Dtype::F32 | Dtype::F8_E8M0 => {
            let idx = row * tensor.shape()[1] + col;
            scalar_f32(tensor.dtype(), tensor.data(), idx)
        }
        Dtype::F8_E4M3 => {
            let idx = row * tensor.shape()[1] + col;
            let value = scalar_f32(tensor.dtype(), tensor.data(), idx)?;
            Ok(value
                * block_scale_f32(
                    scale.context("missing FP8 scale")?,
                    row,
                    col,
                    rows,
                    logical_cols,
                )?)
        }
        Dtype::I8 => {
            let packed_cols = tensor.shape()[1];
            let packed = tensor.data()[row * packed_cols + col / 2];
            let nibble = if col % 2 == 0 {
                packed & 0x0f
            } else {
                (packed >> 4) & 0x0f
            };
            Ok(decode_fp4_e2m1(nibble)
                * block_scale_f32(
                    scale.context("missing FP4 scale")?,
                    row,
                    col,
                    rows,
                    logical_cols,
                )?)
        }
        dtype => bail!("unsupported DeepSeek V4 matrix dtype {dtype:?}"),
    }
}

fn block_scale_f32(
    scale: &safetensors::tensor::TensorView<'_>,
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
) -> Result<f32> {
    ensure!(
        scale.shape().len() == 2,
        "DeepSeek V4 scale tensor must be 2D, got {:?}",
        scale.shape()
    );
    let scale_rows = scale.shape()[0];
    let scale_cols = scale.shape()[1];
    ensure!(scale_rows > 0 && scale_cols > 0, "empty scale tensor");
    let block_h = rows.div_ceil(scale_rows).max(1);
    let block_w = cols.div_ceil(scale_cols).max(1);
    let scale_row = (row / block_h).min(scale_rows - 1);
    let scale_col = (col / block_w).min(scale_cols - 1);
    scalar_f32(
        scale.dtype(),
        scale.data(),
        scale_row * scale_cols + scale_col,
    )
}

fn scalar_f32(dtype: Dtype, data: &[u8], idx: usize) -> Result<f32> {
    match dtype {
        Dtype::BF16 => {
            let offset = idx * 2;
            ensure!(offset + 2 <= data.len(), "BF16 read out of range");
            Ok(bf16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]])).to_f32())
        }
        Dtype::F32 => {
            let offset = idx * 4;
            ensure!(offset + 4 <= data.len(), "F32 read out of range");
            Ok(f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]))
        }
        Dtype::F8_E4M3 => {
            ensure!(idx < data.len(), "F8_E4M3 read out of range");
            Ok(decode_fp8_e4m3fn(data[idx]))
        }
        Dtype::F8_E8M0 => {
            ensure!(idx < data.len(), "F8_E8M0 read out of range");
            Ok(decode_f8_e8m0(data[idx]))
        }
        dtype => bail!("cannot read dtype {dtype:?} as f32 scalar"),
    }
}

fn find_tensor<'a>(
    shards: &'a [SafeTensors<'a>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<safetensors::tensor::TensorView<'a>> {
    if let Some(&idx) = weight_map.get(name) {
        return shards[idx]
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("failed to load tensor {name}: {e}"));
    }
    for shard in shards {
        if let Ok(tensor) = shard.tensor(name) {
            return Ok(tensor);
        }
    }
    bail!("tensor {name} not found in any shard")
}

fn decode_f8_e8m0(bits: u8) -> f32 {
    f32::from_bits((bits as u32) << 23)
}

fn decode_fp8_e4m3fn(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 3) & 0x0f;
    let mant = bits & 0x07;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2.0_f32.powi(-6)
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2.0_f32.powi(exp as i32 - 7)
    }
}

fn decode_fp4_e2m1(bits: u8) -> f32 {
    let sign = if bits & 0x08 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 1) & 0x03;
    let mant = bits & 0x01;
    if exp == 0 {
        sign * (mant as f32 * 0.5)
    } else {
        sign * (1.0 + mant as f32 * 0.5) * 2.0_f32.powi(exp as i32 - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{TensorView, serialize};

    fn single_shard(
        name: &str,
        weight: TensorView<'_>,
        scale_name: Option<&str>,
        scale: Option<TensorView<'_>>,
    ) -> (Vec<u8>, HashMap<String, usize>) {
        let mut tensors = vec![(name.to_string(), weight)];
        let mut map = HashMap::from([(name.to_string(), 0)]);
        if let (Some(scale_name), Some(scale)) = (scale_name, scale) {
            tensors.push((scale_name.to_string(), scale));
            map.insert(scale_name.to_string(), 0);
        }
        (serialize(tensors, None).unwrap(), map)
    }

    #[test]
    fn dequantizes_fp8_block_scaled_matrix() {
        let weight_bytes = [0x38_u8, 0xb8, 0x40, 0xc0];
        let scale_bytes = [127_u8];
        let weight = TensorView::new(Dtype::F8_E4M3, vec![2, 2], &weight_bytes).unwrap();
        let scale = TensorView::new(Dtype::F8_E8M0, vec![1, 1], &scale_bytes).unwrap();
        let (buf, map) = single_shard("a.weight", weight, Some("a.scale"), Some(scale));
        let shards = vec![SafeTensors::deserialize(&buf).unwrap()];

        let (host, rows, cols) =
            dsv4_matrix_host_bf16_for_test(&shards, &map, "a.weight", None).unwrap();

        assert_eq!((rows, cols), (2, 2));
        let values = host.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
        assert_eq!(values, vec![1.0, -1.0, 2.0, -2.0]);
    }

    #[test]
    fn dequantizes_packed_fp4_column_shard() {
        let weight_bytes = [0x21_u8, 0xb3, 0x40, 0x08];
        let scale_bytes = [127_u8];
        let weight = TensorView::new(Dtype::I8, vec![2, 2], &weight_bytes).unwrap();
        let scale = TensorView::new(Dtype::F8_E8M0, vec![1, 1], &scale_bytes).unwrap();
        let (buf, map) = single_shard("e.weight", weight, Some("e.scale"), Some(scale));
        let shards = vec![SafeTensors::deserialize(&buf).unwrap()];
        let tp = TpLoadContext::row(1, 2, 4).unwrap();

        let (host, rows, cols) =
            dsv4_matrix_host_bf16_for_test(&shards, &map, "e.weight", Some(&tp)).unwrap();

        assert_eq!((rows, cols), (2, 2));
        let values = host.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
        assert_eq!(values, vec![1.0, -1.5, 4.0, -0.0]);
    }
}
