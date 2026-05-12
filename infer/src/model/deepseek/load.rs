//! DeepSeek V4 safetensors loading helpers.
//!
//! The public V4 Flash checkpoint stores most dense weights as block-scaled
//! FP8 and routed experts as packed FP4-in-I8. The serving path keeps those
//! tensors in their raw format and uploads the companion E8M0 block scales.

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

pub(super) fn load_dsv4_matrix_raw(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<DeviceMatrix> {
    load_dsv4_matrix_raw_sharded(ctx, shards, weight_map, name, None)
}

pub(super) fn load_dsv4_matrix_raw_sharded(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    tp: Option<&TpLoadContext>,
) -> Result<DeviceMatrix> {
    let shard = MatrixShard::from_tp(tp);
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    ensure!(
        shape.len() == 2,
        "{name}: expected 2D tensor, got shape {:?}",
        shape
    );
    match tensor.dtype() {
        Dtype::F8_E4M3 | Dtype::I8 => {
            load_dsv4_block_scaled_matrix_raw(ctx, shards, weight_map, name, tensor, shard)
        }
        Dtype::BF16 | Dtype::F32 | Dtype::F8_E8M0 => {
            let (host, rows, cols) = load_dsv4_matrix_host_bf16(shards, weight_map, name, shard)?;
            DeviceMatrix::from_host(ctx, &host, rows, cols)
                .with_context(|| format!("uploading DeepSeek V4 dense matrix {name}"))
        }
        dtype => bail!("unsupported DeepSeek V4 raw matrix dtype {dtype:?} for {name}"),
    }
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

fn load_dsv4_block_scaled_matrix_raw(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    tensor: safetensors::tensor::TensorView<'_>,
    shard: MatrixShard,
) -> Result<DeviceMatrix> {
    let shape = tensor.shape();
    let rows = shape[0];
    let physical_cols = shape[1];
    let logical_cols = match tensor.dtype() {
        Dtype::I8 => physical_cols * 2,
        Dtype::F8_E4M3 => physical_cols,
        dtype => bail!("{name}: expected FP8/FP4 tensor, got {dtype:?}"),
    };
    let row_range = shard.row_range(rows)?;
    let col_range = shard.col_range(logical_cols)?;
    let out_rows = row_range.len();
    let out_cols = col_range.len();

    let scale_name = name
        .strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.scale"))
        .with_context(|| format!("{name}: quantized DSv4 tensor must end with .weight"))?;
    let scale = find_tensor(shards, weight_map, &scale_name)
        .with_context(|| format!("{name}: missing block scale tensor {scale_name}"))?;
    ensure!(
        scale.dtype() == Dtype::F8_E8M0,
        "{scale_name}: expected F8_E8M0 scale tensor, got {:?}",
        scale.dtype()
    );
    let (scale_rows, scale_cols, scale_bytes) =
        copy_scale_shard(&scale, &row_range, &col_range, rows, logical_cols)
            .with_context(|| format!("copying block scales for {name}"))?;

    match tensor.dtype() {
        Dtype::F8_E4M3 => {
            let weight_bytes =
                copy_fp8_matrix_shard(tensor.data(), rows, physical_cols, &row_range, &col_range)?;
            DeviceMatrix::from_dsv4_fp8_block_scaled(
                ctx,
                &weight_bytes,
                &scale_bytes,
                out_rows,
                out_cols,
                scale_rows,
                scale_cols,
            )
            .with_context(|| format!("uploading DeepSeek V4 raw FP8 matrix {name}"))
        }
        Dtype::I8 => {
            let weight_bytes =
                copy_fp4_matrix_shard(tensor.data(), rows, physical_cols, &row_range, &col_range)?;
            DeviceMatrix::from_dsv4_fp4_block_scaled(
                ctx,
                &weight_bytes,
                &scale_bytes,
                out_rows,
                out_cols,
                scale_rows,
                scale_cols,
            )
            .with_context(|| format!("uploading DeepSeek V4 raw FP4 matrix {name}"))
        }
        _ => unreachable!(),
    }
}

fn copy_scale_shard(
    scale: &safetensors::tensor::TensorView<'_>,
    row_range: &Range<usize>,
    col_range: &Range<usize>,
    total_rows: usize,
    total_cols: usize,
) -> Result<(usize, usize, Vec<u8>)> {
    ensure!(
        scale.shape().len() == 2,
        "DeepSeek V4 scale tensor must be 2D, got {:?}",
        scale.shape()
    );
    let scale_rows = scale.shape()[0];
    let scale_cols = scale.shape()[1];
    ensure!(scale_rows > 0 && scale_cols > 0, "empty scale tensor");
    ensure!(
        scale.data().len() == scale_rows * scale_cols,
        "E8M0 scale data len {} != shape {}x{}",
        scale.data().len(),
        scale_rows,
        scale_cols
    );
    let block_h = total_rows.div_ceil(scale_rows).max(1);
    let block_w = total_cols.div_ceil(scale_cols).max(1);
    ensure!(
        row_range.start == 0 || row_range.start.is_multiple_of(block_h),
        "row shard {:?} is not aligned to DSv4 scale block height {block_h}",
        row_range
    );
    ensure!(
        row_range.end == total_rows || row_range.end.is_multiple_of(block_h),
        "row shard {:?} is not aligned to DSv4 scale block height {block_h}",
        row_range
    );
    ensure!(
        col_range.start == 0 || col_range.start.is_multiple_of(block_w),
        "column shard {:?} is not aligned to DSv4 scale block width {block_w}",
        col_range
    );
    ensure!(
        col_range.end == total_cols || col_range.end.is_multiple_of(block_w),
        "column shard {:?} is not aligned to DSv4 scale block width {block_w}",
        col_range
    );
    let scale_row_start = row_range.start / block_h;
    let scale_row_end = row_range.end.div_ceil(block_h).min(scale_rows);
    let scale_col_start = col_range.start / block_w;
    let scale_col_end = col_range.end.div_ceil(block_w).min(scale_cols);
    let out_scale_rows = scale_row_end - scale_row_start;
    let out_scale_cols = scale_col_end - scale_col_start;
    ensure!(
        out_scale_rows > 0 && out_scale_cols > 0,
        "empty DSv4 scale shard for rows {:?} cols {:?}",
        row_range,
        col_range
    );

    let mut out = Vec::with_capacity(out_scale_rows * out_scale_cols);
    for scale_row in scale_row_start..scale_row_end {
        let start = scale_row * scale_cols + scale_col_start;
        let end = scale_row * scale_cols + scale_col_end;
        out.extend_from_slice(&scale.data()[start..end]);
    }
    Ok((out_scale_rows, out_scale_cols, out))
}

fn copy_fp8_matrix_shard(
    data: &[u8],
    rows: usize,
    cols: usize,
    row_range: &Range<usize>,
    col_range: &Range<usize>,
) -> Result<Vec<u8>> {
    ensure!(
        data.len() == rows * cols,
        "FP8 data len {} != shape {}x{}",
        data.len(),
        rows,
        cols
    );
    let mut out = Vec::with_capacity(row_range.len() * col_range.len());
    for row in row_range.clone() {
        let start = row * cols + col_range.start;
        let end = row * cols + col_range.end;
        out.extend_from_slice(&data[start..end]);
    }
    Ok(out)
}

fn copy_fp4_matrix_shard(
    data: &[u8],
    rows: usize,
    packed_cols: usize,
    row_range: &Range<usize>,
    col_range: &Range<usize>,
) -> Result<Vec<u8>> {
    ensure!(
        col_range.start.is_multiple_of(2) && col_range.end.is_multiple_of(2),
        "FP4 column shard {:?} must start/end on packed-byte boundaries",
        col_range
    );
    ensure!(
        data.len() == rows * packed_cols,
        "FP4 data len {} != shape {}x{}",
        data.len(),
        rows,
        packed_cols
    );
    let byte_start = col_range.start / 2;
    let byte_end = col_range.end / 2;
    let mut out = Vec::with_capacity(row_range.len() * (byte_end - byte_start));
    for row in row_range.clone() {
        let start = row * packed_cols + byte_start;
        let end = row * packed_cols + byte_end;
        out.extend_from_slice(&data[start..end]);
    }
    Ok(out)
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
    use cuda_kernels::tensor::WeightFormat;
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

    #[test]
    fn loads_raw_fp8_row_shard() {
        let ctx = DeviceContext::new().unwrap();
        let weight_bytes = [
            0x38_u8, 0xb8, 0x40, 0xc0, 0x38, 0xb8, 0x40, 0xc0, 0x38, 0xb8, 0x40, 0xc0, 0x38, 0xb8,
            0x40, 0xc0,
        ];
        let scale_bytes = [127_u8, 128, 129, 130];
        let weight = TensorView::new(Dtype::F8_E4M3, vec![4, 4], &weight_bytes).unwrap();
        let scale = TensorView::new(Dtype::F8_E8M0, vec![2, 2], &scale_bytes).unwrap();
        let (buf, map) = single_shard("a.weight", weight, Some("a.scale"), Some(scale));
        let shards = vec![SafeTensors::deserialize(&buf).unwrap()];
        let tp = TpLoadContext::column(1, 2, 4).unwrap();

        let matrix =
            load_dsv4_matrix_raw_sharded(&ctx, &shards, &map, "a.weight", Some(&tp)).unwrap();

        assert_eq!(matrix.weight_format(), WeightFormat::Dsv4Fp8BlockScaled);
        assert_eq!((matrix.rows, matrix.cols), (2, 4));
        assert_eq!((matrix.dsv4_scale_rows, matrix.dsv4_scale_cols), (1, 2));
        let qweight = matrix.qweight.as_ref().unwrap();
        let qweight_host = ctx.stream.clone_dtoh(qweight).unwrap();
        let qweight_bytes = qweight_host.iter().map(|v| *v as u8).collect::<Vec<_>>();
        assert_eq!(qweight_bytes, weight_bytes[8..16]);
        let scale_host = ctx
            .stream
            .clone_dtoh(matrix.dsv4_scales.as_ref().unwrap())
            .unwrap();
        assert_eq!(scale_host, vec![129, 130]);
    }

    #[test]
    fn loads_raw_fp4_column_shard() {
        let ctx = DeviceContext::new().unwrap();
        let weight_bytes = [0x21_u8, 0xb3, 0x40, 0x08];
        let scale_bytes = [127_u8, 128, 129, 130];
        let weight = TensorView::new(Dtype::I8, vec![2, 2], &weight_bytes).unwrap();
        let scale = TensorView::new(Dtype::F8_E8M0, vec![2, 2], &scale_bytes).unwrap();
        let (buf, map) = single_shard("e.weight", weight, Some("e.scale"), Some(scale));
        let shards = vec![SafeTensors::deserialize(&buf).unwrap()];
        let tp = TpLoadContext::row(1, 2, 4).unwrap();

        let matrix =
            load_dsv4_matrix_raw_sharded(&ctx, &shards, &map, "e.weight", Some(&tp)).unwrap();

        assert_eq!(matrix.weight_format(), WeightFormat::Dsv4Fp4BlockScaled);
        assert_eq!((matrix.rows, matrix.cols), (2, 2));
        assert_eq!((matrix.dsv4_scale_rows, matrix.dsv4_scale_cols), (2, 1));
        let qweight = matrix.qweight.as_ref().unwrap();
        let qweight_host = ctx.stream.clone_dtoh(qweight).unwrap();
        let qweight_bytes = qweight_host.iter().map(|v| *v as u8).collect::<Vec<_>>();
        assert_eq!(qweight_bytes, vec![0xb3, 0x08]);
        let scale_host = ctx
            .stream
            .clone_dtoh(matrix.dsv4_scales.as_ref().unwrap())
            .unwrap();
        assert_eq!(scale_host, vec![128, 130]);
    }
}
