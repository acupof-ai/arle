use anyhow::{Result, anyhow, bail, ensure};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::WeightFormat;
use cudarc::driver::{DevicePtr, DevicePtrMut};

fn fp8_f32_scale_shape(weight: &DeviceMatrix) -> Result<(i32, i32, i32, i32)> {
    match weight.weight_format {
        WeightFormat::Fp8BlockScaled => {
            ensure!(
                weight.quant_scale_rows > 0
                    && weight.quant_scale_cols > 0
                    && weight.quant_block_m > 0
                    && weight.quant_block_k > 0,
                "fp8_block_scaled missing scale/block metadata: scale={}x{}, block={}x{}",
                weight.quant_scale_rows,
                weight.quant_scale_cols,
                weight.quant_block_m,
                weight.quant_block_k
            );
            Ok((
                weight.quant_scale_rows as i32,
                weight.quant_scale_cols as i32,
                weight.quant_block_m as i32,
                weight.quant_block_k as i32,
            ))
        }
        WeightFormat::Fp8PerShard => {
            ensure!(
                weight.quant_scale_rows == 1 && weight.quant_scale_cols == 1,
                "fp8_per_shard dispatch currently supports one resident shard scale, got {}x{}",
                weight.quant_scale_rows,
                weight.quant_scale_cols
            );
            Ok((1, 1, weight.rows as i32, weight.cols as i32))
        }
        other => Err(anyhow!(
            "expected FP8 f32-scale resident quant format, got {other}"
        )),
    }
}

pub(super) fn gemm_batch(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();

    unsafe {
        match weight.weight_format {
            WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled => {
                let qw = weight
                    .qweight
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing qweight", weight.weight_format))?;
                let scales = weight
                    .dsv4_scales
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing dsv4_scales", weight.weight_format))?;
                let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
                let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
                let res = match weight.weight_format {
                    WeightFormat::Dsv4Fp8BlockScaled => ffi::dsv4_fp8_gemv_batch_cuda(
                        qw_ptr as *const u8,
                        scales_ptr as *const u8,
                        x_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        x.seq_len as i32,
                        weight.rows as i32,
                        weight.cols as i32,
                        weight.dsv4_scale_rows as i32,
                        weight.dsv4_scale_cols as i32,
                        stream,
                    ),
                    WeightFormat::Dsv4Fp4BlockScaled => ffi::dsv4_fp4_gemv_batch_cuda(
                        qw_ptr as *const u8,
                        scales_ptr as *const u8,
                        x_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        x.seq_len as i32,
                        weight.rows as i32,
                        weight.cols as i32,
                        weight.dsv4_scale_rows as i32,
                        weight.dsv4_scale_cols as i32,
                        stream,
                    ),
                    _ => unreachable!(),
                };
                res.result()?;
            }
            WeightFormat::Fp8BlockScaled | WeightFormat::Fp8PerShard => {
                let qw = weight
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing qweight_u8", weight.weight_format))?;
                let scales = weight
                    .scale_f32
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing scale_f32", weight.weight_format))?;
                let (scale_rows, scale_cols, block_m, block_k) = fp8_f32_scale_shape(weight)?;
                let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
                let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
                ffi::gemv_fp8_block_scaled_batch_cuda(
                    qw_ptr as *const u8,
                    scales_ptr as *const f32,
                    x_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    x.seq_len as i32,
                    weight.rows as i32,
                    weight.cols as i32,
                    scale_rows,
                    scale_cols,
                    block_m,
                    block_k,
                    stream,
                )
                .result()?;
            }
            WeightFormat::Fp4E2M1Group => {
                ensure!(
                    weight.quant_scale_rows == weight.rows,
                    "fp4_e2m1_group scale rows {} != weight rows {}",
                    weight.quant_scale_rows,
                    weight.rows
                );
                ensure!(
                    weight.group_size > 0
                        && weight.quant_scale_cols == weight.cols / weight.group_size,
                    "fp4_e2m1_group scale cols {} incompatible with cols {} group_size {}",
                    weight.quant_scale_cols,
                    weight.cols,
                    weight.group_size
                );
                let qw = weight
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("fp4_e2m1_group missing qweight_u8"))?;
                let scales = weight
                    .qscale_fp8
                    .as_ref()
                    .ok_or_else(|| anyhow!("fp4_e2m1_group missing qscale_fp8"))?;
                let global = weight
                    .scale_f32
                    .as_ref()
                    .ok_or_else(|| anyhow!("fp4_e2m1_group missing scale_f32 global scale"))?;
                ensure!(
                    global.len() == 1,
                    "fp4_e2m1_group dispatch currently supports one global scale, got {}",
                    global.len()
                );
                let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
                let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
                let (global_ptr, _gg) = global.device_ptr(&ctx.stream);
                ffi::gemv_fp4_e2m1_group_batch_cuda(
                    qw_ptr as *const u8,
                    scales_ptr as *const u8,
                    global_ptr as *const f32,
                    x_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    x.seq_len as i32,
                    weight.rows as i32,
                    weight.cols as i32,
                    weight.group_size as i32,
                    weight.quant_scale_cols as i32,
                    stream,
                )
                .result()?;
            }
            other => bail!("gemm_batch unsupported resident quant weight format {other}"),
        }
    }
    Ok(())
}

pub(super) fn gemv(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &DeviceVec,
    out: &mut DeviceVec,
) -> Result<()> {
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();

    unsafe {
        match weight.weight_format {
            WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled => {
                let qw = weight
                    .qweight
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing qweight", weight.weight_format))?;
                let scales = weight
                    .dsv4_scales
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing dsv4_scales", weight.weight_format))?;
                let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
                let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
                let res = match weight.weight_format {
                    WeightFormat::Dsv4Fp8BlockScaled => ffi::dsv4_fp8_gemv_cuda(
                        qw_ptr as *const u8,
                        scales_ptr as *const u8,
                        x_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        weight.rows as i32,
                        weight.cols as i32,
                        weight.dsv4_scale_rows as i32,
                        weight.dsv4_scale_cols as i32,
                        stream,
                    ),
                    WeightFormat::Dsv4Fp4BlockScaled => ffi::dsv4_fp4_gemv_cuda(
                        qw_ptr as *const u8,
                        scales_ptr as *const u8,
                        x_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        weight.rows as i32,
                        weight.cols as i32,
                        weight.dsv4_scale_rows as i32,
                        weight.dsv4_scale_cols as i32,
                        stream,
                    ),
                    _ => unreachable!(),
                };
                res.result()?;
            }
            WeightFormat::Fp8BlockScaled | WeightFormat::Fp8PerShard => {
                let qw = weight
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing qweight_u8", weight.weight_format))?;
                let scales = weight
                    .scale_f32
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing scale_f32", weight.weight_format))?;
                let (scale_rows, scale_cols, block_m, block_k) = fp8_f32_scale_shape(weight)?;
                let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
                let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
                ffi::gemv_fp8_block_scaled_cuda(
                    qw_ptr as *const u8,
                    scales_ptr as *const f32,
                    x_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    weight.rows as i32,
                    weight.cols as i32,
                    scale_rows,
                    scale_cols,
                    block_m,
                    block_k,
                    stream,
                )
                .result()?;
            }
            WeightFormat::Fp4E2M1Group => {
                ensure!(
                    weight.quant_scale_rows == weight.rows,
                    "fp4_e2m1_group scale rows {} != weight rows {}",
                    weight.quant_scale_rows,
                    weight.rows
                );
                ensure!(
                    weight.group_size > 0
                        && weight.quant_scale_cols == weight.cols / weight.group_size,
                    "fp4_e2m1_group scale cols {} incompatible with cols {} group_size {}",
                    weight.quant_scale_cols,
                    weight.cols,
                    weight.group_size
                );
                let qw = weight
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("fp4_e2m1_group missing qweight_u8"))?;
                let scales = weight
                    .qscale_fp8
                    .as_ref()
                    .ok_or_else(|| anyhow!("fp4_e2m1_group missing qscale_fp8"))?;
                let global = weight
                    .scale_f32
                    .as_ref()
                    .ok_or_else(|| anyhow!("fp4_e2m1_group missing scale_f32 global scale"))?;
                ensure!(
                    global.len() == 1,
                    "fp4_e2m1_group dispatch currently supports one global scale, got {}",
                    global.len()
                );
                let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
                let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
                let (global_ptr, _gg) = global.device_ptr(&ctx.stream);
                ffi::gemv_fp4_e2m1_group_cuda(
                    qw_ptr as *const u8,
                    scales_ptr as *const u8,
                    global_ptr as *const f32,
                    x_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    weight.rows as i32,
                    weight.cols as i32,
                    weight.group_size as i32,
                    weight.quant_scale_cols as i32,
                    stream,
                )
                .result()?;
            }
            other => bail!("gemv unsupported resident quant weight format {other}"),
        }
    }
    Ok(())
}
