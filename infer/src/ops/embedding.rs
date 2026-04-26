use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::WeightFormat;

/// Embedding lookup reading token_id from `decode_meta[0]` (CUDA Graph safe)
pub fn embedding_decode_into(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    decode_meta: &CudaSlice<i32>,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(embed.cols, out.len);

    let (meta_ptr, _gm) = decode_meta.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();

    let result = match embed.weight_format() {
        WeightFormat::DenseBf16 => {
            let (embed_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
            unsafe {
                ffi::embedding_decode_cuda(
                    embed_ptr as *const ffi::Half,
                    meta_ptr as *const i32,
                    out_ptr as *mut ffi::Half,
                    embed.cols as i32,
                    stream,
                )
            }
        }
        WeightFormat::W8A16 => {
            let qweight = embed
                .qweight
                .as_ref()
                .context("W8A16 embedding missing qweight")?;
            let qscales = embed
                .qscales
                .as_ref()
                .context("W8A16 embedding missing qscales")?;
            let (w_ptr, _gw) = qweight.device_ptr(&ctx.stream);
            let (s_ptr, _gs) = qscales.device_ptr(&ctx.stream);
            unsafe {
                ffi::q8_embedding_decode_cuda(
                    w_ptr as *const i8,
                    s_ptr as *const ffi::Half,
                    meta_ptr as *const i32,
                    out_ptr as *mut ffi::Half,
                    embed.cols as i32,
                    embed.group_size as i32,
                    stream,
                )
            }
        }
        WeightFormat::GgufQ3K
        | WeightFormat::GgufQ4K
        | WeightFormat::GgufQ5K
        | WeightFormat::GgufQ6K => {
            let qweight = embed
                .qweight
                .as_ref()
                .context("packed GGUF embedding missing qweight")?;
            let (w_ptr, _gw) = qweight.device_ptr(&ctx.stream);
            let w_ptr = w_ptr as *const u8;
            let meta_ptr = meta_ptr as *const i32;
            let out_ptr = out_ptr as *mut ffi::Half;
            unsafe {
                match embed.weight_format() {
                    WeightFormat::GgufQ3K => ffi::q3k_embedding_decode_cuda(
                        w_ptr,
                        meta_ptr,
                        out_ptr,
                        embed.cols as i32,
                        stream,
                    ),
                    WeightFormat::GgufQ4K => ffi::q4k_embedding_decode_cuda(
                        w_ptr,
                        meta_ptr,
                        out_ptr,
                        embed.cols as i32,
                        stream,
                    ),
                    WeightFormat::GgufQ5K => ffi::q5k_embedding_decode_cuda(
                        w_ptr,
                        meta_ptr,
                        out_ptr,
                        embed.cols as i32,
                        stream,
                    ),
                    WeightFormat::GgufQ6K => ffi::q6k_embedding_decode_cuda(
                        w_ptr,
                        meta_ptr,
                        out_ptr,
                        embed.cols as i32,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
        }
        other => anyhow::bail!("embedding_decode_into does not support {other}"),
    };
    result.result()?;

    Ok(())
}

/// Batched embedding lookup
pub fn embedding_batch(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids_gpu: &CudaSlice<i32>,
    out: &mut HiddenStates,
) -> Result<()> {
    let (t_ptr, _gt) = token_ids_gpu.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();

    let result = match embed.weight_format() {
        WeightFormat::DenseBf16 => {
            let (e_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
            unsafe {
                ffi::embedding_batched_cuda(
                    e_ptr as *const ffi::Half,
                    t_ptr as *const i32,
                    o_ptr as *mut ffi::Half,
                    embed.cols as i32,
                    out.seq_len as i32,
                    stream,
                )
            }
        }
        WeightFormat::W8A16 => {
            let qweight = embed
                .qweight
                .as_ref()
                .context("W8A16 embedding missing qweight")?;
            let qscales = embed
                .qscales
                .as_ref()
                .context("W8A16 embedding missing qscales")?;
            let (w_ptr, _gw) = qweight.device_ptr(&ctx.stream);
            let (s_ptr, _gs) = qscales.device_ptr(&ctx.stream);
            unsafe {
                ffi::q8_embedding_batched_cuda(
                    w_ptr as *const i8,
                    s_ptr as *const ffi::Half,
                    t_ptr as *const i32,
                    o_ptr as *mut ffi::Half,
                    embed.cols as i32,
                    out.seq_len as i32,
                    embed.group_size as i32,
                    stream,
                )
            }
        }
        WeightFormat::GgufQ3K
        | WeightFormat::GgufQ4K
        | WeightFormat::GgufQ5K
        | WeightFormat::GgufQ6K => {
            let qweight = embed
                .qweight
                .as_ref()
                .context("packed GGUF embedding missing qweight")?;
            let (w_ptr, _gw) = qweight.device_ptr(&ctx.stream);
            let w_ptr = w_ptr as *const u8;
            let t_ptr = t_ptr as *const i32;
            let o_ptr = o_ptr as *mut ffi::Half;
            unsafe {
                match embed.weight_format() {
                    WeightFormat::GgufQ3K => ffi::q3k_embedding_batched_cuda(
                        w_ptr,
                        t_ptr,
                        o_ptr,
                        embed.cols as i32,
                        out.seq_len as i32,
                        stream,
                    ),
                    WeightFormat::GgufQ4K => ffi::q4k_embedding_batched_cuda(
                        w_ptr,
                        t_ptr,
                        o_ptr,
                        embed.cols as i32,
                        out.seq_len as i32,
                        stream,
                    ),
                    WeightFormat::GgufQ5K => ffi::q5k_embedding_batched_cuda(
                        w_ptr,
                        t_ptr,
                        o_ptr,
                        embed.cols as i32,
                        out.seq_len as i32,
                        stream,
                    ),
                    WeightFormat::GgufQ6K => ffi::q6k_embedding_batched_cuda(
                        w_ptr,
                        t_ptr,
                        o_ptr,
                        embed.cols as i32,
                        out.seq_len as i32,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
        }
        other => anyhow::bail!("embedding_batch does not support {other}"),
    };
    result.result()?;

    Ok(())
}
