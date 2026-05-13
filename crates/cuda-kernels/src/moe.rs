//! DeepSeek MoE routing helper kernels.

use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::DeviceContext;

#[allow(clippy::too_many_arguments)]
pub fn dsv4_mask_indices_by_ep_i64(
    ctx: &DeviceContext,
    indices: &CudaSlice<i64>,
    masked_indices: &mut CudaSlice<i64>,
    num_tokens: usize,
    num_topk: usize,
    experts_per_ep_rank: usize,
    experts_per_moe_dp_group: usize,
    num_tp_ranks: usize,
    tp_rank: usize,
) -> Result<()> {
    ensure_mask_args(
        indices.len(),
        masked_indices.len(),
        num_tokens,
        num_topk,
        experts_per_ep_rank,
        experts_per_moe_dp_group,
        num_tp_ranks,
        tp_rank,
    )?;

    let (indices_ptr, _g0) = indices.device_ptr(&ctx.stream);
    let (masked_ptr, _g1) = masked_indices.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::dsv4_mask_indices_by_ep_i64_cuda(
            indices_ptr as *const i64,
            masked_ptr as *mut i64,
            num_tokens as i32,
            num_topk as i32,
            experts_per_ep_rank as i32,
            experts_per_moe_dp_group as i32,
            num_tp_ranks as i32,
            tp_rank as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dsv4_mask_indices_by_ep_i32(
    ctx: &DeviceContext,
    indices: &CudaSlice<i32>,
    masked_indices: &mut CudaSlice<i32>,
    num_tokens: usize,
    num_topk: usize,
    experts_per_ep_rank: usize,
    experts_per_moe_dp_group: usize,
    num_tp_ranks: usize,
    tp_rank: usize,
) -> Result<()> {
    ensure_mask_args(
        indices.len(),
        masked_indices.len(),
        num_tokens,
        num_topk,
        experts_per_ep_rank,
        experts_per_moe_dp_group,
        num_tp_ranks,
        tp_rank,
    )?;

    let (indices_ptr, _g0) = indices.device_ptr(&ctx.stream);
    let (masked_ptr, _g1) = masked_indices.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::dsv4_mask_indices_by_ep_i32_cuda(
            indices_ptr as *const i32,
            masked_ptr as *mut i32,
            num_tokens as i32,
            num_topk as i32,
            experts_per_ep_rank as i32,
            experts_per_moe_dp_group as i32,
            num_tp_ranks as i32,
            tp_rank as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ensure_mask_args(
    input_len: usize,
    output_len: usize,
    num_tokens: usize,
    num_topk: usize,
    experts_per_ep_rank: usize,
    experts_per_moe_dp_group: usize,
    num_tp_ranks: usize,
    tp_rank: usize,
) -> Result<()> {
    let expected = num_tokens
        .checked_mul(num_topk)
        .ok_or_else(|| anyhow::anyhow!("num_tokens * num_topk overflows"))?;
    ensure!(
        input_len >= expected && output_len >= expected,
        "mask_indices_by_ep buffers too small: input={} output={} expected={}",
        input_len,
        output_len,
        expected
    );
    ensure!(experts_per_ep_rank > 0, "experts_per_ep_rank must be > 0");
    ensure!(
        experts_per_moe_dp_group >= experts_per_ep_rank,
        "experts_per_moe_dp_group ({experts_per_moe_dp_group}) must be >= experts_per_ep_rank ({experts_per_ep_rank})"
    );
    ensure!(num_tp_ranks > 0, "num_tp_ranks must be > 0");
    ensure!(
        tp_rank < num_tp_ranks,
        "tp_rank {tp_rank} must be < num_tp_ranks {num_tp_ranks}"
    );
    ensure!(expected <= i32::MAX as usize, "mask input too large");
    ensure!(
        experts_per_ep_rank <= i32::MAX as usize
            && experts_per_moe_dp_group <= i32::MAX as usize
            && num_tp_ranks <= i32::MAX as usize
            && tp_rank <= i32::MAX as usize,
        "mask parameter exceeds i32 kernel ABI"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        indices: &[i64],
        experts_per_ep_rank: i64,
        experts_per_moe_dp_group: i64,
        num_tp_ranks: i64,
        tp_rank: i64,
    ) -> Vec<i64> {
        indices
            .iter()
            .map(|&raw| {
                if raw < 0 || ((raw / experts_per_ep_rank) % num_tp_ranks) != tp_rank {
                    return -1;
                }
                let mut value = raw - tp_rank * experts_per_ep_rank;
                let dp_rank = value / experts_per_moe_dp_group;
                value -= dp_rank * (experts_per_moe_dp_group - experts_per_ep_rank);
                if value < 0 { -1 } else { value }
            })
            .collect()
    }

    #[test]
    fn dsv4_mask_indices_by_ep_i64_matches_tilekernels_formula() {
        let ctx = DeviceContext::new().expect("CUDA context");
        let host = vec![-1, 0, 1, 3, 4, 7, 8, 12, 15, 16, 19, 23, 24, 31];
        let input = ctx.stream.clone_htod(&host).expect("H2D input");
        let mut output = ctx
            .stream
            .alloc_zeros::<i64>(host.len())
            .expect("alloc output");
        dsv4_mask_indices_by_ep_i64(&ctx, &input, &mut output, 2, 7, 4, 8, 2, 1)
            .expect("mask_indices_by_ep");
        ctx.sync().expect("sync");
        let got = ctx.stream.clone_dtoh(&output).expect("D2H output");
        let expected = reference(&host, 4, 8, 2, 1);
        assert_eq!(got, expected);
    }
}
