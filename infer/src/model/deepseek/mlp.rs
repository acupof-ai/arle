//! DeepSeek V4 MoE FFN scaffold.
//!
//! The local V4 1B checkpoint uses routed experts plus a shared expert on each
//! layer. Phase 0.5 records the correct tensor shape; Phase 1 supplies the
//! shared CUDA MoE primitive, and Phase 2A wires this block into forward.

#[cfg(feature = "cuda")]
use anyhow::{Result, ensure};
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
use crate::ops;

/// One SwiGLU expert: `w2(silu(w1(x)) * w3(x))`.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4Expert {
    pub(super) w1: DeviceMatrix,
    pub(super) w2: DeviceMatrix,
    pub(super) w3: DeviceMatrix,
}

#[cfg(feature = "cuda")]
impl DeepseekV4Expert {
    /// Run one DeepSeek V4 SwiGLU expert on a packed `[tokens, hidden]` row block.
    pub(super) fn forward(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        ensure!(
            self.w1.cols == hidden.hidden_dim && self.w3.cols == hidden.hidden_dim,
            "DeepSeek V4 expert input width mismatch: hidden_dim={} w1.cols={} w3.cols={}",
            hidden.hidden_dim,
            self.w1.cols,
            self.w3.cols
        );
        ensure!(
            self.w1.rows == self.w3.rows && self.w2.cols == self.w1.rows,
            "DeepSeek V4 expert intermediate mismatch: w1.rows={} w3.rows={} w2.cols={}",
            self.w1.rows,
            self.w3.rows,
            self.w2.cols
        );

        let gate = ops::gemm(ctx, &self.w1, hidden)?;
        let up = ops::gemm(ctx, &self.w3, hidden)?;
        let mut act = HiddenStates::zeros(ctx, self.w1.rows, hidden.seq_len)?;
        ops::dsv4_swiglu_clamped_batch_into(ctx, &gate, &up, &mut act, swiglu_limit)?;
        ops::gemm(ctx, &self.w2, &act)
    }
}

/// V4 routed MoE block plus optional shared expert.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4MoeBlock {
    pub(super) gate_weight: DeviceMatrix,
    pub(super) gate_bias: Option<DeviceVec>,
    /// Hash-router table for early layers. The exact integer storage type is
    /// finalized with the Phase 2A loader; Phase 0.5 only validates the tensor
    /// name and keeps the field explicit.
    pub(super) gate_tid2eid: Option<CudaSlice<i64>>,
    pub(super) experts: Vec<DeepseekV4Expert>,
    pub(super) shared_experts: Option<DeepseekV4Expert>,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)] // method called from forward.rs once MoE kernels land
impl DeepseekV4MoeBlock {
    /// Run routed V4 MoE for a packed `[tokens, hidden]` row block.
    pub(super) fn forward(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        ensure!(
            self.experts.is_empty(),
            "DeepSeek V4 routed MoE combine is not wired yet; local experts loaded={}",
            self.experts.len()
        );
        let shared = self
            .shared_experts
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 MoE block has no shared expert"))?;
        shared.forward(ctx, hidden, swiglu_limit)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use half::bf16;

    fn bf16_vec(values: &[f32]) -> Vec<bf16> {
        values.iter().map(|&value| bf16::from_f32(value)).collect()
    }

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    #[test]
    fn expert_forward_runs_clamped_swiglu_on_gpu() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, -2.0, 0.5, 3.0]))?,
            hidden_dim: 2,
            seq_len: 2,
        };
        let expert = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    1.0, 0.0, //
                    0.0, 1.0, //
                    1.0, 1.0,
                ]),
                3,
                2,
            )?,
            w2: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    1.0, 0.0, 0.5, //
                    0.0, 1.0, -1.0,
                ]),
                2,
                3,
            )?,
            w3: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    0.5, 0.0, //
                    0.0, -1.0, //
                    1.0, -1.0,
                ]),
                3,
                2,
            )?,
        };

        let out = expert.forward(&ctx, &hidden, 2.0)?;
        let out_host = ctx.stream.clone_dtoh(&out.data)?;
        ctx.sync()?;

        let inputs = [[1.0_f32, -2.0_f32], [0.5_f32, 3.0_f32]];
        let mut expected = Vec::new();
        for x in inputs {
            let gate = [x[0], x[1], x[0] + x[1]];
            let up = [0.5 * x[0], -x[1], x[0] - x[1]];
            let act = [
                silu(gate[0].min(2.0)) * up[0].clamp(-2.0, 2.0),
                silu(gate[1].min(2.0)) * up[1].clamp(-2.0, 2.0),
                silu(gate[2].min(2.0)) * up[2].clamp(-2.0, 2.0),
            ];
            expected.push(act[0] + 0.5 * act[2]);
            expected.push(act[1] - act[2]);
        }

        for (idx, got) in out_host.iter().enumerate() {
            assert!(
                (got.to_f32() - expected[idx]).abs() < 0.05,
                "idx={idx} expected={} got={}",
                expected[idx],
                got.to_f32()
            );
        }
        Ok(())
    }
}
