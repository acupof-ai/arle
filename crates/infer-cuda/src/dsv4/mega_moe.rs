use super::*;

#[cfg(all(feature = "cuda", feature = "nccl"))]
pub(crate) struct Dsv4MegaMoeTransport {
    pub(crate) workspace: crate::tp::SymmetricIpcBuffer,
    pub(crate) owned_out: CudaSlice<half::bf16>,
    pub(crate) layout: cuda_kernels::moe::Sm90MegaMoeWorkspaceLayout,
    pub(crate) shape: cuda_kernels::moe::Sm90MegaMoeShape,
    pub(crate) fast_math: bool,
    pub(crate) enable_pdl: bool,
    epoch: std::sync::Mutex<(u64, Option<(u64, usize)>)>,
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
impl Dsv4MegaMoeTransport {
    pub(super) fn assert_collective_values(&self, model: &Dsv4Model, values: &[u64]) -> Result<()> {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let gathered = model.tp.all_gather_bytes(&model.ctx, &bytes, bytes.len())?;
        ensure!(
            gathered.chunks_exact(bytes.len()).all(|peer| peer == bytes),
            "DSv4 MegaMoE specialization differs across TP ranks"
        );
        Ok(())
    }

    pub(super) fn assert_static_spec(
        &self,
        model: &Dsv4Model,
        activation_clamp: f32,
    ) -> Result<()> {
        fn env_i32(name: &str, fallback: i32) -> i32 {
            std::env::var(name)
                .ok()
                .filter(|value| !value.is_empty())
                .and_then(|value| value.parse().ok())
                .unwrap_or(fallback)
        }

        let values = [
            u64::try_from(self.shape.num_ranks)?,
            u64::try_from(self.shape.num_experts)?,
            u64::try_from(self.shape.requested_max_tokens_per_rank)?,
            u64::try_from(self.shape.num_topk)?,
            u64::try_from(self.shape.hidden)?,
            u64::try_from(self.shape.intermediate_hidden)?,
            u64::from(activation_clamp.to_bits()),
            u64::from(self.fast_math),
            u64::from(self.enable_pdl),
            env_i32("DG_NUM_SMS", i32::try_from(model.ctx.sm_count())?) as u32 as u64,
            u64::from(env_i32("DG_SM90_FP8_SWAP_AB", 1) != 0),
        ];
        self.assert_collective_values(model, &values)
    }

    pub(super) fn begin_forward(&self, model: &Dsv4Model, num_tokens: usize) -> Result<u64> {
        self.assert_collective_values(model, &[u64::try_from(num_tokens)?])?;
        let mut epoch = self
            .epoch
            .lock()
            .map_err(|_| anyhow!("DSv4 MegaMoE forward epoch lock poisoned"))?;
        epoch.0 = epoch.0.wrapping_add(1);
        let id = epoch.0;
        epoch.1 = Some((id, num_tokens));
        Ok(id)
    }

    pub(crate) fn assert_forward_epoch(&self, epoch_id: u64, num_tokens: usize) -> Result<()> {
        let epoch = self
            .epoch
            .lock()
            .map_err(|_| anyhow!("DSv4 MegaMoE forward epoch lock poisoned"))?;
        ensure!(
            epoch.1 == Some((epoch_id, num_tokens)),
            "DSv4 MegaMoE layer has no matching top-level forward epoch"
        );
        Ok(())
    }
}

impl Dsv4Model {
    pub(crate) fn begin_mega_moe_forward(&self, _num_tokens: usize) -> Result<Option<u64>> {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(mega_moe) = &self.mega_moe {
            return mega_moe.begin_forward(self, _num_tokens).map(Some);
        }
        Ok(None)
    }

    /// Per-forward owned-token cap when the deepep_ll MoE transport is booted
    /// (`None` ⇒ unbounded). Core caps decode rows + prefill chunk tokens to this
    /// so the LL dispatch buffer is never overrun.
    pub(crate) fn max_tokens_per_step(&self) -> Option<usize> {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(mega_moe) = &self.mega_moe {
            return mega_moe
                .shape
                .requested_max_tokens_per_rank
                .checked_mul(mega_moe.shape.num_ranks);
        }
        #[cfg(feature = "deepep")]
        {
            self.deepep
                .as_ref()
                .and_then(|t| t.max_owned_tokens_per_forward())
        }
        #[cfg(not(feature = "deepep"))]
        {
            None
        }
    }

    pub(crate) fn boot_mega_moe(&mut self, _requested_max_tokens_per_rank: usize) -> Result<()> {
        if !matches!(
            crate::runtime_flags::dsv4_moe_transport()?,
            crate::runtime_flags::Dsv4MoeTransport::MegaMoe
        ) {
            return Ok(());
        }
        #[cfg(not(all(feature = "cuda", feature = "nccl")))]
        anyhow::bail!("ARLE_DSV4_MOE_TRANSPORT=mega_moe requires infer-cuda features cuda,nccl");
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        {
            ensure!(
                self.mega_moe.is_none(),
                "DSv4 MegaMoE transport already booted"
            );
            ensure!(
                self.tp.config().world_size > 1,
                "ARLE_DSV4_MOE_TRANSPORT=mega_moe requires TP world_size > 1"
            );
            let first = self
                .layers
                .iter()
                .find_map(|layer| layer.moe.as_ref())
                .ok_or_else(|| anyhow!("DSv4 MegaMoE requires at least one routed MoE layer"))?;
            ensure!(
                self.layers
                    .iter()
                    .filter_map(|layer| layer.moe.as_ref())
                    .all(|moe| {
                        moe.hidden_dim == first.hidden_dim
                            && moe.intermediate == first.intermediate
                            && moe.num_groups == first.num_groups
                    }),
                "DSv4 MegaMoE requires one routed-expert shape across all layers"
            );
            if let Some(mtp) = &self.mtp {
                let moe = mtp
                    .layer
                    .moe
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 MegaMoE MTP layer has no routed experts"))?;
                ensure!(
                    moe.hidden_dim == first.hidden_dim
                        && moe.intermediate == first.intermediate
                        && moe.num_groups == first.num_groups,
                    "DSv4 MegaMoE MTP routed-expert shape differs from base layers"
                );
            }
            let shape = cuda_kernels::moe::Sm90MegaMoeShape {
                num_ranks: self.tp.config().world_size,
                num_experts: self.moe_config.num_experts,
                requested_max_tokens_per_rank: _requested_max_tokens_per_rank,
                num_topk: self.moe_config.top_k,
                hidden: first.hidden_dim,
                intermediate_hidden: first.intermediate,
            };
            let layout = cuda_kernels::moe::sm90_mega_moe_workspace_layout(shape)?;
            let bytes = usize::try_from(layout.num_bytes)?;
            let workspace = self.tp.alloc_symmetric_ipc(&self.ctx, bytes)?;
            ensure!(
                workspace.bytes() == bytes,
                "DSv4 MegaMoE workspace bytes {} != layout {bytes}",
                workspace.bytes()
            );
            let mega_moe = Dsv4MegaMoeTransport {
                workspace,
                owned_out: self
                    .ctx
                    .stream
                    .alloc_zeros::<half::bf16>(
                        _requested_max_tokens_per_rank
                            .checked_mul(first.hidden_dim)
                            .ok_or_else(|| anyhow!("DSv4 MegaMoE output scratch overflow"))?,
                    )
                    .map_err(|error| {
                        anyhow!("DSv4 MegaMoE output scratch alloc failed: {error}")
                    })?,
                layout,
                shape,
                fast_math: true,
                enable_pdl: false,
                epoch: std::sync::Mutex::new((0, None)),
            };
            mega_moe.assert_static_spec(self, self.config.swiglu_limit)?;
            self.mega_moe = Some(mega_moe);
            Ok(())
        }
    }
}
