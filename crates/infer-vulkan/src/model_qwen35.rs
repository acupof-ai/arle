/// A loaded Qwen3.5 hybrid model with all weights resident on the Vulkan device.
///
/// `ResidentWeights<'a>`'s `DeviceBuffer`s borrow the [`vulkan_sys::VulkanContext`]
/// they were allocated on, so the context must outlive every forward call. The
/// model lives until process exit, so we store a **leaked** `&'static` context
/// (`Box::leak`) and pin the resident weights to that `'static` lifetime — the
/// context (and its GPU allocations) are released only when the process ends.
#[cfg(feature = "vulkan")]
pub struct VulkanQwen35Model {
    pub config: qwen35_spec::Qwen35Config,
    ctx: &'static vulkan_sys::VulkanContext,
    weights: crate::loader::upload::ResidentWeights<'static>,
    state: crate::forward::Qwen35ForwardState,
    /// Persistent decode resources (perf-parity Steps 3+4): the GEMV activation
    /// arena, the compile-once `KernelCache`, and the record-many/submit-once
    /// `CommandRecorder`. Built once in [`Self::load`] and threaded into every
    /// `forward_token` so the hot path never re-allocs scratch, rebuilds a
    /// pipeline, or drains the queue per op.
    decode: crate::forward::DecodeResources<'static>,
}

#[cfg(feature = "vulkan")]
impl VulkanQwen35Model {
    /// Load every Qwen3.5 weight resident onto `ctx`'s device.
    ///
    /// Maps the GGUF metadata to a [`qwen35_spec::Qwen35Config`], plans the
    /// per-tensor residency for all `num_hidden_layers` blocks, and uploads the
    /// plan (K-quant/Q8_0 packed, F16/BF16→F16, F32→F32, token_embd host-side).
    /// `ctx` must be `'static` (leak it at the call site) so the resident
    /// `DeviceBuffer`s outlive any single forward call.
    pub fn load(
        ctx: &'static vulkan_sys::VulkanContext,
        gguf: &infer_gguf::gguf::GgufFile,
    ) -> anyhow::Result<Self> {
        let config = crate::config::qwen35_config_from_gguf(gguf)?;
        let plan = crate::loader::plan_model(gguf, config.num_hidden_layers)?;
        let weights = crate::loader::upload::upload_plan(ctx, gguf, &plan)?;
        let state = crate::forward::Qwen35ForwardState::new(&config);
        let decode = crate::forward::DecodeResources::new(ctx, &config)?;
        Ok(Self {
            config,
            ctx,
            weights,
            state,
            decode,
        })
    }

    /// Reset the per-slot recurrent + KV state for a fresh generation. Zeros both
    /// the host `Qwen35ForwardState` (the host linear-attention oracle path) and
    /// the device-resident gated-delta + conv state (the on-device path), so a
    /// fresh sequence starts clean regardless of which path is selected.
    pub fn reset_state(&mut self) {
        self.state.reset();
        if let Err(e) = self.decode.reset_linear_state() {
            panic!("reset device linear state: {e}");
        }
    }

    /// Drain the accumulated GEMV timing `(submit_secs, other_secs, gemv_count)`
    /// from the decode resources and reset it — lets a timed decode attribute
    /// time between the GPU submits and the host prep/readback around them.
    pub fn take_decode_profile(&mut self) -> (f64, f64, u64) {
        self.decode.take_profile()
    }

    pub fn decode_submit_count(&self) -> u64 {
        self.decode.submit_count()
    }

    /// Number of device-resident weight tensors (token_embd is host-side and not
    /// counted). Lets a load smoke-test assert the model actually landed.
    pub fn resident_tensor_count(&self) -> usize {
        self.weights.tensors.len()
    }

    pub fn resident_device_bytes(&self) -> u64 {
        self.weights
            .tensors
            .values()
            .map(|t| t.buffer.len() as u64)
            .sum()
    }

    pub fn device_name(&self) -> &str {
        self.ctx.device_name()
    }

    /// Numeric forward for one token at `start_pos`, returning logits `[vocab]`.
    ///
    /// Heavy matmuls run on-device (proven Q8_0 GEMV); the elementwise / norm /
    /// attention / gated-delta math runs on the host in f32. See
    /// [`crate::forward`] for the contract. This single-slot lane runs the
    /// uncached full-prefix path: `start_pos` must equal the materialized
    /// sequence length (advanced here), so feed a sequence's tokens in order
    /// (call [`Self::reset_state`] between sequences). `slot` / `epoch` are
    /// accepted for executor-signature parity but unused by this single-slot
    /// state.
    pub fn forward_token(
        &mut self,
        _slot: usize,
        _epoch: u64,
        token: u32,
        start_pos: usize,
    ) -> anyhow::Result<Vec<f32>> {
        crate::forward::forward_token(
            self.ctx,
            &self.config,
            &self.weights,
            &mut self.decode,
            &mut self.state,
            token,
            start_pos,
        )
    }
}
