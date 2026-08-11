//! Qwen3.5 hybrid Vulkan forward contract.
//!
//! Authoritative source: `crates/infer-cuda/src/qwen35.rs`. Qwen3.5
//! alternates gated-delta linear attention with periodic full attention
//! (production pattern is 3 linear layers then 1 full-attention layer).
//! P4 pins the order and recurrent state contract; numeric Vulkan kernels for
//! conv4 + recurrent gated delta are not present in the vendored GLSL corpus.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35Op {
    TokenEmbedding,
    InputRmsNorm,
    LinearInProjQkv,
    LinearInProjZ,
    LinearInProjB,
    LinearInProjA,
    DepthwiseConv4,
    GatedDeltaRecurrent,
    GatedOutputRmsNorm,
    LinearOutProj,
    FullQProjWithGate,
    FullKProj,
    FullVProj,
    FullQNorm,
    FullKNorm,
    FullRope,
    FullAttentionHd256,
    FullAttentionGate,
    FullOProj,
    AttentionResidual,
    PostAttentionRmsNorm,
    MlpGateProj,
    MlpUpProj,
    SwiGlu,
    MlpDownProj,
    MlpResidual,
    FinalRmsNorm,
    LmHead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35LauncherKind {
    TokenEmbedding,
    QuantizedGemv,
    RmsNorm,
    Rope,
    DepthwiseConv4,
    GatedDeltaRecurrent,
    FullAttentionHd256,
    SwiGlu,
    Add,
}

pub const QWEN35_LINEAR_LAYER_OPS: &[Qwen35Op] = &[
    Qwen35Op::InputRmsNorm,
    Qwen35Op::LinearInProjQkv,
    Qwen35Op::LinearInProjZ,
    Qwen35Op::LinearInProjB,
    Qwen35Op::LinearInProjA,
    Qwen35Op::DepthwiseConv4,
    Qwen35Op::GatedDeltaRecurrent,
    Qwen35Op::GatedOutputRmsNorm,
    Qwen35Op::LinearOutProj,
    Qwen35Op::AttentionResidual,
    Qwen35Op::PostAttentionRmsNorm,
    Qwen35Op::MlpGateProj,
    Qwen35Op::MlpUpProj,
    Qwen35Op::SwiGlu,
    Qwen35Op::MlpDownProj,
    Qwen35Op::MlpResidual,
];

pub const QWEN35_FULL_LAYER_OPS: &[Qwen35Op] = &[
    Qwen35Op::InputRmsNorm,
    Qwen35Op::FullQProjWithGate,
    Qwen35Op::FullKProj,
    Qwen35Op::FullVProj,
    Qwen35Op::FullQNorm,
    Qwen35Op::FullKNorm,
    Qwen35Op::FullRope,
    Qwen35Op::FullAttentionHd256,
    Qwen35Op::FullAttentionGate,
    Qwen35Op::FullOProj,
    Qwen35Op::AttentionResidual,
    Qwen35Op::PostAttentionRmsNorm,
    Qwen35Op::MlpGateProj,
    Qwen35Op::MlpUpProj,
    Qwen35Op::SwiGlu,
    Qwen35Op::MlpDownProj,
    Qwen35Op::MlpResidual,
];

pub fn qwen35_forward_ops(layer_types: &[qwen35_spec::LayerType]) -> Vec<Qwen35Op> {
    std::iter::once(Qwen35Op::TokenEmbedding)
        .chain(layer_types.iter().flat_map(|lt| match lt {
            qwen35_spec::LayerType::LinearAttention => QWEN35_LINEAR_LAYER_OPS.iter().copied(),
            qwen35_spec::LayerType::FullAttention => QWEN35_FULL_LAYER_OPS.iter().copied(),
        }))
        .chain([Qwen35Op::FinalRmsNorm, Qwen35Op::LmHead])
        .collect()
}

pub fn qwen35_launcher_kind(op: Qwen35Op) -> Qwen35LauncherKind {
    match op {
        Qwen35Op::TokenEmbedding => Qwen35LauncherKind::TokenEmbedding,
        Qwen35Op::LinearInProjQkv
        | Qwen35Op::LinearInProjZ
        | Qwen35Op::LinearInProjB
        | Qwen35Op::LinearInProjA
        | Qwen35Op::LinearOutProj
        | Qwen35Op::FullQProjWithGate
        | Qwen35Op::FullKProj
        | Qwen35Op::FullVProj
        | Qwen35Op::FullOProj
        | Qwen35Op::MlpGateProj
        | Qwen35Op::MlpUpProj
        | Qwen35Op::MlpDownProj
        | Qwen35Op::LmHead => Qwen35LauncherKind::QuantizedGemv,
        Qwen35Op::InputRmsNorm
        | Qwen35Op::GatedOutputRmsNorm
        | Qwen35Op::FullQNorm
        | Qwen35Op::FullKNorm
        | Qwen35Op::PostAttentionRmsNorm
        | Qwen35Op::FinalRmsNorm => Qwen35LauncherKind::RmsNorm,
        Qwen35Op::DepthwiseConv4 => Qwen35LauncherKind::DepthwiseConv4,
        Qwen35Op::GatedDeltaRecurrent => Qwen35LauncherKind::GatedDeltaRecurrent,
        Qwen35Op::FullRope => Qwen35LauncherKind::Rope,
        Qwen35Op::FullAttentionHd256 | Qwen35Op::FullAttentionGate => {
            Qwen35LauncherKind::FullAttentionHd256
        }
        Qwen35Op::SwiGlu => Qwen35LauncherKind::SwiGlu,
        Qwen35Op::AttentionResidual | Qwen35Op::MlpResidual => Qwen35LauncherKind::Add,
    }
}

pub fn qwen35_launcher_sequence(layer_types: &[qwen35_spec::LayerType]) -> Vec<Qwen35LauncherKind> {
    qwen35_forward_ops(layer_types)
        .into_iter()
        .map(qwen35_launcher_kind)
        .collect()
}

#[cfg(feature = "vulkan")]
pub fn qwen35_kernel_for_launcher(kind: Qwen35LauncherKind) -> vulkan_kernels::Kernel {
    match kind {
        Qwen35LauncherKind::TokenEmbedding => vulkan_kernels::Kernel::GetRows,
        Qwen35LauncherKind::QuantizedGemv => vulkan_kernels::Kernel::GemvQ4K,
        Qwen35LauncherKind::RmsNorm => vulkan_kernels::Kernel::RmsNorm,
        Qwen35LauncherKind::Rope => vulkan_kernels::Kernel::RopeNeox,
        Qwen35LauncherKind::DepthwiseConv4 => vulkan_kernels::Kernel::Qwen35SsmConv,
        Qwen35LauncherKind::GatedDeltaRecurrent => vulkan_kernels::Kernel::Qwen35GatedDeltaNet,
        Qwen35LauncherKind::FullAttentionHd256 => vulkan_kernels::Kernel::FlashAttn,
        Qwen35LauncherKind::SwiGlu => vulkan_kernels::Kernel::SwiGlu,
        Qwen35LauncherKind::Add => vulkan_kernels::Kernel::Add,
    }
}

#[cfg(feature = "vulkan")]
pub fn qwen35_full_attention_spec(
    config: &qwen35_spec::Qwen35Config,
) -> vulkan_kernels::FlashAttentionSpec {
    vulkan_kernels::FlashAttentionSpec::f32_f16(config.head_dim as u32)
}

#[cfg(feature = "vulkan")]
pub fn qwen35_forward_kernel_sequence(
    layer_types: &[qwen35_spec::LayerType],
) -> Vec<vulkan_kernels::Kernel> {
    qwen35_launcher_sequence(layer_types)
        .into_iter()
        .map(qwen35_kernel_for_launcher)
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35HybridCounts {
    pub linear_layers: usize,
    pub full_layers: usize,
}

pub fn hybrid_counts(layer_types: &[qwen35_spec::LayerType]) -> Qwen35HybridCounts {
    let full_layers = layer_types
        .iter()
        .filter(|&&layer| layer == qwen35_spec::LayerType::FullAttention)
        .count();
    Qwen35HybridCounts {
        linear_layers: layer_types.len() - full_layers,
        full_layers,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35LinearStateShape {
    pub recurrent_f32: usize,
    pub conv_bf16: usize,
}

pub fn linear_state_shape(config: &qwen35_spec::Qwen35Config) -> Qwen35LinearStateShape {
    Qwen35LinearStateShape {
        recurrent_f32: config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim,
        conv_bf16: (2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim)
            * config.linear_conv_kernel_dim.saturating_sub(1),
    }
}

pub const QWEN35_MUTATED_RECURRENT_BUFFERS: &[&str] = &[
    "slot.linear.gated_delta_state_f32",
    "slot.linear.conv4_ring_bf16",
    "slot.full.k_cache_bf16",
    "slot.full.v_cache_bf16",
];

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
    /// Per-slot recurrent + KV state for the (single-slot) numeric forward.
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

    /// Total `vkQueueSubmit` calls issued by the decode recorder so far — lets a
    /// timed decode report submits/token (perf-parity Step 4).
    pub fn decode_submit_count(&self) -> u64 {
        self.decode.submit_count()
    }

    /// Number of device-resident weight tensors (token_embd is host-side and not
    /// counted). Lets a load smoke-test assert the model actually landed.
    pub fn resident_tensor_count(&self) -> usize {
        self.weights.tensors.len()
    }

    /// Total bytes of device-resident weights across all tensors.
    pub fn resident_device_bytes(&self) -> u64 {
        self.weights
            .tensors
            .values()
            .map(|t| t.buffer.len() as u64)
            .sum()
    }

    /// The Vulkan device this model is resident on.
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
