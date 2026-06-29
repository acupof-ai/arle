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

#[cfg(test)]
mod tests {
    use super::*;
    use qwen35_spec::LayerType;

    fn qwen35_config() -> qwen35_spec::Qwen35Config {
        qwen35_spec::Qwen35Config::from_json_str(
            r#"{
                "text_config": {
                    "hidden_size": 512,
                    "intermediate_size": 1024,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "vocab_size": 1000,
                    "rms_norm_eps": 1e-6,
                    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 128,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 32,
                    "linear_value_head_dim": 128,
                    "rope_parameters": {
                        "rope_theta": 1000000.0,
                        "partial_rotary_factor": 1.0
                    },
                    "eos_token_id": 248044,
                    "bos_token_id": 248000,
                    "tie_word_embeddings": false,
                    "max_position_embeddings": 32768,
                    "num_experts": 8,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 256,
                    "shared_expert_intermediate_size": 512,
                    "decoder_sparse_step": 1
                }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn qwen35_three_linear_one_full_pattern_is_pinned() {
        let layers = [
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
        ];
        let counts = hybrid_counts(&layers);
        assert_eq!(counts.linear_layers, 4);
        assert_eq!(counts.full_layers, 1);

        let ops = qwen35_forward_ops(&layers);
        assert_eq!(ops.first(), Some(&Qwen35Op::TokenEmbedding));
        assert_eq!(ops.last(), Some(&Qwen35Op::LmHead));
        assert!(ops.contains(&Qwen35Op::DepthwiseConv4));
        assert!(ops.contains(&Qwen35Op::GatedDeltaRecurrent));
        assert!(ops.contains(&Qwen35Op::FullAttentionHd256));
        assert!(ops.contains(&Qwen35Op::FullAttentionGate));
        assert!(ops.contains(&Qwen35Op::SwiGlu));
    }

    #[test]
    fn qwen35_recurrent_buffers_are_explicit() {
        assert_eq!(
            QWEN35_MUTATED_RECURRENT_BUFFERS,
            &[
                "slot.linear.gated_delta_state_f32",
                "slot.linear.conv4_ring_bf16",
                "slot.full.k_cache_bf16",
                "slot.full.v_cache_bf16"
            ]
        );
    }

    #[test]
    fn qwen35_launcher_sequence_covers_linear_and_full_attention() {
        let layers = [
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ];
        let kinds = qwen35_launcher_sequence(&layers);
        assert_eq!(kinds.first(), Some(&Qwen35LauncherKind::TokenEmbedding));
        assert!(kinds.contains(&Qwen35LauncherKind::DepthwiseConv4));
        assert!(kinds.contains(&Qwen35LauncherKind::GatedDeltaRecurrent));
        assert!(kinds.contains(&Qwen35LauncherKind::FullAttentionHd256));
        assert!(kinds.contains(&Qwen35LauncherKind::Rope));
        assert!(kinds.contains(&Qwen35LauncherKind::SwiGlu));
        assert_eq!(kinds.len(), qwen35_forward_ops(&layers).len());
    }

    #[test]
    fn qwen35_linear_state_shape_uses_conv_kernel_minus_one() {
        let cfg = qwen35_config();
        let shape = linear_state_shape(&cfg);
        assert_eq!(shape.recurrent_f32, 32 * 128 * 128);
        assert_eq!(shape.conv_bf16, (2 * 16 * 128 + 32 * 128) * 3);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn qwen35_fused_launcher_classes_map_to_kernels() {
        assert_eq!(
            qwen35_kernel_for_launcher(Qwen35LauncherKind::DepthwiseConv4),
            vulkan_kernels::Kernel::Qwen35SsmConv
        );
        assert_eq!(
            qwen35_kernel_for_launcher(Qwen35LauncherKind::GatedDeltaRecurrent),
            vulkan_kernels::Kernel::Qwen35GatedDeltaNet
        );
        assert_eq!(
            qwen35_kernel_for_launcher(Qwen35LauncherKind::FullAttentionHd256),
            vulkan_kernels::Kernel::FlashAttn
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn qwen35_full_attention_spec_uses_config_head_dim() {
        let cfg = qwen35_config();
        let spec = qwen35_full_attention_spec(&cfg);
        assert_eq!(spec.specialization_u32().len(), 16);
        assert_eq!(spec.specialization_u32()[3], (3, 256));
        assert_eq!(spec.specialization_u32()[4], (4, 256));
        assert_eq!(spec.specialization_u32()[12], (12, 1));
        assert_eq!(spec.specialization_u32()[14], (14, 2));
    }

    /// On-box numeric forward over the REAL dense 27B GGUF. `#[ignore]` by
    /// default; run with
    /// `cargo test -p infer-vulkan --features vulkan --lib -- --ignored --nocapture`.
    /// Skips cleanly if the model file or a Vulkan device is absent.
    ///
    /// Asserts the logits are FINITE and length == vocab, prints the argmax
    /// token id and the top-5 logits, then greedy-decodes ~10 tokens.
    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "needs the on-box Qwen3.6-27B GGUF + a Vulkan device"]
    fn qwen35_27b_numeric_forward_is_finite() {
        let path = std::path::Path::new(r"C:\Users\Asus\models\qwen3.6\Qwen3.6-27B-Q8_0.gguf");
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            return;
        }
        let ctx: &'static vulkan_sys::VulkanContext = match vulkan_sys::VulkanContext::create() {
            Ok(c) => Box::leak(Box::new(c)),
            Err(e) => {
                eprintln!("no Vulkan device available ({e}); skipping 27B forward test");
                return;
            }
        };
        eprintln!("ARLE Vulkan 27B forward on: {}", ctx.device_name());
        for (i, (size, dev_local)) in ctx.memory_heaps().iter().enumerate() {
            eprintln!("  heap {i}: {} MB, device_local={dev_local}", size >> 20);
        }

        let gguf = infer_gguf::gguf::GgufFile::open(path).expect("open 27B GGUF");
        let mut model = VulkanQwen35Model::load(ctx, &gguf).expect("load 27B onto device");
        eprintln!(
            "loaded 27B: {} resident tensors, {} device bytes, vocab {}",
            model.resident_tensor_count(),
            model.resident_device_bytes(),
            model.config.vocab_size
        );

        // A handful of arbitrary in-range token ids (no tokenizer here): a small
        // prompt-shaped prefix. The forward runs the uncached full-prefix path,
        // so feed positions 0..n in order.
        let prompt: [u32; 5] = [785, 6722, 315, 9625, 374]; // arbitrary, < vocab
        let vocab = model.config.vocab_size;

        let mut last_logits: Vec<f32> = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            let tok = tok % vocab as u32;
            last_logits = model
                .forward_token(0, 0, tok, pos)
                .unwrap_or_else(|e| panic!("forward_token(pos={pos}) failed: {e}"));
            assert_eq!(last_logits.len(), vocab, "logits length must equal vocab");
            assert!(
                last_logits.iter().all(|v| v.is_finite()),
                "logits at pos {pos} contain NaN/Inf"
            );
        }

        // Top-5 of the final prompt token's logits.
        let argmax = argmax_of(&last_logits);
        let mut idx: Vec<usize> = (0..last_logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| {
            last_logits[b]
                .partial_cmp(&last_logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        eprintln!(
            "argmax token id = {argmax} (logit {:+.4})",
            last_logits[argmax]
        );
        eprintln!("top-5 logits:");
        for &i in idx.iter().take(5) {
            eprintln!("  token {i:>6} -> {:+.4}", last_logits[i]);
        }

        // Greedy-decode ~10 tokens continuing from the prompt.
        let mut decoded: Vec<u32> = Vec::new();
        let mut next = argmax as u32;
        let mut pos = prompt.len();
        for _ in 0..10 {
            decoded.push(next);
            let logits = model
                .forward_token(0, 0, next, pos)
                .unwrap_or_else(|e| panic!("greedy forward_token(pos={pos}) failed: {e}"));
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "greedy logits at pos {pos} contain NaN/Inf"
            );
            next = argmax_of(&logits) as u32;
            pos += 1;
        }
        eprintln!("greedy-decoded 10 token ids: {decoded:?}");
    }

    /// The TRUE "跑通" proof: load the real 27B, build the tokenizer from the
    /// GGUF's embedded vocab, greedy-generate a continuation for a real English
    /// and a real Chinese prompt, and PRINT the decoded text. Coherent output
    /// proves the forward math (rope / attention / gated-delta / norms / GEMV)
    /// is numerically correct; finite logits alone do not.
    ///
    /// Run with:
    /// `cargo test -p infer-vulkan --features vulkan --lib qwen35_27b_generates_coherent_text -- --ignored --nocapture`
    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "needs the on-box Qwen3.6-27B GGUF + a Vulkan device"]
    fn qwen35_27b_generates_coherent_text() {
        use infer_gguf::tokenizer::GgufTokenizer;

        let path = std::path::Path::new(r"C:\Users\Asus\models\qwen3.6\Qwen3.6-27B-Q8_0.gguf");
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            return;
        }
        let ctx: &'static vulkan_sys::VulkanContext = match vulkan_sys::VulkanContext::create() {
            Ok(c) => Box::leak(Box::new(c)),
            Err(e) => {
                eprintln!("no Vulkan device available ({e}); skipping coherence test");
                return;
            }
        };
        eprintln!("ARLE Vulkan 27B coherence on: {}", ctx.device_name());

        let gguf = infer_gguf::gguf::GgufFile::open(path).expect("open 27B GGUF");
        let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("build tokenizer from GGUF");
        let eos = tokenizer.eos_token_id;
        let mut model = VulkanQwen35Model::load(ctx, &gguf).expect("load 27B onto device");
        let vocab = model.config.vocab_size;
        eprintln!(
            "loaded 27B: vocab {}, tokenizer vocab {}, eos {:?}",
            vocab,
            tokenizer.vocab_size(),
            eos
        );

        // Greedy-generate `max_new` tokens continuing `prompt`, returning the
        // generated ids (excludes the prompt). Resets recurrent + KV state first.
        let generate = |model: &mut VulkanQwen35Model, prompt: &str, max_new: usize| -> Vec<u32> {
            model.reset_state();
            let prompt_ids = tokenizer.encode(prompt, false).expect("encode prompt");
            assert!(!prompt_ids.is_empty(), "prompt encoded to no tokens");
            eprintln!(
                "\nprompt {prompt:?} -> {} tokens {prompt_ids:?}",
                prompt_ids.len()
            );

            // Prefill: feed prompt tokens in order; keep the final token's logits.
            let mut last_logits = Vec::new();
            for (pos, &tok) in prompt_ids.iter().enumerate() {
                last_logits = model
                    .forward_token(0, 0, tok, pos)
                    .unwrap_or_else(|e| panic!("prefill forward_token(pos={pos}) failed: {e}"));
                assert_eq!(last_logits.len(), vocab, "logits length must equal vocab");
                assert!(
                    last_logits.iter().all(|v| v.is_finite()),
                    "prefill logits at pos {pos} contain NaN/Inf"
                );
            }

            let mut generated = Vec::new();
            let mut pos = prompt_ids.len();
            // Time the decode loop (the perf-parity payoff): count generated
            // tokens and the wall time spent in `forward_token`, then report
            // s/token and tok/s vs the llama.cpp 27B bar (7.2 tok/s).
            let decode_start = std::time::Instant::now();
            let submits_before = model.decode_submit_count();
            let mut decode_tokens = 0usize;
            for _ in 0..max_new {
                let next = argmax_of(&last_logits) as u32;
                if Some(next) == eos {
                    eprintln!("  hit eos at gen step {}", generated.len());
                    break;
                }
                generated.push(next);
                last_logits = model
                    .forward_token(0, 0, next, pos)
                    .unwrap_or_else(|e| panic!("gen forward_token(pos={pos}) failed: {e}"));
                assert!(
                    last_logits.iter().all(|v| v.is_finite()),
                    "gen logits at pos {pos} contain NaN/Inf"
                );
                decode_tokens += 1;
                pos += 1;
            }
            let elapsed = decode_start.elapsed().as_secs_f64();
            if decode_tokens > 0 {
                let spt = elapsed / decode_tokens as f64;
                let tps = decode_tokens as f64 / elapsed;
                eprintln!(
                    "  DECODE PERF: {decode_tokens} tokens in {elapsed:.2}s = \
                     {spt:.3} s/token = {tps:.3} tok/s (llama.cpp 27B = 7.2 tok/s / 0.139 s/token)"
                );
                let (submit_s, other_s, gemv_n) = model.take_decode_profile();
                let submits = model.decode_submit_count() - submits_before;
                eprintln!(
                    "  GEMV BREAKDOWN: {gemv_n} GEMVs, {submit_s:.2}s in submit_and_wait + \
                     {other_s:.2}s host prep/readback ({:.2}s/token submit, {:.2}s/token other; \
                     rest = host elementwise/norm)",
                    submit_s / decode_tokens as f64,
                    other_s / decode_tokens as f64,
                );
                eprintln!(
                    "  SUBMITS: {submits} vkQueueSubmit over {decode_tokens} tokens = {:.1} submits/token",
                    submits as f64 / decode_tokens as f64,
                );
            }
            generated
        };

        // ~24 new tokens each (≈ 2 min/prompt at ~5s/token + load). Reduce if
        // the box is slow; 16 still demonstrates coherence.
        let prompts = ["The capital of France is", "请用一句话介绍你自己。"];
        let mut all_ok = true;
        for prompt in prompts {
            let gen_ids = generate(&mut model, prompt, 24);
            let continuation = tokenizer
                .decode(&gen_ids, true)
                .expect("decode continuation");
            eprintln!("=== PROMPT: {prompt}");
            eprintln!("=== CONTINUATION: {continuation}");
            eprintln!("=== GEN IDS: {gen_ids:?}");

            assert!(!gen_ids.is_empty(), "no tokens generated for {prompt:?}");
            // Soft degeneracy check: not the same token repeated for the whole run.
            let degenerate = gen_ids.windows(2).all(|w| w[0] == w[1]);
            if degenerate {
                eprintln!("!!! DEGENERATE: single token repeated for {prompt:?}");
                all_ok = false;
            }
        }
        assert!(
            all_ok,
            "at least one prompt produced degenerate (single-token-repeat) output — \
             inspect the printed continuations for the numeric symptom"
        );
    }

    #[cfg(feature = "vulkan")]
    fn argmax_of(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}
