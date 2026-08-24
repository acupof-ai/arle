//! Qwen3.6 MoE Vulkan forward contract.
//!
//! Qwen3.6 shares the Qwen3.5 hybrid attention substrate and adds sparse MoE
//! layers. This module pins the routing/expert-mix sequence so the Vulkan lane
//! does not collapse MoE into the dense Qwen3.5 MLP path.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36MoeOp {
    RouterGemv,
    TopK,
    NormalizeTopK,
    RoutedExpertGate,
    RoutedExpertUp,
    SwiGlu,
    RoutedExpertDown,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
    ExpertWeightedSum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36MoeLauncherKind {
    RouterGemv,
    HostTopK,
    QuantizedExpertGemv,
    SwiGlu,
    ExpertWeightedAdd,
}

pub const QWEN36_MOE_OPS: &[Qwen36MoeOp] = &[
    Qwen36MoeOp::RouterGemv,
    Qwen36MoeOp::TopK,
    Qwen36MoeOp::NormalizeTopK,
    Qwen36MoeOp::RoutedExpertGate,
    Qwen36MoeOp::RoutedExpertUp,
    Qwen36MoeOp::SwiGlu,
    Qwen36MoeOp::RoutedExpertDown,
    Qwen36MoeOp::SharedExpertGate,
    Qwen36MoeOp::SharedExpertUp,
    Qwen36MoeOp::SharedExpertDown,
    Qwen36MoeOp::ExpertWeightedSum,
];

#[cfg(feature = "vulkan")]
pub fn qwen36_kernel_for_launcher(kind: Qwen36MoeLauncherKind) -> Option<vulkan_kernels::Kernel> {
    Some(match kind {
        Qwen36MoeLauncherKind::RouterGemv | Qwen36MoeLauncherKind::QuantizedExpertGemv => {
            vulkan_kernels::Kernel::GemvQ4K
        }
        Qwen36MoeLauncherKind::SwiGlu => vulkan_kernels::Kernel::SwiGlu,
        Qwen36MoeLauncherKind::ExpertWeightedAdd => vulkan_kernels::Kernel::Add,
        Qwen36MoeLauncherKind::HostTopK => return None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen36Route {
    pub expert: usize,
    pub weight: f32,
}

/// Router selection for one token, transcribed from llama.cpp's `build_moe_ffn`
/// with `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` (llama-graph.cpp:1226-1328):
///
///   1. `probs = softmax(logits)` over ALL `n_expert` experts.
///   2. top-`top_k` experts by `probs` (the unbiased softmax probabilities).
///   3. each selected expert's weight = its softmax `prob`.
///   4. if `norm_topk_prob`: divide the selected weights by their sum (clamped
///      to F16-min to avoid div-by-zero, matching `ggml_clamp(.., 6.1e-5)`).
///
/// This is softmax-over-ALL-then-topk(-then-renorm) — NOT topk-then-softmax,
/// which would pick the same experts but assign different weights. norm_topk
/// renormalizes the kept subset so the routed mix is a convex combination.
pub fn qwen36_topk_routes(logits: &[f32], top_k: usize, norm_topk_prob: bool) -> Vec<Qwen36Route> {
    if top_k == 0 || logits.is_empty() {
        return Vec::new();
    }
    // 1. softmax over all experts (numerically stabilized).
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    // 2. top-k by softmax prob (ties broken by lower expert id, as ggml's stable
    //    argsort keeps the earlier index for equal keys).
    let mut scored: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    scored.sort_by(|(a_idx, a), (b_idx, b)| b.total_cmp(a).then_with(|| a_idx.cmp(b_idx)));
    scored.truncate(top_k.min(scored.len()));
    // 3 + 4. weights = selected probs, optionally renormalized to sum 1. The
    // clamp floor is F16-min (2^-14), matching llama.cpp's `ggml_clamp`.
    if norm_topk_prob {
        const F16_MIN: f32 = 6.103_516e-5; // 2^-14
        let denom = scored.iter().map(|(_, p)| *p).sum::<f32>().max(F16_MIN);
        for (_, p) in &mut scored {
            *p /= denom;
        }
    }
    scored
        .into_iter()
        .map(|(expert, weight)| Qwen36Route { expert, weight })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen36MoeShape {
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
}

impl Qwen36MoeShape {
    pub fn from_config(config: &qwen35_spec::Qwen35Config) -> Option<Self> {
        (config.num_experts > 0).then_some(Self {
            num_experts: config.num_experts,
            experts_per_token: config.num_experts_per_tok,
            moe_intermediate_size: config.moe_intermediate_size,
            shared_expert_intermediate_size: config.shared_expert_intermediate_size,
        })
    }
}

pub const QWEN36_MUTATED_MOE_BUFFERS: &[&str] = &[
    "scratch.router_logits",
    "scratch.topk_ids",
    "scratch.topk_weights",
    "scratch.expert_gate",
    "scratch.expert_up",
    "scratch.expert_act",
    "scratch.expert_down",
    "scratch.shared_expert",
    "scratch.expert_mix",
];

/// A loaded Qwen3.6 MoE model with all weights resident on the Vulkan device.
///
/// Shares the Qwen3.5 loader/config path: the same GGUF→config mapper (it reads
/// the `qwen35moe` arch + `expert_*` keys) and the same residency plan (routed
/// experts are Q4_K/Q5_K kept packed). Stores a leaked `&'static` context so the
/// resident `DeviceBuffer`s outlive any forward call (see [`super::model_qwen35::VulkanQwen35Model`]).
#[cfg(feature = "vulkan")]
pub struct VulkanQwen36Model {
    pub config: qwen35_spec::Qwen35Config,
    ctx: &'static vulkan_sys::VulkanContext,
    weights: crate::loader::upload::ResidentWeights<'static>,
    /// Per-slot recurrent + KV state (shared with the dense Qwen3.5 forward: the
    /// MoE model has the SAME hybrid attention/linear substrate).
    state: crate::forward::Qwen35ForwardState,
    /// Persistent decode resources (GEMV arena / compile-once KernelCache /
    /// record-once CommandRecorder), built once at load and threaded into every
    /// forward — identical to [`super::model_qwen35::VulkanQwen35Model`].
    decode: crate::forward::DecodeResources<'static>,
}

#[cfg(feature = "vulkan")]
impl VulkanQwen36Model {
    /// Load every Qwen3.6 MoE weight resident onto `ctx`'s device via the shared
    /// Qwen3.5 loader/config path. `ctx` must be `'static` (leak it at the call
    /// site) so the resident `DeviceBuffer`s outlive any forward call.
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
    /// the host `Qwen35ForwardState` and the device-resident gated-delta + conv
    /// state (the on-device linear-attention path), so a fresh sequence starts
    /// clean regardless of which path is selected.
    pub fn reset_state(&mut self) {
        self.state.reset();
        if let Err(e) = self.decode.reset_linear_state() {
            panic!("reset device linear state: {e}");
        }
    }

    pub fn take_decode_profile(&mut self) -> (f64, f64, u64) {
        self.decode.take_profile()
    }

    pub fn decode_submit_count(&self) -> u64 {
        self.decode.submit_count()
    }

    /// Number of device-resident weight tensors (token_embd is host-side).
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
    /// Reuses the SHARED [`crate::forward::forward_token`]: the MoE model's
    /// attention / linear / norm / embed / lm_head are identical to the dense
    /// Qwen3.5 path; only the per-layer FFN differs, and the shared forward
    /// already branches `is_moe_layer` → the routed+shared MoE FFN. `slot` /
    /// `epoch` are accepted for executor-signature parity (single-slot state).
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
    #[cfg(feature = "vulkan")]
    use crate::utils::argmax_of;

    #[test]
    fn moe_ops_include_router_routed_and_shared_experts() {
        assert_eq!(QWEN36_MOE_OPS[0], Qwen36MoeOp::RouterGemv);
        assert!(QWEN36_MOE_OPS.contains(&Qwen36MoeOp::TopK));
        assert!(QWEN36_MOE_OPS.contains(&Qwen36MoeOp::RoutedExpertGate));
        assert!(QWEN36_MOE_OPS.contains(&Qwen36MoeOp::RoutedExpertUp));
        assert!(QWEN36_MOE_OPS.contains(&Qwen36MoeOp::RoutedExpertDown));
        assert!(QWEN36_MOE_OPS.contains(&Qwen36MoeOp::SharedExpertGate));
        assert_eq!(QWEN36_MOE_OPS.last(), Some(&Qwen36MoeOp::ExpertWeightedSum));
    }

    #[test]
    fn mutated_moe_scratch_is_enumerated() {
        assert_eq!(QWEN36_MUTATED_MOE_BUFFERS.len(), 9);
        assert!(QWEN36_MUTATED_MOE_BUFFERS.contains(&"scratch.router_logits"));
        assert!(QWEN36_MUTATED_MOE_BUFFERS.contains(&"scratch.expert_mix"));
    }

    #[test]
    fn topk_routes_are_sorted_and_normalized() {
        let routes = qwen36_topk_routes(&[0.0, 3.0, 1.0, 3.0], 2, true);
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].expert, 1);
        assert_eq!(routes[1].expert, 3);
        let sum: f32 = routes.iter().map(|r| r.weight).sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
        assert!((routes[0].weight - 0.5).abs() < 1.0e-6);
        assert!((routes[1].weight - 0.5).abs() < 1.0e-6);

        // Without renormalization the weights are the raw softmax probabilities
        // over ALL experts (here e^3/(2e^3+e^1+e^0) each), so they sum to < 1.
        let raw = qwen36_topk_routes(&[0.0, 3.0, 1.0, 3.0], 2, false);
        let raw_sum: f32 = raw.iter().map(|r| r.weight).sum();
        assert!(raw_sum < 1.0, "raw softmax top-k weights sum to < 1");
        let denom = (3.0f32).exp() * 2.0 + (1.0f32).exp() + 1.0;
        assert!((raw[0].weight - (3.0f32).exp() / denom).abs() < 1.0e-6);
    }

    /// The MoE "跑通" proof: load the real 35B-A3B (arch `qwen35moe`, Q4_K_M),
    /// build the GGUF tokenizer, greedy-generate continuations for an English
    /// and a Chinese prompt, PRINT the text, and report decode tok/s. Coherent
    /// output proves the routed+shared MoE FFN math (softmax router, top-8
    /// experts, per-expert Q4_K/Q5_K GEMV slices, sigmoid-gated shared expert)
    /// matches llama.cpp; the dense 27B path already proves the shared
    /// attention/linear/norm substrate.
    ///
    /// Run with:
    /// `cargo test -p infer-vulkan --features vulkan --lib qwen36_35b_moe_generates_coherent_text -- --ignored --nocapture`
    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "needs the on-box Qwen3.6-35B-A3B MoE GGUF + a Vulkan device"]
    fn qwen36_35b_moe_generates_coherent_text() {
        use infer_gguf::tokenizer::GgufTokenizer;

        let path =
            std::path::Path::new(r"C:\Users\Asus\models\qwen3.6\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            return;
        }
        let ctx: &'static vulkan_sys::VulkanContext = match vulkan_sys::VulkanContext::create() {
            Ok(c) => Box::leak(Box::new(c)),
            Err(e) => {
                eprintln!("no Vulkan device available ({e}); skipping MoE coherence test");
                return;
            }
        };
        eprintln!(
            "ARLE Vulkan 35B-A3B MoE coherence on: {}",
            ctx.device_name()
        );

        let gguf = infer_gguf::gguf::GgufFile::open(path).expect("open 35B-A3B GGUF");
        let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("build tokenizer from GGUF");
        let eos = tokenizer.eos_token_id;
        let mut model = VulkanQwen36Model::load(ctx, &gguf).expect("load 35B-A3B onto device");
        let vocab = model.config.vocab_size;
        eprintln!(
            "loaded 35B-A3B MoE: {} resident tensors, {} device bytes, vocab {}, \
             experts {} (top-{}), {} layers, eos {:?}",
            model.resident_tensor_count(),
            model.resident_device_bytes(),
            vocab,
            model.config.num_experts,
            model.config.num_experts_per_tok,
            model.config.num_hidden_layers,
            eos,
        );

        let generate = |model: &mut VulkanQwen36Model, prompt: &str, max_new: usize| -> Vec<u32> {
            model.reset_state();
            let prompt_ids = tokenizer.encode(prompt, false).expect("encode prompt");
            assert!(!prompt_ids.is_empty(), "prompt encoded to no tokens");
            eprintln!(
                "\nprompt {prompt:?} -> {} tokens {prompt_ids:?}",
                prompt_ids.len()
            );

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
                    "  DECODE PERF: {decode_tokens} tokens in {elapsed:.2}s = {spt:.3} s/token = \
                     {tps:.3} tok/s (llama.cpp 35B-A3B = 47.3 tok/s; dense 27B ARLE = 3.0 s/token)"
                );
                let (submit_s, other_s, gemv_n) = model.take_decode_profile();
                eprintln!(
                    "  GEMV BREAKDOWN: {gemv_n} GEMVs ({:.1}/token), {submit_s:.2}s submit + \
                     {other_s:.2}s host prep/readback ({:.3}s/token submit, {:.3}s/token other)",
                    gemv_n as f64 / decode_tokens as f64,
                    submit_s / decode_tokens as f64,
                    other_s / decode_tokens as f64,
                );
                let submits = model.decode_submit_count() - submits_before;
                eprintln!(
                    "  SUBMITS: {submits} vkQueueSubmit over {decode_tokens} tokens = {:.1} submits/token",
                    submits as f64 / decode_tokens as f64,
                );
            }
            generated
        };

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
            let degenerate = gen_ids.windows(2).all(|w| w[0] == w[1]);
            if degenerate {
                eprintln!("!!! DEGENERATE: single token repeated for {prompt:?}");
                all_ok = false;
            }
        }
        assert!(
            all_ok,
            "at least one prompt produced degenerate output — inspect the printed continuations"
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn moe_launcher_classes_map_to_kernels() {
        assert_eq!(
            qwen36_kernel_for_launcher(Qwen36MoeLauncherKind::RouterGemv),
            Some(vulkan_kernels::Kernel::GemvQ4K)
        );
        assert_eq!(
            qwen36_kernel_for_launcher(Qwen36MoeLauncherKind::SwiGlu),
            Some(vulkan_kernels::Kernel::SwiGlu)
        );
        assert_eq!(
            qwen36_kernel_for_launcher(Qwen36MoeLauncherKind::ExpertWeightedAdd),
            Some(vulkan_kernels::Kernel::Add)
        );
        assert_eq!(
            qwen36_kernel_for_launcher(Qwen36MoeLauncherKind::HostTopK),
            None
        );
    }
}
