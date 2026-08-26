//! Vulkan `BackendExecutor` skeleton.
//!
//! P2 keeps submit/poll behavior identical to the HIP backend: one row per
//! plan,
//! synchronous completion, and host sampling once numeric logits exist. Until
//! a model is loaded, every non-idle plan errors loud.

use anyhow::{Result, bail, ensure};
use infer_plan::{ForwardPlan, SamplingParams, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvPool, PollResult};

use crate::kv_pool::VulkanKvPool;

pub const DEFAULT_PAGE_SIZE: usize = 64;

/// `ARLE_VULKAN_BATCHED_PREFILL=0` forces the per-token prefill loop.
///
/// The batched path is a different arithmetic shape (`mul_mmq` over a chunk vs a
/// chain of GEMVs), so it is not bit-identical to the serial one — the parity
/// gate compares them, and needs a way to run the old path in the same binary.
#[cfg(feature = "vulkan")]
fn batched_prefill_enabled() -> bool {
    !matches!(
        std::env::var("ARLE_VULKAN_BATCHED_PREFILL").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanModelKind {
    Qwen3Dense,
    Dsv4,
    Qwen35Hybrid,
    Qwen36Moe,
    /// Qwen3.8-Flash-Next (`qwen4_exp`). Recognised so it fails by NAME rather
    /// than being mistaken for a plain MoE — it has 512 experts (top-10), a
    /// hyper-connection residual stream 4x the hidden width, a PLE/n-gram
    /// lookup table, and a sparse-attention indexer, none of which the
    /// `Qwen36Moe` path implements.
    ///
    /// The misroute this prevents is silent, not loud: `expert_count > 0` used
    /// to catch it, and the resulting forward would have run
    /// `qwen36_router_topk.comp` with `n_expert = 512` against its `BLOCK 256`
    /// — routing every token through only the first half of the expert table,
    /// with the top-k renormalisation hiding the wrong softmax denominator.
    /// Coherent output, no crash, silently wrong model.
    Qwen4Exp,
}

pub fn classify_vulkan_architecture(
    architecture: &str,
    model_name: Option<&str>,
    expert_count: usize,
) -> VulkanModelKind {
    let arch = architecture.to_ascii_lowercase();
    let name = model_name.unwrap_or_default().to_ascii_lowercase();
    if arch.contains("deepseek4") || arch.contains("deepseek_v4") {
        return VulkanModelKind::Dsv4;
    }
    // `qwen4_exp` BEFORE the expert-count clause below: it reports 512 experts
    // and would otherwise be classified as an ordinary MoE.
    if arch.contains("qwen4_exp") || arch.contains("qwen4exp") {
        return VulkanModelKind::Qwen4Exp;
    }
    // MoE first: `qwen35moe` / `qwen3moe` archs and anything with experts. Note
    // `qwen35moe.contains("qwen35")` is true, so this MUST precede the dense
    // `qwen35` hybrid check below.
    if arch.contains("qwen35moe")
        || arch.contains("qwen3moe")
        || arch.contains("qwen3_moe")
        || expert_count > 0
    {
        return VulkanModelKind::Qwen36Moe;
    }
    // Dense Qwen3.5/3.6 hybrid: the on-box 27B reports `general.architecture =
    // "qwen35"` (the model name is unset/unreliable), so match the arch string
    // as well as any "qwen3.5"/"qwen35" in the name.
    if arch.contains("qwen35") || name.contains("qwen3.5") || name.contains("qwen35") {
        return VulkanModelKind::Qwen35Hybrid;
    }
    VulkanModelKind::Qwen3Dense
}

pub fn classify_vulkan_gguf(gguf: &infer_gguf::gguf::GgufFile) -> Result<VulkanModelKind> {
    let architecture = gguf
        .get_str("general.architecture")
        .unwrap_or("qwen3")
        .to_string();
    let model_name = gguf.get_str("general.name");
    let expert_count = gguf
        .get_usize(&format!("{architecture}.expert_count"))
        .or_else(|| gguf.get_usize(&format!("{architecture}.num_experts")))
        .unwrap_or(0);
    Ok(classify_vulkan_architecture(
        &architecture,
        model_name,
        expert_count,
    ))
}

#[cfg(feature = "vulkan")]
pub enum VulkanLoadedModel {
    Qwen3(Box<crate::model_qwen3::VulkanQwen3Model>),
    Qwen35(Box<crate::model_qwen35::VulkanQwen35Model>),
    Qwen36(Box<crate::model_qwen36::VulkanQwen36Model>),
}

#[cfg(feature = "vulkan")]
impl VulkanLoadedModel {
    fn forward_token(
        &mut self,
        slot: usize,
        epoch: u64,
        token: u32,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        match self {
            Self::Qwen3(model) => model.forward_token(slot, epoch, token, start_pos),
            Self::Qwen35(model) => model.forward_token(slot, epoch, token, start_pos),
            Self::Qwen36(model) => model.forward_token(slot, epoch, token, start_pos),
        }
    }

    /// Materialize `tokens` in one GEMM-shaped batched pass, returning the LAST
    /// token's logits — or `None` when this model has no batched path, in which
    /// case the caller falls back to the per-token loop.
    ///
    /// Only the Qwen3.5 hybrid has one so far. Prefill is where the per-token
    /// loop hurts most: every layer's weights are re-read from LPDDR5X once per
    /// TOKEN, so a GEMV chain runs at memory bandwidth no matter how many tokens
    /// are queued. Batching turns each projection into a `mul_mmq` over the whole
    /// chunk, amortizing the weight read across `T` rows.
    fn forward_tokens_batched(
        &mut self,
        slot: usize,
        epoch: u64,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Option<Vec<f32>>> {
        match self {
            Self::Qwen35(model) => model.forward_tokens(slot, epoch, tokens, start_pos),
            _ => Ok(None),
        }
    }

    /// Leading prefix of `tokens` this model already holds materialized. Only
    /// the Qwen3.5 hybrid tracks its resident sequence; the rest recompute.
    fn cached_prefix_len(&self, tokens: &[u32]) -> usize {
        match self {
            Self::Qwen35(model) => model.cached_prefix_len(tokens),
            _ => 0,
        }
    }

    fn adopt_cached_prefix(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
    ) -> Result<()> {
        match self {
            Self::Qwen35(model) => model.adopt_cached_prefix(slot, tokens, matched_len),
            _ => bail!("this Vulkan model has no position-0 prefix store"),
        }
    }

    fn materialize_finish(&mut self, slot: usize, tokens: &[u32]) -> Result<()> {
        match self {
            Self::Qwen35(model) => model.materialize_finish(slot, tokens),
            _ => Ok(()),
        }
    }
}

#[derive(Debug)]
pub enum VulkanInflight {
    Ready(StepOutput),
}

#[derive(Default)]
pub struct VulkanExecutor {
    #[cfg(feature = "vulkan")]
    model: Option<VulkanLoadedModel>,
    stop_tokens: Vec<u32>,
}

impl VulkanExecutor {
    #[must_use]
    pub fn unloaded() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn has_model(&self) -> bool {
        #[cfg(feature = "vulkan")]
        {
            self.model.is_some()
        }
        #[cfg(not(feature = "vulkan"))]
        {
            false
        }
    }

    /// `(resident_tensor_count, resident_device_bytes)` for the loaded model, or
    /// `None` if no model is loaded. Lets a load smoke-test assert the weights
    /// actually landed on the device.
    #[cfg(feature = "vulkan")]
    #[must_use]
    pub fn resident_stats(&self) -> Option<(usize, u64)> {
        let model = self.model.as_ref()?;
        Some(match model {
            VulkanLoadedModel::Qwen35(m) => (m.resident_tensor_count(), m.resident_device_bytes()),
            VulkanLoadedModel::Qwen36(m) => (m.resident_tensor_count(), m.resident_device_bytes()),
            _ => return None,
        })
    }

    #[cfg(feature = "vulkan")]
    #[must_use]
    pub fn device_name(&self) -> Option<&str> {
        let model = self.model.as_ref()?;
        Some(match model {
            VulkanLoadedModel::Qwen35(m) => m.device_name(),
            VulkanLoadedModel::Qwen36(m) => m.device_name(),
            _ => return None,
        })
    }

    fn forward_tokens(
        &mut self,
        slot: usize,
        epoch: u64,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        ensure!(
            !tokens.is_empty(),
            "Vulkan forward requires at least one token"
        );
        #[cfg(feature = "vulkan")]
        if let Some(model) = self.model.as_mut() {
            // Multi-token steps take the batched path when the model has one.
            // A 1-token step IS decode, which the per-token path already
            // records optimally, so don't pay the chunk staging for it.
            if tokens.len() > 1
                && batched_prefill_enabled()
                && let Some(logits) =
                    model.forward_tokens_batched(slot, epoch, tokens, start_pos)?
            {
                return Ok(infer_plan::sample_token(&logits, params, position));
            }
            let mut logits = Vec::new();
            let t0 = std::time::Instant::now();
            for (i, &token) in tokens.iter().enumerate() {
                logits = model.forward_token(slot, epoch, token, start_pos + i)?;
            }
            if tokens.len() > 1 {
                let secs = t0.elapsed().as_secs_f64();
                log::info!(
                    "vulkan per-token prefill: {} tok @ {start_pos} in {secs:.3}s ({:.1} tok/s)",
                    tokens.len(),
                    tokens.len() as f64 / secs.max(f64::MIN_POSITIVE),
                );
            }
            return Ok(infer_plan::sample_token(&logits, params, position));
        }
        let _ = (slot, epoch, start_pos, params, position);
        bail!(
            "Vulkan executor has no model loaded{}",
            if cfg!(feature = "vulkan") {
                " (Qwen3 numeric forward is pending shader/residency bring-up)"
            } else {
                " (built without the `vulkan` feature)"
            }
        )
    }
}

impl BackendExecutor for VulkanExecutor {
    type Inflight = VulkanInflight;

    fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<VulkanInflight> {
        if plan.is_idle() {
            return Ok(VulkanInflight::Ready(StepOutput { tokens: Vec::new() }));
        }
        let row_count = plan.prefill_rows.len() + plan.decode_rows.len();
        ensure!(
            row_count == 1,
            "Vulkan executor supports exactly one prefill or decode row, got {row_count}"
        );

        if let Some(row) = plan.prefill_rows.first() {
            let epoch = kv.slot_epoch(row.slot);
            let position = row.end_pos() as u64;
            let token = self.forward_tokens(
                row.slot,
                epoch,
                &row.tokens,
                row.start_pos,
                &row.params,
                position,
            )?;
            return Ok(VulkanInflight::Ready(StepOutput {
                tokens: vec![SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    top_logprobs: Vec::new(),
                    finish: None,
                }],
            }));
        }

        if let Some(row) = plan.decode_rows.first() {
            let epoch = kv.slot_epoch(row.slot);
            let position = (row.kv_seq_len + 1) as u64;
            let token = self.forward_tokens(
                row.slot,
                epoch,
                &[row.last_token],
                row.kv_seq_len,
                &row.params,
                position,
            )?;
            return Ok(VulkanInflight::Ready(StepOutput {
                tokens: vec![SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    top_logprobs: Vec::new(),
                    finish: None,
                }],
            }));
        }
        bail!("Vulkan executor received a non-idle plan with no rows")
    }

    fn poll(&mut self, inflight: VulkanInflight) -> Result<PollResult<VulkanInflight>> {
        match inflight {
            VulkanInflight::Ready(output) => Ok(PollResult::Ready(output)),
        }
    }

    fn model_stop_token_ids(&self) -> Vec<u32> {
        self.stop_tokens.clone()
    }

    fn prefix_reuse(&mut self) -> Option<&mut dyn infer_seam::PrefixReuse> {
        Some(self)
    }
}

/// Prefix reuse for the single-slot Vulkan lane.
///
/// The page-radix route is **fail-closed on purpose**: this lane's device KV is
/// one flat `[layer, kv_head, pos, head_dim]` buffer indexed by ABSOLUTE
/// position ([`crate::forward::DeviceKvCache`]), so a host page id names no
/// device bytes and re-attaching pages at a new position would serve another
/// sequence's KV. `reusable_prefix_blocks` returning 0 states that, and matches
/// what the engine already assumed when this executor reported no
/// `prefix_reuse` capability at all.
///
/// What IS reusable is the sequence the lane is holding right now, at the
/// positions it already occupies — the position-0 seam
/// ([`infer_seam::PrefixReuse::cached_prefix_match_len`]). That covers the case
/// that actually costs users minutes: turn N+1 of a conversation, whose prompt
/// is turn N's prompt plus what turn N generated.
impl infer_seam::PrefixReuse for VulkanExecutor {
    /// Zero: see the type doc — host pages do not name device KV here.
    fn reusable_prefix_blocks(&self, _blocks: &[infer_seam::PrefixBlock]) -> usize {
        0
    }

    fn reusable_prefix_blocks_for_prompt(
        &self,
        blocks: &[infer_seam::PrefixBlock],
        _tokens: &[u32],
    ) -> usize {
        self.reusable_prefix_blocks(blocks)
    }

    /// Nothing below the seam is keyed to page ids, so eviction needs no mirror
    /// drop.
    fn release_prefix_pages(&mut self, _pages: &[u32]) {}

    fn release_provisional_prefix_pages(&mut self, _pages: &[u32]) {}

    fn cached_prefix_match_len(&self, tokens: &[u32]) -> Result<usize> {
        #[cfg(feature = "vulkan")]
        if let Some(model) = self.model.as_ref() {
            return Ok(model.cached_prefix_len(tokens));
        }
        let _ = tokens;
        Ok(0)
    }

    fn restore_cached_prefix(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        _slot_pages: &[u32],
    ) -> Result<()> {
        #[cfg(feature = "vulkan")]
        if let Some(model) = self.model.as_mut() {
            return model.adopt_cached_prefix(slot, tokens, matched_len);
        }
        let _ = (slot, tokens, matched_len);
        bail!("Vulkan executor has no model loaded")
    }

    /// Unreachable while `reusable_prefix_blocks` is 0 (the engine only calls
    /// this after a page-radix attach). `matched_len` is the answer that means
    /// "restored exactly the page-aligned prefix", i.e. no change.
    fn restore_prefix_sidecar(
        &mut self,
        _slot: usize,
        _tokens: &[u32],
        matched_len: usize,
        _prefix_pages: &[u32],
    ) -> Result<usize> {
        Ok(matched_len)
    }

    /// Feed the one token this request sampled but never fed, so the resident
    /// sequence covers the finished turn exactly and the next turn resumes past
    /// the whole generated region rather than one token short of it.
    fn capture_finish_frontier(
        &mut self,
        slot: usize,
        tokens: &[u32],
        _slot_pages: &[u32],
    ) -> Result<()> {
        #[cfg(feature = "vulkan")]
        if let Some(model) = self.model.as_mut() {
            return model.materialize_finish(slot, tokens);
        }
        let _ = (slot, tokens);
        Ok(())
    }

    /// No radix publish to ride: the resident sequence IS the store.
    fn save_prefix_sidecar(
        &mut self,
        _slot: usize,
        _tokens: &[u32],
        _matched_len: usize,
        _prefix_pages: &[u32],
        _slot_pages: &[u32],
        _newly_cached: &[u32],
    ) -> Result<()> {
        Ok(())
    }
}

pub fn load_qwen3_gguf(
    path: impl AsRef<std::path::Path>,
    num_slots: usize,
    max_seq_len: usize,
) -> Result<(VulkanExecutor, VulkanKvPool)> {
    ensure!(num_slots > 0, "Vulkan load requires at least one slot");
    ensure!(max_seq_len > 0, "Vulkan load requires max_seq_len > 0");
    let gguf = infer_gguf::gguf::GgufFile::open(&path)?;
    let kind = classify_vulkan_gguf(&gguf)?;
    #[cfg(feature = "vulkan")]
    {
        // Build the right model with all weights resident on the GPU. The
        // model's `ResidentWeights` hold `DeviceBuffer`s that borrow the
        // `VulkanContext`, and the model must outlive every forward call, so we
        // leak a `'static` context (released only at process exit — acceptable
        // for a process-lifetime resident model).
        let model = match kind {
            VulkanModelKind::Qwen35Hybrid => {
                let ctx: &'static vulkan_sys::VulkanContext =
                    Box::leak(Box::new(vulkan_sys::VulkanContext::create()?));
                let model = crate::model_qwen35::VulkanQwen35Model::load(ctx, &gguf)?;
                VulkanLoadedModel::Qwen35(Box::new(model))
            }
            VulkanModelKind::Qwen36Moe => {
                let ctx: &'static vulkan_sys::VulkanContext =
                    Box::leak(Box::new(vulkan_sys::VulkanContext::create()?));
                let model = crate::model_qwen36::VulkanQwen36Model::load(ctx, &gguf)?;
                VulkanLoadedModel::Qwen36(Box::new(model))
            }
            other => bail!(
                "Vulkan {other:?} GGUF load is not wired yet \
                 (only Qwen35Hybrid / Qwen36Moe land resident); \
                 host GGUF parse + classification succeeded"
            ),
        };
        let stop_tokens = match &model {
            VulkanLoadedModel::Qwen35(m) => m.config.stop_token_ids.clone(),
            VulkanLoadedModel::Qwen36(m) => m.config.stop_token_ids.clone(),
            _ => Vec::new(),
        };

        // Page the KV cache so `num_slots` sequences can each reach `max_seq_len`.
        let pages_per_slot = max_seq_len.div_ceil(DEFAULT_PAGE_SIZE);
        let total_pages = num_slots * pages_per_slot;
        let pool = VulkanKvPool::new(num_slots, total_pages, DEFAULT_PAGE_SIZE, max_seq_len);

        let exec = VulkanExecutor {
            model: Some(model),
            stop_tokens,
        };
        Ok((exec, pool))
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (num_slots, max_seq_len, kind);
        Err(anyhow::anyhow!(
            "Vulkan backend not compiled: rebuild with --features vulkan \
             (host stage validated: GGUF parsed)"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_plan::{DecodeRow, ForwardMode, PrefillRow};

    fn one_row_plan(decode: bool) -> ForwardPlan {
        ForwardPlan {
            mode: if decode {
                ForwardMode::Decode
            } else {
                ForwardMode::Prefill
            },
            decode_rows: if decode {
                vec![DecodeRow {
                    slot: 0,
                    last_token: 7,
                    kv_seq_len: 3,
                    params: SamplingParams::default(),
                    penalty_history: None,
                    penalty_prompt_len: 0,
                }]
            } else {
                Vec::new()
            },
            prefill_rows: if decode {
                Vec::new()
            } else {
                vec![PrefillRow {
                    slot: 0,
                    tokens: vec![1, 2, 3],
                    start_pos: 0,
                    total_tokens: 3,
                    params: SamplingParams::default(),
                    penalty_history: None,
                    penalty_prompt_len: 0,
                }]
            },
        }
    }

    fn pool() -> VulkanKvPool {
        VulkanKvPool::new(2, 8, DEFAULT_PAGE_SIZE, 256)
    }

    #[test]
    fn classifies_vulkan_model_families_from_architecture() {
        assert_eq!(
            classify_vulkan_architecture("deepseek4", None, 0),
            VulkanModelKind::Dsv4
        );
        assert_eq!(
            classify_vulkan_architecture("qwen3moe", None, 64),
            VulkanModelKind::Qwen36Moe
        );
        assert_eq!(
            classify_vulkan_architecture("qwen3", Some("Qwen3.5-4B"), 0),
            VulkanModelKind::Qwen35Hybrid
        );
        assert_eq!(
            classify_vulkan_architecture("qwen3moe", Some("Qwen3.5-MoE-A2B"), 64),
            VulkanModelKind::Qwen36Moe
        );
        assert_eq!(
            classify_vulkan_architecture("qwen3", None, 0),
            VulkanModelKind::Qwen3Dense
        );
        // The regression this guards: `qwen4_exp` reports 512 experts, so the
        // `expert_count > 0` clause claimed it as `Qwen36Moe` and the model ran
        // on a path whose router shader tops out at 256 experts.
        assert_eq!(
            classify_vulkan_architecture("qwen4_exp", Some("Qwen3.8-Flash-Next"), 512),
            VulkanModelKind::Qwen4Exp
        );
        assert_eq!(
            classify_vulkan_architecture("qwen4_exp", None, 0),
            VulkanModelKind::Qwen4Exp
        );
    }

    #[test]
    fn idle_plan_returns_empty_output() {
        let mut exec = VulkanExecutor::unloaded();
        let mut pool = pool();
        let inflight = exec.submit(&ForwardPlan::idle(), &mut pool).unwrap();
        match exec.poll(inflight).unwrap() {
            PollResult::Ready(out) => assert!(out.tokens.is_empty()),
            PollResult::NotReady(_) => panic!("P2 resolves synchronously"),
        }
    }

    #[test]
    fn unloaded_executor_errors_clearly() {
        let mut exec = VulkanExecutor::unloaded();
        let mut pool = pool();
        let err = exec.submit(&one_row_plan(false), &mut pool).unwrap_err();
        assert!(err.to_string().contains("no model loaded"), "{err}");
        let err = exec.submit(&one_row_plan(true), &mut pool).unwrap_err();
        assert!(err.to_string().contains("no model loaded"), "{err}");
    }

    /// On-box load test against the REAL Qwen3.6-27B GGUF. `#[ignore]` by
    /// default (loads ~26 GB onto the GPU); run with
    /// `cargo test -p infer-vulkan --features vulkan --lib -- --ignored --nocapture`.
    /// Skips gracefully if the model file or a Vulkan device is absent. Proves
    /// the 27B lands resident on the 8060S via ARLE (the step before the
    /// numeric forward, which still bails).
    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "needs the on-box Qwen3.6-27B GGUF + ~26 GB VRAM"]
    fn loads_real_27b_resident_on_device() {
        let path = std::path::Path::new(r"C:\Users\Asus\models\qwen3.6\Qwen3.6-27B-Q8_0.gguf");
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            return;
        }
        if let Err(e) = vulkan_sys::VulkanContext::create() {
            eprintln!("skip: no Vulkan device available ({e})");
            return;
        }

        let started = std::time::Instant::now();
        let (exec, _pool) =
            load_qwen3_gguf(path, 1, 4096).expect("load_qwen3_gguf on the real 27B must succeed");
        let elapsed = started.elapsed();

        assert!(exec.has_model(), "executor must hold a loaded model");
        let (tensor_count, device_bytes) = exec
            .resident_stats()
            .expect("loaded model must report resident stats");
        assert!(
            tensor_count > 0,
            "model must have at least one device-resident tensor"
        );

        let device_gb = device_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
        eprintln!(
            "PASS: loaded Qwen3.6-27B resident on {}: {} tensors, {:.2} GiB device bytes, {:.1}s",
            exec.device_name().unwrap_or("<unknown>"),
            tensor_count,
            device_gb,
            elapsed.as_secs_f64(),
        );
    }
}
