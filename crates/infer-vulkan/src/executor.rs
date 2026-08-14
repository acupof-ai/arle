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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanModelKind {
    Qwen3Dense,
    Dsv4,
    Qwen35Hybrid,
    Qwen36Moe,
    Gemma4,
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
    if arch.contains("gemma4") || name.contains("gemma-4") || name.contains("gemma4") {
        return VulkanModelKind::Gemma4;
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
    Dsv4(Box<crate::model_dsv4::VulkanDsv4Model>),
    Qwen35(Box<crate::model_qwen35::VulkanQwen35Model>),
    Qwen36(Box<crate::model_qwen36::VulkanQwen36Model>),
    Gemma4(Box<crate::model_gemma4::VulkanGemma4Model>),
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
            Self::Dsv4(model) => model.forward_token(slot, epoch, token, start_pos),
            Self::Qwen35(model) => model.forward_token(slot, epoch, token, start_pos),
            Self::Qwen36(model) => model.forward_token(slot, epoch, token, start_pos),
            Self::Gemma4(model) => model.forward_token(slot, epoch, token, start_pos),
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

    /// Whether a model is resident on the device.
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

    /// Device name of the loaded model, or `None` if no model is loaded.
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
            let mut logits = Vec::new();
            for (i, &token) in tokens.iter().enumerate() {
                logits = model.forward_token(slot, epoch, token, start_pos + i)?;
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
            let position = (row.start_pos + row.tokens.len()) as u64;
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
            microbatch: None,
            spec: None,
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
            classify_vulkan_architecture("gemma4", None, 0),
            VulkanModelKind::Gemma4
        );
        assert_eq!(
            classify_vulkan_architecture("qwen3", None, 0),
            VulkanModelKind::Qwen3Dense
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
        // Skip cleanly if no Vulkan device is available on this box.
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
