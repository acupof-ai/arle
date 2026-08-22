//! [`moe_config_from_qwen35`] maps a [`qwen35_spec::Qwen35Config`] onto the
//! device-independent [`infer_moe::MoeConfig`] (= [`infer_moe::MoeConfig::qwen36`]),
//! so the GPU kernel and the CPU reference share one config. [`ExpertSplit`]
//! holds the EP per-rank expert ownership (single-GPU keeps all experts local).

use anyhow::{Result, ensure};
use infer_moe::MoeConfig;
use qwen35_spec::Qwen35Config;

///
/// # Errors
/// Errors if the config is not a MoE checkpoint (`num_experts == 0`) or the
/// resulting [`MoeConfig`] fails [`MoeConfig::validate`].
pub fn moe_config_from_qwen35(config: &Qwen35Config) -> Result<MoeConfig> {
    ensure!(
        config.is_moe(),
        "qwen3.5 config is dense (num_experts == 0); no MoE router to build"
    );
    let moe = MoeConfig::qwen36(
        config.num_experts,
        config.num_experts_per_tok,
        config.norm_topk_prob,
        config.hidden_size,
    );
    moe.validate()?;
    Ok(moe)
}

/// Experts `0..num_experts` are split evenly across `ep_size` ranks. Rank `r`
/// owns the contiguous block `[local_expert_start, local_expert_start +
/// experts_per_rank)`. Single-GPU (`ep_size == 1`) keeps every expert local.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSplit {
    pub num_experts: usize,
    pub ep_size: usize,
    pub ep_rank: usize,
    pub experts_per_rank: usize,
    pub local_expert_start: usize,
}

impl ExpertSplit {
    ///
    /// # Errors
    /// Errors if `ep_size == 0`, `ep_rank >= ep_size`, or `num_experts` is not
    /// divisible by `ep_size` (the EP partition must be uniform, matching the
    /// `dsv4_*` local-expert kernels' fixed `experts_per_rank`).
    pub fn new(num_experts: usize, ep_size: usize, ep_rank: usize) -> Result<Self> {
        ensure!(ep_size > 0, "ep_size must be > 0");
        ensure!(
            ep_rank < ep_size,
            "ep_rank ({ep_rank}) must be < ep_size ({ep_size})"
        );
        ensure!(
            num_experts.is_multiple_of(ep_size),
            "num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
        );
        let experts_per_rank = num_experts / ep_size;
        Ok(Self {
            num_experts,
            ep_size,
            ep_rank,
            experts_per_rank,
            local_expert_start: ep_rank * experts_per_rank,
        })
    }

    #[must_use]
    pub fn single(num_experts: usize) -> Self {
        Self {
            num_experts,
            ep_size: 1,
            ep_rank: 0,
            experts_per_rank: num_experts,
            local_expert_start: 0,
        }
    }

    #[must_use]
    pub fn local_expert_end(&self) -> usize {
        self.local_expert_start + self.experts_per_rank
    }

    #[must_use]
    pub fn owns(&self, global_expert: usize) -> bool {
        (self.local_expert_start..self.local_expert_end()).contains(&global_expert)
    }
}
