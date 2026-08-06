//! MoE config bridge qwen35-spec → infer-moe + per-rank expert-split math.
//!
//! [`moe_config_from_qwen35`] maps a [`qwen35_spec::Qwen35Config`] onto the
//! device-independent [`infer_moe::MoeConfig`] (= [`infer_moe::MoeConfig::qwen36`]),
//! so the GPU kernel and the CPU reference share one config. [`ExpertSplit`]
//! holds the EP per-rank expert ownership (single-GPU keeps all experts local).

use anyhow::{Result, ensure};
use infer_moe::MoeConfig;
use qwen35_spec::Qwen35Config;

/// Build the [`infer_moe::MoeConfig`] router description from a Qwen3.5/3.6
/// checkpoint config.
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

/// Per-rank expert ownership for expert parallelism (EP).
///
/// Experts `0..num_experts` are split evenly across `ep_size` ranks. Rank `r`
/// owns the contiguous block `[local_expert_start, local_expert_start +
/// experts_per_rank)`. Single-GPU (`ep_size == 1`) keeps every expert local.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSplit {
    /// Total routed experts across all EP ranks.
    pub num_experts: usize,
    /// Number of EP ranks the experts are split across.
    pub ep_size: usize,
    /// This rank's index within the EP group (`0 <= ep_rank < ep_size`).
    pub ep_rank: usize,
    /// Experts owned by each rank (`num_experts / ep_size`).
    pub experts_per_rank: usize,
    /// Global index of the first expert this rank owns.
    pub local_expert_start: usize,
}

impl ExpertSplit {
    /// Compute the per-rank expert split.
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

    /// Single-GPU split: this rank owns every expert.
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

    /// Exclusive end of this rank's owned-expert range.
    #[must_use]
    pub fn local_expert_end(&self) -> usize {
        self.local_expert_start + self.experts_per_rank
    }

    /// Whether the given global expert index is owned by this rank.
    #[must_use]
    pub fn owns(&self, global_expert: usize) -> bool {
        (self.local_expert_start..self.local_expert_end()).contains(&global_expert)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_moe::{ScoringFunc, TopkMethod, route};

    // Minimal MoE config built directly; only the bridge math is under test.
    fn moe_config_fixture(num_experts: usize, top_k: usize, norm: bool) -> Qwen35Config {
        Qwen35Config {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            vocab_size: 100,
            rms_norm_eps: 1e-6,
            stop_token_ids: vec![0],
            bos_token_id: None,
            eos_token_id: 0,
            tie_word_embeddings: true,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            linear_num_key_heads: 1,
            linear_key_head_dim: 4,
            linear_num_value_heads: 1,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            rope_theta: 10000.0,
            rope_scaling: None,
            partial_rotary_factor: 1.0,
            rotary_dim: 4,
            rope_cache_len_hint: Some(128),
            layer_types: vec![
                qwen35_spec::LayerType::FullAttention,
                qwen35_spec::LayerType::FullAttention,
            ],
            num_experts,
            num_experts_per_tok: top_k,
            decoder_sparse_step: 1,
            moe_intermediate_size: 32,
            shared_expert_intermediate_size: 32,
            norm_topk_prob: norm,
            mlp_only_layers: vec![],
            full_attn_gated: true,
        }
    }

    #[test]
    fn bridge_maps_qwen36_router_fields() {
        let cfg = moe_config_fixture(8, 2, true);
        let moe = moe_config_from_qwen35(&cfg).unwrap();
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.top_k, 2);
        assert!(moe.norm_topk_prob);
        assert_eq!(moe.hidden_size, 16);
        // Qwen3.6 semantics: plain softmax, greedy, scaling 1.0, one shared expert.
        assert_eq!(moe.scoring_func, ScoringFunc::Softmax);
        assert_eq!(moe.topk_method, TopkMethod::Greedy);
        assert_eq!(moe.routed_scaling_factor, 1.0);
        assert_eq!(moe.num_shared_experts, 1);
        assert!(moe.n_group.is_none());
    }

    #[test]
    fn bridged_config_routes_through_infer_moe() {
        // The mapped config must drive infer_moe::route consistently with the
        // 17 infer-moe tests' Qwen3.6 expectations: greedy top-k softmax, then
        // norm_topk_prob renorm to sum=1.
        let cfg = moe_config_fixture(4, 2, true);
        let moe = moe_config_from_qwen35(&cfg).unwrap();
        // Fixed logit vector over 4 experts; experts 3 and 2 are the top-2.
        let logits = [0.1f32, 0.2, 3.0, 4.0];
        let decisions = route(&logits, &[], &moe).unwrap();
        assert_eq!(decisions.len(), 1);
        let d = &decisions[0];
        assert_eq!(d.expert_ids(), vec![3, 2]);
        // norm_topk_prob ⇒ the two selected weights renormalize to sum 1.
        let sum: f32 = d.weights().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights should sum to 1, got {sum}"
        );
    }

    #[test]
    fn bridge_rejects_dense_config() {
        let dense = moe_config_fixture(0, 0, true);
        assert!(moe_config_from_qwen35(&dense).is_err());
    }

    #[test]
    fn expert_split_single_gpu_owns_all() {
        let split = ExpertSplit::single(128);
        assert_eq!(split.experts_per_rank, 128);
        assert_eq!(split.local_expert_start, 0);
        assert_eq!(split.local_expert_end(), 128);
        assert!(split.owns(0));
        assert!(split.owns(127));
    }

    #[test]
    fn expert_split_even_partition_across_ep_ranks() {
        // 128 experts over EP=8 → 16 experts/rank, contiguous, non-overlapping.
        let mut covered = 0;
        let mut expected_start = 0;
        for ep_rank in 0..8 {
            let s = ExpertSplit::new(128, 8, ep_rank).unwrap();
            assert_eq!(s.experts_per_rank, 16);
            assert_eq!(s.local_expert_start, expected_start);
            assert_eq!(s.local_expert_end(), expected_start + 16);
            // Rank owns its block and not its neighbours'.
            assert!(s.owns(expected_start));
            assert!(s.owns(expected_start + 15));
            assert!(!s.owns(expected_start.wrapping_sub(1)) || ep_rank == 0);
            covered += s.experts_per_rank;
            expected_start += 16;
        }
        assert_eq!(covered, 128);
        assert_eq!(expected_start, 128);
    }

    #[test]
    fn expert_split_dsv4_256_experts_ep8_owns_32_per_rank() {
        // DSv4-Flash: 256 routed experts over EP=8 → 32 experts/rank, contiguous.
        let mut covered = 0;
        for ep_rank in 0..8 {
            let s = ExpertSplit::new(256, 8, ep_rank).unwrap();
            assert_eq!(s.experts_per_rank, 32, "rank {ep_rank} owns 256/8");
            assert_eq!(s.local_expert_start, ep_rank * 32);
            assert_eq!(s.local_expert_end(), ep_rank * 32 + 32);
            assert!(s.owns(ep_rank * 32) && s.owns(ep_rank * 32 + 31));
            assert!(!s.owns(ep_rank * 32 + 32)); // next rank's first expert
            covered += s.experts_per_rank;
        }
        assert_eq!(
            covered, 256,
            "all 256 experts owned exactly once across EP=8"
        );
    }

    #[test]
    fn expert_split_rejects_indivisible_and_bad_rank() {
        assert!(ExpertSplit::new(10, 4, 0).is_err()); // 10 not divisible by 4
        assert!(ExpertSplit::new(8, 0, 0).is_err()); // ep_size 0
        assert!(ExpertSplit::new(8, 4, 4).is_err()); // ep_rank out of range
    }
}
