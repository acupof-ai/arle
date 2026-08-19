//! MoE routing / gating configuration — the device-independent description of
//! one MoE block's router, covering the knobs both legacy routers read (DSv4:
//! `noaux_tc` bias selection, no grouping; Qwen3.6: softmax + greedy top-k, one
//! shared expert). Group-limited routing fields exist for a future grouped
//! kernel but are wired by neither ARLE router.

use crate::error::{Result, bail};

/// [`Self::scoring_kind`] matches the CUDA `dsv4_route` `scoring_kind` arg (0/1/2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScoringFunc {
    Softmax,
    Sigmoid,
    /// `scores[e] = sqrt(softplus(logits[e]))` (DeepSeek-V4 `sqrtsoftplus`).
    SqrtSoftplus,
}

impl ScoringFunc {
    /// The CUDA `dsv4_route` `scoring_kind` integer for this scoring func.
    #[must_use]
    pub fn scoring_kind(self) -> i32 {
        match self {
            Self::Softmax => 0,
            Self::Sigmoid => 1,
            Self::SqrtSoftplus => 2,
        }
    }
}

/// - [`TopkMethod::Greedy`]: plain top-k over the scores (Qwen3.6).
/// - [`TopkMethod::NoAuxTc`]: DSv4 no-aux-loss top-k — selection uses the
///   bias-corrected key `scores[e] + bias[e]`, the weight reads un-biased
///   `scores[e]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TopkMethod {
    Greedy,
    NoAuxTc,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    /// Routed experts (one router logit each).
    pub num_experts: usize,
    /// Experts selected per token (`num_experts_per_tok`).
    pub top_k: usize,
    /// Scoring function; the selection bias presence comes from `topk_method`.
    pub scoring_func: ScoringFunc,
    pub topk_method: TopkMethod,
    /// Renormalize the selected top-k weights to sum to 1.
    ///
    /// - **softmax**: drives Qwen3.6's separate post-renorm; the DSv4 reference +
    ///   kernel never renorm inside the router (denom pinned to 1.0).
    /// - **sigmoid / sqrtsoftplus**: DSv4 always normalizes regardless of this
    ///   flag (`denom = selected_sum + 1e-9`); the flag only documents intent.
    pub norm_topk_prob: bool,
    /// Final multiplicative scaling on every routed weight (Qwen3.6 = 1.0).
    pub routed_scaling_factor: f32,
    /// Group-limited routing: expert-group count (DeepSeek-V2/V3 `n_group`).
    /// `None` ⇒ no grouping. See [`crate::route::group_limited_mask`].
    pub n_group: Option<usize>,
    /// Groups kept per token; `Some` iff `n_group` is `Some`.
    pub topk_group: Option<usize>,
    /// Router projection input dim; carried for shape validation (this crate
    /// routes from pre-computed logits, never multiplies the gate).
    pub hidden_size: usize,
}

impl MoeConfig {
    /// Qwen3.6 router: softmax, greedy top-k, no bias, scaling 1.0, one
    /// sigmoid-gated shared expert.
    #[must_use]
    pub fn qwen36(
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
        hidden_size: usize,
    ) -> Self {
        Self {
            num_experts,
            top_k,
            scoring_func: ScoringFunc::Softmax,
            topk_method: TopkMethod::Greedy,
            norm_topk_prob,
            routed_scaling_factor: 1.0,
            n_group: None,
            topk_group: None,
            hidden_size,
        }
    }

    /// DSv4 router: `sqrtsoftplus` scoring, `noaux_tc` top-k (selection bias is
    /// a runtime gate tensor, not a config field), config `routed_scaling_factor`,
    /// DSv4 router: `sqrtsoftplus` scoring, `noaux_tc` top-k (selection bias is
    /// a runtime gate tensor, not a config field), config `routed_scaling_factor`.
    /// DSv4-Flash ships no group-limited routing.
    #[must_use]
    pub fn dsv4(
        num_experts: usize,
        top_k: usize,
        routed_scaling_factor: f32,
        hidden_size: usize,
    ) -> Self {
        Self {
            num_experts,
            top_k,
            scoring_func: ScoringFunc::SqrtSoftplus,
            topk_method: TopkMethod::NoAuxTc,
            // sqrtsoftplus always normalizes the selected weights; the flag is
            // documentary for this path (see route.rs `normalize` derivation).
            norm_topk_prob: true,
            routed_scaling_factor,
            n_group: None,
            topk_group: None,
            hidden_size,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_experts == 0 {
            bail!("MoE config requires num_experts > 0");
        }
        if self.top_k == 0 {
            bail!("MoE config requires top_k (num_experts_per_tok) > 0");
        }
        if self.top_k > self.num_experts {
            bail!(
                "num_experts_per_tok ({}) must not exceed num_experts ({})",
                self.top_k,
                self.num_experts
            );
        }
        match (self.n_group, self.topk_group) {
            (None, None) => {}
            (Some(n_group), Some(topk_group)) => {
                if n_group == 0 {
                    bail!("group-limited routing requires n_group > 0");
                }
                if !self.num_experts.is_multiple_of(n_group) {
                    bail!(
                        "num_experts ({}) must be divisible by n_group ({n_group})",
                        self.num_experts
                    );
                }
                if topk_group == 0 || topk_group > n_group {
                    bail!("topk_group ({topk_group}) must be in 1..=n_group ({n_group})");
                }
                // top_k must be reachable from the kept groups.
                let experts_per_group = self.num_experts / n_group;
                if self.top_k > topk_group * experts_per_group {
                    bail!(
                        "top_k ({}) exceeds capacity of topk_group*experts_per_group ({})",
                        self.top_k,
                        topk_group * experts_per_group
                    );
                }
            }
            _ => bail!("n_group and topk_group must both be set or both unset"),
        }
        Ok(())
    }
}
