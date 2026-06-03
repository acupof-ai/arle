//! MoE routing / gating configuration.
//!
//! `MoeConfig` is the device-independent description of one MoE block's router,
//! collecting exactly the knobs the legacy routers read:
//!
//! - **DSv4** (`crates/deepseek-spec/src/v4.rs`): `n_routed_experts`,
//!   `n_shared_experts`, `num_experts_per_tok`, `routed_scaling_factor`,
//!   `norm_topk_prob`, `scoring_func`, `topk_method`. DSv4 in this codebase
//!   selects with a learned score-correction bias (`topk_method = "noaux_tc"`)
//!   and has **no** group-limited routing (`n_group` / `topk_group` absent).
//! - **Qwen3.6** (`infer/src/model/qwen35/moe.rs`): `num_experts`,
//!   `num_experts_per_tok`, `norm_topk_prob`, one always-on shared expert.
//!   Routes with plain softmax + argmax top-k, no bias, scaling = 1.0.
//!
//! Group-limited routing (`n_group` / `topk_group`) is the upstream
//! DeepSeek-V2/V3 mechanism. It is **not** wired by either legacy ARLE router,
//! but the fields + [`crate::route`] grouping helper are provided so the same
//! foundation can verify a grouped GPU kernel if a future checkpoint sets them.

use crate::error::{Result, bail};

/// How the per-expert router logits are turned into selection scores.
///
/// Ported from `DeepSeekV4Config::router_scores_from_logits`
/// (`crates/deepseek-spec/src/v4.rs:305`). The `&str` config value maps:
/// `"softmax" → Softmax`, `"sigmoid" → Sigmoid`, `"sqrtsoftplus" → SqrtSoftplus`.
///
/// The integer discriminants match the CUDA `dsv4_route` kernel's
/// `scoring_kind` argument (`crates/cuda-kernels/csrc/moe/dsv4_route.cu:196`):
/// `0 = softmax`, `1 = sigmoid`, `2 = sqrt(softplus)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ScoringFunc {
    /// `scores = stable_softmax(logits)` over all experts.
    Softmax,
    /// `scores[e] = sigmoid(logits[e])` (independent per expert).
    Sigmoid,
    /// `scores[e] = sqrt(softplus(logits[e]))` (DeepSeek-V4 `sqrtsoftplus`).
    SqrtSoftplus,
}

impl ScoringFunc {
    /// Parse the legacy `scoring_func` config string. Mirrors the match arms in
    /// `DeepSeekV4Config::router_scores_from_logits`.
    pub fn from_config_str(s: &str) -> Result<Self> {
        match s {
            "softmax" => Ok(Self::Softmax),
            "sigmoid" => Ok(Self::Sigmoid),
            "sqrtsoftplus" => Ok(Self::SqrtSoftplus),
            other => bail!("unsupported DSV4 router scoring_func `{other}`"),
        }
    }

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

/// How the top-k experts are selected from the per-expert scores.
///
/// - [`TopkMethod::Greedy`]: plain top-k over the scores themselves. This is
///   the Qwen3.6 path (`softmax → argmax top-k`).
/// - [`TopkMethod::NoAuxTc`]: DeepSeek-V4 no-aux-loss top-k. Selection uses the
///   bias-corrected key `scores[e] + bias[e]` (the `e_score_correction_bias`),
///   while the *weight* still reads the un-biased `scores[e]`. Ported from
///   `topk_indices_by_score` + `moe_routes_from_scores`
///   (`crates/deepseek-spec/src/v4.rs:319`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TopkMethod {
    /// Top-k directly over the scores (Qwen3.6).
    Greedy,
    /// No-aux-loss top-k with score-correction bias (DSv4 `noaux_tc`).
    NoAuxTc,
}

impl TopkMethod {
    /// Parse the legacy `topk_method` config string. DSv4 ships `"noaux_tc"`;
    /// any plain greedy method maps to [`TopkMethod::Greedy`].
    pub fn from_config_str(s: &str) -> Result<Self> {
        match s {
            "noaux_tc" => Ok(Self::NoAuxTc),
            "greedy" | "group_limited_greedy" => Ok(Self::Greedy),
            other => bail!("unsupported MoE topk_method `{other}`"),
        }
    }
}

/// Device-independent description of one MoE block's router.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MoeConfig {
    /// Number of routed experts (DSv4 `n_routed_experts` / Qwen3.6
    /// `num_experts`). The router emits one logit per routed expert.
    pub num_experts: usize,
    /// Number of always-on shared experts that run for every token in addition
    /// to the routed top-k (DSv4 `n_shared_experts`; Qwen3.6 has exactly one
    /// sigmoid-gated shared expert, expressed as `1` here). The shared experts
    /// do not participate in routing — this field is carried so consumers can
    /// account for them; see [`MoeConfig::has_shared_expert`].
    pub num_shared_experts: usize,
    /// Experts selected per token (DSv4 / Qwen3.6 `num_experts_per_tok`).
    pub top_k: usize,
    /// Per-expert score → selection-key correction bias presence. DSv4
    /// `noaux_tc` selection adds a learned `e_score_correction_bias[e]` to the
    /// score before top-k (`bias` passed to `moe_routes_from_scores`). When
    /// `None`, selection is over the raw scores. Qwen3.6 passes no bias (its
    /// CUDA path supplies an all-zero buffer, the additive identity).
    pub scoring_func: ScoringFunc,
    /// Top-k selection rule (`noaux_tc` vs greedy).
    pub topk_method: TopkMethod,
    /// Renormalize the selected top-k weights to sum to 1 per token.
    ///
    /// Semantics differ subtly by scoring func, faithfully matching the legacy:
    /// - **softmax** (`ScoringFunc::Softmax`): the DSv4 reference + the CUDA
    ///   `dsv4_route` kernel *never* renormalize inside the router (denom is
    ///   pinned to `1.0`). Qwen3.6 applies the renorm as a *separate* step
    ///   (`qwen36_renorm_topk_weights`) gated on `norm_topk_prob`. So for the
    ///   softmax path this flag drives that explicit post-renorm.
    /// - **sigmoid / sqrtsoftplus**: the DSv4 reference *always* normalizes the
    ///   selected scores (`denom = selected_sum + 1e-9`) regardless of this
    ///   flag — see [`crate::route`]. This field then only documents intent.
    pub norm_topk_prob: bool,
    /// Final multiplicative scaling applied to every routed weight
    /// (DSv4 `routed_scaling_factor`; Qwen3.6 uses `1.0`).
    pub routed_scaling_factor: f32,
    /// Group-limited routing: number of expert groups. Upstream DeepSeek-V2/V3
    /// `n_group`. `None` ⇒ no grouping (the ARLE DSv4 + Qwen3.6 path). When set,
    /// experts are partitioned into `n_group` equal contiguous groups and only
    /// `topk_group` groups are eligible — see [`crate::route::group_limited_mask`].
    pub n_group: Option<usize>,
    /// Group-limited routing: number of groups kept per token (`topk_group`).
    /// Must be `Some` iff `n_group` is `Some`.
    pub topk_group: Option<usize>,
    /// Router (gate) projection dims: `[num_experts, hidden_size]`. The router
    /// is `logits = gate @ x` → `[num_experts]` per token. Carried for shape
    /// validation against a GPU kernel; this crate routes from pre-computed
    /// logits and does not multiply the gate matrix.
    pub hidden_size: usize,
}

impl MoeConfig {
    /// Construct the Qwen3.6 router config: plain softmax, greedy top-k, no
    /// bias, scaling = 1.0, one sigmoid-gated shared expert, optional
    /// `norm_topk_prob`. Mirrors `infer/src/model/qwen35/moe.rs`.
    #[must_use]
    pub fn qwen36(
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
        hidden_size: usize,
    ) -> Self {
        Self {
            num_experts,
            num_shared_experts: 1,
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

    /// Whether at least one always-on shared expert runs for every token.
    #[must_use]
    pub fn has_shared_expert(&self) -> bool {
        self.num_shared_experts > 0
    }

    /// Validate the config. Mirrors the legacy `DeepSeekV4Config::validate`
    /// MoE checks (`n_routed_experts != 0`, `num_experts_per_tok != 0`,
    /// `num_experts_per_tok <= n_routed_experts`).
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
