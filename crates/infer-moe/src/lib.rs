//! `infer-moe` — pure, CPU-verifiable, device-independent MoE routing / gating
//! math (the reference the GPU kernel is verified against).
//!
//! # Which model uses which routing rule
//!
//! | Aspect            | DSv4 (`deepseek-spec/src/v4.rs`)        | Qwen3.6 (`qwen35/moe.rs`)        |
//! |-------------------|-----------------------------------------|----------------------------------|
//! | Scoring func      | `softmax` / `sigmoid` / `sqrtsoftplus`  | `softmax` (plain)                |
//! | Selection         | top-k by `score + bias` (`noaux_tc`)    | greedy top-k by score (no bias)  |
//! | Weight denom      | `1.0` (softmax) / `selected_sum+1e-9`   | `1.0`, then optional norm renorm |
//! | `norm_topk_prob`  | softmax: no renorm; else always norm    | optional separate renorm step    |
//! | `routed_scaling`  | config (`routed_scaling_factor`)        | `1.0`                            |
//! | shared experts    | `n_shared_experts` always-on            | exactly one, sigmoid-gated       |
//! | `n_group`/`topk_group` | **absent** (no group-limited routing) | **absent**                  |
//!
//! Group-limited routing (`n_group`/`topk_group`) is the DeepSeek-V2/V3 mechanism
//! neither ARLE router wires; the fields + [`group_limited_mask`] exist so this
//! reference can verify a grouped kernel if a future checkpoint sets them.

#[path = "config.rs"]
mod config;
#[path = "error.rs"]
mod error;
#[path = "route.rs"]
mod route;

pub use config::{MoeConfig, ScoringFunc, TopkMethod};
pub use error::{MoeError, Result};
pub use route::{
    ExpertWeight, RoutingDecision, group_limited_mask, route, route_token, scores_from_logits,
    sigmoid, stable_softmax, stable_softplus,
};
