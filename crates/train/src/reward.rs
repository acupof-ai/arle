//! Stepwise reward tooling for multi-turn episodes (M4.2).
//!
//! Converts a per-turn reward vector into per-turn returns (MC with discount
//! γ) and fans those returns out to per-position values aligned with an
//! Episode's `response_mask` / `turn_boundaries`. Positions outside agent
//! turns receive 0.

/// Subtract `penalty` from each turn flagged as a failure (malformed tool
/// call, parse error, etc.) before MC discounting. Preserves vector length;
/// positions with `failures[i] == false` pass through unchanged. `penalty`
/// is expected to be non-negative — it is subtracted regardless of sign.
pub fn apply_turn_penalty(per_turn_rewards: &[f32], failures: &[bool], penalty: f32) -> Vec<f32> {
    assert_eq!(
        per_turn_rewards.len(),
        failures.len(),
        "rewards len {} != failures len {}",
        per_turn_rewards.len(),
        failures.len(),
    );
    per_turn_rewards
        .iter()
        .zip(failures.iter())
        .map(|(reward, failed)| if *failed { *reward - penalty } else { *reward })
        .collect()
}

/// Monte-Carlo return at each turn: `G_t = Σ_{k ≥ t} γ^{k - t} · r_k`.
pub fn discounted_returns(per_turn_rewards: &[f32], gamma: f32) -> Vec<f32> {
    let mut returns = vec![0.0f32; per_turn_rewards.len()];
    let mut running = 0.0f32;
    for (index, reward) in per_turn_rewards.iter().enumerate().rev() {
        running = reward + gamma * running;
        returns[index] = running;
    }
    returns
}

/// Expand per-turn returns to a per-position vector of length `seq_len`.
/// Positions inside `[agent_start, agent_end)` for a turn get that turn's
/// return; all others remain 0.
pub fn returns_to_per_position(
    returns: &[f32],
    turn_boundaries: &[(usize, usize)],
    seq_len: usize,
) -> Vec<f32> {
    assert_eq!(
        returns.len(),
        turn_boundaries.len(),
        "returns len {} != turn_boundaries len {}",
        returns.len(),
        turn_boundaries.len(),
    );
    let mut out = vec![0.0f32; seq_len];
    for (return_value, (agent_start, agent_end)) in returns.iter().zip(turn_boundaries.iter()) {
        for slot in &mut out[*agent_start..*agent_end] {
            *slot = *return_value;
        }
    }
    out
}

/// Per-group advantage normalization on scalar returns:
/// `(return - mean_group) / (std_group + eps)`.
pub fn group_normalize(returns: &[f32], group_size: usize) -> Vec<f32> {
    assert!(group_size > 0, "group size must be positive");
    assert_eq!(
        returns.len() % group_size,
        0,
        "return count must be divisible by group size",
    );
    let mut out = Vec::with_capacity(returns.len());
    for group in returns.chunks(group_size) {
        let mean = group.iter().sum::<f32>() / group.len() as f32;
        let variance = group
            .iter()
            .map(|value| (*value - mean).powi(2))
            .sum::<f32>()
            / group.len() as f32;
        let std = variance.sqrt();
        for value in group {
            out.push((*value - mean) / (std + 1.0e-6));
        }
    }
    out
}
