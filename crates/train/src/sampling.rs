use crate::dataset::LcgRng;

pub fn sample_categorical(
    logits_data: &[f32],
    batch_shape: (usize, usize),
    vocab: usize,
    temperature: f32,
    rng: &mut LcgRng,
) -> (Vec<usize>, Vec<f32>) {
    let rows = batch_shape.0 * batch_shape.1;
    assert_eq!(
        logits_data.len(),
        rows * vocab,
        "categorical sampler expects logits shaped as [B, S, V]",
    );

    let mut sampled_ids = Vec::with_capacity(rows);
    let mut chosen_log_probs = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * vocab;
        let slice = &logits_data[base..base + vocab];
        let (index, log_prob) = if temperature <= 0.0 {
            let index = argmax(slice);
            (index, log_prob_at_index(slice, 1.0, index))
        } else {
            let index = sample_row(slice, temperature, rng);
            (index, log_prob_at_index(slice, temperature, index))
        };
        sampled_ids.push(index);
        chosen_log_probs.push(log_prob);
    }

    (sampled_ids, chosen_log_probs)
}

pub(crate) fn log_prob_at_index(logits: &[f32], temperature: f32, index: usize) -> f32 {
    let inv_temp = 1.0 / temperature;
    let max_scaled = logits
        .iter()
        .map(|value| *value * inv_temp)
        .fold(f32::NEG_INFINITY, f32::max);
    let denom = logits
        .iter()
        .map(|value| ((*value * inv_temp) - max_scaled).exp())
        .sum::<f32>();
    (logits[index] * inv_temp) - max_scaled - denom.ln()
}

fn sample_row(logits: &[f32], temperature: f32, rng: &mut LcgRng) -> usize {
    let inv_temp = 1.0 / temperature;
    let max_scaled = logits
        .iter()
        .map(|value| *value * inv_temp)
        .fold(f32::NEG_INFINITY, f32::max);
    let denom = logits
        .iter()
        .map(|value| ((*value * inv_temp) - max_scaled).exp())
        .sum::<f32>();

    let target = rng.next_u64() as f64 / (u64::MAX as f64 + 1.0);
    let mut cumulative = 0.0_f64;
    for (index, value) in logits.iter().enumerate() {
        let prob = (((*value * inv_temp) - max_scaled).exp() / denom) as f64;
        cumulative += prob;
        if target < cumulative {
            return index;
        }
    }

    logits.len().saturating_sub(1)
}

fn argmax(logits: &[f32]) -> usize {
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in logits.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best_index = index;
        }
    }
    best_index
}
