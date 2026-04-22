use crate::dataset::LcgRng;

pub fn sample_categorical(
    logits_data: &[f32],
    batch_shape: (usize, usize),
    vocab: usize,
    temperature: f32,
    rng: &mut LcgRng,
) -> (Vec<usize>, Vec<f32>) {
    let rows = batch_shape.0 * batch_shape.1;
    let mut sampled_ids = Vec::with_capacity(rows);
    let mut chosen_log_probs = Vec::with_capacity(rows);
    sample_categorical_into(
        &mut sampled_ids,
        &mut chosen_log_probs,
        logits_data,
        batch_shape,
        vocab,
        temperature,
        rng,
    );

    (sampled_ids, chosen_log_probs)
}

pub fn sample_categorical_into(
    sampled_ids: &mut Vec<usize>,
    chosen_log_probs: &mut Vec<f32>,
    logits_data: &[f32],
    batch_shape: (usize, usize),
    vocab: usize,
    temperature: f32,
    rng: &mut LcgRng,
) {
    let rows = batch_shape.0 * batch_shape.1;
    assert_eq!(
        logits_data.len(),
        rows * vocab,
        "categorical sampler expects logits shaped as [B, S, V]",
    );

    sampled_ids.clear();
    chosen_log_probs.clear();
    sampled_ids.reserve(rows.saturating_sub(sampled_ids.capacity()));
    chosen_log_probs.reserve(rows.saturating_sub(chosen_log_probs.capacity()));
    for row in 0..rows {
        let base = row * vocab;
        let slice = &logits_data[base..base + vocab];
        let (index, log_prob) = sample_row_with_log_prob(slice, temperature, rng);
        sampled_ids.push(index);
        chosen_log_probs.push(log_prob);
    }
}

pub fn sample_categorical_rows_into(
    sampled_ids: &mut Vec<usize>,
    chosen_log_probs: &mut Vec<f32>,
    logits_data: &[f32],
    batch_shape: (usize, usize),
    vocab: usize,
    temperature: f32,
    rngs: &mut [LcgRng],
) {
    let rows = batch_shape.0 * batch_shape.1;
    assert_eq!(
        logits_data.len(),
        rows * vocab,
        "categorical sampler expects logits shaped as [B, S, V]",
    );
    assert_eq!(
        rngs.len(),
        rows,
        "categorical sampler expects one RNG per row",
    );

    sampled_ids.clear();
    chosen_log_probs.clear();
    sampled_ids.reserve(rows.saturating_sub(sampled_ids.capacity()));
    chosen_log_probs.reserve(rows.saturating_sub(chosen_log_probs.capacity()));
    for (row, rng) in rngs.iter_mut().enumerate() {
        let base = row * vocab;
        let slice = &logits_data[base..base + vocab];
        let (index, log_prob) = sample_row_with_log_prob(slice, temperature, rng);
        sampled_ids.push(index);
        chosen_log_probs.push(log_prob);
    }
}

fn sample_row_with_log_prob(logits: &[f32], temperature: f32, rng: &mut LcgRng) -> (usize, f32) {
    if temperature <= 0.0 {
        let index = argmax(logits);
        return (index, log_prob_at_index(logits, 1.0, index));
    }

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
        let scaled = (*value * inv_temp) - max_scaled;
        let prob = (scaled.exp() / denom) as f64;
        cumulative += prob;
        if target < cumulative {
            return (index, scaled - denom.ln());
        }
    }

    let index = logits.len().saturating_sub(1);
    let scaled = (logits[index] * inv_temp) - max_scaled;
    (index, scaled - denom.ln())
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

#[cfg(test)]
mod tests {
    use super::{log_prob_at_index, sample_categorical, sample_categorical_rows_into};
    use crate::dataset::LcgRng;

    #[test]
    fn sampled_log_prob_matches_reference_at_temperature() {
        let logits = [0.1, 1.7, -0.4, 0.8, 0.0, 2.2];
        let mut rng = LcgRng::seed(12345);
        let (ids, log_probs) = sample_categorical(&logits, (1, 1), logits.len(), 0.7, &mut rng);
        assert_eq!(ids.len(), 1);
        assert_eq!(log_probs.len(), 1);
        let expected = log_prob_at_index(&logits, 0.7, ids[0]);
        assert!((log_probs[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn greedy_sampled_log_prob_matches_reference() {
        let logits = [0.5, 0.25, 3.0, -1.0];
        let mut rng = LcgRng::seed(7);
        let (ids, log_probs) = sample_categorical(&logits, (1, 1), logits.len(), 0.0, &mut rng);
        assert_eq!(ids, vec![2]);
        let expected = log_prob_at_index(&logits, 1.0, 2);
        assert!((log_probs[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn per_row_rng_sampling_matches_serial_sampling() {
        let vocab = 4;
        let logits = [
            0.1, 0.9, -0.3, 0.2, //
            1.2, -0.7, 0.3, 0.8, //
            -0.5, 0.6, 0.4, 0.0,
        ];
        let mut sampled_ids = Vec::new();
        let mut chosen_log_probs = Vec::new();
        let mut rngs = vec![LcgRng::seed(11), LcgRng::seed(22), LcgRng::seed(33)];
        let mut serial_rngs = rngs.clone();

        sample_categorical_rows_into(
            &mut sampled_ids,
            &mut chosen_log_probs,
            &logits,
            (3, 1),
            vocab,
            0.7,
            &mut rngs,
        );

        let mut expected_ids = Vec::new();
        let mut expected_log_probs = Vec::new();
        for (row, rng) in serial_rngs.iter_mut().enumerate() {
            let base = row * vocab;
            let (index, log_prob) =
                super::sample_row_with_log_prob(&logits[base..base + vocab], 0.7, rng);
            expected_ids.push(index);
            expected_log_probs.push(log_prob);
        }

        assert_eq!(sampled_ids, expected_ids);
        assert_eq!(chosen_log_probs, expected_log_probs);
        for (actual_rng, expected_rng) in rngs.iter_mut().zip(serial_rngs.iter_mut()) {
            assert_eq!(actual_rng.next_u64(), expected_rng.next_u64());
        }
    }
}
