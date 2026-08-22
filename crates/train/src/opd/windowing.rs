use autograd::{Tape, TensorId, TensorStore, ops::slice};

use crate::qwen35::SequenceWindow;

use super::{OpdError, OpdKlMask, Result};

pub(super) fn sequence_windows(
    total_positions: usize,
    window_size: usize,
) -> Result<Vec<SequenceWindow>> {
    if total_positions == 0 {
        return Err(OpdError::InvalidInput(
            "OPD windowed logits path requires at least one position. Hint: \
             pass a non-empty prompt/completion sequence before enabling \
             --logits-window-size."
                .to_owned(),
        ));
    }
    if window_size == 0 {
        return Err(OpdError::InvalidInput(
            "OPD logits window size must be > 0 when set. Hint: pass \
             --logits-window-size 64 or omit the flag for full logits."
                .to_owned(),
        ));
    }
    let mut windows = Vec::new();
    let mut start = 0usize;
    while start < total_positions {
        let end = start.saturating_add(window_size).min(total_positions);
        windows.push(SequenceWindow { start, end });
        start = end;
    }
    Ok(windows)
}

pub(super) fn sequence_windows_for_range(
    range: KlLogitRange,
    window_size: usize,
) -> Result<Vec<SequenceWindow>> {
    sequence_windows(range.len(), window_size).map(|windows| {
        windows
            .into_iter()
            .map(|window| SequenceWindow {
                start: range.start + window.start,
                end: range.start + window.end,
            })
            .collect()
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct KlLogitRange {
    pub(super) start: usize,
    pub(super) end: usize,
}

impl KlLogitRange {
    pub(super) fn len(self) -> usize {
        self.end - self.start
    }
}

pub(super) fn kl_logit_range(
    mask: OpdKlMask,
    prompt_len: usize,
    sequence_len: usize,
) -> Result<KlLogitRange> {
    match mask {
        OpdKlMask::Full => {
            if sequence_len == 0 {
                return Err(OpdError::InvalidInput(
                    "OPD full KL mask requires at least one sequence position.".to_owned(),
                ));
            }
            Ok(KlLogitRange {
                start: 0,
                end: sequence_len,
            })
        }
        OpdKlMask::CompletionOnly => {
            if prompt_len == 0 {
                return Err(OpdError::InvalidInput(
                    "OPD completion-only KL mask requires a non-empty prompt.".to_owned(),
                ));
            }
            if sequence_len <= prompt_len {
                return Err(OpdError::InvalidInput(format!(
                    "OPD completion-only KL mask requires at least one completion token, \
                     got prompt_len={prompt_len} and sequence_len={sequence_len}. \
                     Hint: set --rollout-len > 0 or use --opd-kl-mask full."
                )));
            }
            Ok(KlLogitRange {
                start: prompt_len - 1,
                end: sequence_len - 1,
            })
        }
    }
}

pub(super) fn slice_logits_for_kl(
    logits: TensorId,
    range: KlLogitRange,
    vocab: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let starts = [0, range.start, 0];
    let ends = [1, range.end, vocab];
    slice(logits, &starts, &ends, store, tape).map_err(OpdError::from)
}

/// `masked_positions` are the predicting positions `p` (sorted ascending) whose
/// next token is LLM-generated — the ones that receive KL loss. Tool/environment
/// tokens leave gaps, so consecutive `p`s form runs; scoring one run's logit tile
/// at a time keeps tool-token positions out of the loss (mirroring the masked-CE
/// path) while bounding each `[1, window, vocab]` tile to `window_size` rows.
pub(super) fn masked_gkd_windows(
    masked_positions: &[usize],
    window_size: usize,
) -> Vec<SequenceWindow> {
    let mut windows = Vec::new();
    let mut i = 0;
    while i < masked_positions.len() {
        let start = masked_positions[i];
        let mut end = start + 1;
        let mut j = i + 1;
        while j < masked_positions.len() && masked_positions[j] == end {
            end += 1;
            j += 1;
        }
        let mut s = start;
        while s < end {
            let e = (s + window_size).min(end);
            windows.push(SequenceWindow { start: s, end: e });
            s = e;
        }
        i = j;
    }
    windows
}
