use anyhow::Result;

use crate::server_engine::FinishReason;
use crate::types::{BlockId, RequestId};

use super::DraftMode;

/// Logical per-row decode work selected by the shared scheduler.
///
/// The row describes what to execute, not how a backend should locate or
/// launch KV. CUDA lowering re-derives paged block tables from `req_id`; Metal
/// uses `logical_kv_offset` for packed-varlen offsets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalDecodeRow {
    pub row_index: usize,
    pub req_id: RequestId,
    pub input_token: u32,
    pub logical_kv_offset: usize,
}

impl LogicalDecodeRow {
    pub fn new(
        row_index: usize,
        req_id: RequestId,
        input_token: u32,
        logical_kv_offset: usize,
    ) -> Self {
        Self {
            row_index,
            req_id,
            input_token,
            logical_kv_offset,
        }
    }
}

/// Logical prefill chunk selected for one scheduler tick.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalPrefillRow {
    pub row_index: usize,
    pub req_id: RequestId,
    pub input_tokens: Vec<u32>,
    pub prompt_start: usize,
    pub prompt_end: usize,
    pub prompt_len: usize,
    pub logical_kv_start: usize,
    pub logical_kv_end: usize,
}

impl LogicalPrefillRow {
    pub fn new(
        row_index: usize,
        req_id: RequestId,
        input_tokens: Vec<u32>,
        prompt_start: usize,
        prompt_end: usize,
        prompt_len: usize,
    ) -> Self {
        Self::with_logical_kv_span(
            row_index,
            req_id,
            input_tokens,
            prompt_start,
            prompt_end,
            prompt_len,
            prompt_start,
            prompt_end,
        )
    }

    pub fn with_logical_kv_span(
        row_index: usize,
        req_id: RequestId,
        input_tokens: Vec<u32>,
        prompt_start: usize,
        prompt_end: usize,
        prompt_len: usize,
        logical_kv_start: usize,
        logical_kv_end: usize,
    ) -> Self {
        Self {
            row_index,
            req_id,
            input_tokens,
            prompt_start,
            prompt_end,
            prompt_len,
            logical_kv_start,
            logical_kv_end,
        }
    }

    pub fn token_count(&self) -> usize {
        self.input_tokens.len()
    }
}

/// CPU-visible logical shape for one scheduler tick.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LogicalBatchShape {
    pub decode_rows: usize,
    pub prefill_rows: usize,
    pub total_rows: usize,
    pub decode_tokens: usize,
    pub prefill_tokens: usize,
    pub scheduled_tokens: usize,
}

impl LogicalBatchShape {
    pub fn from_rows(decode_rows: &[LogicalDecodeRow], prefill_rows: &[LogicalPrefillRow]) -> Self {
        let decode_tokens = decode_rows.len();
        let prefill_tokens = prefill_rows
            .iter()
            .map(LogicalPrefillRow::token_count)
            .sum::<usize>();
        Self {
            decode_rows: decode_rows.len(),
            prefill_rows: prefill_rows.len(),
            total_rows: decode_rows.len() + prefill_rows.len(),
            decode_tokens,
            prefill_tokens,
            scheduled_tokens: decode_tokens + prefill_tokens,
        }
    }
}

/// Unified logical serving plan for one scheduler tick.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LogicalServePlan {
    pub decode_rows: Vec<LogicalDecodeRow>,
    pub prefill_rows: Vec<LogicalPrefillRow>,
    pub batch_shape: LogicalBatchShape,
    pub spec_rows: Vec<LogicalSpecDecodeRow>,
    pub sparse_views: Vec<LogicalSparseDraftView>,
}

impl LogicalServePlan {
    pub fn new(decode_rows: Vec<LogicalDecodeRow>, prefill_rows: Vec<LogicalPrefillRow>) -> Self {
        let batch_shape = LogicalBatchShape::from_rows(&decode_rows, &prefill_rows);
        Self {
            decode_rows,
            prefill_rows,
            batch_shape,
            spec_rows: Vec::new(),
            sparse_views: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_spec_rows(mut self, spec_rows: Vec<LogicalSpecDecodeRow>) -> Self {
        self.spec_rows = spec_rows;
        self
    }

    #[must_use]
    pub fn with_sparse_views(mut self, sparse_views: Vec<LogicalSparseDraftView>) -> Self {
        self.sparse_views = sparse_views;
        self
    }

    pub fn idle() -> Self {
        Self::default()
    }

    pub fn is_idle(&self) -> bool {
        self.decode_rows.is_empty()
            && self.prefill_rows.is_empty()
            && self.spec_rows.is_empty()
            && self.sparse_views.is_empty()
    }
}

/// Spec-decode annotation for a selected decode row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalSpecDecodeRow {
    pub decode_row_index: usize,
    pub draft_tokens: Vec<u32>,
    pub draft_mode: DraftMode,
}

impl LogicalSpecDecodeRow {
    pub fn new(decode_row_index: usize, draft_tokens: Vec<u32>, draft_mode: DraftMode) -> Self {
        Self {
            decode_row_index,
            draft_tokens,
            draft_mode,
        }
    }
}

/// Sparse-KV draft-view annotation for a selected decode row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalSparseDraftView {
    pub decode_row_index: usize,
    pub allowed_block_ids: Vec<BlockId>,
    pub recent_window_tokens: usize,
}

impl LogicalSparseDraftView {
    pub fn new(
        decode_row_index: usize,
        allowed_block_ids: Vec<BlockId>,
        recent_window_tokens: usize,
    ) -> Self {
        Self {
            decode_row_index,
            allowed_block_ids,
            recent_window_tokens,
        }
    }
}

/// One generated token returned by a backend lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeneratedToken {
    pub token_id: u32,
}

impl GeneratedToken {
    pub fn new(token_id: u32) -> Self {
        Self { token_id }
    }
}

/// Backend lowering contract for a logical scheduler plan.
pub trait LogicalPlanLowering {
    fn execute(&mut self, plan: &LogicalServePlan) -> Result<LogicalStepOutput>;
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LogicalStepOutput {
    pub generated_tokens: Vec<(usize, GeneratedToken)>,
    pub finished_rows: Vec<(usize, FinishReason)>,
    pub spec_acceptance: Vec<(usize, usize, usize)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_plan_computes_batch_shape() {
        let plan = LogicalServePlan::new(
            vec![
                LogicalDecodeRow::new(0, RequestId(7), 111, 32),
                LogicalDecodeRow::new(1, RequestId(8), 222, 48),
            ],
            vec![LogicalPrefillRow::new(
                2,
                RequestId(9),
                vec![1, 2, 3, 4],
                8,
                12,
                20,
            )],
        );

        assert_eq!(
            plan.batch_shape,
            LogicalBatchShape {
                decode_rows: 2,
                prefill_rows: 1,
                total_rows: 3,
                decode_tokens: 2,
                prefill_tokens: 4,
                scheduled_tokens: 6,
            }
        );
        assert!(!plan.is_idle());
    }

    #[test]
    fn logical_prefill_row_preserves_backend_neutral_kv_span() {
        let row = LogicalPrefillRow::with_logical_kv_span(
            0,
            RequestId(3),
            vec![10, 11],
            4,
            6,
            16,
            128,
            130,
        );

        assert_eq!(row.prompt_start, 4);
        assert_eq!(row.prompt_end, 6);
        assert_eq!(row.logical_kv_start, 128);
        assert_eq!(row.logical_kv_end, 130);
        assert_eq!(row.token_count(), 2);
    }

    #[test]
    fn logical_lowering_round_trips_rows_in_plan_order() {
        #[derive(Default)]
        struct RecordingLowering {
            seen: Vec<LogicalServePlan>,
        }

        impl LogicalPlanLowering for RecordingLowering {
            fn execute(&mut self, plan: &LogicalServePlan) -> Result<LogicalStepOutput> {
                self.seen.push(plan.clone());
                Ok(LogicalStepOutput {
                    generated_tokens: plan
                        .decode_rows
                        .iter()
                        .map(|row| (row.row_index, GeneratedToken::new(row.input_token + 1)))
                        .collect(),
                    finished_rows: vec![(2, FinishReason::Length)],
                    spec_acceptance: plan
                        .spec_rows
                        .iter()
                        .map(|row| {
                            (
                                row.decode_row_index,
                                row.draft_tokens.len(),
                                row.draft_tokens.len(),
                            )
                        })
                        .collect(),
                })
            }
        }

        let plan = LogicalServePlan::new(
            vec![
                LogicalDecodeRow::new(0, RequestId(1), 40, 11),
                LogicalDecodeRow::new(1, RequestId(2), 50, 13),
            ],
            vec![LogicalPrefillRow::new(
                2,
                RequestId(3),
                vec![60, 61, 62],
                0,
                3,
                9,
            )],
        )
        .with_spec_rows(vec![LogicalSpecDecodeRow::new(
            1,
            vec![51, 52],
            DraftMode::SelfSpec,
        )])
        .with_sparse_views(vec![LogicalSparseDraftView::new(
            0,
            vec![BlockId(4), BlockId(5)],
            16,
        )]);

        let mut lowering = RecordingLowering::default();
        let output = lowering.execute(&plan).expect("lowering succeeds");

        assert_eq!(lowering.seen, vec![plan]);
        assert_eq!(
            output.generated_tokens,
            vec![(0, GeneratedToken::new(41)), (1, GeneratedToken::new(51)),]
        );
        assert_eq!(output.finished_rows, vec![(2, FinishReason::Length)]);
        assert_eq!(output.spec_acceptance, vec![(1, 2, 2)]);
    }

    #[test]
    fn idle_plan_has_empty_shape() {
        let plan = LogicalServePlan::idle();

        assert!(plan.is_idle());
        assert_eq!(plan.batch_shape, LogicalBatchShape::default());
    }
}
