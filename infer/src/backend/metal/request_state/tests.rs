//! Tests for `request_state.rs`.
//!
//! Split out of `request_state.rs` (pure structural refactor — no behavior change).

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct FakeDriver {
        prefill_outputs: VecDeque<Option<u32>>,
        decode_outputs: VecDeque<u32>,
        cleanup_calls: Arc<AtomicUsize>,
    }

    impl FakeDriver {
        fn new(
            prefill_outputs: impl IntoIterator<Item = Option<u32>>,
            decode_outputs: impl IntoIterator<Item = u32>,
            cleanup_calls: Arc<AtomicUsize>,
        ) -> Self {
            Self {
                prefill_outputs: prefill_outputs.into_iter().collect(),
                decode_outputs: decode_outputs.into_iter().collect(),
                cleanup_calls,
            }
        }
    }

    impl StepDriver for FakeDriver {
        fn prefill_token(&mut self, _token: u32, _terminal_prompt: bool) -> Result<Option<u32>> {
            self.prefill_outputs
                .pop_front()
                .context("missing fake prefill output")
        }

        fn decode_token(&mut self, _token: u32) -> Result<u32> {
            self.decode_outputs
                .pop_front()
                .context("missing fake decode output")
        }

        fn cleanup(&mut self) -> Result<()> {
            self.cleanup_calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn prefill_chunk_only_emits_a_token_once_the_prompt_finishes() {
        let cleanup_calls = Arc::new(AtomicUsize::new(0));
        let mut state = ResumableRequestState::new(
            FakeDriver::new([None, None, None, Some(41)], [], cleanup_calls.clone()),
            vec![1, 2, 3, 4],
            3,
            vec![],
            99,
            false,
        )
        .expect("state");

        let first = state.prefill_chunk(2).expect("chunk 1");
        assert_eq!(
            first,
            PrefillChunkResult {
                processed_tokens: 2,
                emitted_token: None,
                phase: MetalRequestPhase::Prefill,
                finish_reason: None,
            }
        );
        assert_eq!(state.prompt_progress(), 2);
        assert_eq!(state.generated_tokens(), 0);
        assert_eq!(cleanup_calls.load(Ordering::Relaxed), 0);

        let second = state.prefill_chunk(2).expect("chunk 2");
        assert_eq!(second.processed_tokens, 2);
        assert_eq!(second.emitted_token, Some(41));
        assert_eq!(second.phase, MetalRequestPhase::Decode);
        assert_eq!(state.prompt_progress(), 4);
        assert_eq!(state.generated_tokens(), 1);
        assert_eq!(cleanup_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn length_finish_cleans_up_after_decode_step() {
        let cleanup_calls = Arc::new(AtomicUsize::new(0));
        let mut state = ResumableRequestState::new(
            FakeDriver::new([Some(7)], [8], cleanup_calls.clone()),
            vec![1],
            2,
            vec![],
            99,
            false,
        )
        .expect("state");

        let prefill = state.prefill_chunk(1).expect("prefill");
        assert_eq!(prefill.emitted_token, Some(7));
        assert_eq!(state.phase(), MetalRequestPhase::Decode);
        assert_eq!(state.generated_tokens(), 1);
        assert_eq!(cleanup_calls.load(Ordering::Relaxed), 0);

        let decoded = state.decode_step().expect("decode");
        assert_eq!(decoded, Some(8));
        assert_eq!(state.phase(), MetalRequestPhase::Finished);
        assert_eq!(state.finish_reason(), Some("length"));
        assert_eq!(state.generated_tokens(), 2);
        assert_eq!(cleanup_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn stop_token_from_prefill_finishes_immediately() {
        let cleanup_calls = Arc::new(AtomicUsize::new(0));
        let mut state = ResumableRequestState::new(
            FakeDriver::new([Some(42)], [], cleanup_calls.clone()),
            vec![5],
            4,
            vec![42],
            99,
            false,
        )
        .expect("state");

        let prefill = state.prefill_chunk(1).expect("prefill");
        assert_eq!(prefill.emitted_token, Some(42));
        assert_eq!(prefill.phase, MetalRequestPhase::Finished);
        assert_eq!(prefill.finish_reason, Some("stop"));
        assert_eq!(state.finish_reason(), Some("stop"));
        assert_eq!(cleanup_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn dropping_incomplete_state_still_runs_cleanup_once() {
        let cleanup_calls = Arc::new(AtomicUsize::new(0));
        {
            let state = ResumableRequestState::new(
                FakeDriver::new([None, None], [], cleanup_calls.clone()),
                vec![1, 2],
                4,
                vec![],
                99,
                false,
            )
            .expect("state");
            assert_eq!(state.phase(), MetalRequestPhase::Prefill);
        }

        assert_eq!(cleanup_calls.load(Ordering::Relaxed), 1);
    }
}
