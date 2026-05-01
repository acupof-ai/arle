//! Tests for `request_state.rs`.
//!
//! Split out of `request_state.rs` (pure structural refactor — no behavior change).

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::super::*;
    use crate::backend::metal::mlx::{Dtype, MlxArray, eval};
    use crate::test_support::metal_test_guard;

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

    #[test]
    fn qwen35_prefix_snapshot_disk_payload_roundtrips_arrays() {
        let _guard = metal_test_guard();
        let snapshot = Qwen35PrefixSnapshot {
            token_ids: vec![101, 102],
            kv_flat: vec![MlxArray::from_slice_i32(&[10, 20], &[2])],
            gdr_flat: vec![MlxArray::from_slice_f32(&[1.5, 2.5], &[2])],
            cache_len: 2,
            kv_capacity: 4,
        };

        let payload = snapshot.encode_for_disk(b"qwen35-test").expect("encode");
        assert_eq!(
            snapshot
                .estimated_disk_payload_len(b"qwen35-test")
                .expect("estimate"),
            payload.len() as u64
        );
        let restored =
            Qwen35PrefixSnapshot::decode_from_disk(&payload, b"qwen35-test").expect("decode");
        eval(&[&restored.kv_flat[0], &restored.gdr_flat[0]]);

        assert_eq!(restored.token_ids, vec![101, 102]);
        assert_eq!(restored.cache_len, 2);
        assert_eq!(restored.kv_capacity, 4);
        assert_eq!(restored.kv_flat.len(), 1);
        assert_eq!(restored.gdr_flat.len(), 1);
        assert_eq!(restored.kv_flat[0].shape(), &[2]);
        assert_eq!(restored.kv_flat[0].dtype(), Dtype::Int32);
        assert_eq!(restored.kv_flat[0].as_slice_i32(), vec![10, 20]);
        assert_eq!(restored.gdr_flat[0].shape(), &[2]);
        assert_eq!(restored.gdr_flat[0].dtype(), Dtype::Float32);
        assert_eq!(restored.gdr_flat[0].as_slice_f32(), &[1.5, 2.5]);
    }

    #[test]
    fn qwen35_prefix_snapshot_disk_payload_rejects_wrong_model_fingerprint() {
        let _guard = metal_test_guard();
        let snapshot = Qwen35PrefixSnapshot {
            token_ids: vec![101],
            kv_flat: vec![MlxArray::from_slice_i32(&[10], &[1])],
            gdr_flat: Vec::new(),
            cache_len: 1,
            kv_capacity: 1,
        };

        let payload = snapshot.encode_for_disk(b"qwen35-a").expect("encode");
        assert!(Qwen35PrefixSnapshot::decode_from_disk(&payload, b"qwen35-b").is_err());
    }

    #[test]
    fn qwen35_prefix_snapshot_disk_payload_rejects_corrupt_body() {
        let _guard = metal_test_guard();
        let snapshot = Qwen35PrefixSnapshot {
            token_ids: vec![101],
            kv_flat: vec![MlxArray::from_slice_i32(&[10], &[1])],
            gdr_flat: Vec::new(),
            cache_len: 1,
            kv_capacity: 1,
        };

        let mut payload = snapshot.encode_for_disk(b"qwen35-a").expect("encode");
        let last = payload.last_mut().expect("payload body byte");
        *last ^= 0x01;

        let err = match Qwen35PrefixSnapshot::decode_from_disk(&payload, b"qwen35-a") {
            Ok(_) => panic!("corrupt body should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("checksum"),
            "unexpected corrupt-body error: {err:#}"
        );
    }

    #[test]
    fn qwen35_prefix_snapshot_metadata_checksum_covers_header_fields() {
        let _guard = metal_test_guard();
        let snapshot = Qwen35PrefixSnapshot {
            token_ids: vec![101],
            kv_flat: vec![MlxArray::from_slice_i32(&[10], &[1])],
            gdr_flat: Vec::new(),
            cache_len: 1,
            kv_capacity: 1,
        };

        let payload = snapshot.encode_for_disk(b"qwen35-a").expect("encode");
        let (mut header, _body) =
            decode_qwen35_prefix_snapshot_header(&payload, b"qwen35-a", false)
                .expect("decode header");
        header.kv_capacity += 1;

        let err = validate_qwen35_prefix_snapshot_metadata_checksum(&header)
            .expect_err("tampered header metadata should fail");
        assert!(
            err.to_string().contains("metadata checksum"),
            "unexpected metadata-checksum error: {err:#}"
        );
    }

    #[test]
    fn qwen35_prefix_snapshot_disk_payload_rejects_wrong_model_before_body() {
        let _guard = metal_test_guard();
        let snapshot = Qwen35PrefixSnapshot {
            token_ids: vec![101],
            kv_flat: vec![MlxArray::from_slice_i32(&[10], &[1])],
            gdr_flat: Vec::new(),
            cache_len: 1,
            kv_capacity: 1,
        };

        let mut payload = snapshot.encode_for_disk(b"qwen35-a").expect("encode");
        payload.pop().expect("payload body byte");

        let wrong_model = match Qwen35PrefixSnapshot::decode_from_disk(&payload, b"qwen35-b") {
            Ok(_) => panic!("wrong model should be rejected"),
            Err(err) => err,
        };
        assert!(
            wrong_model.to_string().contains("model fingerprint"),
            "unexpected wrong-model error: {wrong_model:#}"
        );

        let truncated_body = match Qwen35PrefixSnapshot::decode_from_disk(&payload, b"qwen35-a") {
            Ok(_) => panic!("truncated body should be rejected"),
            Err(err) => err,
        };
        assert!(
            truncated_body.to_string().contains("truncated"),
            "unexpected truncated-body error: {truncated_body:#}"
        );
    }

    #[test]
    fn qwen35_prefix_snapshot_disk_payload_peeks_tokens_without_import() {
        let _guard = metal_test_guard();
        let snapshot = Qwen35PrefixSnapshot {
            token_ids: vec![101, 102, 103],
            kv_flat: vec![MlxArray::from_slice_i32(&[10, 20, 30], &[3])],
            gdr_flat: Vec::new(),
            cache_len: 3,
            kv_capacity: 4,
        };

        let payload = snapshot.encode_for_disk(b"qwen35-a").expect("encode");
        let token_ids =
            Qwen35PrefixSnapshot::peek_disk_token_ids(&payload, b"qwen35-a").expect("peek");
        assert_eq!(token_ids, vec![101, 102, 103]);

        let err = Qwen35PrefixSnapshot::peek_disk_token_ids(&payload, b"qwen35-b")
            .expect_err("wrong model should fail");
        assert!(
            err.to_string().contains("model fingerprint"),
            "unexpected peek error: {err:#}"
        );
    }

    #[test]
    fn longest_reusable_aligned_prefix_keeps_full_aligned_prompt() {
        assert_eq!(longest_reusable_aligned_prefix_len(32, 32, 16), 32);
        assert_eq!(longest_reusable_aligned_prefix_len(33, 33, 16), 32);
        assert_eq!(longest_reusable_aligned_prefix_len(15, 15, 16), 0);
        assert_eq!(longest_reusable_aligned_prefix_len(48, 40, 16), 32);
        assert_eq!(longest_reusable_aligned_prefix_len(32, 32, 0), 0);
    }

    #[test]
    fn qwen35_disk_publish_prefix_lens_include_importable_block_aligned_fallback() {
        assert_eq!(qwen35_disk_publish_prefix_lens(32, 32, 16), vec![16, 32]);
        assert_eq!(qwen35_disk_publish_prefix_lens(33, 33, 16), vec![32]);
        assert_eq!(qwen35_disk_publish_prefix_lens(16, 16, 16), vec![16]);
        assert_eq!(
            qwen35_disk_publish_prefix_lens(15, 15, 16),
            Vec::<usize>::new()
        );
        assert_eq!(qwen35_disk_publish_prefix_lens(48, 40, 16), vec![32]);
        assert_eq!(
            qwen35_disk_publish_prefix_lens(32, 32, 0),
            Vec::<usize>::new()
        );
    }
}
