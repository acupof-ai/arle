//! The verify-path selftest gate: prove `forward_tokens_verify` agrees with
//! `forward_tokens` and is insensitive to a forced-wrong draft.

use super::*;

impl Dsv4CudaExecutor {
    /// The selftest drives `forward_tokens` directly (no `prepare_kv_batch`), so
    /// the band and device page tables must be materialized here.
    fn selftest_prefill(
        &mut self,
        slot_idx: usize,
        prompt: &[u32],
        params: &SamplingParams,
    ) -> Result<u32> {
        self.kv_adapter.flashmla_free_slot(slot_idx)?;
        self.slots[slot_idx].reset(&self.model.ctx, &mut self.kv_adapter)?;
        self.kv_adapter.prepare_direct_forward(
            &self.model.ctx,
            slot_idx,
            prompt.len() + crate::dsv4::MAX_SPEC_DRAFT_DEPTH + 2,
        )?;
        self.kv_adapter.zero_slot_band(&self.model.ctx, slot_idx)?;
        if self.kv_adapter.take_device_table_dirty(slot_idx) {
            self.slots[slot_idx]
                .refresh_flashmla_device_page_tables(&self.model.ctx, &self.kv_adapter)?;
        }
        self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            prompt,
            0,
            params,
            prompt.len() as u64,
            infer_plan::PenaltyHistory::default(),
        )
    }

    pub(crate) fn verify_forward_selftest(&mut self, prompt: &[u32]) -> Result<()> {
        ensure!(
            !prompt.is_empty(),
            "DSv4 verify-forward selftest requires a non-empty prompt"
        );
        let slot_idx = 0;
        let params = SamplingParams::default();
        let start_pos = prompt.len();

        let token_a = self.selftest_prefill(slot_idx, prompt, &params)?;
        let verify_one = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a],
            start_pos,
            (start_pos + 1) as u64,
        )?;

        let token_a_again = self.selftest_prefill(slot_idx, prompt, &params)?;
        ensure!(
            token_a == token_a_again,
            "DSv4 verify selftest prefill token drifted: {token_a} != {token_a_again}"
        );
        let normal_one = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a],
            start_pos,
            &params,
            (start_pos + 1) as u64,
            infer_plan::PenaltyHistory::default(),
        )?;
        ensure!(
            verify_one.argmax.first().copied() == Some(normal_one),
            "DSv4 verify selftest one-token mismatch: verify={:?} normal={normal_one}",
            verify_one.argmax
        );

        let token_a = self.selftest_prefill(slot_idx, prompt, &params)?;
        let verify_one = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a],
            start_pos,
            (start_pos + 1) as u64,
        )?;
        let token_b = verify_one.argmax[0];
        let mut wrong_b = token_b.wrapping_add(2);
        if wrong_b == token_b {
            wrong_b = token_b.wrapping_add(3);
        }

        let token_a = self.selftest_prefill(slot_idx, prompt, &params)?;
        let verify_two = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a, wrong_b],
            start_pos,
            (start_pos + 1) as u64,
        )?;
        ensure!(
            verify_two.argmax.first() == verify_one.argmax.first(),
            "DSv4 verify selftest two-token row0 mismatch: one={:?} two={:?}",
            verify_one.argmax,
            verify_two.argmax
        );

        // No col1/bonus gate here: col1 on a forced-wrong draft is discarded in
        // real decode, and comparing it is confounded by the M=2-vs-M=1 FP8
        // kernel path. See errors/2026-06-08-dsv4-batched-verify-col1-wrong.md.

        self.kv_adapter.flashmla_free_slot(slot_idx)?;
        self.slots[slot_idx].reset(&self.model.ctx, &mut self.kv_adapter)?;
        self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
        if self.model.tp.config().rank == 0 {
            eprintln!(
                "[dsv4-mtp-selftest] PASS token_a={token_a} token_b={token_b} wrong_b={wrong_b} verify_two={:?}",
                verify_two.argmax
            );
        }
        Ok(())
    }
}
