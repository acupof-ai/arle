/// A loaded Qwen3.5 hybrid model with all weights resident on the Vulkan device.
///
/// `ResidentWeights<'a>`'s `DeviceBuffer`s borrow the [`vulkan_sys::VulkanContext`]
/// they were allocated on, so the context must outlive every forward call. The
/// model lives until process exit, so we store a **leaked** `&'static` context
/// (`Box::leak`) and pin the resident weights to that `'static` lifetime — the
/// context (and its GPU allocations) are released only when the process ends.
#[cfg(feature = "vulkan")]
pub struct VulkanQwen35Model {
    pub config: qwen35_spec::Qwen35Config,
    ctx: &'static vulkan_sys::VulkanContext,
    weights: crate::loader::upload::ResidentWeights<'static>,
    state: crate::forward::Qwen35ForwardState,
    /// The tokens materialized into `state`, in order — the identity of the
    /// resident sequence and the whole basis for prefix reuse. Kept instead of
    /// the old `(slot, epoch)` key because epoch changes on every new request,
    /// including the ones that CONTINUE this sequence and whose entire purpose
    /// is to keep the state. Always the same length as `state.seq_len`.
    resident_tokens: Vec<u32>,
    /// The slot `resident_tokens` belongs to. One lane serves every request in
    /// turn, so another slot must never resume onto this sequence's state.
    resident_slot: Option<usize>,
    /// Persistent decode resources (perf-parity Steps 3+4): the GEMV activation
    /// arena, the compile-once `KernelCache`, and the record-many/submit-once
    /// `CommandRecorder`. Built once in [`Self::load`] and threaded into every
    /// `forward_token` so the hot path never re-allocs scratch, rebuilds a
    /// pipeline, or drains the queue per op.
    decode: crate::forward::DecodeResources<'static>,
}

#[cfg(feature = "vulkan")]
impl VulkanQwen35Model {
    /// Load every Qwen3.5 weight resident onto `ctx`'s device.
    ///
    /// Maps the GGUF metadata to a [`qwen35_spec::Qwen35Config`], plans the
    /// per-tensor residency for all `num_hidden_layers` blocks, and uploads the
    /// plan (K-quant/Q8_0 packed, F16/BF16→F16, F32→F32, token_embd host-side).
    /// `ctx` must be `'static` (leak it at the call site) so the resident
    /// `DeviceBuffer`s outlive any single forward call.
    pub fn load(
        ctx: &'static vulkan_sys::VulkanContext,
        gguf: &infer_gguf::gguf::GgufFile,
    ) -> anyhow::Result<Self> {
        let config = crate::config::qwen35_config_from_gguf(gguf)?;
        let plan = crate::loader::plan_model(gguf, config.num_hidden_layers)?;
        let weights = crate::loader::upload::upload_plan(ctx, gguf, &plan)?;
        let state = crate::forward::Qwen35ForwardState::new(&config);
        let decode = crate::forward::DecodeResources::new(ctx, &config)?;
        Ok(Self {
            config,
            ctx,
            weights,
            state,
            resident_tokens: Vec::new(),
            resident_slot: None,
            decode,
        })
    }

    /// Reset the per-slot recurrent + KV state for a fresh generation. Zeros both
    /// the host `Qwen35ForwardState` (the host linear-attention oracle path) and
    /// the device-resident gated-delta + conv state (the on-device path), so a
    /// fresh sequence starts clean regardless of which path is selected.
    pub fn reset_state(&mut self) {
        self.state.reset();
        self.resident_tokens.clear();
        if let Err(e) = self.decode.reset_linear_state() {
            panic!("reset device linear state: {e}");
        }
    }

    /// Drain the accumulated GEMV timing `(submit_secs, other_secs, gemv_count)`
    /// from the decode resources and reset it — lets a timed decode attribute
    /// time between the GPU submits and the host prep/readback around them.
    pub fn take_decode_profile(&mut self) -> (f64, f64, u64) {
        self.decode.take_profile()
    }

    pub fn decode_submit_count(&self) -> u64 {
        self.decode.submit_count()
    }

    /// Number of device-resident weight tensors (token_embd is host-side and not
    /// counted). Lets a load smoke-test assert the model actually landed.
    pub fn resident_tensor_count(&self) -> usize {
        self.weights.tensors.len()
    }

    pub fn resident_device_bytes(&self) -> u64 {
        self.weights
            .tensors
            .values()
            .map(|t| t.buffer.len() as u64)
            .sum()
    }

    pub fn device_name(&self) -> &str {
        self.ctx.device_name()
    }

    /// Numeric forward for one token at `start_pos`, returning logits `[vocab]`.
    ///
    /// Heavy matmuls run on-device (proven Q8_0 GEMV); the elementwise / norm /
    /// attention / gated-delta math runs on the host in f32. See
    /// [`crate::forward`] for the contract: `start_pos` must equal the
    /// materialized sequence length (advanced here), so feed a sequence's
    /// tokens in order.
    ///
    /// `start_pos == 0` means "a new sequence starts here" and resets the
    /// carried state. That is the only reset trigger: keying it off `epoch`
    /// (as this did before) also reset on the prefix-RESTORE path, where the
    /// engine hands a fresh request a `start_pos` in the middle of the sequence
    /// this lane already holds and the entire point is to keep the state.
    /// `epoch` is now unused — `resident_slot` plus the position check below
    /// carry the same protection.
    pub fn forward_token(
        &mut self,
        slot: usize,
        _epoch: u64,
        token: u32,
        start_pos: usize,
    ) -> anyhow::Result<Vec<f32>> {
        if start_pos == 0 {
            // Without this a second request reuses slot 0 with start_pos 0 while
            // the state still carries the first request's seq_len, and the length
            // check in `forward::forward_token` takes the whole server down.
            self.reset_state();
            self.resident_slot = Some(slot);
        } else if self.resident_slot != Some(slot) {
            // A resume can only ever be a resume of THIS lane's one sequence.
            anyhow::bail!(
                "Vulkan lane holds slot {:?}'s sequence; slot {slot} asked to \
                 resume at {start_pos}",
                self.resident_slot
            );
        }
        let logits = crate::forward::forward_token(
            self.ctx,
            &self.config,
            &self.weights,
            &mut self.decode,
            &mut self.state,
            token,
            start_pos,
        )?;
        self.resident_tokens.push(token);
        Ok(logits)
    }

    /// Batched prefill: materialize `tokens` starting at `start_pos` in one
    /// GEMM-shaped pass and return the logits of the LAST token only.
    ///
    /// Same contract and same bookkeeping as calling [`Self::forward_token`] once
    /// per token — `resident_tokens` / `resident_slot` advance identically, so
    /// prefix reuse ([`Self::cached_prefix_match_len`]) is unaffected. The
    /// difference is arithmetic intensity: each weight is read once per CHUNK
    /// instead of once per token, turning the memory-bound GEMV loop into a
    /// compute-bound `mul_mmq`.
    ///
    /// Returns `Ok(None)` when the batched path does not cover this model (MoE
    /// layers, or a linear layer whose `ssm_alpha`/`ssm_beta` is not resident
    /// quantized) — the caller must then fall back to the per-token loop. That
    /// check happens BEFORE anything is recorded: a mid-chunk bail would be
    /// unrecoverable, since the KV cache and the gated-delta state advance in
    /// place.
    pub fn forward_tokens(
        &mut self,
        slot: usize,
        _epoch: u64,
        tokens: &[u32],
        start_pos: usize,
    ) -> anyhow::Result<Option<Vec<f32>>> {
        if tokens.is_empty() {
            return Ok(None);
        }
        if let Some(reason) =
            crate::prefill::prefill_unsupported_reason(&self.config, &self.weights)
        {
            // Once per process: the caller's fallback is silent, and "batched
            // declined" looks exactly like "batched ran and was slow" in a
            // wall-clock A/B.
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                log::warn!("vulkan batched prefill unavailable, using per-token loop: {reason}");
            });
            return Ok(None);
        }
        if start_pos == 0 {
            self.reset_state();
            self.resident_slot = Some(slot);
        } else if self.resident_slot != Some(slot) {
            anyhow::bail!(
                "Vulkan lane holds slot {:?}'s sequence; slot {slot} asked to \
                 resume at {start_pos}",
                self.resident_slot
            );
        }
        let logits = crate::prefill::forward_prefill(
            self.ctx,
            &self.config,
            &self.weights,
            &mut self.decode,
            &mut self.state,
            tokens,
            start_pos,
        )?;
        self.resident_tokens.extend_from_slice(tokens);
        Ok(Some(logits))
    }

    /// Length of the longest leading prefix of `tokens` this lane can resume
    /// from without recomputing it — the position-0 reuse seam
    /// ([`infer_seam::PrefixReuse::cached_prefix_match_len`]).
    ///
    /// It is all-or-nothing: either the whole resident sequence, or zero. The
    /// full-attention KV is positional and genuinely holds `[0, len)`, but the
    /// gated-delta recurrence is a running fold with no rewind and no snapshot,
    /// so state that has already consumed a diverging token is worth nothing —
    /// a PARTIAL match cannot be served, only a continuation.
    ///
    /// An exact match returns 0 as well: the caller must still forward at least
    /// one token to sample from, and rewinding by that one token is the same
    /// impossible rewind.
    pub fn cached_prefix_len(&self, tokens: &[u32]) -> usize {
        let len = self.resident_tokens.len();
        if self.resident_slot.is_none() || len == 0 || len >= tokens.len() {
            return 0;
        }
        if tokens[..len] == self.resident_tokens[..] {
            len
        } else {
            0
        }
    }

    /// Adopt the resident sequence as `slot`'s restored prefix of length
    /// `matched_len`. Nothing moves: the KV planes and the recurrent state
    /// already ARE what this prefix produced. Re-derives the match rather than
    /// trusting `matched_len`, since accepting a wrong one would decode against
    /// another conversation's recurrence.
    pub fn adopt_cached_prefix(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
    ) -> anyhow::Result<()> {
        let actual = self.cached_prefix_len(tokens);
        anyhow::ensure!(
            actual == matched_len && matched_len > 0,
            "cached-prefix restore for slot {slot} asked for {matched_len} tokens, \
             lane holds {actual}"
        );
        self.resident_slot = Some(slot);
        Ok(())
    }

    /// Materialize the one token a finishing request sampled but never fed, so
    /// the resident sequence covers `tokens` exactly and the next turn resumes
    /// past the whole generated region instead of stopping one short of it.
    ///
    /// Deliberately narrow: it runs a single forward, and only when the lane is
    /// exactly one token behind `tokens`. Anything further behind means this is
    /// not our sequence (or the engine re-planned), and catching up would cost
    /// a full prefill at finish time for a prefix that may never be reused.
    pub fn materialize_finish(&mut self, slot: usize, tokens: &[u32]) -> anyhow::Result<()> {
        let len = self.resident_tokens.len();
        if self.resident_slot != Some(slot)
            || tokens.len() != len + 1
            || tokens[..len] != self.resident_tokens[..]
        {
            return Ok(());
        }
        self.forward_token(slot, 0, tokens[len], len)?;
        Ok(())
    }
}
