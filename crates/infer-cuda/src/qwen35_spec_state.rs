use super::*;

/// Per-request speculative-decode state for the Qwen3.6 NextN-MTP draft head.
///
/// Holds (a) the draft head's **fresh per-block** K/V cache — the head is a
/// single full-attention layer that attends only over the current draft chain,
/// seeded each block from the last-accepted trunk hidden (the trunk context is
/// baked into that hidden via the `fc` concat, not re-attended); and (b) the
/// pre-verify **snapshot** of the trunk's linear-attn recurrent + conv state,
/// restored on a rejected draft. Allocated only when spec-decode is on, one per
/// concurrent slot, so the baseline decode path never pays for it.
#[allow(dead_code)] // head_k/head_v read by mtp_forward_level; snap by spec_step (next increments)
pub(crate) struct Qwen35SpecSlotState {
    /// `(depth+1)*kv_dim` bf16, rewritten each draft block.
    pub(crate) head_k: DeviceVec,
    pub(crate) head_v: DeviceVec,
    /// Pre-verify snapshot of trunk linear-attn recurrent states, sized like [`Qwen35SlotState::gdr_states`].
    pub(crate) gdr_snap: Vec<CudaSlice<f32>>,
    /// Pre-verify snapshot of trunk linear-attn conv rings (bf16).
    pub(crate) conv_snap: Vec<DeviceVec>,
    /// Per-linear-layer capture of verify forward's gated-delta inputs, for partial-accept linear-only replay (see [`Qwen35LinearCapture`]).
    pub(crate) capture: Qwen35LinearCapture,
    /// Shared by draft + two verify rows: zero per-token argmax allocs, verify argmax stays on-device.
    pub(crate) argmax_scratch: CudaSlice<i32>,
    /// Sampled-mode buffers (lazy on first temp>0 step; greedy unused). Shapes by head cap `spec_draft_tokens.max(1)`:
    /// `q_probs [cap, vocab] f32`, `p_probs [cap+1, vocab] f32`, `sample_tok [1]`, `accept_out [2]`,
    /// `chain_draft [cap]`, `u_accept [cap]`, `u_residual [cap+1]`.
    pub(crate) q_probs: SliceSlot<f32>,
    pub(crate) p_probs: SliceSlot<f32>,
    pub(crate) sample_tok: SliceSlot<i32>,
    pub(crate) accept_out: SliceSlot<i32>,
    pub(crate) chain_draft: SliceSlot<i32>,
    pub(crate) u_accept: SliceSlot<f32>,
    pub(crate) u_residual: SliceSlot<f32>,
}

/// Pointer/length staging for [`Qwen35Model::batched_copy`].
#[derive(Default)]
pub(crate) struct Qwen35CopyScratch {
    pub(crate) ptrs: SliceSlot<u64>,
    pub(crate) lens: SliceSlot<i32>,
    pub(crate) host: Vec<u64>,
    pub(crate) hlen: Vec<i32>,
}

/// Per-slot buffer addresses for varlen replay kernels; flat so staging is one H2D.
/// Re-staged every tick (accepted set changes).
#[derive(Default)]
pub(crate) struct Qwen35ReplayTables {
    pub(crate) ptrs: SliceSlot<u64>,
    pub(crate) row_len: SliceSlot<i32>,
    pub(crate) host: Vec<u64>,
    pub(crate) layout: ReplayLayout,
}

/// Where each table sits in the flat staging buffer — single source for host writer and device reader.
#[derive(Clone, Copy, Default)]
pub(crate) struct ReplayLayout {
    pub(crate) base: u64,
    pub(crate) stride: usize,
    pub(crate) batch: usize,
}

impl ReplayLayout {
    pub(crate) fn at(&self, kind: usize, li: usize) -> usize {
        kind * self.stride + li * self.batch
    }

    /// Device address of table `kind`'s `[B]` row for linear layer `li`.
    pub(crate) fn table(&self, kind: usize, li: usize) -> u64 {
        self.base + (self.at(kind, li) as u64) * 8
    }
}

/// [`Qwen35ReplayTables`] kinds, in layout order.
pub(crate) const TBL_QKV: usize = 0;
pub(crate) const TBL_B: usize = 1;
pub(crate) const TBL_A: usize = 2;
pub(crate) const TBL_CONV: usize = 3;
pub(crate) const TBL_GDR: usize = 4;
pub(crate) const REPLAY_TABLES: usize = 5;

impl Qwen35ReplayTables {
    pub(crate) fn stage(
        &mut self,
        ctx: &DeviceContext,
        slots: &mut [&mut Qwen35SlotState],
        captures: &[&Qwen35LinearCapture],
        ks: &[usize],
        num_linear: usize,
    ) -> Result<()> {
        let b = slots.len();
        let lay = ReplayLayout {
            base: 0,
            stride: num_linear * b,
            batch: b,
        };
        self.layout = lay;
        self.host.clear();
        self.host.resize(REPLAY_TABLES * lay.stride, 0);
        for li in 0..num_linear {
            for (s, slot) in slots.iter_mut().enumerate() {
                let mut put = |kind: usize, addr: u64| self.host[lay.at(kind, li) + s] = addr;
                put(TBL_QKV, captures[s].qkv[li].data.device_ptr(&ctx.stream).0);
                put(TBL_B, captures[s].b_proj[li].data.device_ptr(&ctx.stream).0);
                put(TBL_A, captures[s].a_proj[li].data.device_ptr(&ctx.stream).0);
                put(
                    TBL_CONV,
                    slot.conv_states[li].data.device_ptr_mut(&ctx.stream).0,
                );
                put(TBL_GDR, slot.gdr_states[li].device_ptr_mut(&ctx.stream).0);
            }
        }
        let dst = self.ptrs.get(ctx, self.host.len())?;
        ctx.stream
            .memcpy_htod(&self.host, dst)
            .map_err(|e| anyhow!("H2D replay pointer tables: {e}"))?;
        let lens: Vec<i32> = ks.iter().map(|k| (k + 1) as i32).collect();
        let dst = self.row_len.get(ctx, b)?;
        ctx.stream
            .memcpy_htod(&lens, dst)
            .map_err(|e| anyhow!("H2D replay row lengths: {e}"))?;
        Ok(())
    }
}

/// Per-linear-layer capture of gated-delta-rule inputs from the spec verify forward.
///
/// On partial accept (`k < depth`) the trunk linear state must sit at post-`[pending, d1..dk]`.
/// The old path re-ran a FULL `depth+1`-wide trunk forward purely for that recurrent
/// side-effect (21-47 ms/macro-step on H20 real-fp8). GDR + conv1d state is a pure function
/// of per-layer inputs (post-in_proj `qkv` PRE-conv1d, plus `b`/`a` gate projections); those
/// inputs already encode the full-stack residual because verify produced them with the real
/// trunk. So we cache them during verify and re-run ONLY conv1d + recurrent GDR over rows
/// `[0..=k]` on partial accept — bit-identical to verify's first `k+1` steps, skipping
/// full-attn, MLP/MoE, final norm, and lm_head.
///
/// Caches are token-major `[(depth+1), width]` bf16 (token `t` at `t*width`), so rows
/// `[0..=k]` slice contiguously as `[0..(k+1)*width]`.
#[allow(dead_code)] // populated by linear_attention under capture; read by replay_linear_only
pub(crate) struct Qwen35LinearCapture {
    /// `num_linear`; per-row stride = `len / (depth+1)`.
    pub(crate) rows: usize,
    /// Post-in_proj fused `[q|k|v]` (PRE-conv1d) for all `depth+1` rows; feeds `conv1d_prefill_cuda` on replay.
    pub(crate) qkv: Vec<DeviceVec>,
    /// `in_proj_b` projection (one scalar per local v-head) for all rows.
    pub(crate) b_proj: Vec<DeviceVec>,
    /// `in_proj_a` projection (one scalar per local v-head) for all rows.
    pub(crate) a_proj: Vec<DeviceVec>,
}

#[allow(dead_code)] // consumed by mtp_forward_level + spec_step (next increments)
impl Qwen35SpecSlotState {
    /// Snapshot trunk linear state before verify (for reject rollback).
    pub(crate) fn snapshot_trunk(
        &mut self,
        ctx: &DeviceContext,
        slot: &Qwen35SlotState,
    ) -> Result<()> {
        slot.snapshot_linear_into(ctx, &mut self.gdr_snap, &mut self.conv_snap)
    }

    pub(crate) fn restore_trunk(
        &self,
        ctx: &DeviceContext,
        slot: &mut Qwen35SlotState,
    ) -> Result<()> {
        slot.restore_linear_from(ctx, &self.gdr_snap, &self.conv_snap)
    }

    /// Append this slot's (snapshot, live) linear-state addresses into `gdr`/`conv`;
    /// caller issues one batched copy for the whole speculative batch.
    pub(crate) fn linear_state_addrs(
        &mut self,
        ctx: &DeviceContext,
        slot: &mut Qwen35SlotState,
        bytes: (usize, usize),
        gdr: &mut (Vec<u64>, Vec<u64>),
        conv: &mut (Vec<u64>, Vec<u64>),
    ) -> Result<()> {
        ensure!(
            self.gdr_snap.len() == slot.gdr_states.len()
                && self.conv_snap.len() == slot.conv_states.len(),
            "spec snapshot scratch sized {}/{} != slot linear layers {}/{}",
            self.gdr_snap.len(),
            self.conv_snap.len(),
            slot.gdr_states.len(),
            slot.conv_states.len()
        );
        // batched copy takes a size, not a slice — every pair must match.
        ensure!(
            self.gdr_snap.iter().all(|b| b.len() * 4 == bytes.0)
                && slot.gdr_states.iter().all(|b| b.len() * 4 == bytes.0)
                && self.conv_snap.iter().all(|b| b.len * 2 == bytes.1)
                && slot.conv_states.iter().all(|b| b.len * 2 == bytes.1),
            "spec linear state buffers are not uniformly {}/{} bytes",
            bytes.0,
            bytes.1
        );
        for (s, l) in self.gdr_snap.iter_mut().zip(slot.gdr_states.iter_mut()) {
            gdr.0.push(s.device_ptr_mut(&ctx.stream).0);
            gdr.1.push(l.device_ptr_mut(&ctx.stream).0);
        }
        for (s, l) in self.conv_snap.iter_mut().zip(slot.conv_states.iter_mut()) {
            conv.0.push(s.data.device_ptr_mut(&ctx.stream).0);
            conv.1.push(l.data.device_ptr_mut(&ctx.stream).0);
        }
        Ok(())
    }

    pub(crate) fn argmax_scratch_mut(&mut self) -> &mut CudaSlice<i32> {
        &mut self.argmax_scratch
    }
}
