//! Persistent forward-workspace buffer slots (exact-shape reuse cache).
//!
//! The Qwen3.5/3.6 hybrid forward used to perform ~425 fresh device
//! allocations PER CALL (`HiddenStates::zeros` / `DeviceVec::zeros` /
//! `clone_htod` per layer per token). These slots give every such buffer a
//! persistent home: a slot owns at most one device buffer tagged with its
//! current shape; `get` returns the cached buffer when the requested shape
//! matches (skipping the alloc + zero-fill) and re-allocates — zeroed, exactly
//! like the per-call `*::zeros` it replaces — when the shape changes.
//!
//! WHY exact-shape instead of a capacity-sized arena: several consumers derive
//! semantics from `CudaSlice::len()`, not from the logical shape —
//! `TpRuntime::all_reduce_sum` sizes the NCCL/one-shot message (and PICKS the
//! one-shot-vs-NCCL path) from `buf.data.len()`, and `clone_dtoh`/`memcpy_dtod`
//! length-check the whole slice. A capacity-sized buffer would change the
//! collective message length and algorithm on TP>=2 (a numerics hazard, bf16
//! reduction order) and over-read on D2H. Exact-shape buffers are
//! byte-identical to the per-call path by construction.
//!
//! Shape-change cadence: decode loops at seq_len==1 (zero allocations in
//! steady state); chunked prefill reuses across uniform chunks and re-allocates
//! once on the remainder chunk and once on the prefill->decode transition.
//! Worst case (interleaved prefill/decode rows at c>=2) degenerates to one
//! allocation per shape flip — never worse than the old always-allocate path.
//!
//! SAFETY (disabled cudarc event tracking): dropping the old buffer on a shape
//! change frees it immediately. This is safe because every forward ends in a
//! host sync (token sampling D2H / explicit `ctx.sync`) BEFORE the next call
//! can request a different shape, so no in-flight kernel can still reference
//! the dropped buffer. Reuse within/between calls is ordered by the single
//! compute stream.
//!
//! ZEROING contract: a slot reuse skips the zero-fill, so every consumer must
//! either (a) fully overwrite the buffer before any read (the default —
//! callers document the proof per buffer), or (b) explicitly request
//! `get_zeroed` / `neg1_filled` when the kernel ASSUMES initialized memory
//! (atomicAdd counters, scatter targets with partial coverage).

use anyhow::{Result, anyhow};
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates};
use cudarc::driver::{CudaSlice, DevicePtrMut, DeviceRepr, ValidAsZeroBits};

/// One cached [`HiddenStates`] buffer (`[hidden_dim, seq_len]` bf16).
#[derive(Default)]
pub(crate) struct HiddenSlot(Option<HiddenStates>);

impl HiddenSlot {
    /// Shape-matched reuse (NO zero-fill on reuse — caller proves every element
    /// is written before read), fresh zeroed allocation on mismatch.
    pub(crate) fn get(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<&mut HiddenStates> {
        let reusable = matches!(
            &self.0,
            Some(buf) if buf.hidden_dim == hidden_dim && buf.seq_len == seq_len
        );
        if !reusable {
            self.0 = Some(HiddenStates::zeros(ctx, hidden_dim, seq_len)?);
        }
        self.0
            .as_mut()
            .ok_or_else(|| anyhow!("workspace hidden slot empty after fill"))
    }

    /// Like [`Self::get`] but guaranteed all-zero on EVERY call (memset on
    /// reuse) — for buffers whose unwritten regions are read downstream
    /// (e.g. the EP-partial MoE scatter target).
    pub(crate) fn get_zeroed(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<&mut HiddenStates> {
        let reused = matches!(
            &self.0,
            Some(buf) if buf.hidden_dim == hidden_dim && buf.seq_len == seq_len
        );
        let buf = self.get(ctx, hidden_dim, seq_len)?;
        if reused {
            ctx.stream
                .memset_zeros(&mut buf.data)
                .map_err(|e| anyhow!("workspace hidden slot memset failed: {e}"))?;
        }
        Ok(buf)
    }

    /// Drop the cached buffer (frees the VRAM; e.g. OPD weight time-share).
    pub(crate) fn release(&mut self) {
        self.0 = None;
    }

    /// Borrow the cached buffer, if any.
    pub(crate) fn as_ref(&self) -> Option<&HiddenStates> {
        self.0.as_ref()
    }
}

/// One cached [`DeviceVec`] buffer (`[len]` bf16).
#[derive(Default)]
pub(crate) struct VecSlot(Option<DeviceVec>);

impl VecSlot {
    /// Length-matched reuse (NO zero-fill on reuse — caller proves full
    /// overwrite before read), fresh zeroed allocation on mismatch.
    pub(crate) fn get(&mut self, ctx: &DeviceContext, len: usize) -> Result<&mut DeviceVec> {
        let reusable = matches!(&self.0, Some(buf) if buf.len == len);
        if !reusable {
            self.0 = Some(DeviceVec::zeros(ctx, len)?);
        }
        self.0
            .as_mut()
            .ok_or_else(|| anyhow!("workspace vec slot empty after fill"))
    }

    pub(crate) fn release(&mut self) {
        self.0 = None;
    }
}

/// One cached typed device buffer (`CudaSlice<T>`).
#[derive(Default)]
pub(crate) struct SliceSlot<T>(Option<CudaSlice<T>>);

impl<T: DeviceRepr + ValidAsZeroBits> SliceSlot<T> {
    /// Length-matched reuse (NO re-init on reuse — caller proves every read
    /// element is written first), fresh zeroed allocation on mismatch.
    pub(crate) fn get(&mut self, ctx: &DeviceContext, len: usize) -> Result<&mut CudaSlice<T>> {
        let reusable = matches!(&self.0, Some(buf) if buf.len() == len);
        if !reusable {
            self.0 = Some(
                ctx.stream
                    .alloc_zeros::<T>(len)
                    .map_err(|e| anyhow!("workspace slice alloc failed: {e}"))?,
            );
        }
        self.0
            .as_mut()
            .ok_or_else(|| anyhow!("workspace slice slot empty after fill"))
    }

    /// Like [`Self::get`] but guaranteed all-zero on EVERY call (memset on
    /// reuse) — for atomicAdd accumulators (MoE counts/cursors).
    pub(crate) fn get_zeroed(
        &mut self,
        ctx: &DeviceContext,
        len: usize,
    ) -> Result<&mut CudaSlice<T>> {
        let reused = matches!(&self.0, Some(buf) if buf.len() == len);
        let buf = self.get(ctx, len)?;
        if reused {
            ctx.stream
                .memset_zeros(buf)
                .map_err(|e| anyhow!("workspace slice memset failed: {e}"))?;
        }
        Ok(buf)
    }

    /// Like [`Self::upload`] but skips even the H2D copy when the cached
    /// length matches. ONLY for content that is a pure function of the length
    /// (e.g. an identity index table) — the caller guarantees a length-matched
    /// `host` is byte-identical to what was uploaded before.
    pub(crate) fn upload_const(
        &mut self,
        ctx: &DeviceContext,
        host: &[T],
    ) -> Result<&CudaSlice<T>> {
        if matches!(&self.0, Some(buf) if buf.len() == host.len()) {
            return self
                .0
                .as_ref()
                .ok_or_else(|| anyhow!("workspace slice slot empty after match"));
        }
        self.upload(ctx, host)
    }

    /// Upload `host` into the slot (H2D copy into the reused buffer when the
    /// length matches; alloc + copy otherwise). Replaces per-call `clone_htod`.
    pub(crate) fn upload(&mut self, ctx: &DeviceContext, host: &[T]) -> Result<&CudaSlice<T>> {
        match &mut self.0 {
            Some(buf) if buf.len() == host.len() => {
                ctx.stream
                    .memcpy_htod(host, buf)
                    .map_err(|e| anyhow!("workspace H2D upload failed: {e}"))?;
            }
            slot => {
                *slot = Some(
                    ctx.stream
                        .clone_htod(host)
                        .map_err(|e| anyhow!("workspace H2D alloc-upload failed: {e}"))?,
                );
            }
        }
        self.0
            .as_ref()
            .ok_or_else(|| anyhow!("workspace slice slot empty after upload"))
    }

    pub(crate) fn release(&mut self) {
        self.0 = None;
    }
}

impl SliceSlot<i32> {
    /// Like `get` but guaranteed all `-1` on EVERY call (device memset 0xFF) —
    /// the EP-partial MoE pack sentinel (the scatter kernel skips `< 0` slots;
    /// a stale/zero tail would alias route slot 0).
    pub(crate) fn neg1_filled(
        &mut self,
        ctx: &DeviceContext,
        len: usize,
    ) -> Result<&mut CudaSlice<i32>> {
        let buf = self.get(ctx, len)?;
        let n_bytes = len * std::mem::size_of::<i32>();
        let (ptr, guard) = buf.device_ptr_mut(&ctx.stream);
        // SAFETY: `ptr` is a live device allocation of `n_bytes` on this stream.
        unsafe {
            cudarc::driver::result::memset_d8_async(ptr, 0xFF, n_bytes, ctx.stream.cu_stream())
                .map_err(|e| anyhow!("workspace -1 sentinel memset failed: {e}"))?;
        }
        drop(guard);
        Ok(buf)
    }
}

/// One cached PINNED host buffer, exact-length reuse.
///
/// A `Vec` destination makes the driver page-lock and free a staging buffer on
/// every `memcpy_dtoh`. In a 30.75 s pure-decode window that was 1.64 s of
/// `cuMemHostAlloc` + `cuMemFreeHost` across 3552 calls — 15% of the window,
/// paired 1:1 with the readbacks, to move a handful of i32 tokens.
pub(crate) struct PinnedSlot<T>(Option<cudarc::driver::PinnedHostSlice<T>>);

impl<T> Default for PinnedSlot<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T: DeviceRepr> PinnedSlot<T> {
    pub(crate) fn get(
        &mut self,
        ctx: &DeviceContext,
        len: usize,
    ) -> Result<&mut cudarc::driver::PinnedHostSlice<T>> {
        if !matches!(&self.0, Some(buf) if buf.len() == len) {
            // SAFETY: written only by the D2H whose completion the caller waits
            // on before reading; freed with the slot.
            self.0 = Some(unsafe {
                ctx.ctx
                    .alloc_pinned::<T>(len)
                    .map_err(|e| anyhow!("alloc pinned staging failed: {e}"))?
            });
        }
        self.0
            .as_mut()
            .ok_or_else(|| anyhow!("pinned slot empty after fill"))
    }
}
