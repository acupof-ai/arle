use super::*;

/// Host image of one whole slot for G3 capacity spill (mirror of
/// [`crate::dsv4::Dsv4SlotImage`]): the slot's full-attn KV pages plus the
/// per-linear-layer recurrent + conv state and the materialized length. Every
/// device buffer the slot owns is captured here byte-for-byte — a missed buffer
/// is a silently-wrong restore. `k_caches`/`v_caches` are NOT captured: the
/// paged default leaves them empty (full-attn KV is paged), asserted at capture.
pub(crate) struct Qwen35SlotImage {
    /// Full-attn KV bytes in slot-logical page order (from `copy_pages_to_host`).
    pub(crate) full_attn_pages: Vec<u8>,
    /// Logical page count the bytes cover (drives swap-in page allocation).
    pub(crate) full_attn_page_count: usize,
    /// `[num_linear]` gated-delta recurrent states (f32), D2H verbatim.
    pub(crate) gdr_host: Vec<Vec<f32>>,
    /// `[num_linear]` conv1d rings (bf16), D2H verbatim.
    pub(crate) conv_host: Vec<Vec<bf16>>,
    /// Materialized full-attn length the image was captured at.
    pub(crate) seq_len: usize,
}

pub(crate) struct Qwen35SlotState {
    /// `[num_full_layers]` contiguous K caches, each `max_seq_len*kv_dim` bf16.
    /// EMPTY by default (full-attn KV is paged); populated only by the legacy
    /// contiguous lane when explicitly requested.
    pub(crate) k_caches: Vec<DeviceVec>,
    pub(crate) v_caches: Vec<DeviceVec>,
    /// `[num_linear_layers]` gated-delta recurrent states (`Vh*Kd*Vd` f32).
    pub(crate) gdr_states: Vec<CudaSlice<f32>>,
    /// `[num_linear_layers]` conv1d rings (`qkv_dim*(kernel-1)` bf16).
    pub(crate) conv_states: Vec<DeviceVec>,
    /// Reusable pinned staging for [`Self::snapshot_recurrent`]. A `Vec`/`[T]`
    /// destination makes every `memcpy_dtoh` a full stream synchronize
    /// (cudarc `SyncOnDrop::Sync`); a pinned slice syncs on its own event.
    pub(crate) gdr_pinned: Vec<PinnedHostSlice<f32>>,
    pub(crate) conv_pinned: Vec<PinnedHostSlice<bf16>>,
    /// True once `acquire_recurrent` has run for the current occupant (even
    /// when `num_linear == 0` and the state vecs are empty). Guards the
    /// forward's `has_recurrent()` chokepoint against a missed acquire.
    pub(crate) recurrent_acquired: bool,
    /// Tokens materialized into the caches so far (full-attn kv_len).
    pub(crate) seq_len: usize,
}

/// A detached, reusable recurrent state block (`(gdr_states, conv_states)`) on
/// the executor's free-list. Released back by a finished request and popped by
/// the next one — same dims for every slot, so any block fits any slot.
pub(crate) type RecurrentBlock = (Vec<CudaSlice<f32>>, Vec<DeviceVec>);

/// One committed token + its behavior logprob under the filtered sampling
/// dist (`None` = uncaptured: greedy / delta policy).
pub(crate) type CommittedToken = (u32, Option<f32>);

/// D2H snapshot of the recurrent state at a prefix boundary.
/// Used by the sidecar prefix-cache to restore the recurrent layers
/// when reusing a Qwen3.5/3.6 hybrid prefix via the page-radix path.
#[derive(Clone)]
pub(crate) struct Qwen35RecurrentSnapshot {
    /// `[num_linear]` gated-delta states (f32), copied verbatim from device.
    pub(crate) gdr: Vec<Vec<f32>>,
    /// `[num_linear]` conv1d rings (bf16), copied verbatim from device.
    pub(crate) conv: Vec<Vec<bf16>>,
}

impl Qwen35SlotImage {
    /// Approximate device-state byte size of a whole-slot image (the unit the G3
    /// `KvTierStore` budgets one entry against). Used only to size the tier's
    /// count cap — exactness isn't required, but it must scale with the image.
    pub(crate) fn dram_bytes(&self) -> usize {
        self.full_attn_pages.len()
            + self.gdr_host.iter().map(|v| v.len() * 4).sum::<usize>()
            + self.conv_host.iter().map(|v| v.len() * 2).sum::<usize>()
            + 8
    }

    /// Flatten the whole-slot image into ONE length-prefixed byte buffer for the
    /// G3 [`kv_native_sys::KvTierStore`] (opaque-`u64`-keyed transport). The
    /// exact byte-inverse of [`Self::from_bytes`] — proven field-for-field in the
    /// `slot_image_byte_inverse` unit test. No serde: a small fixed header
    /// (`seq_len`, `full_attn_page_count`, `full_attn_pages` byte length, the two
    /// linear counts) followed by the full-attn bytes, then each gdr vec's
    /// `[len:u64][f32...]` and each conv vec's `[len:u64][bf16...]`. The payload
    /// regions are bulk byte copies in host order — the buffer never leaves this
    /// box (host DRAM / local NVMe), and per-element `to_le_bytes` cost 37M
    /// 4-byte appends per park (~150 ms), the dominant swap stall.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.dram_bytes() + 64);
        buf.extend_from_slice(&(self.seq_len as u64).to_le_bytes());
        buf.extend_from_slice(&(self.full_attn_page_count as u64).to_le_bytes());
        buf.extend_from_slice(&(self.full_attn_pages.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.gdr_host.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.conv_host.len() as u64).to_le_bytes());
        buf.extend_from_slice(&self.full_attn_pages);
        for gdr in &self.gdr_host {
            buf.extend_from_slice(&(gdr.len() as u64).to_le_bytes());
            // SAFETY: f32 has no padding; the byte view aliases exactly len*4.
            buf.extend_from_slice(unsafe {
                std::slice::from_raw_parts(gdr.as_ptr() as *const u8, gdr.len() * 4)
            });
        }
        for conv in &self.conv_host {
            buf.extend_from_slice(&(conv.len() as u64).to_le_bytes());
            // SAFETY: bf16 is #[repr(transparent)] over u16; byte view is valid.
            buf.extend_from_slice(unsafe {
                std::slice::from_raw_parts(conv.as_ptr() as *const u8, conv.len() * 2)
            });
        }
        buf
    }

    /// Reconstruct a whole-slot image from [`Self::to_bytes`] — the exact
    /// byte-inverse. A cursor walks the fixed header then the four sized regions;
    /// any short/over-long buffer (a corrupt or foreign payload) errors rather
    /// than restore garbage.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut pos = 0usize;
        let take_u64 = |pos: &mut usize| -> Result<u64> {
            let end = pos
                .checked_add(8)
                .ok_or_else(|| anyhow!("slot image header overflow"))?;
            let slice = bytes
                .get(*pos..end)
                .ok_or_else(|| anyhow!("slot image truncated reading u64 at {pos}"))?;
            *pos = end;
            Ok(u64::from_le_bytes(slice.try_into().expect("8 bytes")))
        };
        let seq_len = take_u64(&mut pos)? as usize;
        let full_attn_page_count = take_u64(&mut pos)? as usize;
        let full_attn_len = take_u64(&mut pos)? as usize;
        let num_gdr = take_u64(&mut pos)? as usize;
        let num_conv = take_u64(&mut pos)? as usize;
        let pages_end = pos
            .checked_add(full_attn_len)
            .ok_or_else(|| anyhow!("slot image full-attn length overflow"))?;
        let full_attn_pages = bytes
            .get(pos..pages_end)
            .ok_or_else(|| anyhow!("slot image truncated reading full-attn pages"))?
            .to_vec();
        pos = pages_end;
        let mut gdr_host = Vec::with_capacity(num_gdr);
        for _ in 0..num_gdr {
            let len = take_u64(&mut pos)? as usize;
            let end = pos
                .checked_add(len * 4)
                .ok_or_else(|| anyhow!("slot image gdr length overflow"))?;
            let raw = bytes
                .get(pos..end)
                .ok_or_else(|| anyhow!("slot image truncated reading gdr state"))?;
            let mut v = vec![0f32; len];
            // SAFETY: f32 has no padding; the byte view aliases exactly len*4.
            unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len * 4) }
                .copy_from_slice(raw);
            gdr_host.push(v);
            pos = end;
        }
        let mut conv_host = Vec::with_capacity(num_conv);
        for _ in 0..num_conv {
            let len = take_u64(&mut pos)? as usize;
            let end = pos
                .checked_add(len * 2)
                .ok_or_else(|| anyhow!("slot image conv length overflow"))?;
            let raw = bytes
                .get(pos..end)
                .ok_or_else(|| anyhow!("slot image truncated reading conv state"))?;
            let mut v = vec![bf16::ZERO; len];
            // SAFETY: bf16 is #[repr(transparent)] over u16; byte view is valid.
            unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len * 2) }
                .copy_from_slice(raw);
            conv_host.push(v);
            pos = end;
        }
        ensure!(
            pos == bytes.len(),
            "slot image has {} trailing bytes after deserialize",
            bytes.len() - pos
        );
        Ok(Self {
            full_attn_pages,
            full_attn_page_count,
            gdr_host,
            conv_host,
            seq_len,
        })
    }
}

pub(crate) fn alloc_recurrent_block(
    ctx: &DeviceContext,
    num_linear: usize,
    gdr_state_len: usize,
    conv_len: usize,
) -> Result<RecurrentBlock> {
    let (gdr, conv) = (0..num_linear)
        .map(|_| {
            // SAFETY: zero_recurrent runs before any read of this state.
            let g = unsafe { ctx.stream.alloc::<f32>(gdr_state_len) }
                .map_err(|e| anyhow!("alloc gated-delta state failed: {e}"))?;
            // SAFETY: same as above.
            let c = unsafe { DeviceVec::uninit(ctx, conv_len) }?;
            Ok((g, c))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip::<_, _, Vec<_>, Vec<_>>();
    Ok((gdr, conv))
}

impl Qwen35RecurrentSnapshot {
    /// Approximate host byte size (for cap accounting).
    #[allow(dead_code)]
    pub(crate) fn host_bytes(&self) -> usize {
        self.gdr.iter().map(|v| v.len() * 4).sum::<usize>()
            + self.conv.iter().map(|v| v.len() * 2).sum::<usize>()
    }

    /// Flatten for the sidecar tier store — exact byte-inverse of
    /// [`Self::from_bytes`]. Header `[num_gdr][num_conv]` (u64 LE), then each vec
    /// `[len:u64][elems...]`. No full-attention KV: restore mirrors the radix
    /// prefix's own device pages.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        // Every stride boundary of every prefill serializes the WHOLE recurrent
        // state, so this runs inside the tick, not behind it — surface the cost.
        let started = std::time::Instant::now();
        let mut buf = Vec::with_capacity(self.host_bytes() + 64);
        buf.extend_from_slice(&(self.gdr.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.conv.len() as u64).to_le_bytes());
        for gdr in &self.gdr {
            buf.extend_from_slice(&(gdr.len() as u64).to_le_bytes());
            // SAFETY: f32 has no padding; the byte view aliases exactly len*4.
            buf.extend_from_slice(unsafe {
                std::slice::from_raw_parts(gdr.as_ptr() as *const u8, gdr.len() * 4)
            });
        }
        for conv in &self.conv {
            buf.extend_from_slice(&(conv.len() as u64).to_le_bytes());
            // SAFETY: bf16 is #[repr(transparent)] over u16; byte view is valid.
            buf.extend_from_slice(unsafe {
                std::slice::from_raw_parts(conv.as_ptr() as *const u8, conv.len() * 2)
            });
        }
        log::info!(
            "recurrent sidecar serialize: {:.1} MiB in {:.1} ms",
            buf.len() as f64 / (1024.0 * 1024.0),
            started.elapsed().as_secs_f64() * 1e3
        );
        buf
    }

    /// Reconstruct a snapshot from [`Self::to_bytes`] — the exact byte-inverse.
    /// Any short/over-long buffer (corrupt or foreign payload) errors rather than
    /// restore garbage, so the caller falls through to clean recompute.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut pos = 0usize;
        let take_u64 = |pos: &mut usize| -> Result<u64> {
            let end = pos
                .checked_add(8)
                .ok_or_else(|| anyhow!("recurrent snapshot header overflow"))?;
            let slice = bytes
                .get(*pos..end)
                .ok_or_else(|| anyhow!("recurrent snapshot truncated reading u64 at {pos}"))?;
            *pos = end;
            Ok(u64::from_le_bytes(slice.try_into().expect("8 bytes")))
        };
        let num_gdr = take_u64(&mut pos)? as usize;
        let num_conv = take_u64(&mut pos)? as usize;
        let mut gdr = Vec::with_capacity(num_gdr);
        for _ in 0..num_gdr {
            let len = take_u64(&mut pos)? as usize;
            let end = pos
                .checked_add(len * 4)
                .ok_or_else(|| anyhow!("recurrent snapshot gdr length overflow"))?;
            let raw = bytes
                .get(pos..end)
                .ok_or_else(|| anyhow!("recurrent snapshot truncated reading gdr state"))?;
            let mut v = vec![0f32; len];
            // SAFETY: f32 has no padding; the byte view aliases exactly len*4.
            unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len * 4) }
                .copy_from_slice(raw);
            gdr.push(v);
            pos = end;
        }
        let mut conv = Vec::with_capacity(num_conv);
        for _ in 0..num_conv {
            let len = take_u64(&mut pos)? as usize;
            let end = pos
                .checked_add(len * 2)
                .ok_or_else(|| anyhow!("recurrent snapshot conv length overflow"))?;
            let raw = bytes
                .get(pos..end)
                .ok_or_else(|| anyhow!("recurrent snapshot truncated reading conv state"))?;
            let mut v = vec![bf16::ZERO; len];
            // SAFETY: bf16 is #[repr(transparent)] over u16; byte view is valid.
            unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len * 2) }
                .copy_from_slice(raw);
            conv.push(v);
            pos = end;
        }
        ensure!(
            pos == bytes.len(),
            "recurrent snapshot has {} trailing bytes after deserialize",
            bytes.len() - pos
        );
        Ok(Self { gdr, conv })
    }
}

/// FNV-1a hash of a token id slice — used to key the recurrent sidecar.
pub(crate) fn hash_prefix_tokens(tokens: &[u32]) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut h = FNV_OFFSET;
    for &t in tokens {
        let bytes = t.to_le_bytes();
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

impl Qwen35SlotState {
    /// A fresh idle slot: allocate NOTHING. The recurrent state (~147 MiB) is a
    /// fixed-size per-request state — not token-addressable like the paged
    /// full-attn KV — so it draws from a request-grained free-list pool
    /// ([`RecurrentBlock`]) lazily on activation ([`Self::acquire_recurrent`]),
    /// not upfront per `num_slots`. Idle slots cost zero recurrent HBM; the win
    /// is partial-load footprint + the foundation for future L2 spill. At full
    /// concurrency the pool grows to `num_slots`, identical to the old upfront
    /// reservation. `k_caches`/`v_caches` stay empty (full-attn KV is paged).
    pub(crate) fn new_linear_only() -> Self {
        Self {
            k_caches: Vec::new(),
            v_caches: Vec::new(),
            gdr_states: Vec::new(),
            conv_states: Vec::new(),
            gdr_pinned: Vec::new(),
            conv_pinned: Vec::new(),
            recurrent_acquired: false,
            seq_len: 0,
        }
    }

    /// Activate this slot's recurrent state for a fresh request (the
    /// `start_pos == 0` prefill): pop a reusable block from `pool` (free-list
    /// reuse, no alloc churn) or allocate fresh, then ZERO it. Idempotent —
    /// no-op if already allocated, so a chunked-prefill's later chunks
    /// (`start_pos > 0`) never re-zero and wipe the prefix's recurrent state;
    /// the zero happens ONLY on fresh acquisition. MUST run before any forward
    /// reads `gdr_states`.
    pub(crate) fn acquire_recurrent(
        &mut self,
        ctx: &DeviceContext,
        num_linear: usize,
        gdr_state_len: usize,
        conv_len: usize,
        pool: &mut Vec<RecurrentBlock>,
    ) -> Result<()> {
        if !self.gdr_states.is_empty() {
            self.recurrent_acquired = true;
            return Ok(()); // already active (a chunked-prefill continuation)
        }
        let (gdr, conv) = match pool.pop() {
            Some(block) => block,
            None => alloc_recurrent_block(ctx, num_linear, gdr_state_len, conv_len)?,
        };
        self.gdr_states = gdr;
        self.conv_states = conv;
        self.recurrent_acquired = true;
        self.seq_len = 0;
        // Zero on acquisition (a pooled block carries the prior occupant's
        // state; a fresh alloc is already zero but re-zeroing is cheap/uniform).
        self.zero_recurrent(ctx)
    }

    /// Return this slot's recurrent block to the free-list, leaving the slot's
    /// fields empty. Called ONLY at request finish (the slot's prior occupant is
    /// fully done — no in-flight forward references it), so the block is safe to
    /// hand to the next request.
    pub(crate) fn release_recurrent(&mut self, pool: &mut Vec<RecurrentBlock>) {
        self.recurrent_acquired = false;
        if self.gdr_states.is_empty() {
            return;
        }
        let gdr = std::mem::take(&mut self.gdr_states);
        let conv = std::mem::take(&mut self.conv_states);
        pool.push((gdr, conv));
        // The pinned staging is per-slot and sized to the device buffers just
        // returned; holding it across the idle window pins ~147 MiB/slot host
        // RAM for a snapshot that may never come. `ensure_snapshot_staging`
        // rebuilds it on the next snapshot.
        self.gdr_pinned.clear();
        self.conv_pinned.clear();
        self.seq_len = 0;
    }

    /// Zero the recurrent + conv-ring state in place (the per-request fresh
    /// start). Does not touch the full-attn cursor. No-op when unallocated.
    pub(crate) fn zero_recurrent(&mut self, ctx: &DeviceContext) -> Result<()> {
        for s in &mut self.gdr_states {
            ctx.stream
                .memset_zeros(s)
                .map_err(|e| anyhow!("memset gated-delta state failed: {e}"))?;
        }
        for c in &mut self.conv_states {
            ctx.stream
                .memset_zeros(&mut c.data)
                .map_err(|e| anyhow!("memset conv state failed: {e}"))?;
        }
        Ok(())
    }

    pub(crate) fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Whether this slot's recurrent state is resident (acquired). A forward
    /// that reads `gdr_states` MUST see this true — a false here means an
    /// `acquire_recurrent` hook was missed at the request's `start_pos == 0`.
    pub(crate) fn has_recurrent(&self) -> bool {
        self.recurrent_acquired
    }

    /// D2H snapshot of the current recurrent state. A full-attn-only model
    /// (`num_linear == 0`) has empty `gdr_states`/`conv_states` but still reaches
    /// the prefix-cache sidecar snapshot path — return an empty snapshot (no
    /// device work) so `restore_recurrent_from_snapshot` (0==0 dims, no-op zips)
    /// stays consistent.
    pub(crate) fn snapshot_recurrent(
        &mut self,
        ctx: &DeviceContext,
    ) -> Result<Qwen35RecurrentSnapshot> {
        if self.gdr_states.is_empty() {
            return Ok(Qwen35RecurrentSnapshot {
                gdr: Vec::new(),
                conv: Vec::new(),
            });
        }
        self.ensure_snapshot_staging(ctx)?;
        for (state, dst) in self.gdr_states.iter().zip(self.gdr_pinned.iter_mut()) {
            ctx.stream
                .memcpy_dtoh(state, dst)
                .map_err(|e| anyhow!("gdr D2H failed: {e}"))?;
        }
        for (state, dst) in self.conv_states.iter().zip(self.conv_pinned.iter_mut()) {
            ctx.stream
                .memcpy_dtoh(&state.data, dst)
                .map_err(|e| anyhow!("conv D2H failed: {e}"))?;
        }
        // No stream synchronize: `as_slice` waits on each pinned buffer's own
        // event, and stream order already keeps the next chunk off the state.
        let gdr = self
            .gdr_pinned
            .iter()
            .map(|p| {
                p.as_slice()
                    .map(<[f32]>::to_vec)
                    .map_err(|e| anyhow!("gdr pinned read failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let conv = self
            .conv_pinned
            .iter()
            .map(|p| {
                p.as_slice()
                    .map(<[bf16]>::to_vec)
                    .map_err(|e| anyhow!("conv pinned read failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Qwen35RecurrentSnapshot { gdr, conv })
    }

    /// Allocate the pinned snapshot staging once per occupant. Sizes follow the
    /// live state vectors, which `acquire_recurrent` fixes for the request.
    pub(crate) fn ensure_snapshot_staging(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.gdr_pinned.len() == self.gdr_states.len()
            && self.conv_pinned.len() == self.conv_states.len()
        {
            return Ok(());
        }
        self.gdr_pinned.clear();
        self.conv_pinned.clear();
        for state in &self.gdr_states {
            // SAFETY: the buffer is written only by the D2H above, after which
            // `synchronize` has completed it; freed with the slot state.
            self.gdr_pinned.push(unsafe {
                ctx.ctx
                    .alloc_pinned::<f32>(state.len())
                    .map_err(|e| anyhow!("alloc pinned gdr staging failed: {e}"))?
            });
        }
        for state in &self.conv_states {
            // SAFETY: as above.
            self.conv_pinned.push(unsafe {
                ctx.ctx
                    .alloc_pinned::<bf16>(state.data.len())
                    .map_err(|e| anyhow!("alloc pinned conv staging failed: {e}"))?
            });
        }
        Ok(())
    }

    /// H2D restore from a sidecar snapshot. The slot MUST have acquired recurrent
    /// buffers before calling (call `acquire_recurrent` first). Errors if
    /// dims mismatch (stale snapshot from a different checkpoint).
    pub(crate) fn restore_recurrent_from_snapshot(
        &mut self,
        ctx: &DeviceContext,
        snap: &Qwen35RecurrentSnapshot,
    ) -> Result<()> {
        ensure!(
            snap.gdr.len() == self.gdr_states.len() && snap.conv.len() == self.conv_states.len(),
            "recurrent sidecar dim mismatch: snapshot gdr={}/conv={} vs slot gdr={}/conv={}",
            snap.gdr.len(),
            snap.conv.len(),
            self.gdr_states.len(),
            self.conv_states.len()
        );
        for (s, h) in self.gdr_states.iter_mut().zip(&snap.gdr) {
            ctx.stream
                .memcpy_htod(h, s)
                .map_err(|e| anyhow!("gdr H2D restore failed: {e}"))?;
        }
        for (c, h) in self.conv_states.iter_mut().zip(&snap.conv) {
            ctx.stream
                .memcpy_htod(h, &mut c.data)
                .map_err(|e| anyhow!("conv H2D restore failed: {e}"))?;
        }
        ctx.stream
            .synchronize()
            .map_err(|e| anyhow!("sync after recurrent restore: {e}"))?;
        Ok(())
    }

    /// Advance the materialized length by `n` tokens. The captured decode
    /// graph's caller uses this: the graph body
    /// ([`Qwen35Model::forward_decode_step_captured`]) is host-state-free —
    /// replay re-launches only GPU work — so the host-side length advance
    /// happens exactly once per step at the call site, never inside the
    /// captured closure.
    pub(crate) fn advance_seq_len(&mut self, n: usize) {
        self.seq_len += n;
    }

    /// Snapshot the linear-attention recurrent + conv-ring state into caller
    /// scratch, before a speculative verify pass. The 48 gated-delta layers
    /// advance their state **in place, content-based, no position index**
    /// (`gated_delta_rule_decode_cuda` / `_prefill_recurrent_cuda`), so they do
    /// NOT self-heal under a seq_len rewind — they must be restored on reject.
    /// The full-attn K/V caches are position-indexed and self-heal via the
    /// rewind, so they are intentionally not copied here.
    pub(crate) fn snapshot_linear_into(
        &self,
        ctx: &DeviceContext,
        gdr_snap: &mut [CudaSlice<f32>],
        conv_snap: &mut [DeviceVec],
    ) -> Result<()> {
        ensure!(
            gdr_snap.len() == self.gdr_states.len() && conv_snap.len() == self.conv_states.len(),
            "spec snapshot scratch sized {}/{} != slot linear layers {}/{}",
            gdr_snap.len(),
            conv_snap.len(),
            self.gdr_states.len(),
            self.conv_states.len()
        );
        for (dst, src) in gdr_snap.iter_mut().zip(self.gdr_states.iter()) {
            ctx.stream
                .memcpy_dtod(src, dst)
                .map_err(|e| anyhow!("snapshot gated-delta state failed: {e}"))?;
        }
        for (dst, src) in conv_snap.iter_mut().zip(self.conv_states.iter()) {
            ctx.stream
                .memcpy_dtod(&src.data, &mut dst.data)
                .map_err(|e| anyhow!("snapshot conv state failed: {e}"))?;
        }
        Ok(())
    }

    /// Restore the linear-attention recurrent + conv-ring state from a snapshot
    /// taken by [`Self::snapshot_linear_into`] (speculative verify rejected).
    pub(crate) fn restore_linear_from(
        &mut self,
        ctx: &DeviceContext,
        gdr_snap: &[CudaSlice<f32>],
        conv_snap: &[DeviceVec],
    ) -> Result<()> {
        ensure!(
            gdr_snap.len() == self.gdr_states.len() && conv_snap.len() == self.conv_states.len(),
            "spec restore scratch sized {}/{} != slot linear layers {}/{}",
            gdr_snap.len(),
            conv_snap.len(),
            self.gdr_states.len(),
            self.conv_states.len()
        );
        for (dst, src) in self.gdr_states.iter_mut().zip(gdr_snap.iter()) {
            ctx.stream
                .memcpy_dtod(src, dst)
                .map_err(|e| anyhow!("restore gated-delta state failed: {e}"))?;
        }
        for (dst, src) in self.conv_states.iter_mut().zip(conv_snap.iter()) {
            ctx.stream
                .memcpy_dtod(&src.data, &mut dst.data)
                .map_err(|e| anyhow!("restore conv state failed: {e}"))?;
        }
        Ok(())
    }

    /// Rewind the full-attn cache cursor (speculative verify accepted `len`
    /// tokens). Stale rows `[len, prev_seq_len)` are position-indexed and get
    /// overwritten by the next real token at that position — no copy needed.
    pub(crate) fn set_seq_len(&mut self, len: usize) {
        self.seq_len = len;
    }

    /// Free the per-slot full-attn contiguous K/V caches. Since the shared-paged
    /// migration the DEFAULT build never allocates them (`new_linear_only` leaves
    /// `k_caches`/`v_caches` empty), so this is a no-op there. The linear-attn
    /// recurrent + conv-ring state (`gdr_states`/`conv_states`) is untouched.
    #[allow(dead_code)] // legacy contiguous-lane helper; the paged default never allocates these.
    pub(crate) fn free_full_attn_caches(&mut self) {
        self.k_caches = Vec::new();
        self.v_caches = Vec::new();
    }

    /// Serialize this slot's COMPLETE device state into a host image for G3
    /// whole-slot spill, then FREE the device buffers (pages back to
    /// `full_attn_kv`, recurrent block back to `pool`). The engine frees the slot
    /// right after `demote_slot`, so the trailing `ctx.sync()` (inside
    /// `copy_pages_to_host` for pages, explicit here for the recurrent D2H) makes
    /// the host image complete before any device buffer is reused.
    ///
    /// Captured buffers (every device buffer the slot owns, proven complete):
    ///   (a) full-attn KV pages — `copy_pages_to_host`, then drop the mirror
    ///       (the host pool frees them right after `demote_slot`).
    ///   (b) `gdr_states[0..num_linear]` (f32) — `clone_dtoh` each.
    ///   (c) `conv_states[0..num_linear]` (bf16) — `clone_dtoh` each.
    ///   (d) `seq_len`.
    /// Then `release_recurrent` returns the recurrent block to the free-list.
    /// `k_caches`/`v_caches` are asserted empty (paged default) — not captured.
    pub(crate) fn swap_out_image(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        full_attn_kv: &mut PagedKVPool,
        recurrent_pool: &mut Vec<RecurrentBlock>,
    ) -> Result<Qwen35SlotImage> {
        ensure!(
            self.k_caches.is_empty() && self.v_caches.is_empty(),
            "Qwen3.6 whole-slot swap requires the paged full-attn default; \
             the legacy contiguous K/V caches are not captured (slot {slot})"
        );
        ensure!(
            self.seq_len == full_attn_kv.seq_len(slot),
            "Qwen3.6 swap-out slot {slot} seq_len {} != pool seq_len {}",
            self.seq_len,
            full_attn_kv.seq_len(slot)
        );
        // (a) Full-attn KV pages (slot-logical order). `copy_pages_to_host` ends
        // in `ctx.sync()`, so the host bytes are complete here.
        let pages = full_attn_kv.page_indices(slot).to_vec();
        let full_attn_pages = full_attn_kv.copy_pages_to_host(ctx, &pages)?;
        let full_attn_page_count = pages.len();
        // (b) + (c) recurrent + conv D2H (stream-ordered; sync below covers them).
        let gdr_host = self
            .gdr_states
            .iter()
            .map(|s| {
                ctx.stream
                    .clone_dtoh(s)
                    .map_err(|e| anyhow!("Qwen3.6 swap gdr-state D2H failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let conv_host = self
            .conv_states
            .iter()
            .map(|c| {
                ctx.stream
                    .clone_dtoh(&c.data)
                    .map_err(|e| anyhow!("Qwen3.6 swap conv-state D2H failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        // The clone_dtoh copies above are stream-ordered; drain before the host
        // image is stored/read.
        ctx.sync()?;
        let image = Qwen35SlotImage {
            full_attn_pages,
            full_attn_page_count,
            gdr_host,
            conv_host,
            seq_len: self.seq_len,
        };
        // Free every device buffer now that the image owns the state.
        full_attn_kv.mirror_slot(slot, &[], 0)?;
        self.release_recurrent(recurrent_pool);
        self.seq_len = 0;
        Ok(image)
    }

    /// Restore a whole-slot image — the exact byte-inverse of
    /// [`Self::swap_out_image`]. Mirror the host pages the engine re-allocated,
    /// acquire a recurrent block, H2D the captured bytes verbatim (the SAME
    /// session restores its OWN state), set `seq_len`. The engine resumes
    /// decode immediately after `promote_slot`, so the trailing `ctx.sync()`
    /// makes the device restore complete before the host image can be dropped.
    pub(crate) fn swap_in_image(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        full_attn_kv: &mut PagedKVPool,
        recurrent_pool: &mut Vec<RecurrentBlock>,
        num_linear: usize,
        gdr_state_len: usize,
        conv_len: usize,
        image: &Qwen35SlotImage,
        slot_pages: &[u32],
    ) -> Result<()> {
        // A scheduler-free slot may still hold its FINISHED previous occupant's
        // device state: this arm vacates lazily at the next position-0 prefill,
        // and a swap re-admission is exactly such a fresh occupancy. Release the
        // stale state the same way `submit_prefill_row` does (#134 — the old
        // empty-slot ensure here cost one graceful recompute per rotation pair).
        if self.has_recurrent() {
            self.release_recurrent(recurrent_pool);
        }
        if full_attn_kv.seq_len(slot) != 0 {
            full_attn_kv.mirror_slot(slot, &[], 0)?;
        }
        self.seq_len = 0;
        ensure!(
            image.gdr_host.len() == num_linear && image.conv_host.len() == num_linear,
            "Qwen3.6 swap image linear count {}/{} != num_linear {num_linear}",
            image.gdr_host.len(),
            image.conv_host.len()
        );
        // (a) Mirror the captured page count, H2D the full-attn bytes.
        ensure!(
            slot_pages.len() == image.full_attn_page_count,
            "Qwen3.6 swap-in host slot holds {} pages != captured {}",
            slot_pages.len(),
            image.full_attn_page_count
        );
        full_attn_kv.mirror_slot(slot, slot_pages, image.seq_len)?;
        full_attn_kv.copy_pages_from_host(ctx, slot_pages, &image.full_attn_pages)?;
        // (b) + (c) acquire a fresh recurrent block (alloc+zero) then H2D-restore.
        self.acquire_recurrent(ctx, num_linear, gdr_state_len, conv_len, recurrent_pool)?;
        for (dst, src) in self.gdr_states.iter_mut().zip(&image.gdr_host) {
            ctx.stream
                .memcpy_htod(src, dst)
                .map_err(|e| anyhow!("Qwen3.6 swap gdr-state H2D failed: {e}"))?;
        }
        for (dst, src) in self.conv_states.iter_mut().zip(&image.conv_host) {
            ctx.stream
                .memcpy_htod(src, &mut dst.data)
                .map_err(|e| anyhow!("Qwen3.6 swap conv-state H2D failed: {e}"))?;
        }
        // (d) materialized length (both the slot and the pool's allocator agree).
        self.seq_len = image.seq_len;
        // H2D complete before the host image can be dropped (matches DSv4).
        ctx.sync()?;
        Ok(())
    }
}

#[allow(dead_code)] // consumed by mtp_forward_level + spec_step (next increments)

impl DenseMlp {
    /// SwiGLU intermediate width (half the fused projection's output rows).
    pub(crate) fn inter_dim(&self) -> usize {
        self.gate_up_proj.rows / 2
    }
}
