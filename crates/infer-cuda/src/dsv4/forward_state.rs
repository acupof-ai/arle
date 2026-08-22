use super::*;

pub(super) struct Dsv4SpecVerifyScratch {
    /// Allocated row capacity — the ceiling `set_rows` validates against.
    pub(super) rows: usize,
    pub(super) embeddings: HiddenStates,
    pub(super) initial_stream: HiddenStates,
    pub(super) layers: Vec<Dsv4SpecVerifyLayerScratch>,
}

pub(super) struct Dsv4SpecVerifyLayerScratch {
    pub(super) attn_normed: HiddenStates,
    pub(super) attn_out: HiddenStates,
    pub(super) attn_stream: HiddenStates,
    pub(super) ffn_normed: HiddenStates,
    pub(super) moe_out: HiddenStates,
    pub(super) moe_with_shared: HiddenStates,
    pub(super) ffn_stream: HiddenStates,
}

impl Dsv4SpecVerifyScratch {
    pub(super) fn new(model: &Dsv4Model) -> Result<Self> {
        let hidden_size = model.config.hidden_size;
        let stream_dim = hidden_size * model.config.hc_mult;
        let rows = model.spec_verify_rows();
        let layers: Vec<_> = (0..model.layers.len())
            .map(|_| Dsv4SpecVerifyLayerScratch::new(&model.ctx, hidden_size, stream_dim, rows))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            rows,
            // SAFETY: uninit device scratch; fully written before first read.
            embeddings: unsafe { HiddenStates::uninit(&model.ctx, hidden_size, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            initial_stream: unsafe { HiddenStates::uninit(&model.ctx, stream_dim, rows)? },
            layers,
        })
    }

    pub(super) fn set_rows(&mut self, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.rows,
            "DSv4 spec-verify scratch rows {rows} exceed allocated {}",
            self.rows
        );
        self.embeddings.seq_len = rows;
        self.initial_stream.seq_len = rows;
        for layer in &mut self.layers {
            layer.set_rows(rows);
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn device_bytes(&self) -> usize {
        self.embeddings.device_bytes()
            + self.initial_stream.device_bytes()
            + self
                .layers
                .iter()
                .map(Dsv4SpecVerifyLayerScratch::device_bytes)
                .sum::<usize>()
    }
}

impl Dsv4SpecVerifyLayerScratch {
    pub(super) fn new(
        ctx: &DeviceContext,
        hidden_size: usize,
        stream_dim: usize,
        rows: usize,
    ) -> Result<Self> {
        Ok(Self {
            // SAFETY: uninit device scratch; fully written before first read.
            attn_normed: unsafe { HiddenStates::uninit(ctx, hidden_size, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            attn_out: unsafe { HiddenStates::uninit(ctx, hidden_size, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            attn_stream: unsafe { HiddenStates::uninit(ctx, stream_dim, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            ffn_normed: unsafe { HiddenStates::uninit(ctx, hidden_size, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            moe_out: unsafe { HiddenStates::uninit(ctx, hidden_size, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            moe_with_shared: unsafe { HiddenStates::uninit(ctx, hidden_size, rows)? },
            // SAFETY: uninit device scratch; fully written before first read.
            ffn_stream: unsafe { HiddenStates::uninit(ctx, stream_dim, rows)? },
        })
    }

    pub(super) fn set_rows(&mut self, rows: usize) {
        self.attn_normed.seq_len = rows;
        self.attn_out.seq_len = rows;
        self.attn_stream.seq_len = rows;
        self.ffn_normed.seq_len = rows;
        self.moe_out.seq_len = rows;
        self.moe_with_shared.seq_len = rows;
        self.ffn_stream.seq_len = rows;
    }

    #[allow(dead_code)]
    pub(super) fn device_bytes(&self) -> usize {
        self.attn_normed.device_bytes()
            + self.attn_out.device_bytes()
            + self.attn_stream.device_bytes()
            + self.ffn_normed.device_bytes()
            + self.moe_out.device_bytes()
            + self.moe_with_shared.device_bytes()
            + self.ffn_stream.device_bytes()
    }
}

/// `DeviceContext` disables cudarc's implicit event tracking, so without an owner
/// living until the final host-sync sample, Rust can drop and reuse allocations
/// the stream still reads. Default-off: `CudaSlice::clone()` is a device-to-device
/// copy, so enabling it adds tens of thousands of D2D calls per decode window.
pub(crate) struct Dsv4ForwardKeepalive {
    active: bool,
    bf16: Vec<CudaSlice<half::bf16>>,
    f32: Vec<CudaSlice<f32>>,
    i32: Vec<CudaSlice<i32>>,
    #[cfg(feature = "deepep")]
    i64: Vec<CudaSlice<i64>>,
    u32: Vec<CudaSlice<u32>>,
    u8: Vec<CudaSlice<u8>>,
}

impl Dsv4ForwardKeepalive {
    pub(crate) fn new(active: bool) -> Self {
        Self {
            active,
            bf16: Vec::with_capacity(512),
            f32: Vec::with_capacity(256),
            i32: Vec::with_capacity(128),
            #[cfg(feature = "deepep")]
            i64: Vec::with_capacity(32),
            u32: Vec::with_capacity(16),
            u8: Vec::with_capacity(128),
        }
    }

    pub(crate) fn keep_hidden(&mut self, value: &HiddenStates) {
        if !self.active {
            return;
        }
        self.bf16.push(value.data.clone());
    }

    pub(crate) fn keep_vec(&mut self, value: &DeviceVec) {
        if !self.active {
            return;
        }
        self.bf16.push(value.data.clone());
    }

    pub(crate) fn keep_f32(&mut self, value: &CudaSlice<f32>) {
        if !self.active {
            return;
        }
        self.f32.push(value.clone());
    }

    pub(crate) fn keep_i32(&mut self, value: &CudaSlice<i32>) {
        if !self.active {
            return;
        }
        self.i32.push(value.clone());
    }

    #[cfg(feature = "deepep")]
    pub(crate) fn keep_i64(&mut self, value: &CudaSlice<i64>) {
        if !self.active {
            return;
        }
        self.i64.push(value.clone());
    }

    pub(crate) fn keep_u8(&mut self, value: &CudaSlice<u8>) {
        if !self.active {
            return;
        }
        self.u8.push(value.clone());
    }

    pub(crate) fn keep_u32(&mut self, value: &CudaSlice<u32>) {
        if !self.active {
            return;
        }
        self.u32.push(value.clone());
    }

    pub(super) fn len(&self) -> usize {
        let len =
            self.bf16.len() + self.f32.len() + self.i32.len() + self.u32.len() + self.u8.len();
        #[cfg(feature = "deepep")]
        let len = len + self.i64.len();
        len
    }
}
