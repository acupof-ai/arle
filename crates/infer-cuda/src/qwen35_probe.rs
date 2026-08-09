use super::*;

pub(crate) mod conv_probe {
    use super::*;
    use std::cell::RefCell;

    pub(crate) struct Capture {
        pub(crate) seq_len: usize,
        pub(crate) channels: usize,
        pub(crate) kernel_size: usize,
        pub(crate) input: Vec<bf16>,
        pub(crate) weight: Vec<bf16>,
        pub(crate) output: Vec<bf16>,
        pub(crate) pre_state: Vec<bf16>,
        pub(crate) post_state: Vec<bf16>,
    }

    thread_local! {
        static CAPTURES: RefCell<Option<Vec<Capture>>> = const { RefCell::new(None) };
    }

    pub(crate) fn arm() {
        CAPTURES.with(|captures| {
            assert!(
                captures.replace(Some(Vec::new())).is_none(),
                "conv probe already armed"
            );
        });
    }

    pub(crate) fn drain() -> Vec<Capture> {
        CAPTURES.with(|captures| captures.borrow_mut().take().expect("conv probe not armed"))
    }

    pub(crate) fn disarm() {
        CAPTURES.with(|captures| {
            captures.borrow_mut().take();
        });
    }

    pub(crate) struct Pending {
        seq_len: usize,
        channels: usize,
        kernel_size: usize,
        input: Vec<bf16>,
        weight: Vec<bf16>,
        pre_state: Vec<bf16>,
    }

    pub(crate) fn begin(
        ctx: &DeviceContext,
        linear_idx: usize,
        seq_len: usize,
        channels: usize,
        kernel_size: usize,
        input: &CudaView<'_, bf16>,
        weight: &DeviceVec,
        state: &DeviceVec,
    ) -> Result<Option<Pending>> {
        // 只捕获第一个 linear-attention 层的 conv：同一层的 conv 算术在所有层相同，
        // 一层足以验证正确性，避免下载每一层的 state。
        let needed = linear_idx == 0 && CAPTURES.with(|captures| captures.borrow().is_some());
        if !needed {
            return Ok(None);
        }
        let input = ctx
            .stream
            .clone_dtoh(input)
            .map_err(|e| anyhow!("conv input D2H failed: {e}"))?;
        let weight = ctx
            .stream
            .clone_dtoh(&weight.data)
            .map_err(|e| anyhow!("conv weight D2H failed: {e}"))?;
        let pre_state = ctx
            .stream
            .clone_dtoh(&state.data)
            .map_err(|e| anyhow!("conv pre-state D2H failed: {e}"))?;
        ctx.sync()?;
        Ok(Some(Pending {
            seq_len,
            channels,
            kernel_size,
            input,
            weight,
            pre_state,
        }))
    }

    pub(crate) fn finish(
        ctx: &DeviceContext,
        pending: Option<Pending>,
        output: &CudaViewMut<'_, bf16>,
        state: &DeviceVec,
    ) -> Result<()> {
        let Some(pending) = pending else {
            return Ok(());
        };
        let output = ctx
            .stream
            .clone_dtoh(output)
            .map_err(|e| anyhow!("conv output D2H failed: {e}"))?;
        let post_state = ctx
            .stream
            .clone_dtoh(&state.data)
            .map_err(|e| anyhow!("conv post-state D2H failed: {e}"))?;
        ctx.sync()?;
        CAPTURES.with(|captures| {
            captures
                .borrow_mut()
                .as_mut()
                .expect("conv probe disarmed during capture")
                .push(Capture {
                    seq_len: pending.seq_len,
                    channels: pending.channels,
                    kernel_size: pending.kernel_size,
                    input: pending.input,
                    weight: pending.weight,
                    output,
                    pre_state: pending.pre_state,
                    post_state,
                });
        });
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod gdr_probe {
    use super::*;
    use std::cell::RefCell;

    pub(crate) struct Capture {
        pub(crate) seq_len: usize,
        pub(crate) num_k_heads: usize,
        pub(crate) num_v_heads: usize,
        pub(crate) key_dim: usize,
        pub(crate) val_dim: usize,
        pub(crate) qkv: Vec<bf16>,
        pub(crate) b_proj: Vec<bf16>,
        pub(crate) a_proj: Vec<bf16>,
        pub(crate) dt_bias: Vec<bf16>,
        pub(crate) a_log: Vec<f32>,
        pub(crate) pre_state: Vec<f32>,
        pub(crate) output: Vec<bf16>,
        pub(crate) post_state: Vec<f32>,
    }

    thread_local! {
        static CAPTURES: RefCell<Option<Vec<Capture>>> = const { RefCell::new(None) };
    }

    pub(crate) fn arm() {
        CAPTURES.with(|captures| {
            assert!(
                captures.replace(Some(Vec::new())).is_none(),
                "gdr probe already armed"
            );
        });
    }

    pub(crate) fn drain() -> Vec<Capture> {
        CAPTURES.with(|captures| captures.borrow_mut().take().expect("gdr probe not armed"))
    }

    pub(crate) fn disarm() {
        CAPTURES.with(|captures| {
            captures.borrow_mut().take();
        });
    }

    pub(crate) struct Pending {
        seq_len: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        key_dim: usize,
        val_dim: usize,
        qkv: Vec<bf16>,
        b_proj: Vec<bf16>,
        a_proj: Vec<bf16>,
        dt_bias: Vec<bf16>,
        a_log: Vec<f32>,
        pre_state: Vec<f32>,
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin(
        ctx: &DeviceContext,
        linear_idx: usize,
        seq_len: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        key_dim: usize,
        val_dim: usize,
        qkv: &CudaViewMut<'_, bf16>,
        b_proj: &CudaView<'_, bf16>,
        a_proj: &CudaView<'_, bf16>,
        dt_bias: &DeviceVec,
        a_log: &CudaSlice<f32>,
        state: &CudaSlice<f32>,
    ) -> Result<Option<Pending>> {
        let needed = linear_idx == 0 && CAPTURES.with(|captures| captures.borrow().is_some());
        if !needed {
            return Ok(None);
        }
        let qkv = ctx
            .stream
            .clone_dtoh(qkv)
            .map_err(|e| anyhow!("gdr qkv D2H failed: {e}"))?;
        let b_proj = ctx
            .stream
            .clone_dtoh(b_proj)
            .map_err(|e| anyhow!("gdr b_proj D2H failed: {e}"))?;
        let a_proj = ctx
            .stream
            .clone_dtoh(a_proj)
            .map_err(|e| anyhow!("gdr a_proj D2H failed: {e}"))?;
        let dt_bias = ctx
            .stream
            .clone_dtoh(&dt_bias.data)
            .map_err(|e| anyhow!("gdr dt_bias D2H failed: {e}"))?;
        let a_log = ctx
            .stream
            .clone_dtoh(a_log)
            .map_err(|e| anyhow!("gdr a_log D2H failed: {e}"))?;
        let pre_state = ctx
            .stream
            .clone_dtoh(state)
            .map_err(|e| anyhow!("gdr pre-state D2H failed: {e}"))?;
        ctx.sync()?;
        Ok(Some(Pending {
            seq_len,
            num_k_heads,
            num_v_heads,
            key_dim,
            val_dim,
            qkv,
            b_proj,
            a_proj,
            dt_bias,
            a_log,
            pre_state,
        }))
    }

    pub(crate) fn finish(
        ctx: &DeviceContext,
        pending: Option<Pending>,
        output: &CudaViewMut<'_, bf16>,
        state: &CudaSlice<f32>,
    ) -> Result<()> {
        let Some(pending) = pending else {
            return Ok(());
        };
        let output = ctx
            .stream
            .clone_dtoh(output)
            .map_err(|e| anyhow!("gdr output D2H failed: {e}"))?;
        let post_state = ctx
            .stream
            .clone_dtoh(state)
            .map_err(|e| anyhow!("gdr post-state D2H failed: {e}"))?;
        ctx.sync()?;
        CAPTURES.with(|captures| {
            captures
                .borrow_mut()
                .as_mut()
                .expect("gdr probe disarmed during capture")
                .push(Capture {
                    seq_len: pending.seq_len,
                    num_k_heads: pending.num_k_heads,
                    num_v_heads: pending.num_v_heads,
                    key_dim: pending.key_dim,
                    val_dim: pending.val_dim,
                    qkv: pending.qkv,
                    b_proj: pending.b_proj,
                    a_proj: pending.a_proj,
                    dt_bias: pending.dt_bias,
                    a_log: pending.a_log,
                    pre_state: pending.pre_state,
                    output,
                    post_state,
                });
        });
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod prep_probe {
    use super::*;
    use std::cell::RefCell;

    #[allow(dead_code)]
    pub(crate) struct Capture {
        pub(crate) seq_len: usize,
        pub(crate) num_q_heads: usize,
        pub(crate) num_kv_heads: usize,
        pub(crate) head_dim: usize,
        pub(crate) rotary_dim: usize,
        pub(crate) rms_eps: f32,
        pub(crate) start_pos: i32,
        pub(crate) q_full: Vec<bf16>,
        pub(crate) k_batch: Vec<bf16>,
        pub(crate) q_norm: Vec<bf16>,
        pub(crate) k_norm: Vec<bf16>,
        pub(crate) cos: Vec<bf16>,
        pub(crate) sin: Vec<bf16>,
        pub(crate) q_prepped: Vec<bf16>,
    }

    thread_local! {
        static CAPTURES: RefCell<Option<Vec<Capture>>> = const { RefCell::new(None) };
    }

    pub(crate) fn arm() {
        CAPTURES.with(|captures| {
            assert!(
                captures.replace(Some(Vec::new())).is_none(),
                "prep probe already armed"
            );
        });
    }

    pub(crate) fn drain() -> Vec<Capture> {
        CAPTURES.with(|captures| captures.borrow_mut().take().expect("prep probe not armed"))
    }

    pub(crate) fn disarm() {
        CAPTURES.with(|captures| {
            captures.borrow_mut().take();
        });
    }

    pub(crate) struct Pending {
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rms_eps: f32,
        start_pos: i32,
        q_full: Vec<bf16>,
        k_batch: Vec<bf16>,
        q_norm: Vec<bf16>,
        k_norm: Vec<bf16>,
        cos: Vec<bf16>,
        sin: Vec<bf16>,
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin(
        ctx: &DeviceContext,
        full_idx: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rms_eps: f32,
        q_full: &CudaSlice<bf16>,
        k_batch: &CudaSlice<bf16>,
        q_norm: &DeviceVec,
        k_norm: &DeviceVec,
        cos: &DeviceVec,
        sin: &DeviceVec,
        start_pos: &CudaSlice<i32>,
    ) -> Result<Option<Pending>> {
        let needed = full_idx == 0 && CAPTURES.with(|captures| captures.borrow().is_some());
        if !needed {
            return Ok(None);
        }
        let q_full = ctx
            .stream
            .clone_dtoh(q_full)
            .map_err(|e| anyhow!("prep q_full D2H failed: {e}"))?;
        let k_batch = ctx
            .stream
            .clone_dtoh(k_batch)
            .map_err(|e| anyhow!("prep k_batch D2H failed: {e}"))?;
        let q_norm = ctx
            .stream
            .clone_dtoh(&q_norm.data)
            .map_err(|e| anyhow!("prep q_norm D2H failed: {e}"))?;
        let k_norm = ctx
            .stream
            .clone_dtoh(&k_norm.data)
            .map_err(|e| anyhow!("prep k_norm D2H failed: {e}"))?;
        let cos = ctx
            .stream
            .clone_dtoh(&cos.data)
            .map_err(|e| anyhow!("prep cos D2H failed: {e}"))?;
        let sin = ctx
            .stream
            .clone_dtoh(&sin.data)
            .map_err(|e| anyhow!("prep sin D2H failed: {e}"))?;
        let start_pos = ctx
            .stream
            .clone_dtoh(start_pos)
            .map_err(|e| anyhow!("prep start_pos D2H failed: {e}"))?;
        ctx.sync()?;
        Ok(Some(Pending {
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            rms_eps,
            start_pos: start_pos[0],
            q_full,
            k_batch,
            q_norm,
            k_norm,
            cos,
            sin,
        }))
    }

    pub(crate) fn finish(
        ctx: &DeviceContext,
        pending: Option<Pending>,
        q_prepped: &HiddenStates,
    ) -> Result<()> {
        let Some(pending) = pending else {
            return Ok(());
        };
        let q_prepped = ctx
            .stream
            .clone_dtoh(&q_prepped.data)
            .map_err(|e| anyhow!("prep q_prepped D2H failed: {e}"))?;
        ctx.sync()?;
        CAPTURES.with(|captures| {
            captures
                .borrow_mut()
                .as_mut()
                .expect("prep probe disarmed during capture")
                .push(Capture {
                    seq_len: pending.seq_len,
                    num_q_heads: pending.num_q_heads,
                    num_kv_heads: pending.num_kv_heads,
                    head_dim: pending.head_dim,
                    rotary_dim: pending.rotary_dim,
                    rms_eps: pending.rms_eps,
                    start_pos: pending.start_pos,
                    q_full: pending.q_full,
                    k_batch: pending.k_batch,
                    q_norm: pending.q_norm,
                    k_norm: pending.k_norm,
                    cos: pending.cos,
                    sin: pending.sin,
                    q_prepped,
                });
        });
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod attn_probe {
    use super::*;
    use std::cell::RefCell;

    pub(crate) struct Capture {
        pub(crate) seq_len: usize,
        pub(crate) num_q_heads: usize,
        pub(crate) num_kv_heads: usize,
        pub(crate) head_dim: usize,
        pub(crate) rotary_dim: usize,
        pub(crate) q_prepped: Vec<bf16>,
        pub(crate) k_raw: Vec<bf16>,
        pub(crate) v_raw: Vec<bf16>,
        pub(crate) k_norm: Vec<bf16>,
        pub(crate) cos: Vec<bf16>,
        pub(crate) sin: Vec<bf16>,
        pub(crate) rms_eps: f32,
        pub(crate) start_pos: i32,
        pub(crate) attn_out: Vec<bf16>,
    }

    thread_local! {
        static CAPTURES: RefCell<Option<Vec<Capture>>> = const { RefCell::new(None) };
    }

    pub(crate) fn arm() {
        CAPTURES.with(|captures| {
            assert!(
                captures.replace(Some(Vec::new())).is_none(),
                "attn probe already armed"
            );
        });
    }

    pub(crate) fn drain() -> Vec<Capture> {
        CAPTURES.with(|captures| captures.borrow_mut().take().expect("attn probe not armed"))
    }

    pub(crate) fn disarm() {
        CAPTURES.with(|captures| {
            captures.borrow_mut().take();
        });
    }

    pub(crate) struct Pending {
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        q_prepped: Vec<bf16>,
        k_raw: Vec<bf16>,
        v_raw: Vec<bf16>,
        k_norm: Vec<bf16>,
        cos: Vec<bf16>,
        sin: Vec<bf16>,
        rms_eps: f32,
        start_pos: i32,
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin(
        ctx: &DeviceContext,
        full_idx: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        q_prepped: &CudaSlice<bf16>,
        k_raw: &CudaSlice<bf16>,
        v_raw: &CudaSlice<bf16>,
        k_norm: &DeviceVec,
        cos: &DeviceVec,
        sin: &DeviceVec,
        rms_eps: f32,
        start_pos: &CudaSlice<i32>,
    ) -> Result<Option<Pending>> {
        let needed = full_idx == 0 && CAPTURES.with(|captures| captures.borrow().is_some());
        if !needed {
            return Ok(None);
        }
        let q_prepped = ctx
            .stream
            .clone_dtoh(q_prepped)
            .map_err(|e| anyhow!("attn q_prepped D2H failed: {e}"))?;
        let k_raw = ctx
            .stream
            .clone_dtoh(k_raw)
            .map_err(|e| anyhow!("attn k_raw D2H failed: {e}"))?;
        let v_raw = ctx
            .stream
            .clone_dtoh(v_raw)
            .map_err(|e| anyhow!("attn v_raw D2H failed: {e}"))?;
        let k_norm = ctx
            .stream
            .clone_dtoh(&k_norm.data)
            .map_err(|e| anyhow!("attn k_norm D2H failed: {e}"))?;
        let cos = ctx
            .stream
            .clone_dtoh(&cos.data)
            .map_err(|e| anyhow!("attn cos D2H failed: {e}"))?;
        let sin = ctx
            .stream
            .clone_dtoh(&sin.data)
            .map_err(|e| anyhow!("attn sin D2H failed: {e}"))?;
        let start_pos = ctx
            .stream
            .clone_dtoh(start_pos)
            .map_err(|e| anyhow!("attn start_pos D2H failed: {e}"))?;
        ctx.sync()?;
        Ok(Some(Pending {
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            q_prepped,
            k_raw,
            v_raw,
            k_norm,
            cos,
            sin,
            rms_eps,
            start_pos: start_pos[0],
        }))
    }

    pub(crate) fn finish(
        ctx: &DeviceContext,
        pending: Option<Pending>,
        attn_out: &HiddenStates,
    ) -> Result<()> {
        let Some(pending) = pending else {
            return Ok(());
        };
        let attn_out = ctx
            .stream
            .clone_dtoh(&attn_out.data)
            .map_err(|e| anyhow!("attn attn_out D2H failed: {e}"))?;
        ctx.sync()?;
        CAPTURES.with(|captures| {
            captures
                .borrow_mut()
                .as_mut()
                .expect("attn probe disarmed during capture")
                .push(Capture {
                    seq_len: pending.seq_len,
                    num_q_heads: pending.num_q_heads,
                    num_kv_heads: pending.num_kv_heads,
                    head_dim: pending.head_dim,
                    rotary_dim: pending.rotary_dim,
                    q_prepped: pending.q_prepped,
                    k_raw: pending.k_raw,
                    v_raw: pending.v_raw,
                    k_norm: pending.k_norm,
                    cos: pending.cos,
                    sin: pending.sin,
                    rms_eps: pending.rms_eps,
                    start_pos: pending.start_pos,
                    attn_out,
                });
        });
        Ok(())
    }
}
