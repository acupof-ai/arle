use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    backend::Device,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn rope(
    x: TensorId,
    cos: TensorId,
    sin: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Route `x` through the lazy `backend.rope` whenever a live device handle
    // is available — `Dirty::Device` or `Dirty::Both` (device is cheaper).
    // Wider than the silu / softmax / exp dispatch (Dirty::Device only)
    // because rope follows a rank-3 matmul + rank-4 reshape; if the reshape
    // ever goes lazy its output lands Dirty::Both and we want to stay
    // on-device. cos/sin stay host: Qwen's rope caches are per-seq-len and
    // the uploads are tiny vs. the 4-D rotation. Backward stays eager;
    // `tape.backward`'s pre-walk batch-flush handles Dirty::Device outputs.
    let has_device_handle = {
        let t = store.tensor(x)?;
        t.device_handle.is_some() && t.dirty != Dirty::Host
    };
    let can_use_device_rope = {
        let x_shape = store.tensor(x)?.shape.clone();
        let cos_shape = store.tensor(cos)?.shape.clone();
        validate_shapes(&x_shape, &cos_shape, &store.tensor(sin)?.shape)?;
        // Partial rotary (cos rows < head/2) rides the device kernel on CUDA
        // only — the Metal graph builder is full-rotary-only.
        cos_shape[1] * 2 == x_shape[3]
            || (cos_shape[1] * 2 < x_shape[3] && store.backend().device() == Device::Cuda)
    };
    if has_device_handle && can_use_device_rope {
        rope_device_lazy(x, cos, sin, store, tape)
    } else {
        rope_host_eager(x, cos, sin, store, tape)
    }
}

fn rope_device_lazy(
    x: TensorId,
    cos: TensorId,
    sin: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // cos/sin are host-resident seq-len-keyed caches; a device-resident
    // caller pays one readback each, behavior stays correct.
    store.ensure_host(cos)?;
    store.ensure_host(sin)?;
    store.ensure_device(x)?;

    let x_shape = store.tensor(x)?.shape.clone();
    let cos_shape = store.tensor(cos)?.shape.clone();
    let sin_shape = store.tensor(sin)?.shape.clone();
    validate_shapes(&x_shape, &cos_shape, &sin_shape)?;

    let x_handle = store
        .tensor(x)?
        .device_handle
        .as_ref()
        .ok_or(AutogradError::TapeInvariant(
            "rope: ensure_device left x without a device handle",
        ))?
        .clone();
    let cos_data = store.tensor(cos)?.data.clone();
    let sin_data = store.tensor(sin)?.data.clone();

    let out_handle = store
        .backend()
        .rope(&x_handle, &x_shape, &cos_data, &sin_data)?;
    let output_id = store.alloc_device_tensor(x_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::RoPE,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::RoPECtx { cos, sin },
    }
    .record(store, tape)?;

    Ok(output_id)
}

fn rope_host_eager(
    x: TensorId,
    cos: TensorId,
    sin: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_host(x)?;
    store.ensure_host(cos)?;
    store.ensure_host(sin)?;
    let x_tensor = store.tensor_host(x)?;
    let cos_tensor = store.tensor_host(cos)?;
    let sin_tensor = store.tensor_host(sin)?;
    validate_shapes(&x_tensor.shape, &cos_tensor.shape, &sin_tensor.shape)?;

    let output = if cos_tensor.shape[1] * 2 == x_tensor.shape[3] {
        store.backend().rope_forward(
            &x_tensor.data,
            &x_tensor.shape,
            &cos_tensor.data,
            &sin_tensor.data,
        )?
    } else {
        crate::backend::cpu_rope_forward(
            &x_tensor.data,
            &x_tensor.shape,
            &cos_tensor.data,
            &sin_tensor.data,
        )?
    };

    let output_id = store.alloc(Tensor::new(output, x_tensor.shape.clone(), false)?);
    TapeEntry {
        op: BackwardOp::RoPE,
        output_id,
        input_ids: smallvec![x],
        saved: SavedContext::RoPECtx { cos, sin },
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub(crate) fn rope_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("rope missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::RoPECtx { cos, sin } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "rope backward missing saved context",
        ));
    };

    let x_shape = store.tensor(x)?.shape.clone();
    // cos/sin are host caches — ensure_host is a no-op on the canonical
    // path; reading host data here matches the forward's contract.
    store.ensure_host(cos)?;
    store.ensure_host(sin)?;
    let cos_tensor = store.tensor_host(cos)?;
    let sin_tensor = store.tensor_host(sin)?;
    validate_shapes(&x_shape, &cos_tensor.shape, &sin_tensor.shape)?;
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    if upstream_shape != x_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: x_shape.clone(),
            got: upstream_shape,
        });
    }

    // The kernel handles partial rotary (tail passthrough) — a full-head
    // `==` gate here would make every Qwen3.6 full-attn q/k grad fall to
    // host and cascade the upstream chain (Transpose/Slice backwards) onto
    // the CPU.
    let device_path_ok = {
        let upstream = store.tensor(output_grad_id)?;
        upstream.dirty != Dirty::Host
            && upstream.device_handle.is_some()
            && (cos_tensor.shape[1] * 2 == x_shape[3]
                || (cos_tensor.shape[1] * 2 < x_shape[3]
                    && store.backend().device() == Device::Cuda))
    };
    if device_path_ok {
        let upstream_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let grad_handle = store.backend().rope_backward_device(
            &upstream_handle,
            &x_shape,
            &cos_tensor.data,
            &sin_tensor.data,
        )?;
        let grad_id = store.alloc_device_tensor(x_shape, grad_handle)?;
        return Ok(smallvec![(x, grad_id)]);
    }

    // Host fallback (CPU/Metal). rope backward is rope forward with sin
    // negated:
    //   forward:  y0 = x0*cos - x1*sin,   y1 = x1*cos + x0*sin
    //   backward: gx0 = gy0*cos + gy1*sin, gx1 = gy1*cos - gy0*sin
    //           = rope_forward(gy, cos, -sin)
    let upstream = store.tensor_host(output_grad_id)?;
    let neg_sin = store.backend().neg_forward(&sin_tensor.data)?;
    let grad_x = if cos_tensor.shape[1] * 2 == x_shape[3] {
        store
            .backend()
            .rope_forward(&upstream.data, &x_shape, &cos_tensor.data, &neg_sin)?
    } else {
        crate::backend::cpu_rope_forward(&upstream.data, &x_shape, &cos_tensor.data, &neg_sin)?
    };

    let grad_id = store.alloc(Tensor::new(grad_x, x_shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

fn validate_shapes(x_shape: &[usize], cos_shape: &[usize], sin_shape: &[usize]) -> Result<()> {
    if x_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: x_shape.len(),
        });
    }
    if cos_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: cos_shape.len(),
        });
    }
    if sin_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: sin_shape.len(),
        });
    }

    let head_dim = x_shape[3];
    if !head_dim.is_multiple_of(2) {
        return Err(AutogradError::InvalidRank {
            expected: "even head dim",
            got: head_dim,
        });
    }

    if cos_shape[0] != x_shape[2] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![x_shape[2], head_dim / 2],
            got: cos_shape.to_vec(),
        });
    }
    if sin_shape != cos_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: cos_shape.to_vec(),
            got: sin_shape.to_vec(),
        });
    }
    let rotary_half_dim = cos_shape[1];
    if rotary_half_dim == 0 || rotary_half_dim > head_dim / 2 {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![x_shape[2], head_dim / 2],
            got: cos_shape.to_vec(),
        });
    }

    Ok(())
}
