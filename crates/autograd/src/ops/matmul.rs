use smallvec::smallvec;
use std::collections::HashSet;

use crate::{
    AutogradError, Result,
    backend::{
        Device, cpu_matmul_bt_backward, matmul_bt_output_shape as backend_matmul_bt_output_shape,
        matmul_output_shape as backend_matmul_output_shape,
    },
    ops::chunk_accum::{ChunkSum, SeqAccum},
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Dirty, Tensor, TensorId, TensorStore},
};

pub fn matmul(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_device(a)?;
    store.ensure_device(b)?;
    let a_handle = store
        .tensor(a)?
        .device_handle
        .as_ref()
        .expect("ensure_device")
        .clone();
    let b_handle = store
        .tensor(b)?
        .device_handle
        .as_ref()
        .expect("ensure_device")
        .clone();
    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    let (out_handle, out_shape) = store
        .backend()
        .matmul(&a_handle, &a_shape, &b_handle, &b_shape)?;
    let output_id = store.alloc_device_tensor(out_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::Matmul,
        output_id,
        input_ids: smallvec![a, b],
        saved: SavedContext::MatmulCtx { a, b },
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub fn matmul_bt(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    matmul_bt_with_site(a, b, store, tape, "matmul_bt")
}

pub fn matmul_bt_with_site(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
    site: &'static str,
) -> Result<TensorId> {
    store.ensure_device(a)?;
    store.ensure_device(b)?;
    let a_handle = store
        .tensor(a)?
        .device_handle
        .as_ref()
        .expect("ensure_device")
        .clone();
    let b_handle = store
        .tensor(b)?
        .device_handle
        .as_ref()
        .expect("ensure_device")
        .clone();
    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    let (out_handle, out_shape) = store
        .backend()
        .matmul_bt(&a_handle, &a_shape, &b_handle, &b_shape)?;
    let output_id = store.alloc_device_tensor(out_shape, out_handle)?;

    TapeEntry {
        op: BackwardOp::MatmulBT,
        output_id,
        input_ids: smallvec![a, b],
        saved: SavedContext::MatmulBTCtx { a, b, site },
    }
    .record(store, tape)?;

    Ok(output_id)
}

pub(crate) fn matmul_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::MatmulCtx { a, b } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "matmul backward missing saved context",
        ));
    };

    // Shape gate — borrow-only so we don't force a host readback when both
    // sides are still device-resident. Cloning here would panic on
    // `Dirty::Device` and (worse for `Dirty::Both`) materialise the 1 GB
    // upstream gradient on host even when the device path is taken.
    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    let need_grad_a = store.tensor(a)?.requires_grad;
    let need_grad_b = store.tensor(b)?.requires_grad;
    let expected_shape = matmul_output_shape(&a_shape, &b_shape)?;
    if upstream_shape != expected_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_shape,
            got: upstream_shape,
        });
    }

    let mut grads = GradPairs::new();
    if !need_grad_a && !need_grad_b {
        return Ok(grads);
    }
    match (a_shape.len(), b_shape.len()) {
        (2, 2) | (3, 3) => {}
        _ => {
            return Err(AutogradError::InvalidRank {
                expected: "both operands must be rank-2 or rank-3",
                got: a_shape.len().max(b_shape.len()),
            });
        }
    }

    // When all three operands are still device-resident, dispatch through
    // `matmul_backward_device` so the saved hidden / weight buffers and the
    // upstream gradient never round-trip through host — the LM-head GEMM's
    // `grad_out: &[f32]` is otherwise the single largest (1 GB) readback per
    // step. Heal host-resident operands first (mirrors matmul_bt_backward) so
    // one upstream host grad (e.g. from a host concat backward) doesn't demote
    // this GEMM — and everything downstream — to the host contract.
    if store.backend().device() != Device::Cpu {
        store.ensure_device(a)?;
        store.ensure_device(b)?;
        store.ensure_device(output_grad_id)?;
    }
    let device_path_ok = {
        let a_t = store.tensor(a)?;
        let b_t = store.tensor(b)?;
        let g_t = store.tensor(output_grad_id)?;
        a_t.dirty != Dirty::Host
            && a_t.device_handle.is_some()
            && b_t.dirty != Dirty::Host
            && b_t.device_handle.is_some()
            && g_t.dirty != Dirty::Host
            && g_t.device_handle.is_some()
    };
    if device_path_ok {
        let a_handle = store
            .tensor(a)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let b_handle = store
            .tensor(b)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let g_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let (grad_a_handle, grad_b_handle) = store.backend().matmul_backward_device(
            &a_handle,
            &a_shape,
            &b_handle,
            &b_shape,
            &g_handle,
            &upstream_shape,
            need_grad_a,
            need_grad_b,
        )?;
        if let Some(handle) = grad_a_handle {
            let grad_id = store.alloc_device_tensor(a_shape.clone(), handle)?;
            grads.push((a, grad_id));
        }
        if let Some(handle) = grad_b_handle {
            let grad_id = store.alloc_device_tensor(b_shape.clone(), handle)?;
            grads.push((b, grad_id));
        }
        return Ok(grads);
    }

    // Host-eager fallback. Any operand without a device handle (or already
    // marked `Dirty::Host`) drops the whole call back onto the legacy
    // `matmul_backward(&[f32], …)` contract, matching pre-P2 behaviour.
    let a_tensor = store.tensor_host(a)?;
    let b_tensor = store.tensor_host(b)?;
    let upstream = store.tensor_host(output_grad_id)?;
    let (grad_a_data, grad_b_data) = store.backend().matmul_backward(
        &a_tensor.data,
        &a_tensor.shape,
        &b_tensor.data,
        &b_tensor.shape,
        &upstream.data,
        &upstream.shape,
        need_grad_a,
        need_grad_b,
    )?;
    if need_grad_a {
        let grad_id = store.alloc(Tensor::new(grad_a_data, a_tensor.shape.clone(), false)?);
        grads.push((a, grad_id));
    }
    if need_grad_b {
        let grad_id = store.alloc(Tensor::new(grad_b_data, b_tensor.shape.clone(), false)?);
        grads.push((b, grad_id));
    }

    Ok(grads)
}

pub(crate) fn matmul_bt_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::MatmulBTCtx { a, b, site } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "matmul_bt backward missing saved context",
        ));
    };

    let a_shape = store.tensor(a)?.shape.clone();
    let b_shape = store.tensor(b)?.shape.clone();
    let upstream_shape = store.tensor(output_grad_id)?.shape.clone();
    let need_grad_a = store.tensor(a)?.requires_grad;
    let need_grad_b = store.tensor(b)?.requires_grad;
    let expected_shape = matmul_bt_output_shape(&a_shape, &b_shape)?;
    if upstream_shape != expected_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_shape,
            got: upstream_shape,
        });
    }
    if let Ok(raw_threshold) = std::env::var("ARLE_MATMUL_BT_BWD_TRACE_MIN_ELEMS")
        && let Ok(threshold) = raw_threshold.parse::<usize>()
        && upstream_shape.iter().product::<usize>() >= threshold
    {
        let a_tensor = store.tensor(a)?;
        let b_tensor = store.tensor(b)?;
        let g_tensor = store.tensor(output_grad_id)?;
        eprintln!(
            "arle_matmul_bt_bwd_trace site={site} a={a_shape:?} b={b_shape:?} \
             grad={upstream_shape:?} need_a={need_grad_a} need_b={need_grad_b} \
             dirty_a={:?} dirty_b={:?} dirty_g={:?} has_dev_a={} has_dev_b={} has_dev_g={}",
            a_tensor.dirty,
            b_tensor.dirty,
            g_tensor.dirty,
            a_tensor.device_handle.is_some(),
            b_tensor.device_handle.is_some(),
            g_tensor.device_handle.is_some()
        );
    }

    let mut grads = GradPairs::new();
    if !need_grad_a && !need_grad_b {
        return Ok(grads);
    }
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2",
            got: a_shape.len().max(b_shape.len()),
        });
    }

    if store.backend().device() != Device::Cpu {
        store.ensure_device(a)?;
        store.ensure_device(b)?;
        store.ensure_device(output_grad_id)?;
    }

    let device_path_ok = {
        let a_t = store.tensor(a)?;
        let b_t = store.tensor(b)?;
        let g_t = store.tensor(output_grad_id)?;
        a_t.dirty != Dirty::Host
            && a_t.device_handle.is_some()
            && b_t.dirty != Dirty::Host
            && b_t.device_handle.is_some()
            && g_t.dirty != Dirty::Host
            && g_t.device_handle.is_some()
    };
    if device_path_ok {
        if let Some(tiled) = matmul_bt_lora_backward_tiled(
            a,
            b,
            output_grad_id,
            site,
            &a_shape,
            &b_shape,
            &upstream_shape,
            need_grad_a,
            need_grad_b,
            store,
        )? {
            return Ok(tiled);
        }
        let a_handle = store
            .tensor(a)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let b_handle = store
            .tensor(b)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let g_handle = store
            .tensor(output_grad_id)?
            .device_handle
            .as_ref()
            .expect("checked above")
            .clone();
        let (grad_a_handle, grad_b_handle) = store.backend().matmul_bt_backward_device(
            &a_handle,
            &a_shape,
            &b_handle,
            &b_shape,
            &g_handle,
            &upstream_shape,
            need_grad_a,
            need_grad_b,
        )?;
        if let Some(handle) = grad_a_handle {
            let grad_id = store.alloc_device_tensor(a_shape.clone(), handle)?;
            grads.push((a, grad_id));
        }
        if let Some(handle) = grad_b_handle {
            let grad_id = store.alloc_device_tensor(b_shape.clone(), handle)?;
            grads.push((b, grad_id));
        }
        return Ok(grads);
    }

    let a_tensor = store.tensor_host(a)?;
    let b_tensor = store.tensor_host(b)?;
    let upstream = store.tensor_host(output_grad_id)?;
    let (grad_a_data, grad_b_data) = cpu_matmul_bt_backward(
        &a_tensor.data,
        &a_tensor.shape,
        &b_tensor.data,
        &b_tensor.shape,
        &upstream.data,
        &upstream.shape,
        need_grad_a,
        need_grad_b,
    )?;
    if need_grad_a {
        let grad_id = store.alloc(Tensor::new(grad_a_data, a_tensor.shape.clone(), false)?);
        grads.push((a, grad_id));
    }
    if need_grad_b {
        let grad_id = store.alloc(Tensor::new(grad_b_data, b_tensor.shape.clone(), false)?);
        grads.push((b, grad_id));
    }

    Ok(grads)
}

fn lora_linear_backward_tile_rows() -> usize {
    crate::runtime_flags::lora_linear_bwd_tile_rows()
}

fn is_lora_matmul_site(site: &str) -> bool {
    site.ends_with(".lora_a") || site.ends_with(".lora_b")
}

#[allow(clippy::too_many_arguments)]
fn matmul_bt_lora_backward_tiled(
    a: TensorId,
    b: TensorId,
    output_grad_id: TensorId,
    site: &'static str,
    a_shape: &[usize],
    b_shape: &[usize],
    upstream_shape: &[usize],
    need_grad_a: bool,
    need_grad_b: bool,
    store: &mut TensorStore,
) -> Result<Option<GradPairs>> {
    if !is_lora_matmul_site(site) {
        return Ok(None);
    }
    let tile_rows = lora_linear_backward_tile_rows();
    if a_shape[0] <= tile_rows {
        return Ok(None);
    }

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    let mut tape = Tape::new();
    tape.set_enabled(false);
    let live_before = store.live_ids().into_iter().collect::<HashSet<_>>();
    let mut grad_a = need_grad_a
        .then(|| SeqAccum::new(vec![m, k], 0, store))
        .transpose()?;
    let mut grad_b_acc = ChunkSum::new();

    let mut row0 = 0usize;
    while row0 < m {
        let row1 = (row0 + tile_rows).min(m);
        let rows = row1 - row0;
        let a_chunk = crate::ops::slice(a, &[row0, 0], &[row1, k], store, &mut tape)?;
        let g_chunk = crate::ops::slice(output_grad_id, &[row0, 0], &[row1, n], store, &mut tape)?;
        store.ensure_device(a_chunk)?;
        store.ensure_device(g_chunk)?;
        let a_handle = store
            .tensor(a_chunk)?
            .device_handle
            .as_ref()
            .expect("ensure_device")
            .clone();
        let b_handle = store
            .tensor(b)?
            .device_handle
            .as_ref()
            .expect("checked by caller")
            .clone();
        let g_handle = store
            .tensor(g_chunk)?
            .device_handle
            .as_ref()
            .expect("ensure_device")
            .clone();
        let a_chunk_shape = vec![rows, k];
        let g_chunk_shape = vec![rows, n];
        let (grad_a_part, grad_b_part) = store.backend().matmul_bt_backward_device(
            &a_handle,
            &a_chunk_shape,
            &b_handle,
            b_shape,
            &g_handle,
            &g_chunk_shape,
            need_grad_a,
            need_grad_b,
        )?;
        if let Some(acc) = grad_a.as_mut() {
            let handle = grad_a_part.ok_or(AutogradError::TapeInvariant(
                "matmul_bt lora tiled backward requested grad_a but a chunk produced none",
            ))?;
            let chunk = store.alloc_device_tensor(a_chunk_shape, handle)?;
            acc.write_rows(row0, chunk, store)?;
        }
        if let Some(handle) = grad_b_part {
            let part = store.alloc_device_tensor(b_shape.to_vec(), handle)?;
            grad_b_acc.add(part, store)?;
        }

        let keep = grad_a
            .as_ref()
            .map(SeqAccum::id)
            .into_iter()
            .chain(grad_b_acc.id())
            .collect();
        store.free_new_except(&live_before, &keep)?;
        row0 = row1;
    }

    let mut grads = GradPairs::new();
    if let Some(acc) = grad_a {
        grads.push((a, acc.finish()));
    }
    if need_grad_b {
        let grad_b = grad_b_acc
            .finish(store)?
            .ok_or(AutogradError::TapeInvariant(
                "matmul_bt lora tiled backward requested grad_b but produced none",
            ))?;
        grads.push((b, grad_b));
    }
    if upstream_shape != [m, n] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![m, n],
            got: upstream_shape.to_vec(),
        });
    }
    Ok(Some(grads))
}

fn matmul_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    backend_matmul_output_shape(a_shape, b_shape)
}

fn matmul_bt_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    backend_matmul_bt_output_shape(a_shape, b_shape)
}
