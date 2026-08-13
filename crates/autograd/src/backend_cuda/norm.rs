use super::*;

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_rms_norm(
    backend: &CudaBackend,
    x: &[f32],
    weight: &[f32],
    shape: &[usize],
    eps: f32,
) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected: usize = shape.iter().product();
    if x.len() != expected {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![x.len()],
        });
    }
    if weight.len() != last_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![last_dim],
            got: vec![weight.len()],
        });
    }
    let rows = expected / last_dim;
    let cols = i32::try_from(last_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda rms_norm cols exceeds i32"))?;
    let d_x = backend.upload_slice(x, shape)?;
    let d_w = backend
        .stream
        .clone_htod(weight)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(expected)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
    launch_rows(
        &backend.stream,
        backend.kernels.function("rms_norm_f32")?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(&d_x)
                .arg(&d_w)
                .arg(&cols)
                .arg(&eps);
            builder
        },
    )?;
    cuda_download(backend, &d_out, expected)
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_rms_norm_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    weight: &[f32],
    shape: &[usize],
    eps: f32,
) -> Result<DeviceHandle> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected = shape_size(shape);
    if weight.len() != last_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![last_dim],
            got: vec![weight.len()],
        });
    }
    let rows = expected / last_dim;
    let cols = i32::try_from(last_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda rms_norm cols exceeds i32"))?;
    let d_w = backend
        .stream
        .clone_htod(weight)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;

    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
    if let DeviceHandle::CudaBf16(storage) = x {
        let d_x = backend.cuda_bf16_storage_slice(storage)?;
        if d_x.len() != expected {
            return Err(AutogradError::DataLengthMismatch {
                len: d_x.len(),
                shape: shape.to_vec(),
                size: expected,
            });
        }
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(expected)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
        let func = backend
            .kernels
            .function_for("rms_norm_f32", TapeDtype::Bf16)?;
        launch_rows(
            &backend.stream,
            &func,
            rows,
            BLOCK,
            SHARED,
            |mut builder| {
                builder
                    .arg(&mut d_out)
                    .arg(d_x)
                    .arg(&d_w)
                    .arg(&cols)
                    .arg(&eps);
                builder
            },
        )?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }
    let d_x = backend.cuda_slice(x, "rms_norm")?;
    if d_x.len() != expected {
        return Err(AutogradError::DataLengthMismatch {
            len: d_x.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(expected)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    launch_rows(
        &backend.stream,
        backend.kernels.function("rms_norm_f32")?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_x)
                .arg(&d_w)
                .arg(&cols)
                .arg(&eps);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

// Device-resident backward for `rms_norm`. Three kernels:
//   1. `rms_norm_inv_rms_f32` — one block per row, reduces sum_sq and
//      emits `inv_rms[rows]` to a device scratch buffer.
//   2. `rms_norm_backward_x_f32` — one block per row, consumes the saved
//      `inv_rms` and reduces `dot` (one shared-mem reduction).
//   3. `rms_norm_backward_w_f32` — one block per column, accumulates
//      `upstream * x * inv_rms` across rows and reduces to grad_w.
// Returned handles are unevaluated; the terminal `eval` belongs to the
// caller.
#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_rms_norm_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    x: &DeviceHandle,
    weight: &DeviceHandle,
    shape: &[usize],
    eps: f32,
    need_grad_x: bool,
    need_grad_w: bool,
) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
    if !need_grad_x && !need_grad_w {
        return Ok((None, None));
    }
    let hidden = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if hidden == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let total = shape_size(shape);
    let rows = total / hidden;

    // Lane follows the saved x's dtype so the adjoint reads what forward saw;
    // grad_w stays f32 (param-grad boundary) in both lanes.
    if let DeviceHandle::CudaBf16(storage) = x {
        let d_x = backend.cuda_bf16_storage_slice(storage)?;
        let d_up_op = backend.bf16_operand(upstream, "rms_norm_backward_device")?;
        let d_up = d_up_op.get();
        let d_w = backend.cuda_slice(weight, "rms_norm_backward_device")?;
        if d_up.len() != total || d_x.len() != total || d_w.len() != hidden {
            return Err(AutogradError::ShapeMismatch {
                expected: shape.to_vec(),
                got: vec![d_up.len()],
            });
        }
        let mut d_inv = backend
            .stream
            .alloc_zeros::<f32>(rows.max(1))
            .map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (rms_norm_backward inv_rms)")
            })?;
        let cols_i = i32::try_from(hidden)
            .map_err(|_| AutogradError::TapeInvariant("cuda rms_norm_backward cols exceeds i32"))?;
        const BLOCK: u32 = 256;
        const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
        if rows > 0 {
            let func = backend
                .kernels
                .function_for("rms_norm_inv_rms_f32", TapeDtype::Bf16)?;
            launch_rows(
                &backend.stream,
                &func,
                rows,
                BLOCK,
                SHARED,
                |mut builder| {
                    builder.arg(&mut d_inv).arg(d_x).arg(&cols_i).arg(&eps);
                    builder
                },
            )?;
        }
        let grad_x = if need_grad_x {
            let mut d_grad = backend.stream.alloc_zeros::<u16>(total).map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (rms_norm_backward grad_x)")
            })?;
            if rows > 0 {
                let func = backend
                    .kernels
                    .function_for("rms_norm_backward_x_f32", TapeDtype::Bf16)?;
                launch_rows(
                    &backend.stream,
                    &func,
                    rows,
                    BLOCK,
                    SHARED,
                    |mut builder| {
                        builder
                            .arg(&mut d_grad)
                            .arg(d_up)
                            .arg(d_x)
                            .arg(d_w)
                            .arg(&d_inv)
                            .arg(&cols_i);
                        builder
                    },
                )?;
            }
            Some(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_grad)))
        } else {
            None
        };
        let grad_w = if need_grad_w {
            let mut d_grad = backend.stream.alloc_zeros::<f32>(hidden).map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (rms_norm_backward grad_w)")
            })?;
            if rows > 0 && hidden > 0 {
                let rows_i = i32::try_from(rows).map_err(|_| {
                    AutogradError::TapeInvariant("cuda rms_norm_backward rows exceeds i32")
                })?;
                let func = backend
                    .kernels
                    .function_for("rms_norm_backward_w_f32", TapeDtype::Bf16)?;
                launch_rows(
                    &backend.stream,
                    &func,
                    hidden,
                    BLOCK,
                    SHARED,
                    |mut builder| {
                        builder
                            .arg(&mut d_grad)
                            .arg(d_up)
                            .arg(d_x)
                            .arg(&d_inv)
                            .arg(&rows_i)
                            .arg(&cols_i);
                        builder
                    },
                )?;
            }
            Some(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
        } else {
            None
        };
        return Ok((grad_x, grad_w));
    }

    let d_up_op = backend.f32_operand(upstream, "rms_norm_backward_device")?;
    let d_x_op = backend.f32_operand(x, "rms_norm_backward_device")?;
    let d_up = d_up_op.get();
    let d_x = d_x_op.get();
    let d_w = backend.cuda_slice(weight, "rms_norm_backward_device")?;
    if d_up.len() != total || d_x.len() != total || d_w.len() != hidden {
        return Err(AutogradError::ShapeMismatch {
            expected: shape.to_vec(),
            got: vec![d_up.len()],
        });
    }

    // Inv_rms scratch buffer.
    let mut d_inv = backend
        .stream
        .alloc_zeros::<f32>(rows.max(1))
        .map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (rms_norm_backward inv_rms)")
        })?;
    if rows > 0 {
        let cols_i = i32::try_from(hidden)
            .map_err(|_| AutogradError::TapeInvariant("cuda rms_norm_backward cols exceeds i32"))?;
        const BLOCK: u32 = 256;
        const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
        launch_rows(
            &backend.stream,
            backend.kernels.function("rms_norm_inv_rms_f32")?,
            rows,
            BLOCK,
            SHARED,
            |mut builder| {
                builder.arg(&mut d_inv).arg(d_x).arg(&cols_i).arg(&eps);
                builder
            },
        )?;
    }

    let grad_x = if need_grad_x {
        let mut d_grad = backend.stream.alloc_zeros::<f32>(total).map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (rms_norm_backward grad_x)")
        })?;
        if rows > 0 {
            let cols_i = i32::try_from(hidden).map_err(|_| {
                AutogradError::TapeInvariant("cuda rms_norm_backward cols exceeds i32")
            })?;
            const BLOCK: u32 = 256;
            const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
            launch_rows(
                &backend.stream,
                backend.kernels.function("rms_norm_backward_x_f32")?,
                rows,
                BLOCK,
                SHARED,
                |mut builder| {
                    builder
                        .arg(&mut d_grad)
                        .arg(d_up)
                        .arg(d_x)
                        .arg(d_w)
                        .arg(&d_inv)
                        .arg(&cols_i);
                    builder
                },
            )?;
        }
        Some(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
    } else {
        None
    };

    let grad_w = if need_grad_w {
        let mut d_grad = backend.stream.alloc_zeros::<f32>(hidden).map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (rms_norm_backward grad_w)")
        })?;
        if rows > 0 && hidden > 0 {
            let rows_i = i32::try_from(rows).map_err(|_| {
                AutogradError::TapeInvariant("cuda rms_norm_backward rows exceeds i32")
            })?;
            let cols_i = i32::try_from(hidden).map_err(|_| {
                AutogradError::TapeInvariant("cuda rms_norm_backward cols exceeds i32")
            })?;
            const BLOCK: u32 = 256;
            const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
            launch_rows(
                &backend.stream,
                backend.kernels.function("rms_norm_backward_w_f32")?,
                hidden,
                BLOCK,
                SHARED,
                |mut builder| {
                    builder
                        .arg(&mut d_grad)
                        .arg(d_up)
                        .arg(d_x)
                        .arg(&d_inv)
                        .arg(&rows_i)
                        .arg(&cols_i);
                    builder
                },
            )?;
        }
        Some(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
    } else {
        None
    };

    Ok((grad_x, grad_w))
}
