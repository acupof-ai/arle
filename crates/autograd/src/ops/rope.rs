use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

pub fn rope(
    x: TensorId,
    cos: TensorId,
    sin: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x_tensor = store.tensor(x)?.clone();
    let cos_tensor = store.tensor(cos)?.clone();
    let sin_tensor = store.tensor(sin)?.clone();
    validate_shapes(&x_tensor.shape, &cos_tensor.shape, &sin_tensor.shape)?;

    let output = store.backend().rope_forward(
        &x_tensor.data,
        &x_tensor.shape,
        &cos_tensor.data,
        &sin_tensor.data,
    )?;

    let output_id = store.alloc(Tensor::new(
        output,
        x_tensor.shape.clone(),
        x_tensor.requires_grad,
    )?);
    if x_tensor.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::RoPE,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::RoPECtx { cos, sin },
        });
    }

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
    let cos_tensor = store.tensor(cos)?.clone();
    let sin_tensor = store.tensor(sin)?.clone();
    validate_shapes(&x_shape, &cos_tensor.shape, &sin_tensor.shape)?;

    let upstream = store.tensor(output_grad_id)?.clone();
    if upstream.shape != x_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: x_shape.clone(),
            got: upstream.shape,
        });
    }

    // rope backward is rope forward with sin negated:
    //   forward:  y0 = x0*cos - x1*sin,   y1 = x1*cos + x0*sin
    //   backward: gx0 = gy0*cos + gy1*sin, gx1 = gy1*cos - gy0*sin
    //           = rope_forward(gy, cos, -sin)
    // Negate on the CPU (cheap; cache only; tiny compared to the 4-D rotation)
    // then dispatch through the backend so Metal / CUDA accelerate the bulk op.
    let neg_sin = store.backend().neg_forward(&sin_tensor.data)?;
    let grad_x =
        store
            .backend()
            .rope_forward(&upstream.data, &x_shape, &cos_tensor.data, &neg_sin)?;

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

    let expected_cache_shape = vec![x_shape[2], head_dim / 2];
    if cos_shape != expected_cache_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_cache_shape.clone(),
            got: cos_shape.to_vec(),
        });
    }
    if sin_shape != expected_cache_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_cache_shape,
            got: sin_shape.to_vec(),
        });
    }

    Ok(())
}
