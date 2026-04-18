use libm::erff;
use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{GpuTensor, TensorId, TensorStore},
};

const INV_SQRT_2: f32 = 0.707_106_77;
const INV_SQRT_2PI: f32 = 0.398_942_3;

pub fn gelu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    let output = input
        .data
        .iter()
        .map(|&value| 0.5 * value * (1.0 + erff(value * INV_SQRT_2)))
        .collect();
    let output_id = store.alloc(GpuTensor::new(
        output,
        input.shape.clone(),
        input.requires_grad,
    )?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Gelu,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::GeluCtx { x },
        });
    }

    Ok(output_id)
}

pub(crate) fn gelu_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::GeluCtx { x } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "gelu backward missing saved input",
        ));
    };
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let input = store.tensor(x)?.clone();
    let upstream = store.tensor(output_grad_id)?.clone();
    if input.shape != upstream.shape {
        return Err(AutogradError::ShapeMismatch {
            expected: input.shape,
            got: upstream.shape,
        });
    }

    let grad = input
        .data
        .iter()
        .zip(upstream.data.iter())
        .map(|(&value, &grad_out)| {
            let erf_term = erff(value * INV_SQRT_2);
            let exp_term = (-0.5 * value * value).exp();
            let derivative = 0.5 * (1.0 + erf_term) + (value * INV_SQRT_2PI * exp_term);
            grad_out * derivative
        })
        .collect();
    let grad_id = store.alloc(GpuTensor::new(grad, input.shape, false)?);
    Ok(smallvec![(x, grad_id)])
}
