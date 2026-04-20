use libm::erff;
use smallvec::smallvec;

use crate::{
    AutogradError, Result,
    tape::{BackwardOp, GradPairs, SavedContext, Tape, TapeEntry},
    tensor::{Tensor, TensorId, TensorStore},
};

const INV_SQRT_2: f32 = 0.707_106_77;
const INV_SQRT_2PI: f32 = 0.398_942_3;

pub fn exp(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    let output = store.backend().exp_forward(&input.data)?;
    let output_id = store.alloc(Tensor::new(
        output,
        input.shape.clone(),
        input.requires_grad,
    )?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Exp,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::Tensor(output_id),
        });
    }

    Ok(output_id)
}

pub fn gelu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    let output = input
        .data
        .iter()
        .map(|&value| 0.5 * value * (1.0 + erff(value * INV_SQRT_2)))
        .collect();
    let output_id = store.alloc(Tensor::new(
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

pub fn silu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    let output = store.backend().silu_forward(&input.data)?;
    let output_id = store.alloc(Tensor::new(
        output,
        input.shape.clone(),
        input.requires_grad,
    )?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Silu,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::SiluCtx { x },
        });
    }

    Ok(output_id)
}

pub fn sigmoid(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    let input = store.tensor(x)?.clone();
    let output: Vec<f32> = input
        .data
        .iter()
        .map(|&value| 1.0 / (1.0 + (-value).exp()))
        .collect();
    let output_id = store.alloc(Tensor::new(
        output,
        input.shape.clone(),
        input.requires_grad,
    )?);

    if input.requires_grad {
        tape.record(TapeEntry {
            op: BackwardOp::Sigmoid,
            output_id,
            input_ids: smallvec![x],
            saved: SavedContext::SigmoidCtx { y: output_id },
        });
    }

    Ok(output_id)
}

pub(crate) fn exp_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("exp missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::Tensor(y_id) = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "exp backward missing saved output",
        ));
    };

    let output = store.tensor(y_id)?.clone();
    let upstream = store.tensor(output_grad_id)?.clone();
    if output.shape != upstream.shape {
        return Err(AutogradError::ShapeMismatch {
            expected: output.shape,
            got: upstream.shape,
        });
    }

    let grad = store.backend().mul_forward(&output.data, &upstream.data)?;
    let grad_id = store.alloc(Tensor::new(grad, output.shape, false)?);
    Ok(smallvec![(x, grad_id)])
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
    let grad_id = store.alloc(Tensor::new(grad, input.shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

pub(crate) fn silu_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let SavedContext::SiluCtx { x } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "silu backward missing saved input",
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
            let sigmoid = 1.0 / (1.0 + (-value).exp());
            let derivative = sigmoid + (value * sigmoid * (1.0 - sigmoid));
            grad_out * derivative
        })
        .collect();
    let grad_id = store.alloc(Tensor::new(grad, input.shape, false)?);
    Ok(smallvec![(x, grad_id)])
}

pub(crate) fn sigmoid_backward(
    entry: &TapeEntry,
    output_grad_id: TensorId,
    store: &mut TensorStore,
) -> Result<GradPairs> {
    let x = *entry
        .input_ids
        .first()
        .ok_or(AutogradError::TapeInvariant("sigmoid missing input"))?;
    if !store.tensor(x)?.requires_grad {
        return Ok(GradPairs::new());
    }

    let SavedContext::SigmoidCtx { y } = entry.saved.clone() else {
        return Err(AutogradError::TapeInvariant(
            "sigmoid backward missing saved output",
        ));
    };

    let output = store.tensor(y)?.clone();
    let upstream = store.tensor(output_grad_id)?.clone();
    if output.shape != upstream.shape {
        return Err(AutogradError::ShapeMismatch {
            expected: output.shape,
            got: upstream.shape,
        });
    }

    let grad = output
        .data
        .iter()
        .zip(upstream.data.iter())
        .map(|(&value, &grad_out)| grad_out * value * (1.0 - value))
        .collect();
    let grad_id = store.alloc(Tensor::new(grad, output.shape, false)?);
    Ok(smallvec![(x, grad_id)])
}
