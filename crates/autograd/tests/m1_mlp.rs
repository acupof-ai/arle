use autograd::{
    Result, Tape, TensorStore,
    module::{Linear, Module},
    ops::{gather_last_dim, gelu, log_softmax, mean, mul_scalar},
    optim::AdamW,
};

#[test]
fn two_layer_mlp_learns_xorish_dataset() -> Result<()> {
    let (inputs, labels) = xorish_dataset();

    let mut store = TensorStore::default();
    let layer1 = Linear::new(2, 32, true, &mut store);
    let layer2 = Linear::new(32, 2, true, &mut store);
    let mut params = layer1.parameters();
    params.extend(layer2.parameters());
    let mut optim = AdamW::new(0.05, (0.9, 0.999), 1e-8, 0.0);

    let x = store.from_slice(&inputs, &[64, 2])?;
    let mut final_loss = f32::INFINITY;

    for _ in 0..100 {
        optim.zero_grad(&params, &mut store);

        let mut tape = Tape::new();
        let hidden = layer1.forward(x, &mut store, &mut tape)?;
        let hidden = gelu(hidden, &mut store, &mut tape)?;
        let logits = layer2.forward(hidden, &mut store, &mut tape)?;
        let log_probs = log_softmax(logits, &mut store, &mut tape)?;
        let nll = gather_last_dim(log_probs, &labels, &mut store, &mut tape)?;
        let neg_nll = mul_scalar(nll, -1.0, &mut store, &mut tape)?;
        let loss = mean(neg_nll, &mut store, &mut tape)?;
        final_loss = store.to_host(loss)?[0];
        tape.backward(loss, &mut store)?;
        optim.step(&params, &mut store);
    }

    assert!(final_loss < 0.5, "final loss was {final_loss}");
    Ok(())
}

fn xorish_dataset() -> (Vec<f32>, Vec<usize>) {
    let prototypes = [
        (-1.0_f32, -1.0_f32, 0_usize),
        (-1.0_f32, 1.0_f32, 1_usize),
        (1.0_f32, -1.0_f32, 1_usize),
        (1.0_f32, 1.0_f32, 0_usize),
    ];

    let mut inputs = Vec::with_capacity(64 * 2);
    let mut labels = Vec::with_capacity(64);
    for repeat in 0..16 {
        let jitter = ((repeat as f32) - 7.5) * 0.02;
        for &(x0, x1, label) in &prototypes {
            inputs.push(x0 + jitter);
            inputs.push(x1 - jitter);
            labels.push(label);
        }
    }

    (inputs, labels)
}
