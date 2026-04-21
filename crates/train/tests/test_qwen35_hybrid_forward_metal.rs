#![cfg(feature = "metal")]

use std::sync::Arc;

use autograd::{Backend, Tape, TensorStore, backend_metal::MetalBackend};
use train::qwen35::Qwen35Model;

mod common;

use common::qwen35_test_support::{TestResult, hybrid_qwen35_config};

#[test]
fn qwen35_hybrid_forward_matches_cpu_on_metal_backend() -> TestResult {
    let cfg = hybrid_qwen35_config();

    let mut cpu_store = TensorStore::default();
    let cpu_model = Qwen35Model::new_for_eval(&cfg, &mut cpu_store)?;
    let mut cpu_tape = Tape::new();
    let cpu_logits = cpu_model.forward_tokens(&[1, 2, 3, 4], &mut cpu_store, &mut cpu_tape)?;
    let cpu_values = cpu_store.to_host(cpu_logits)?;

    let backend: Arc<dyn Backend> = Arc::new(MetalBackend);
    let mut metal_store = TensorStore::with_backend(backend);
    let metal_model = Qwen35Model::new_for_eval(&cfg, &mut metal_store)?;
    let mut metal_tape = Tape::new();
    let metal_logits =
        metal_model.forward_tokens(&[1, 2, 3, 4], &mut metal_store, &mut metal_tape)?;
    let metal_values = metal_store.to_host(metal_logits)?;

    assert_eq!(cpu_values.len(), metal_values.len());
    for (idx, (&cpu, &metal)) in cpu_values.iter().zip(metal_values.iter()).enumerate() {
        let abs_err = (cpu - metal).abs();
        let rel_err = abs_err / cpu.abs().max(1.0e-5);
        assert!(
            abs_err <= 2.0e-4 || rel_err <= 2.0e-4,
            "hybrid cpu/metal forward drift at index {idx}: cpu={cpu} metal={metal} abs={abs_err} rel={rel_err}"
        );
    }

    Ok(())
}
