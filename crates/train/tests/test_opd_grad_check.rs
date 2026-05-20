use autograd::{AutogradError, Tape, Tensor, TensorStore};
use train::loss::kl_distill_loss;

const EPSILON: f32 = 1.0e-3;
const MAX_REL_ERR: f32 = 1.0e-2;

fn kl_loss_value(student_logits: &[f32], teacher_logits: &[f32], shape: &[usize]) -> f32 {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    tape.set_enabled(false);

    let student = store
        .alloc(Tensor::new(student_logits.to_vec(), shape.to_vec(), true).expect("student logits"));
    let teacher = store.alloc(
        Tensor::new(teacher_logits.to_vec(), shape.to_vec(), false).expect("teacher logits"),
    );
    let num_positions = student_logits.len() / shape.last().copied().expect("vocab dim");
    let loss =
        kl_distill_loss(student, teacher, num_positions, &mut store, &mut tape).expect("kl loss");
    store.to_host(loss).expect("loss host value")[0]
}

fn kl_loss_student_grad(
    student_logits: &[f32],
    teacher_logits: &[f32],
    shape: &[usize],
) -> Vec<f32> {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();

    let student = store
        .alloc(Tensor::new(student_logits.to_vec(), shape.to_vec(), true).expect("student logits"));
    let teacher = store.alloc(
        Tensor::new(teacher_logits.to_vec(), shape.to_vec(), false).expect("teacher logits"),
    );
    let num_positions = student_logits.len() / shape.last().copied().expect("vocab dim");
    let loss =
        kl_distill_loss(student, teacher, num_positions, &mut store, &mut tape).expect("kl loss");
    tape.backward(loss, &mut store).expect("backward");

    let grad = store
        .get(student)
        .and_then(|tensor| tensor.grad)
        .expect("student logits gradient");
    store.to_host(grad).expect("gradient host value")
}

#[test]
fn kl_distill_loss_student_logits_grad_matches_finite_difference() {
    let shape = [3, 4];
    let student_logits = vec![
        0.90, -0.10, 0.20, -0.40, -0.30, 0.80, 0.10, -1.00, 1.20, -0.40, 0.50, -0.80,
    ];
    let teacher_logits = vec![
        -0.40, 0.70, -0.20, 0.10, 0.40, -0.60, 0.90, -0.10, -0.70, 0.30, -0.50, 0.80,
    ];

    let analytic = kl_loss_student_grad(&student_logits, &teacher_logits, &shape);
    assert_eq!(analytic.len(), student_logits.len());

    let mut max_rel_err = 0.0_f32;
    let mut worst = (0usize, 0.0_f32, 0.0_f32, 0.0_f32);
    for i in 0..student_logits.len() {
        let mut plus = student_logits.clone();
        plus[i] += EPSILON;
        let mut minus = student_logits.clone();
        minus[i] -= EPSILON;

        let loss_plus = kl_loss_value(&plus, &teacher_logits, &shape);
        let loss_minus = kl_loss_value(&minus, &teacher_logits, &shape);
        let finite_diff = (loss_plus - loss_minus) / (2.0 * EPSILON);
        let abs_err = (analytic[i] - finite_diff).abs();
        let denom = analytic[i].abs().max(finite_diff.abs()).max(1.0e-6);
        let rel_err = abs_err / denom;

        if rel_err > max_rel_err {
            max_rel_err = rel_err;
            worst = (i, analytic[i], finite_diff, abs_err);
        }
    }

    eprintln!("kl_distill_loss finite-diff max_relative_error={max_rel_err:.6}");
    assert!(
        max_rel_err < MAX_REL_ERR,
        "finite-diff gradient check failed: max_rel_err={max_rel_err:.6} \
         at student_logits[{}], analytic={}, finite_diff={}, abs_err={}",
        worst.0,
        worst.1,
        worst.2,
        worst.3
    );
}

#[test]
fn kl_distill_loss_stays_finite_for_wide_range_teacher_logits() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let student = store.alloc(
        Tensor::new(
            vec![0.25, -0.50, 0.75, -1.00, -0.25, 0.50, -0.75, 1.00],
            vec![2, 4],
            true,
        )
        .expect("student logits"),
    );
    let teacher = store.alloc(
        Tensor::new(
            vec![1000.0, 0.0, -1000.0, 500.0, -800.0, 800.0, 0.0, -400.0],
            vec![2, 4],
            false,
        )
        .expect("teacher logits"),
    );

    let loss = kl_distill_loss(student, teacher, 2, &mut store, &mut tape)
        .expect("wide-range teacher logits must not overflow");
    let loss_value = store.to_host(loss).expect("loss host value")[0];
    assert!(loss_value.is_finite(), "loss must be finite: {loss_value}");

    tape.backward(loss, &mut store)
        .expect("wide-range backward must stay finite");
    let grad = store
        .get(student)
        .and_then(|tensor| tensor.grad)
        .expect("student logits gradient");
    let grad_values = store.to_host(grad).expect("gradient host value");
    assert!(
        grad_values.iter().all(|value| value.is_finite()),
        "gradient must be finite: {grad_values:?}"
    );
}

#[test]
fn kl_distill_loss_rejects_mismatched_logit_shapes() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let student = store.alloc(Tensor::new(vec![0.0; 6], vec![2, 3], true).expect("student logits"));
    let teacher =
        store.alloc(Tensor::new(vec![0.0; 8], vec![2, 4], false).expect("teacher logits"));

    let err = kl_distill_loss(student, teacher, 2, &mut store, &mut tape)
        .expect_err("mismatched logits must fail before softmax");

    let AutogradError::TapeInvariant(message) = err else {
        panic!("expected TapeInvariant, got {err:?}");
    };
    assert!(message.contains("student_logits and teacher_logits"));
    assert!(message.contains("identical shapes"));
    assert!(message.contains("matching vocab_size"));
}

#[test]
fn kl_distill_loss_rejects_missing_vocab_axis_with_hint() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let student = store.alloc(Tensor::new(vec![0.0], vec![], true).expect("student scalar"));
    let teacher = store.alloc(Tensor::new(vec![0.0], vec![], false).expect("teacher scalar"));

    let err = kl_distill_loss(student, teacher, 1, &mut store, &mut tape)
        .expect_err("rank-0 logits must fail before softmax");

    let AutogradError::TapeInvariant(message) = err else {
        panic!("expected TapeInvariant, got {err:?}");
    };
    assert!(message.contains("at least one dimension"));
    assert!(message.contains("[..., vocab_size]"));
}

#[test]
fn kl_distill_loss_rejects_zero_vocab_with_hint() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let student = store.alloc(Tensor::new(vec![], vec![2, 0], true).expect("student logits"));
    let teacher = store.alloc(Tensor::new(vec![], vec![2, 0], false).expect("teacher logits"));

    let err = kl_distill_loss(student, teacher, 2, &mut store, &mut tape)
        .expect_err("zero-vocab logits must fail before softmax");

    let AutogradError::TapeInvariant(message) = err else {
        panic!("expected TapeInvariant, got {err:?}");
    };
    assert!(message.contains("vocab"));
    assert!(message.contains("non-zero"));
    assert!(message.contains("config.json"));
}

#[test]
fn kl_distill_loss_rejects_teacher_logits_requiring_grad() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let student = store.alloc(Tensor::new(vec![0.0; 6], vec![2, 3], true).expect("student logits"));
    let teacher = store.alloc(Tensor::new(vec![0.0; 6], vec![2, 3], true).expect("teacher logits"));

    let err = kl_distill_loss(student, teacher, 2, &mut store, &mut tape)
        .expect_err("teacher logits must be frozen");

    let AutogradError::TapeInvariant(message) = err else {
        panic!("expected TapeInvariant, got {err:?}");
    };
    assert!(message.contains("teacher_logits"));
    assert!(message.contains("requires_grad=false"));
    assert!(message.contains("teacher"));
}

#[test]
fn kl_distill_loss_rejects_student_logits_without_grad() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let student =
        store.alloc(Tensor::new(vec![0.0; 6], vec![2, 3], false).expect("student logits"));
    let teacher =
        store.alloc(Tensor::new(vec![0.0; 6], vec![2, 3], false).expect("teacher logits"));

    let err = kl_distill_loss(student, teacher, 2, &mut store, &mut tape)
        .expect_err("frozen student logits must fail before producing a no-grad loss");

    let AutogradError::TapeInvariant(message) = err else {
        panic!("expected TapeInvariant, got {err:?}");
    };
    assert!(message.contains("student_logits"));
    assert!(message.contains("requires_grad=true"));
    assert!(message.contains("frozen student loss"));
}

#[test]
fn kl_distill_loss_rejects_stale_num_positions() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let student = store.alloc(Tensor::new(vec![0.0; 6], vec![2, 3], true).expect("student logits"));
    let teacher =
        store.alloc(Tensor::new(vec![0.0; 6], vec![2, 3], false).expect("teacher logits"));

    let err = kl_distill_loss(student, teacher, 1, &mut store, &mut tape)
        .expect_err("stale num_positions must fail before softmax");

    let AutogradError::TapeInvariant(message) = err else {
        panic!("expected TapeInvariant, got {err:?}");
    };
    assert!(message.contains("num_positions"));
    assert!(message.contains("rollout.len()"));
}
