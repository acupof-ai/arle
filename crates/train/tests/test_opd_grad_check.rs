use autograd::{Tape, Tensor, TensorStore};
use train::loss::kl_distill_loss;

const EPSILON: f32 = 1.0e-3;
const MAX_REL_ERR: f32 = 1.0e-2;

fn kl_loss_value(student_logits: &[f32], teacher_logits: &[f32], shape: &[usize]) -> f32 {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    tape.set_enabled(false);

    let student = store.alloc(
        Tensor::new(student_logits.to_vec(), shape.to_vec(), false).expect("student logits"),
    );
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
