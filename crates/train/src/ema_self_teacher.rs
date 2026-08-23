//! EMA self-teacher for ARLE's Self-OPD (SOPD, #91): a second [`Qwen35Model`]
//! that shares the student's base parameters but carries its own LoRA adapter
//! (the EMA of the student adapter), consumed as a teacher via
//! [`InProcessTeacher`].
//!
//! Snapshot/restore (R2): on a gate-fail the student adapter, the EMA adapter,
//! AND the AdamW moments must roll back together as ONE unit. Restoring only
//! the student adapter would leave the EMA trained against the rejected
//! student state and the optimizer moments stale — the same partial-rollback
//! failure mode as the DSv4-EAGLE truncate that restored only
//! `compressed.seq_len` and corrupted the draft at the boundary.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow};
use autograd::{TensorId, TensorStore, adamw_state::AdamWState, optim::AdamW};

use crate::{
    lora::{LoraConfig, LoraTargetSet},
    qwen35::Qwen35Model,
    teacher_infer::InProcessTeacher,
};

/// Sort by base name so the same adapter lands at the same index across
/// student and EMA.
fn pair_adapters(map: &HashMap<&'static str, TensorId>) -> Result<Vec<(TensorId, TensorId)>> {
    let mut a_by_base: HashMap<&str, TensorId> = HashMap::new();
    let mut b_by_base: HashMap<&str, TensorId> = HashMap::new();

    for (&name, &id) in map {
        if let Some(base) = name.strip_suffix(".lora_a") {
            a_by_base.insert(base, id);
        } else if let Some(base) = name.strip_suffix(".lora_b") {
            b_by_base.insert(base, id);
        } else {
            return Err(anyhow!(
                "adapter name {name:?} ends in neither .lora_a nor .lora_b"
            ));
        }
    }

    let mut bases: Vec<&str> = a_by_base.keys().copied().collect();
    bases.sort_unstable();

    let mut pairs = Vec::with_capacity(bases.len());
    for base in bases {
        let a = *a_by_base
            .get(base)
            .ok_or_else(|| anyhow!("adapter base {base:?} has no .lora_a"))?;
        let b = *b_by_base
            .get(base)
            .ok_or_else(|| anyhow!("adapter base {base:?} has a .lora_a but no .lora_b"))?;
        pairs.push((a, b));
    }

    if pairs.len() != b_by_base.len() {
        return Err(anyhow!(
            "adapter pairing mismatch: {} a-tensors but {} b-tensors",
            pairs.len(),
            b_by_base.len()
        ));
    }

    Ok(pairs)
}

pub struct EmaSelfTeacher {
    model: Qwen35Model,
    /// Layer-ordered (sorted by base name) so index `i` aligns with student.
    adapter_ids: Vec<(TensorId, TensorId)>,
    /// EMA model params MINUS the student's trainable adapter ids.
    /// `share_base_parameters_from` folds the student adapter ids into the EMA
    /// model's `param_ids`, so `all_parameter_ids()` would expose them as
    /// teacher params — which the OPD step rejects (`requires_grad=true`).
    teacher_param_ids: Vec<TensorId>,
}

impl EmaSelfTeacher {
    /// Call immediately after building the student, before any other store
    /// scratch: `new_lora_from_base` calls `store.retain_ids(student ∪ ema)`,
    /// freeing every other store tensor.
    ///
    /// EMA adapter is initialized to an EXACT COPY of the student adapter:
    /// ① resume — a resumed student has a non-zero adapter; ② bilinearity —
    /// the elementwise EMA update on A/B only tracks the student's B·A delta
    /// when EMA and student share the same basis.
    pub fn from_student(
        student: &Qwen35Model,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        store: &mut TensorStore,
    ) -> Result<Self> {
        let model = Qwen35Model::new_lora_from_base(student, lora, target_set, store)?;
        let adapter_ids = pair_adapters(&model.adapter_name_map())?;

        // TensorIds are allocation-stable, so this snapshot stays valid.
        let student_adapter_ids: HashSet<TensorId> =
            student.adapter_name_map().values().copied().collect();
        let teacher_param_ids: Vec<TensorId> = model
            .all_parameter_ids()
            .into_iter()
            .filter(|id| !student_adapter_ids.contains(id))
            .collect();

        let mut teacher = Self {
            model,
            adapter_ids,
            teacher_param_ids,
        };
        teacher.copy_from_student(student, store)?;
        teacher.freeze_adapter(store)?;
        Ok(teacher)
    }

    /// EMA adapter is host-side blended, never autograd-optimized — must be
    /// frozen so `validate_teacher_params` accepts the EMA model.
    fn freeze_adapter(&self, store: &mut TensorStore) -> Result<()> {
        for &(a, b) in &self.adapter_ids {
            store.set_requires_grad(a, false)?;
            store.set_requires_grad(b, false)?;
        }
        Ok(())
    }

    fn student_adapter_pairs(student: &Qwen35Model) -> Vec<(TensorId, TensorId)> {
        pair_adapters(&student.adapter_name_map())
            .expect("student adapter map must pair into (lora_a, lora_b) tuples")
    }

    fn copy_from_student(&mut self, student: &Qwen35Model, store: &mut TensorStore) -> Result<()> {
        let student_pairs = Self::student_adapter_pairs(student);
        if student_pairs.len() != self.adapter_ids.len() {
            return Err(anyhow!(
                "adapter count mismatch: student has {} pairs, EMA has {}",
                student_pairs.len(),
                self.adapter_ids.len()
            ));
        }

        for ((student_a, student_b), &(ema_a, ema_b)) in
            student_pairs.iter().zip(self.adapter_ids.iter())
        {
            copy_tensor(store, *student_a, ema_a)?;
            copy_tensor(store, *student_b, ema_b)?;
        }
        Ok(())
    }

    pub fn update(
        &mut self,
        student: &Qwen35Model,
        store: &mut TensorStore,
        alpha: f32,
    ) -> Result<()> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(anyhow!(
                "EMA alpha must be in (0.0, 1.0), got {alpha} (hint: use ~0.999)"
            ));
        }

        let student_pairs = Self::student_adapter_pairs(student);
        if student_pairs.len() != self.adapter_ids.len() {
            return Err(anyhow!(
                "adapter count mismatch: student has {} pairs, EMA has {}",
                student_pairs.len(),
                self.adapter_ids.len()
            ));
        }

        for ((student_a, student_b), &(ema_a, ema_b)) in
            student_pairs.iter().zip(self.adapter_ids.iter())
        {
            ema_blend(store, *student_a, ema_a, alpha)?;
            ema_blend(store, *student_b, ema_b, alpha)?;
        }
        Ok(())
    }

    /// Base + EMA-adapter, tape off. Exposes only frozen params (not the
    /// student's trainable adapter — see `teacher_param_ids`).
    pub fn as_teacher(&self) -> InProcessTeacher<'_> {
        InProcessTeacher::with_parameter_ids(&self.model, self.teacher_param_ids.clone())
    }
}

/// Student adapter + EMA adapter + AdamW moments — roll back as one unit on
/// gate-fail (partial rollback = stale EMA/moments, the DSv4-EAGLE failure).
pub struct EmaTrainSnapshot {
    student_adapter: Vec<(Vec<f32>, Vec<f32>)>,
    ema_adapter: Vec<(Vec<f32>, Vec<f32>)>,
    adamw: AdamWState,
}

impl EmaSelfTeacher {
    pub fn snapshot(
        &self,
        student: &Qwen35Model,
        optimizer: &AdamW,
        store: &mut TensorStore,
    ) -> Result<EmaTrainSnapshot> {
        let student_pairs = Self::student_adapter_pairs(student);

        let mut student_adapter = Vec::with_capacity(student_pairs.len());
        for (a, b) in &student_pairs {
            student_adapter.push((store.to_host(*a)?, store.to_host(*b)?));
        }

        let mut ema_adapter = Vec::with_capacity(self.adapter_ids.len());
        for (a, b) in &self.adapter_ids {
            ema_adapter.push((store.to_host(*a)?, store.to_host(*b)?));
        }

        let names = student_adapter_names(student);
        let adamw = optimizer.export_state(&names);

        Ok(EmaTrainSnapshot {
            student_adapter,
            ema_adapter,
            adamw,
        })
    }

    /// All three (student adapter, EMA adapter, AdamW moments) roll back as
    /// one unit — partial rollback leaves stale EMA/moments.
    pub fn restore(
        &mut self,
        snapshot: &EmaTrainSnapshot,
        student: &Qwen35Model,
        optimizer: &mut AdamW,
        store: &mut TensorStore,
    ) -> Result<()> {
        let student_pairs = Self::student_adapter_pairs(student);
        if student_pairs.len() != snapshot.student_adapter.len() {
            return Err(anyhow!(
                "restore student adapter count mismatch: live {} pairs, snapshot {}",
                student_pairs.len(),
                snapshot.student_adapter.len()
            ));
        }
        if self.adapter_ids.len() != snapshot.ema_adapter.len() {
            return Err(anyhow!(
                "restore EMA adapter count mismatch: live {} pairs, snapshot {}",
                self.adapter_ids.len(),
                snapshot.ema_adapter.len()
            ));
        }

        for ((a, b), (a_data, b_data)) in student_pairs.iter().zip(snapshot.student_adapter.iter())
        {
            write_tensor(store, *a, a_data)?;
            write_tensor(store, *b, b_data)?;
        }
        for (&(a, b), (a_data, b_data)) in self.adapter_ids.iter().zip(snapshot.ema_adapter.iter())
        {
            write_tensor(store, a, a_data)?;
            write_tensor(store, b, b_data)?;
        }

        let names = student_adapter_names(student);
        // Clear adapter moments before overlaying the snapshot — `import_state`
        // only re-installs snapshotted entries; a rejected step's moments would
        // otherwise persist for the next accepted step.
        let student_ids: Vec<TensorId> = names.iter().map(|(id, _)| *id).collect();
        optimizer.clear_param_state(&student_ids);
        optimizer.import_state(&snapshot.adamw, &names)?;
        Ok(())
    }
}

fn student_adapter_names(student: &Qwen35Model) -> Vec<(TensorId, String)> {
    student
        .adapter_name_map()
        .into_iter()
        .map(|(name, id)| (id, name.to_string()))
        .collect()
}

fn copy_tensor(store: &mut TensorStore, src: TensorId, dst: TensorId) -> Result<()> {
    let data = store.to_host(src)?;
    write_tensor(store, dst, &data)
}

fn write_tensor(store: &mut TensorStore, dst: TensorId, data: &[f32]) -> Result<()> {
    let tensor = store
        .get_mut(dst)
        .ok_or_else(|| anyhow!("EMA: destination tensor {dst} not found in store"))?;
    if tensor.data.len() != data.len() {
        return Err(anyhow!(
            "EMA: length mismatch writing tensor {dst} (shape {:?}, len {}): source len {}",
            tensor.shape,
            tensor.data.len(),
            data.len()
        ));
    }
    tensor.data.copy_from_slice(data);
    Ok(())
}

fn ema_blend(
    store: &mut TensorStore,
    student_id: TensorId,
    ema_id: TensorId,
    alpha: f32,
) -> Result<()> {
    let s = store.to_host(student_id)?;
    let e = store.to_host(ema_id)?;
    if s.len() != e.len() {
        return Err(anyhow!(
            "EMA: length mismatch (student {student_id} len {}, ema {ema_id} len {})",
            s.len(),
            e.len()
        ));
    }
    let mut out = e;
    for (o, &sv) in out.iter_mut().zip(s.iter()) {
        *o = alpha * *o + (1.0 - alpha) * sv;
    }
    write_tensor(store, ema_id, &out)
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use autograd::{TensorStore, optim::AdamW};

    use super::*;
    use crate::{
        lora::{LoraConfig, LoraTargetSet},
        qwen35::{LayerType, Qwen35Config, Qwen35Model},
    };

    type TestResult<T = ()> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

    fn tiny_qwen35_config() -> Qwen35Config {
        Qwen35Config {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            vocab_size: 16,
            rms_norm_eps: 1.0e-6,
            stop_token_ids: vec![15],
            bos_token_id: Some(1),
            eos_token_id: 15,
            tie_word_embeddings: false,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_num_value_heads: 2,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000.0,
            rope_scaling: None,
            partial_rotary_factor: 1.0,
            rotary_dim: 8,
            rope_cache_len_hint: Some(16),
            layer_types: vec![LayerType::FullAttention; 2],
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            mlp_only_layers: Vec::new(),
            full_attn_gated: true,
        }
    }

    fn lora_config() -> LoraConfig {
        LoraConfig {
            rank: 2,
            alpha: 4.0,
        }
    }

    #[test]
    fn ema_update_moves_toward_student_and_restores() -> TestResult {
        let mut store = TensorStore::default();
        let cfg = tiny_qwen35_config();
        let lora = lora_config();
        let target_set = LoraTargetSet::AttentionQv;

        let student = Qwen35Model::new_with_lora_targets(&cfg, lora, target_set, &mut store)?;

        let mut ema = EmaSelfTeacher::from_student(&student, lora, target_set, &mut store)?;

        let student_pairs = EmaSelfTeacher::student_adapter_pairs(&student);
        assert_eq!(
            student_pairs.len(),
            ema.adapter_ids.len(),
            "student and EMA must have the same adapter pair count"
        );
        assert!(
            !ema.adapter_ids.is_empty(),
            "AttentionQv must produce adapters"
        );

        for ((sa, sb), &(ea, eb)) in student_pairs.iter().zip(ema.adapter_ids.iter()) {
            assert_eq!(
                store.to_host(*sa)?,
                store.to_host(ea)?,
                "init lora_a must copy"
            );
            assert_eq!(
                store.to_host(*sb)?,
                store.to_host(eb)?,
                "init lora_b must copy"
            );
            // EMA adapter is a frozen teacher param (EMA-blended host-side, never
            // autograd-optimized) — must be requires_grad=false so the OPD step's
            // validate_teacher_params accepts the EMA model.
            assert!(
                !store.get(ea).expect("ema lora_a present").requires_grad,
                "ema lora_a must be frozen"
            );
            assert!(
                !store.get(eb).expect("ema lora_b present").requires_grad,
                "ema lora_b must be frozen"
            );
        }
        // The student adapter stays trainable — only the EMA copy is frozen.
        for &(sa, sb) in student_pairs.iter() {
            assert!(
                store.get(sa).expect("student lora_a present").requires_grad,
                "student lora_a must stay trainable"
            );
            assert!(
                store.get(sb).expect("student lora_b present").requires_grad,
                "student lora_b must stay trainable"
            );
        }

        // Snapshot the pre-mutation EMA lora_b for the halfway-move assertion.
        let (s_a0, s_b0) = student_pairs[0];
        let (e_a0, e_b0) = ema.adapter_ids[0];
        let ema_b_before = store.to_host(e_b0)?;

        let student_b_len = store.to_host(s_b0)?.len();
        let mutated: Vec<f32> = (0..student_b_len)
            .map(|i| (i as f32) * 0.25 + 1.0)
            .collect();
        {
            let t = store.get_mut(s_b0).expect("student lora_b present");
            t.data.copy_from_slice(&mutated);
        }
        let student_b_after = store.to_host(s_b0)?;

        // update with alpha=0.5 → EMA moves HALFWAY toward student.
        ema.update(&student, &mut store, 0.5)?;
        let ema_b_mid = store.to_host(e_b0)?;

        for ((before, target), got) in ema_b_before
            .iter()
            .zip(student_b_after.iter())
            .zip(ema_b_mid.iter())
        {
            let expected = 0.5 * before + 0.5 * target;
            assert!(
                (got - expected).abs() <= 1.0e-6,
                "EMA must move halfway: before={before} target={target} got={got} expected={expected}"
            );
        }
        // NOT equal to the old EMA nor the new student (since they differ).
        assert_ne!(
            ema_b_mid, ema_b_before,
            "EMA must have moved off its old value"
        );
        assert_ne!(
            ema_b_mid, student_b_after,
            "EMA halfway must differ from student"
        );

        let optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);
        let snap = ema.snapshot(&student, &optimizer, &mut store)?;

        let student_a_snapshot = store.to_host(s_a0)?;
        let student_b_snapshot = store.to_host(s_b0)?;
        let ema_a_snapshot = store.to_host(e_a0)?;
        let ema_b_snapshot = store.to_host(e_b0)?;

        {
            let t = store.get_mut(s_b0).expect("student lora_b present");
            for x in t.data.iter_mut() {
                *x = -7.0;
            }
        }
        {
            let t = store.get_mut(e_b0).expect("ema lora_b present");
            for x in t.data.iter_mut() {
                *x = 13.0;
            }
        }
        {
            let t = store.get_mut(s_a0).expect("student lora_a present");
            for x in t.data.iter_mut() {
                *x = 99.0;
            }
        }

        let mut optimizer = optimizer;
        ema.restore(&snap, &student, &mut optimizer, &mut store)?;

        assert_eq!(
            store.to_host(s_a0)?,
            student_a_snapshot,
            "student lora_a restored"
        );
        assert_eq!(
            store.to_host(s_b0)?,
            student_b_snapshot,
            "student lora_b restored"
        );
        assert_eq!(store.to_host(e_a0)?, ema_a_snapshot, "ema lora_a restored");
        assert_eq!(store.to_host(e_b0)?, ema_b_snapshot, "ema lora_b restored");

        Ok(())
    }

    /// R2 regression: a step REJECTED after the snapshot must not leave its
    /// freshly-created AdamW moments behind. `import_state` only re-installs
    /// snapshotted entries, so `restore` must clear the adapter's optimizer
    /// state first — otherwise the next accepted step reuses stale m/v.
    #[test]
    fn restore_clears_rejected_step_optimizer_moments() -> TestResult {
        let mut store = TensorStore::default();
        let cfg = tiny_qwen35_config();
        let lora = lora_config();
        let target_set = LoraTargetSet::AttentionQv;

        let student = Qwen35Model::new_with_lora_targets(&cfg, lora, target_set, &mut store)?;
        let mut ema = EmaSelfTeacher::from_student(&student, lora, target_set, &mut store)?;

        let student_pairs = EmaSelfTeacher::student_adapter_pairs(&student);
        let (_s_a0, s_b0) = student_pairs[0];

        let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);

        // 1. Snapshot BEFORE the optimizer has any moments for the adapter — the
        //    "first gated step" scenario from the P1 finding.
        let snap = ema.snapshot(&student, &optimizer, &mut store)?;

        // 2. A rejected step: attach a grad to the student adapter and step the
        //    optimizer, which inserts AdamW moments for that id.
        let b_len = store.to_host(s_b0)?.len();
        let grad = store.from_slice(&vec![0.5_f32; b_len], &[b_len])?;
        store
            .tensors
            .get_mut(s_b0)
            .and_then(|slot| slot.as_mut())
            .expect("student adapter tensor present")
            .grad = Some(grad);
        optimizer.step(&[s_b0], &mut store)?;
        // sanity: the rejected step really created a moment entry.
        assert_eq!(
            optimizer.clear_param_state(&[s_b0]),
            1,
            "the rejected step must have created an AdamW moment"
        );
        // re-create it (the sanity check above consumed it) so `restore` has the
        // stale moment to clear.
        store
            .tensors
            .get_mut(s_b0)
            .and_then(|slot| slot.as_mut())
            .expect("student adapter tensor present")
            .grad = Some(grad);
        optimizer.step(&[s_b0], &mut store)?;

        // 3. Restore must clear those moments (absent from the snapshot).
        ema.restore(&snap, &student, &mut optimizer, &mut store)?;

        // 4. Nothing left to remove → restore cleared the rejected moment.
        assert_eq!(
            optimizer.clear_param_state(&[s_b0]),
            0,
            "restore must clear adapter moments the snapshot did not contain"
        );

        Ok(())
    }

    /// P1 regression: the EMA teacher must expose only its own frozen params
    /// (shared base + frozen EMA adapter), NOT the student's trainable adapter.
    /// `share_base_parameters_from` folds the student adapter ids into the EMA
    /// model's `all_parameter_ids()`, which would make the OPD step reject the
    /// teacher (`requires_grad=true`) and the student (param "owned by teacher").
    #[test]
    fn ema_teacher_excludes_trainable_student_adapter() -> TestResult {
        use crate::teacher_infer::TeacherForward;

        let mut store = TensorStore::default();
        let cfg = tiny_qwen35_config();
        let lora = lora_config();
        let target_set = LoraTargetSet::AttentionQv;

        let student = Qwen35Model::new_with_lora_targets(&cfg, lora, target_set, &mut store)?;
        let ema = EmaSelfTeacher::from_student(&student, lora, target_set, &mut store)?;

        let teacher = ema.as_teacher();
        let teacher_ids: HashSet<TensorId> = teacher.parameter_ids().iter().copied().collect();
        assert!(!teacher_ids.is_empty(), "teacher must expose params");

        // Student adapter ids (trainable, student-owned) must be ABSENT.
        let student_adapter: Vec<TensorId> = student.adapter_name_map().values().copied().collect();
        assert!(!student_adapter.is_empty(), "student must have adapters");
        for id in &student_adapter {
            assert!(
                !teacher_ids.contains(id),
                "teacher params must exclude trainable student adapter id {id}"
            );
        }

        // EMA adapter ids must be PRESENT (forward reads them; retention needs them).
        for &(ea, eb) in ema.adapter_ids.iter() {
            assert!(
                teacher_ids.contains(&ea),
                "ema lora_a must be a teacher param"
            );
            assert!(
                teacher_ids.contains(&eb),
                "ema lora_b must be a teacher param"
            );
        }

        // Every exposed teacher param must be frozen — exactly the
        // validate_teacher_params bar the OPD step enforces.
        for &id in teacher.parameter_ids() {
            assert!(
                !store.get(id).expect("teacher param present").requires_grad,
                "teacher param {id} must be frozen (requires_grad=false)"
            );
        }

        Ok(())
    }
}
