//! EMA self-teacher core for ARLE's Self-OPD (SOPD, #91).
//!
//! [`EmaSelfTeacher`] owns a second [`Qwen35Model`] that shares the student's
//! base parameters (via [`Qwen35Model::new_lora_from_base`]) but carries its
//! OWN LoRA adapter — the exponential-moving-average (EMA) of the student
//! adapter. The student trains; after each step the EMA adapter is nudged
//! `θ_ema ← α·θ_ema + (1−α)·θ_student` (host-side, elementwise). The EMA model
//! is consumed as a [`TeacherForward`] (`base + EMA-adapter`, tape off, in the
//! train store) — it does NOT define a new `TeacherForward` impl; it REUSES
//! [`InProcessTeacher`].
//!
//! Snapshot/restore (R2): on a gate-fail the student adapter, the EMA adapter,
//! AND the AdamW moments must roll back together as ONE unit. Restoring only
//! the student adapter (a naive cut) would leave the EMA trained against the
//! rejected student state and the optimizer moments stale — the same
//! partial-rollback failure mode as the DSv4-EAGLE truncate that restored only
//! `compressed.seq_len` and corrupted the draft at the boundary.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use autograd::{TensorId, TensorStore, adamw_state::AdamWState, optim::AdamW};

use crate::{
    lora::{LoraConfig, LoraTargetSet},
    qwen35::Qwen35Model,
    teacher_infer::InProcessTeacher,
};

/// Pair an `adapter_name_map()` into deterministic, layer-ordered
/// `(lora_a, lora_b)` tuples.
///
/// Adapter names are `{base_name}.lora_a` / `{base_name}.lora_b` where
/// `{base_name}` ends in `.weight` (see `lora.rs:123`). We strip the
/// `.lora_a` / `.lora_b` suffix to recover the shared base prefix, match the
/// `a`/`b` siblings by that prefix, and sort by base name so the same logical
/// adapter lands at the same index across the student and the EMA models.
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

/// EMA self-teacher: a base-shared second [`Qwen35Model`] holding the EMA
/// adapter, plus the layer-ordered EMA adapter `(lora_a, lora_b)` TensorIds.
pub struct EmaSelfTeacher {
    /// Base-shared second model holding the EMA adapter.
    model: Qwen35Model,
    /// EMA adapter pairs, layer-ordered (sorted by base name) so index `i`
    /// lines up with [`Self::student_adapter_pairs`] index `i`.
    adapter_ids: Vec<(TensorId, TensorId)>,
}

impl EmaSelfTeacher {
    /// Build the EMA self-teacher from a freshly constructed student model.
    ///
    /// CONSTRUCTION ORDER (REQUIRED): call this immediately after building the
    /// student model and BEFORE allocating any other train-store scratch
    /// tensors. [`Qwen35Model::new_lora_from_base`] calls
    /// `store.retain_ids(student ∪ ema)`, which FREES every other store tensor;
    /// any scratch allocated before this call is silently dropped.
    ///
    /// The EMA adapter is initialized to an EXACT COPY of the student adapter
    /// (via [`Self::copy_from_student`]), NOT relied on to coincide via
    /// name-seeded init. WHY: ① resume-from-checkpoint — a resumed student has
    /// a non-zero adapter, and the EMA must start from it, not from fresh seed;
    /// ② bilinearity — the EMA update is elementwise on `A` and `B` separately,
    /// which only tracks the student's effective `B·A` delta when EMA and
    /// student share the same basis. The exact copy establishes that shared
    /// basis at step 0, and the slow EMA (α≈0.999) keeps the two bases close
    /// enough thereafter for the elementwise update to remain meaningful.
    pub fn from_student(
        student: &Qwen35Model,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        store: &mut TensorStore,
    ) -> Result<Self> {
        let model = Qwen35Model::new_lora_from_base(student, lora, target_set, store)?;
        let adapter_ids = pair_adapters(&model.adapter_name_map())?;
        let mut teacher = Self { model, adapter_ids };
        teacher.copy_from_student(student, store)?;
        Ok(teacher)
    }

    /// Pair the STUDENT's adapter `(lora_a, lora_b)` TensorIds in the same
    /// deterministic order as [`Self::adapter_ids`], so student index `i`
    /// aligns with EMA index `i`.
    pub fn student_adapter_pairs(student: &Qwen35Model) -> Vec<(TensorId, TensorId)> {
        // `from_student` already proved the student's adapter map pairs cleanly
        // (the EMA model is built from the same target_set), so any pairing
        // error here is a logic bug; surface it loudly rather than silently
        // returning a short Vec.
        pair_adapters(&student.adapter_name_map())
            .expect("student adapter map must pair into (lora_a, lora_b) tuples")
    }

    /// Copy the student adapter values verbatim into the EMA adapter (EMA = student).
    pub fn copy_from_student(
        &mut self,
        student: &Qwen35Model,
        store: &mut TensorStore,
    ) -> Result<()> {
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

    /// EMA step `θ_ema ← α·θ_ema + (1−α)·θ_student`, elementwise, host-side,
    /// for every aligned `(a, a)` and `(b, b)` adapter pair.
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

    /// The [`TeacherForward`](crate::teacher_infer::TeacherForward) the OPD step
    /// consumes: base + EMA-adapter, tape off, in the train store.
    pub fn as_teacher(&self) -> InProcessTeacher<'_> {
        InProcessTeacher::new(&self.model)
    }
}

/// Snapshot of everything that must roll back together on a gate-fail (R2).
pub struct EmaTrainSnapshot {
    /// Student adapter `(lora_a data, lora_b data)`, layer-ordered.
    student_adapter: Vec<(Vec<f32>, Vec<f32>)>,
    /// EMA adapter `(lora_a data, lora_b data)`, layer-ordered.
    ema_adapter: Vec<(Vec<f32>, Vec<f32>)>,
    /// AdamW moments for the trainable student params.
    adamw: AdamWState,
}

impl EmaSelfTeacher {
    /// Snapshot the student adapter, the EMA adapter, and the AdamW moments as
    /// one unit. `names` for the optimizer export is the student adapter
    /// `(TensorId, leaked-name.to_string())` list — the optimizer only holds
    /// moments for the trainable student params.
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

    /// Restore the student adapter, the EMA adapter, AND the AdamW moments
    /// together (R2). Restoring ONLY the student adapter would leave the EMA
    /// trained against rejected student state and the AdamW moments stale —
    /// the DSv4-EAGLE partial-rollback failure mode. All three roll back as
    /// one unit.
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
        optimizer.import_state(&snapshot.adamw, &names)?;
        Ok(())
    }
}

/// The student's trainable-adapter `(TensorId, leaked-name.to_string())` list,
/// for AdamW export/import.
fn student_adapter_names(student: &Qwen35Model) -> Vec<(TensorId, String)> {
    student
        .adapter_name_map()
        .into_iter()
        .map(|(name, id)| (id, name.to_string()))
        .collect()
}

/// `dst.data ← src.data`. Asserts equal length (error, not panic).
fn copy_tensor(store: &mut TensorStore, src: TensorId, dst: TensorId) -> Result<()> {
    let data = store.to_host(src)?;
    write_tensor(store, dst, &data)
}

/// `dst.data ← data`. Asserts equal length, including the tensor shape in the
/// error on mismatch.
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

/// `ema.data ← α·ema.data + (1−α)·student.data`, elementwise, host-side.
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
    let mut out = e; // reuse the EMA host buffer
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

        // 1. Student via the normal lora constructor, on the CPU backend store.
        let student = Qwen35Model::new_with_lora_targets(&cfg, lora, target_set, &mut store)?;

        // 2. EMA self-teacher; assert EMA adapter == student adapter at construction.
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
        }

        // Snapshot the pre-mutation EMA lora_b (zeros at init) of pair 0 for the
        // halfway-move assertion.
        let (s_a0, s_b0) = student_pairs[0];
        let (e_a0, e_b0) = ema.adapter_ids[0];
        let ema_b_before = store.to_host(e_b0)?;

        // 3. Mutate the student adapter: write non-zero values into student lora_b.
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

        // 4. Snapshot adapters → mutate student + ema → restore → byte-identity.
        let optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);
        let snap = ema.snapshot(&student, &optimizer, &mut store)?;

        let student_a_snapshot = store.to_host(s_a0)?;
        let student_b_snapshot = store.to_host(s_b0)?;
        let ema_a_snapshot = store.to_host(e_a0)?;
        let ema_b_snapshot = store.to_host(e_b0)?;

        // Scribble over student + ema adapters.
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
}
