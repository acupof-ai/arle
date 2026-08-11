//! Safetensors LoRA-adapter loading for the serve-side student re-merge.
//!
//! Reads the adapter file `train::infer_student::save_lora_adapters` writes
//! (bf16 safetensors keyed by train adapter names like
//! `model.language_model.layers.7.self_attn.q_proj.weight.lora_a`) and builds
//! the [`infer_cuda::StudentLoraUpdate`] the resident re-merge consumes
//! (`--lora-adapters` at serve startup). The name parser is the single source
//! shared with `train`'s per-round `sync_lora_from_store`.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use infer_cuda::{
    StudentLoraLayer, StudentLoraMatrices, StudentLoraProjection, StudentLoraProjectionUpdate,
    StudentLoraUpdate,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraHalf {
    A,
    B,
}

/// Parse a train adapter tensor name like
/// `model.language_model.layers.7.self_attn.q_proj.weight.lora_a` into
/// `(layer_idx, projection, half)`. Returns `None` for non-linear adapters
/// such as norms or conv state tensors.
pub fn parse_student_adapter_name(name: &str) -> Option<(usize, StudentLoraProjection, LoraHalf)> {
    // Two on-disk conventions: the internal `save_lora_adapters` suffix
    // (`…q_proj.lora_a`) and the PEFT layout `--save-lora-adapters` emits
    // (`…q_proj.lora_A.weight`). Accept both.
    let half = if name.ends_with(".lora_a") || name.contains(".lora_A.") {
        LoraHalf::A
    } else if name.ends_with(".lora_b") || name.contains(".lora_B.") {
        LoraHalf::B
    } else {
        return None;
    };
    let parts: Vec<&str> = name.split('.').collect();
    let layers_pos = parts.iter().position(|part| *part == "layers")?;
    let layer_idx: usize = parts.get(layers_pos + 1)?.parse().ok()?;
    let projection = match (*parts.get(layers_pos + 2)?, *parts.get(layers_pos + 3)?) {
        ("self_attn", "q_proj") => StudentLoraProjection::FullQ,
        ("self_attn", "k_proj") => StudentLoraProjection::FullK,
        ("self_attn", "v_proj") => StudentLoraProjection::FullV,
        ("self_attn", "o_proj") => StudentLoraProjection::FullO,
        ("self_attn" | "linear_attn", "in_proj_qkv") => StudentLoraProjection::LinearQkv,
        ("self_attn" | "linear_attn", "in_proj_z") => StudentLoraProjection::LinearZ,
        ("self_attn" | "linear_attn", "in_proj_b") => StudentLoraProjection::LinearB,
        ("self_attn" | "linear_attn", "in_proj_a") => StudentLoraProjection::LinearA,
        ("self_attn" | "linear_attn", "out_proj") => StudentLoraProjection::LinearOut,
        ("mlp", "gate_proj") => StudentLoraProjection::MlpGate,
        ("mlp", "up_proj") => StudentLoraProjection::MlpUp,
        ("mlp", "down_proj") => StudentLoraProjection::MlpDown,
        ("mlp", "gate") => StudentLoraProjection::MoeRouter,
        ("mlp", "shared_expert_gate") => StudentLoraProjection::MoeSharedExpertGate,
        ("mlp", "shared_expert") => match *parts.get(layers_pos + 4)? {
            "gate_proj" => StudentLoraProjection::MoeSharedGate,
            "up_proj" => StudentLoraProjection::MoeSharedUp,
            "down_proj" => StudentLoraProjection::MoeSharedDown,
            _ => return None,
        },
        ("mlp", "experts") => {
            let expert_idx: usize = parts.get(layers_pos + 4)?.parse().ok()?;
            match *parts.get(layers_pos + 5)? {
                "gate_proj" => StudentLoraProjection::MoeExpertGate { expert_idx },
                "up_proj" => StudentLoraProjection::MoeExpertUp { expert_idx },
                "down_proj" => StudentLoraProjection::MoeExpertDown { expert_idx },
                _ => return None,
            }
        }
        _ => return None,
    };
    Some((layer_idx, projection, half))
}

/// One projection's optional A/B host matrices, each `(values, rows, cols)`.
#[derive(Default)]
struct PartialProj {
    a: Option<(Vec<f32>, usize, usize)>,
    b: Option<(Vec<f32>, usize, usize)>,
}

/// Load a train `save_lora_adapters` safetensors file into the
/// [`StudentLoraUpdate`] the resident re-merge consumes. `rank` is read from
/// the tensor shapes (`lora_A` rows); `alpha` comes from the caller
/// (`--lora-alpha`) because the adapter file carries matrices only.
pub fn load_student_lora_update(path: &Path, alpha: f32) -> Result<StudentLoraUpdate> {
    // Accept either the `.safetensors` file directly or a PEFT adapter DIR
    // (`--save-lora-adapters` emits `<dir>/adapter_model.safetensors`), matching
    // the train-side resume loader so a saved adapter loads back into the serve.
    let file = if path.is_dir() {
        path.join("adapter_model.safetensors")
    } else {
        path.to_path_buf()
    };
    let bytes =
        std::fs::read(&file).with_context(|| format!("read LoRA adapters {}", file.display()))?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse safetensors {}", file.display()))?;

    let mut layers: BTreeMap<usize, BTreeMap<StudentLoraProjection, PartialProj>> = BTreeMap::new();
    for (name, view) in tensors.tensors() {
        let Some((layer_idx, projection, half)) = parse_student_adapter_name(&name) else {
            bail!("{}: unsupported adapter tensor name {name}", path.display());
        };
        let shape = view.shape();
        ensure!(
            shape.len() == 2,
            "{name}: expected rank-2 matrix, got shape {shape:?}"
        );
        let values = tensor_to_f32(&name, &view)?;
        let slot = layers
            .entry(layer_idx)
            .or_default()
            .entry(projection)
            .or_default();
        match half {
            LoraHalf::A => slot.a = Some((values, shape[0], shape[1])),
            LoraHalf::B => slot.b = Some((values, shape[0], shape[1])),
        }
    }
    ensure!(
        !layers.is_empty(),
        "{}: no LoRA adapter tensors found",
        path.display()
    );

    // Rank from the shapes: lora_A is [rank, in_features], lora_B is
    // [out_features, rank]; every pair must agree.
    let mut rank: Option<usize> = None;
    let mut out_layers = Vec::with_capacity(layers.len());
    for (layer_idx, projections) in layers {
        let projections = projections
            .into_iter()
            .map(|(projection, partial)| {
                let (Some((a, a_rows, a_cols)), Some((b, b_rows, b_cols))) = (partial.a, partial.b)
                else {
                    bail!("layer {layer_idx} {projection:?} is missing its lora_A or lora_B half");
                };
                let r = *rank.get_or_insert(a_rows);
                ensure!(
                    a_rows == r && b_cols == r,
                    "layer {layer_idx} {projection:?}: rank mismatch (lora_A rows {a_rows}, \
                     lora_B cols {b_cols}, expected {r})"
                );
                Ok(StudentLoraProjectionUpdate {
                    projection,
                    matrices: StudentLoraMatrices {
                        a,
                        b,
                        rank: r,
                        in_features: a_cols,
                        out_features: b_rows,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        out_layers.push(StudentLoraLayer {
            layer_idx,
            projections,
        });
    }

    Ok(StudentLoraUpdate {
        layers: out_layers,
        rank: rank.expect("non-empty layers imply a rank"),
        alpha,
    })
}

/// Decode one safetensors view to host f32 (train saves bf16; accept f32 too).
fn tensor_to_f32(name: &str, view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    let data = view.data();
    match view.dtype() {
        safetensors::Dtype::BF16 => Ok(data
            .chunks_exact(2)
            .map(|pair| half::bf16::from_le_bytes([pair[0], pair[1]]).to_f32())
            .collect()),
        safetensors::Dtype::F32 => Ok(data
            .chunks_exact(4)
            .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
            .collect()),
        other => Err(anyhow!("{name}: unsupported adapter dtype {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{LoraHalf, parse_student_adapter_name};
    use infer_cuda::StudentLoraProjection;

    fn parsed_projection(name: &str) -> Option<StudentLoraProjection> {
        parse_student_adapter_name(name).map(|(_, projection, _)| projection)
    }

    #[test]
    fn parse_adapter_name_covers_all_linear_targets() {
        let prefix = "model.language_model.layers.7";
        let cases = [
            (
                format!("{prefix}.self_attn.q_proj.weight.lora_a"),
                StudentLoraProjection::FullQ,
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight.lora_a"),
                StudentLoraProjection::FullK,
            ),
            (
                format!("{prefix}.self_attn.v_proj.weight.lora_b"),
                StudentLoraProjection::FullV,
            ),
            (
                format!("{prefix}.self_attn.o_proj.weight.lora_a"),
                StudentLoraProjection::FullO,
            ),
            (
                format!("{prefix}.self_attn.in_proj_qkv.weight.lora_a"),
                StudentLoraProjection::LinearQkv,
            ),
            (
                format!("{prefix}.linear_attn.in_proj_z.weight.lora_a"),
                StudentLoraProjection::LinearZ,
            ),
            (
                format!("{prefix}.linear_attn.in_proj_b.weight.lora_a"),
                StudentLoraProjection::LinearB,
            ),
            (
                format!("{prefix}.linear_attn.in_proj_a.weight.lora_a"),
                StudentLoraProjection::LinearA,
            ),
            (
                format!("{prefix}.linear_attn.out_proj.weight.lora_b"),
                StudentLoraProjection::LinearOut,
            ),
            (
                format!("{prefix}.mlp.gate_proj.weight.lora_a"),
                StudentLoraProjection::MlpGate,
            ),
            (
                format!("{prefix}.mlp.up_proj.weight.lora_a"),
                StudentLoraProjection::MlpUp,
            ),
            (
                format!("{prefix}.mlp.down_proj.weight.lora_a"),
                StudentLoraProjection::MlpDown,
            ),
            (
                format!("{prefix}.mlp.gate.weight.lora_a"),
                StudentLoraProjection::MoeRouter,
            ),
            (
                format!("{prefix}.mlp.shared_expert.gate_proj.weight.lora_a"),
                StudentLoraProjection::MoeSharedGate,
            ),
            (
                format!("{prefix}.mlp.shared_expert.up_proj.weight.lora_a"),
                StudentLoraProjection::MoeSharedUp,
            ),
            (
                format!("{prefix}.mlp.shared_expert.down_proj.weight.lora_a"),
                StudentLoraProjection::MoeSharedDown,
            ),
            (
                format!("{prefix}.mlp.shared_expert_gate.weight.lora_a"),
                StudentLoraProjection::MoeSharedExpertGate,
            ),
            (
                format!("{prefix}.mlp.experts.13.gate_proj.weight.lora_a"),
                StudentLoraProjection::MoeExpertGate { expert_idx: 13 },
            ),
            (
                format!("{prefix}.mlp.experts.13.up_proj.weight.lora_a"),
                StudentLoraProjection::MoeExpertUp { expert_idx: 13 },
            ),
            (
                format!("{prefix}.mlp.experts.13.down_proj.weight.lora_b"),
                StudentLoraProjection::MoeExpertDown { expert_idx: 13 },
            ),
        ];
        for (name, expected) in cases {
            assert_eq!(parsed_projection(&name), Some(expected), "{name}");
        }

        let (layer_idx, projection, half) =
            parse_student_adapter_name(&format!("{prefix}.self_attn.q_proj.weight.lora_a"))
                .expect("q_proj parses");
        assert_eq!(layer_idx, 7);
        assert_eq!(projection, StudentLoraProjection::FullQ);
        assert_eq!(half, LoraHalf::A);

        assert!(
            parse_student_adapter_name(&format!("{prefix}.self_attn.q_norm.weight.lora_a"))
                .is_none()
        );
        assert!(
            parse_student_adapter_name(&format!("{prefix}.self_attn.conv1d.weight.lora_a"))
                .is_none()
        );
    }
}
