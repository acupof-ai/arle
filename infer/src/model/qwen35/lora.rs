//! PEFT LoRA adapter loading for Qwen3.5 serve.
//!
//! P1-B intentionally keeps this as a load-time merge path for the 0.8B
//! distilled-student eval loop: adapter deltas are folded into dense BF16
//! full-attention q/v projection weights before serving begins. That avoids
//! changing Qwen3.5 prefill/decode hot paths while still loading the
//! adapter-only checkpoints emitted by `train::qwen35_checkpoint`.

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

use anyhow::{anyhow, bail, ensure, Context, Result};
use half::bf16;
use log::{debug, info, warn};
use memmap2::Mmap;
use safetensors::{
    tensor::{Dtype, TensorView},
    SafeTensors,
};

use cuda_kernels::prelude::{DeviceContext, DeviceMatrix};

#[derive(Debug)]
pub(super) struct Qwen35LoRA {
    pub(super) layers: Vec<LayerLoRA>,
    pub(super) tensor_count: usize,
    pub(super) rank: usize,
    pub(super) alpha: f32,
    pub(super) scale: f32,
}

#[derive(Debug, Default)]
pub(super) struct LayerLoRA {
    pub(super) q_proj: Option<LoraAB>,
    pub(super) v_proj: Option<LoraAB>,
}

#[derive(Debug)]
pub(super) struct LoraAB {
    a: AdapterMatrix,
    b: AdapterMatrix,
}

#[derive(Debug, Clone)]
struct AdapterMatrix {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

pub(super) fn load_peft_lora(lora_path: &str, num_layers: usize) -> Result<Qwen35LoRA> {
    let dir = Path::new(lora_path);
    if !dir.is_dir() {
        bail!("Qwen3.5 LoRA path '{}' is not a directory", lora_path);
    }

    let cfg_path = dir.join("adapter_config.json");
    let cfg_raw =
        fs::read_to_string(&cfg_path).with_context(|| format!("reading {}", cfg_path.display()))?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_raw)
        .with_context(|| format!("parsing {}", cfg_path.display()))?;
    let rank = cfg
        .get("r")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow!("adapter_config.json missing `r`"))? as usize;
    let alpha = cfg
        .get("lora_alpha")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| anyhow!("adapter_config.json missing `lora_alpha`"))? as f32;
    ensure!(rank > 0, "adapter_config.json has r=0");
    let scale = alpha / rank as f32;
    let target_modules = parse_target_modules(&cfg);

    let st_path = dir.join("adapter_model.safetensors");
    let file =
        fs::File::open(&st_path).with_context(|| format!("opening {}", st_path.display()))?;
    // SAFETY: the mmap lives until all borrowed TensorViews are converted
    // into owned AdapterMatrix values.
    let mmap =
        unsafe { Mmap::map(&file) }.with_context(|| format!("mmapping {}", st_path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parsing safetensors header in {}", st_path.display()))?;

    let mut buckets: HashMap<(usize, LoraModule), LoraBucket> = HashMap::new();
    let mut tensor_count = 0usize;
    for (name, view) in st.iter() {
        let Some((layer_idx, module, which)) = parse_peft_key(name) else {
            debug!("qwen35 lora: skipping unrecognized tensor key {name}");
            continue;
        };
        if layer_idx >= num_layers {
            debug!(
                "qwen35 lora: skipping {name} because layer_idx {layer_idx} >= num_layers {num_layers}"
            );
            continue;
        }
        if let Some(target_modules) = target_modules.as_ref() {
            if !target_modules.contains(&module) {
                debug!("qwen35 lora: skipping {name} because {module:?} is not in target_modules");
                continue;
            }
        }
        let matrix = adapter_matrix_from_view(&view).with_context(|| format!("reading {name}"))?;
        if which == Which::A {
            ensure!(
                matrix.rows == rank,
                "{name}: lora_A rows {} != adapter rank {rank}",
                matrix.rows
            );
        } else {
            ensure!(
                matrix.cols == rank,
                "{name}: lora_B cols {} != adapter rank {rank}",
                matrix.cols
            );
        }
        let bucket = buckets.entry((layer_idx, module)).or_default();
        match which {
            Which::A => bucket.a = Some(matrix),
            Which::B => bucket.b = Some(matrix),
        }
        tensor_count += 1;
    }

    let mut layers: Vec<LayerLoRA> = (0..num_layers).map(|_| LayerLoRA::default()).collect();
    let mut loaded_adapters = 0usize;
    for ((layer_idx, module), bucket) in buckets {
        let Some(a) = bucket.a else {
            warn!("qwen35 lora: layer {layer_idx} {module:?} has B without A; skipping");
            continue;
        };
        let Some(b) = bucket.b else {
            warn!("qwen35 lora: layer {layer_idx} {module:?} has A without B; skipping");
            continue;
        };
        let adapter = LoraAB { a, b };
        match module {
            LoraModule::QProj => layers[layer_idx].q_proj = Some(adapter),
            LoraModule::VProj => layers[layer_idx].v_proj = Some(adapter),
            unsupported => {
                warn!(
                    "qwen35 lora: ignoring unsupported module {unsupported:?}; P1-B serves q/v adapters only"
                );
                continue;
            }
        }
        loaded_adapters += 1;
    }

    info!(
        "qwen35 lora: loaded {loaded_adapters} q/v adapters ({tensor_count} tensors) across {num_layers} layers from {lora_path} (r={rank}, alpha={alpha}, scale={scale:.4})"
    );
    Ok(Qwen35LoRA {
        layers,
        tensor_count,
        rank,
        alpha,
        scale,
    })
}

pub(super) fn merge_lora_into_dense_matrix(
    ctx: &DeviceContext,
    matrix: &mut DeviceMatrix,
    adapter: &LoraAB,
    scale: f32,
    label: &str,
) -> Result<()> {
    ensure!(
        matrix.is_dense_bf16(),
        "{label}: Qwen3.5 LoRA serve merge currently requires dense BF16 base weights; got {:?}",
        matrix.weight_format()
    );
    ensure!(
        adapter.a.cols == matrix.cols,
        "{label}: lora_A cols {} != base cols {}",
        adapter.a.cols,
        matrix.cols
    );
    ensure!(
        adapter.b.rows == matrix.rows,
        "{label}: lora_B rows {} != base rows {}",
        adapter.b.rows,
        matrix.rows
    );
    ensure!(
        adapter.a.rows == adapter.b.cols,
        "{label}: LoRA rank mismatch A rows {} != B cols {}",
        adapter.a.rows,
        adapter.b.cols
    );

    let mut host = ctx
        .stream
        .clone_dtoh(&matrix.data)
        .map_err(|err| anyhow!("{label}: D2H base weight copy failed: {err}"))?;
    ctx.sync()?;
    apply_lora_delta_to_host(&mut host, matrix.rows, matrix.cols, adapter, scale)?;
    *matrix = DeviceMatrix::from_host(ctx, &host, matrix.rows, matrix.cols)
        .with_context(|| format!("{label}: upload merged weight"))?;
    Ok(())
}

fn apply_lora_delta_to_host(
    base: &mut [bf16],
    rows: usize,
    cols: usize,
    adapter: &LoraAB,
    scale: f32,
) -> Result<()> {
    ensure!(
        base.len() == rows * cols,
        "base len {} != rows*cols {}",
        base.len(),
        rows * cols
    );
    let rank = adapter.a.rows;
    ensure!(adapter.a.cols == cols, "lora_A cols mismatch");
    ensure!(adapter.b.rows == rows, "lora_B rows mismatch");
    ensure!(adapter.b.cols == rank, "lora_B cols mismatch");

    for row in 0..rows {
        for col in 0..cols {
            let mut delta = 0.0f32;
            for r in 0..rank {
                delta += adapter.b.values[row * rank + r] * adapter.a.values[r * cols + col];
            }
            let idx = row * cols + col;
            base[idx] = bf16::from_f32(base[idx].to_f32() + scale * delta);
        }
    }
    Ok(())
}

fn adapter_matrix_from_view(view: &TensorView<'_>) -> Result<AdapterMatrix> {
    let shape = view.shape();
    ensure!(
        shape.len() == 2,
        "expected rank-2 LoRA matrix, got shape {shape:?}"
    );
    let rows = shape[0];
    let cols = shape[1];
    let elem_count = rows * cols;
    let values = match view.dtype() {
        Dtype::F32 => {
            ensure!(
                view.data().len() == elem_count * 4,
                "F32 matrix byte length mismatch"
            );
            view.data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4 byte chunk")))
                .collect()
        }
        Dtype::BF16 => {
            ensure!(
                view.data().len() == elem_count * 2,
                "BF16 matrix byte length mismatch"
            );
            view.data()
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes(chunk.try_into().expect("2 byte chunk"));
                    bf16::from_bits(bits).to_f32()
                })
                .collect()
        }
        other => bail!("unsupported Qwen3.5 LoRA dtype {other:?}; expected F32 or BF16"),
    };
    Ok(AdapterMatrix { rows, cols, values })
}

fn parse_target_modules(cfg: &serde_json::Value) -> Option<HashSet<LoraModule>> {
    let targets = cfg.get("target_modules")?.as_array()?;
    let mut parsed = HashSet::new();
    for target in targets {
        let Some(target) = target.as_str() else {
            warn!("qwen35 lora: ignoring non-string target_modules entry {target:?}");
            continue;
        };
        match LoraModule::from_str(target) {
            Some(module) => {
                parsed.insert(module);
            }
            None => warn!("qwen35 lora: unknown target_modules entry '{target}', ignoring"),
        }
    }
    Some(parsed)
}

#[derive(Debug, Default)]
struct LoraBucket {
    a: Option<AdapterMatrix>,
    b: Option<AdapterMatrix>,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub(super) enum LoraModule {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

impl LoraModule {
    fn from_str(raw: &str) -> Option<Self> {
        match raw {
            "q_proj" => Some(Self::QProj),
            "k_proj" => Some(Self::KProj),
            "v_proj" => Some(Self::VProj),
            "o_proj" => Some(Self::OProj),
            "gate_proj" => Some(Self::GateProj),
            "up_proj" => Some(Self::UpProj),
            "down_proj" => Some(Self::DownProj),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Which {
    A,
    B,
}

fn parse_peft_key(name: &str) -> Option<(usize, LoraModule, Which)> {
    let parts: Vec<&str> = name.split('.').collect();
    let layers_pos = parts.iter().position(|part| *part == "layers")?;
    let layer_idx = parts.get(layers_pos + 1)?.parse().ok()?;
    let branch = *parts.get(layers_pos + 2)?;
    if branch != "self_attn" && branch != "mlp" {
        return None;
    }
    let module = LoraModule::from_str(parts.get(layers_pos + 3)?)?;
    let which = match *parts.get(layers_pos + 4)? {
        "lora_A" => Which::A,
        "lora_B" => Which::B,
        _ => return None,
    };
    match parts.get(layers_pos + 5) {
        Some(&"weight") | None => {}
        _ => return None,
    }
    Some((layer_idx, module, which))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, View};
    use std::borrow::Cow;
    use std::collections::BTreeMap;

    struct F32Tensor {
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl F32Tensor {
        fn new(shape: Vec<usize>, values: Vec<f32>) -> Self {
            assert_eq!(shape.iter().product::<usize>(), values.len());
            let data = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            Self { shape, data }
        }
    }

    impl View for &F32Tensor {
        fn dtype(&self) -> Dtype {
            Dtype::F32
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    #[test]
    fn parse_accepts_qwen35_language_model_prefix() {
        let got = parse_peft_key(
            "base_model.model.model.language_model.layers.11.self_attn.q_proj.lora_A.weight",
        );
        assert!(matches!(got, Some((11, LoraModule::QProj, Which::A))));
    }

    #[test]
    fn load_peft_lora_counts_qv_adapter_tensors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rank = 2usize;
        let hidden = 3usize;
        let q_out = 4usize;
        let v_out = 2usize;
        let mut storage: BTreeMap<String, F32Tensor> = BTreeMap::new();
        for layer in 0..6 {
            for (module, rows) in [("q_proj", q_out), ("v_proj", v_out)] {
                let a = F32Tensor::new(vec![rank, hidden], vec![0.125; rank * hidden]);
                let b = F32Tensor::new(vec![rows, rank], vec![0.25; rows * rank]);
                storage.insert(
                    format!(
                        "base_model.model.model.language_model.layers.{layer}.self_attn.{module}.lora_A.weight"
                    ),
                    a,
                );
                storage.insert(
                    format!(
                        "base_model.model.model.language_model.layers.{layer}.self_attn.{module}.lora_B.weight"
                    ),
                    b,
                );
            }
        }
        let tensors = storage
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor))
            .collect::<BTreeMap<_, _>>();
        safetensors::serialize_to_file(
            tensors,
            None,
            &dir.path().join("adapter_model.safetensors"),
        )
        .expect("write adapter safetensors");
        fs::write(
            dir.path().join("adapter_config.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "r": rank,
                "lora_alpha": 4.0,
                "target_modules": ["q_proj", "v_proj"],
            }))
            .unwrap(),
        )
        .expect("write adapter config");

        let lora = load_peft_lora(dir.path().to_str().unwrap(), 24).expect("load qwen35 lora");
        assert_eq!(lora.tensor_count, 24);
        assert_eq!(
            lora.layers
                .iter()
                .filter(|layer| layer.q_proj.is_some())
                .count(),
            6
        );
        assert_eq!(
            lora.layers
                .iter()
                .filter(|layer| layer.v_proj.is_some())
                .count(),
            6
        );
        assert!(lora.layers[0].q_proj.is_some());
        assert!(lora.layers[0].v_proj.is_some());
    }

    #[test]
    fn load_peft_lora_respects_target_modules() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rank = 1usize;
        let hidden = 2usize;
        let mut storage: BTreeMap<String, F32Tensor> = BTreeMap::new();
        for module in ["q_proj", "v_proj"] {
            storage.insert(
                format!("base_model.model.model.language_model.layers.3.self_attn.{module}.lora_A.weight"),
                F32Tensor::new(vec![rank, hidden], vec![0.125; rank * hidden]),
            );
            storage.insert(
                format!("base_model.model.model.language_model.layers.3.self_attn.{module}.lora_B.weight"),
                F32Tensor::new(vec![hidden, rank], vec![0.25; hidden * rank]),
            );
        }
        let tensors = storage
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor))
            .collect::<BTreeMap<_, _>>();
        safetensors::serialize_to_file(
            tensors,
            None,
            &dir.path().join("adapter_model.safetensors"),
        )
        .expect("write adapter safetensors");
        fs::write(
            dir.path().join("adapter_config.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "r": rank,
                "lora_alpha": 1.0,
                "target_modules": ["q_proj"],
            }))
            .unwrap(),
        )
        .expect("write adapter config");

        let lora = load_peft_lora(dir.path().to_str().unwrap(), 24).expect("load qwen35 lora");
        assert!(lora.layers[3].q_proj.is_some());
        assert!(lora.layers[3].v_proj.is_none());
        assert_eq!(lora.tensor_count, 2);
    }

    #[test]
    fn apply_lora_delta_to_host_matches_base_plus_scaled_ba() {
        let adapter = LoraAB {
            a: AdapterMatrix {
                rows: 2,
                cols: 3,
                values: vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0],
            },
            b: AdapterMatrix {
                rows: 2,
                cols: 2,
                values: vec![0.25, 2.0, -1.0, 0.5],
            },
        };
        let scale = 0.5f32;
        let mut base = vec![bf16::from_f32(1.0); 6];
        apply_lora_delta_to_host(&mut base, 2, 3, &adapter, scale).expect("merge delta");

        let expected = [
            1.0 + 0.5 * (0.25 * 1.0 + 2.0 * -1.0),
            1.0 + 0.5 * (0.25 * 2.0 + 2.0 * 0.5),
            1.0 + 0.5 * (0.25 * 3.0 + 2.0 * 2.0),
            1.0 + 0.5 * (-1.0 * 1.0 + 0.5 * -1.0),
            1.0 + 0.5 * (-1.0 * 2.0 + 0.5 * 0.5),
            1.0 + 0.5 * (-1.0 * 3.0 + 0.5 * 2.0),
        ];
        for (idx, (got, expected)) in base.iter().zip(expected).enumerate() {
            let expected_bf16 = bf16::from_f32(expected).to_f32();
            assert!(
                (got.to_f32() - expected_bf16).abs() < 1e-6,
                "idx {idx}: got {}, expected {expected_bf16}",
                got.to_f32()
            );
        }
    }
}
