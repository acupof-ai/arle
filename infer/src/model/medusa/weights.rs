use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use safetensors::SafeTensors;

use cuda_kernels::prelude::{DeviceContext, DeviceMatrix};

use super::{Medusa, MedusaConfig, ResidualBlock};

pub fn load_medusa_weights(
    ctx: &DeviceContext,
    config: MedusaConfig,
    weights_path: impl AsRef<Path>,
) -> Result<Medusa> {
    let weights_path = weights_path.as_ref();
    config.validate()?;

    let mut blocks = Vec::with_capacity(config.num_heads);
    let mut lm_heads = Vec::with_capacity(config.num_heads);
    let bundled = read_optional_safetensors(&weights_path.join("medusa_lm_heads.safetensors"))?;

    for head_idx in 0..config.num_heads {
        let block_file = read_optional_safetensors(
            &weights_path.join(format!("medusa_head_{head_idx}_block.safetensors")),
        )?;
        let lm_file = read_optional_safetensors(
            &weights_path.join(format!("medusa_head_{head_idx}_lmhead.safetensors")),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(
                load_matrix_any(
                    ctx,
                    block_file.as_deref().or(bundled.as_deref()),
                    &[
                        format!("layers.{layer_idx}.weight"),
                        format!("medusa_heads.{head_idx}.layers.{layer_idx}.weight"),
                        format!("medusa_head_{head_idx}.layers.{layer_idx}.weight"),
                        format!("head_{head_idx}.block.layers.{layer_idx}.weight"),
                        "weight".to_string(),
                    ],
                    config.hidden_size,
                    config.hidden_size,
                )
                .with_context(|| {
                    format!(
                        "load Medusa head {head_idx} residual layer {layer_idx} from {}",
                        weights_path.display()
                    )
                })?,
            );
        }
        let lm_head = load_matrix_any(
            ctx,
            lm_file.as_deref().or(bundled.as_deref()),
            &[
                "weight".to_string(),
                format!("lm_heads.{head_idx}.weight"),
                format!("medusa_lm_heads.{head_idx}.weight"),
                format!("medusa_heads.{head_idx}.lm_head.weight"),
                format!("head_{head_idx}.lm_head.weight"),
            ],
            config.vocab_size,
            config.hidden_size,
        )
        .with_context(|| {
            format!(
                "load Medusa head {head_idx} lm_head from {}",
                weights_path.display()
            )
        })?;

        blocks.push(ResidualBlock::new(layers));
        lm_heads.push(lm_head);
    }

    Medusa::new(config, blocks, lm_heads)
}

fn read_optional_safetensors(path: &Path) -> Result<Option<Vec<u8>>> {
    if !path.exists() {
        return Ok(None);
    }
    fs::read(path)
        .with_context(|| format!("read {}", path.display()))
        .map(Some)
}

fn load_matrix_any(
    ctx: &DeviceContext,
    tensors: Option<&[u8]>,
    names: &[String],
    rows: usize,
    cols: usize,
) -> Result<DeviceMatrix> {
    let bytes = tensors.ok_or_else(|| anyhow::anyhow!("no Medusa safetensors file found"))?;
    let tensors = SafeTensors::deserialize(bytes).context("deserialize Medusa safetensors")?;
    for name in names {
        if let Ok(view) = tensors.tensor(name) {
            let shape = view.shape();
            if shape.len() != 2 || shape[0] != rows || shape[1] != cols {
                bail!(
                    "Medusa tensor {name} shape {:?}, expected [{rows}, {cols}]",
                    shape
                );
            }
            return DeviceMatrix::from_safetensors(ctx, view.data(), rows, cols)
                .with_context(|| format!("upload Medusa tensor {name}"));
        }
    }
    bail!("missing Medusa tensor; tried {:?}", names)
}
