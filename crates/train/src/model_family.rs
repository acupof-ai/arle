use std::{path::Path, str::FromStr};

use thiserror::Error;

use crate::qwen35::Qwen35Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Auto,
    Qwen35,
}

impl FromStr for ModelFamily {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "qwen35" | "qwen3.5" => Ok(Self::Qwen35),
            _ => Err(format!("unknown model family: {value}")),
        }
    }
}

#[derive(Debug, Error)]
pub enum ModelFamilyError {
    #[error("{0}")]
    Custom(String),
}

pub fn resolve_model_family(
    config_path: &Path,
    requested: ModelFamily,
) -> Result<ModelFamily, ModelFamilyError> {
    match requested {
        ModelFamily::Auto => {
            if Qwen35Config::from_json_file(config_path).is_ok() {
                Ok(ModelFamily::Qwen35)
            } else {
                Err(ModelFamilyError::Custom(format!(
                    "unable to infer model family from {}; qwen3.5 config parser did not accept it",
                    config_path.display()
                )))
            }
        }
        family => Ok(family),
    }
}
