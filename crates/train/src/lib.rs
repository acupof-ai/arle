//! ARLE runtime-led post-training substrate (2026-05-18 OPD-only pivot): one
//! surface — teacher via `infer-api`, student LoRA, rollout→score→LoRA-backward.
//! Two objective families ride it:
//! - **OPD** (`opd`, `self-opd`): teacher / EMA-self-teacher KL distill.
//! - **RFT** (`agent-opd`, `rubric-opd`): execution- or rubric-reward selects
//!   trajectories, trained by masked CE — no teacher, no KL. The `opd` in these
//!   names is the shared substrate, not the objective.
//!
//! "OPD-only" is the positioning (vs the retired pretrain/SFT/GRPO/multi-turn),
//! not a claim every subcommand is KL distillation.

#[path = "aopd_profile.rs"]
pub mod aopd_profile;
#[path = "causal_lm.rs"]
pub mod causal_lm;
#[path = "cc_convert.rs"]
pub mod cc_convert;
#[path = "cc_harness.rs"]
pub mod cc_harness;
#[path = "checkpoint.rs"]
pub mod checkpoint;
#[path = "context_parallel.rs"]
pub mod context_parallel;
#[path = "ema_self_teacher.rs"]
pub mod ema_self_teacher;
pub use autograd::grad_clip;
#[path = "infer_student.rs"]
pub mod infer_student;
#[path = "lora.rs"]
pub mod lora;
#[path = "lora_shard.rs"]
pub mod lora_shard;
#[path = "loss.rs"]
pub mod loss;
#[path = "model_family.rs"]
pub mod model_family;
#[path = "opd.rs"]
pub mod opd;
#[path = "pipeline_parallel.rs"]
pub mod pipeline_parallel;
#[path = "prompts.rs"]
pub mod prompts;
#[path = "qwen35.rs"]
pub mod qwen35;
#[path = "qwen35_checkpoint.rs"]
pub mod qwen35_checkpoint;
#[path = "qwen35_loader.rs"]
pub mod qwen35_loader;
#[path = "tensor_parallel.rs"]
pub mod tensor_parallel;
// CLI-driven runtime toggles (train flags -> statics; no env reads).
#[path = "runtime_flags.rs"]
mod runtime_flags;
pub use runtime_flags::{TrainRuntimeFlags, apply_runtime_flags};
#[path = "rubric.rs"]
pub mod rubric;
#[path = "rubric_opd.rs"]
pub mod rubric_opd;
#[path = "sandbox.rs"]
pub mod sandbox;
#[path = "spawner.rs"]
pub mod spawner;
#[path = "swe_dataset.rs"]
pub mod swe_dataset;
#[path = "teacher_infer.rs"]
pub mod teacher_infer;
#[path = "tokenizer.rs"]
pub mod tokenizer;
#[path = "trainer.rs"]
pub mod trainer;
#[path = "trajectory_scorer.rs"]
pub mod trajectory_scorer;
#[path = "update_strategy.rs"]
pub mod update_strategy;

pub use lora::{LinearWithLora, LoraAdapterConfig, LoraConfig, LoraTargetSet};
pub use trainer::cleanup_after_backward;
