//! ARLE runtime-led post-training substrate.
//!
//! Per the 2026-05-18 OPD-only pivot, ARLE keeps ONE runtime-led training
//! surface (teacher via `infer-api`, student LoRA, shared rollout→score→LoRA
//! backward loop). Two objective families ride that substrate:
//!
//! - **OPD** (`opd`, `self-opd`) — a teacher / EMA-self-teacher scores the
//!   student's on-policy rollout and a KL distill loss drives backward. This is
//!   the "On-Policy Distillation" the pivot is named for.
//! - **RFT / rejection-sampling** (`agent-opd`, `rubric-opd`) — an execution
//!   reward (hidden tests pass) or a rubric judge selects trajectories, trained
//!   by completion/response-masked next-token CE. **No teacher forward, no KL** —
//!   these are on-policy RFT built on the same loop, not distillation. The `opd`
//!   in their names is the shared substrate, not the objective.
//!
//! "OPD-only" is the runtime-led *positioning* (vs the retired pretrain / SFT /
//! GRPO / multi-turn surfaces), not a claim that every subcommand is a KL
//! distillation. See `docs/projects/2026-05-18-opd-only-pivot.md`.

#[path = "agent_opd.rs"]
pub mod agent_opd;
#[path = "causal_lm.rs"]
pub mod causal_lm;
#[path = "cc_convert.rs"]
pub mod cc_convert;
#[path = "checkpoint.rs"]
pub mod checkpoint;
#[path = "control.rs"]
pub mod control;
#[path = "ema_self_teacher.rs"]
pub mod ema_self_teacher;
#[path = "grad_accum.rs"]
pub mod grad_accum;
#[path = "grad_clip.rs"]
pub mod grad_clip;
#[path = "infer_student.rs"]
pub mod infer_student;
#[path = "lora.rs"]
pub mod lora;
#[path = "loss.rs"]
pub mod loss;
#[path = "metrics.rs"]
pub mod metrics;
#[path = "model_family.rs"]
pub mod model_family;
#[path = "moe.rs"]
pub mod moe;
#[path = "opd.rs"]
pub mod opd;
#[path = "prompts.rs"]
pub mod prompts;
#[path = "qwen35.rs"]
pub mod qwen35;
#[path = "qwen35_checkpoint.rs"]
pub mod qwen35_checkpoint;
#[path = "qwen35_loader.rs"]
pub mod qwen35_loader;
#[path = "rubric.rs"]
pub mod rubric;
#[path = "rubric_opd.rs"]
pub mod rubric_opd;
#[path = "sandbox.rs"]
pub mod sandbox;
#[path = "server.rs"]
pub mod server;
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

pub use causal_lm::CausalLm;
pub use grad_accum::GradAccumulator;
pub use lora::{LinearWithLora, LoraAdapterConfig, LoraConfig, LoraTargetSet};
pub use metrics::*;
pub use moe::{MoeConfig, MoeWithLora};
pub use trainer::{
    EvalOutcome, StepCtx, StepOutcome, Trainer, TrainerConfig, cleanup_after_backward,
};
