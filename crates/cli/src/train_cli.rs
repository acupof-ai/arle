use std::process::ExitCode;

use anyhow::Result;
#[cfg(not(feature = "cuda"))]
use anyhow::bail;

use crate::args::{TrainArgs, TrainCommand, TrainOpdArgs, TrainSelfOpdArgs};

#[path = "train_cli/agent_opd.rs"]
mod agent_opd;
#[path = "train_cli/agent_opd_batch.rs"]
mod agent_opd_batch;
#[path = "train_cli/agent_opd_mesh.rs"]
mod agent_opd_mesh;
#[path = "train_cli/agent_opd_window.rs"]
mod agent_opd_window;
#[path = "train_cli/capacity_report.rs"]
mod capacity_report;
#[path = "train_cli/cc_eval.rs"]
mod cc_eval;
#[path = "train_cli/math_opd.rs"]
mod math_opd;
#[path = "train_cli/model_probe.rs"]
mod model_probe;
#[path = "train_cli/nll_eval.rs"]
mod nll_eval;
#[path = "train_cli/opd_checkpoint.rs"]
mod opd_checkpoint;
#[path = "train_cli/opd_driver.rs"]
mod opd_driver;
#[path = "train_cli/opd_engine.rs"]
mod opd_engine;
#[path = "train_cli/opd_prompts.rs"]
mod opd_prompts;
#[path = "train_cli/opd_runtime.rs"]
mod opd_runtime;
#[path = "train_cli/opd_teacher.rs"]
mod opd_teacher;
#[path = "train_cli/replay_records.rs"]
mod replay_records;
#[path = "train_cli/rubric_opd.rs"]
mod rubric_opd;
#[path = "train_cli/w2s.rs"]
mod w2s;

#[cfg(feature = "cuda")]
pub(crate) use model_probe::resolve_local_tokenizer_path;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) use model_probe::run_model;

#[cfg(feature = "cuda")]
use crate::spec_train_target::run_spec_draft;

pub(crate) fn run_train(train: TrainArgs) -> ExitCode {
    // OPD runtime toggles land in the train/autograd statics once, before any
    // model load or step (runtime config = CLI flags, not env).
    match train.command {
        TrainCommand::Env(args) => exit_from_result(capacity_report::run_train_env(args)),
        TrainCommand::EstimateMemory(args) => {
            exit_from_result(capacity_report::run_train_estimate_memory(args))
        }
        TrainCommand::Opd(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            run_opd(args)
        }
        TrainCommand::SelfOpd(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            run_self_opd(args)
        }
        TrainCommand::RubricOpd(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            exit_from_result(rubric_opd::run_rubric_opd_impl(args))
        }
        TrainCommand::AgentOpd(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            exit_from_result(agent_opd::run_agent_opd_impl(*args))
        }
        TrainCommand::MathOpd(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            exit_from_result(math_opd::run_math_opd_impl(*args))
        }
        TrainCommand::CcConvert(args) => {
            exit_from_result(replay_records::run_cc_convert_impl(args))
        }
        TrainCommand::Ppl(args) => exit_from_result(nll_eval::run_ppl(args)),
        TrainCommand::SpecDraft(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            exit_from_result(run_spec_draft(args))
        }
        TrainCommand::W2s(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            exit_from_result(w2s::run_w2s(args))
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn run_spec_draft(_args: crate::args::TrainSpecDraftArgs) -> Result<()> {
    bail!("arle train spec-draft requires the CUDA backend (the trunk forward is CUDA-only)")
}

fn run_opd(args: TrainOpdArgs) -> ExitCode {
    if args.smoke {
        return exit_from_result(opd_driver::run_opd_smoke(args));
    }
    if args.student_model.is_some() {
        return exit_from_result(opd_driver::run_opd_from_dirs(args));
    }
    eprintln!(
        "[arle train opd] error: either `--student-model <dir>` or `--smoke` is required.\n\
         See `arle train opd --help` for the full surface."
    );
    ExitCode::FAILURE
}

fn run_self_opd(args: TrainSelfOpdArgs) -> ExitCode {
    // Reject 0.0, negatives, AND NaN (`NaN <= 0.0` is false, so the is_nan arm).
    if args.gkd_lambda <= 0.0 || args.gkd_lambda.is_nan() {
        eprintln!(
            "[arle train self-opd] error: --gkd-lambda must be > 0 ({} given).\n\
             SOPD cold-starts with the EMA teacher == an exact copy of the student\n\
             (lora_b=0 ⇒ student == EMA == base), so the pure KL term has zero gradient\n\
             and the run is a silent no-op. The bootstrap gradient comes from the λ>0\n\
             GKD CE self-anchor on the rollouts; default is 0.5.",
            args.gkd_lambda
        );
        return ExitCode::FAILURE;
    }
    if args.smoke {
        return exit_from_result(opd_driver::run_self_opd_smoke(args));
    }
    if args.student_model.is_some() {
        return exit_from_result(opd_driver::run_self_opd_from_dir(args));
    }
    eprintln!(
        "[arle train self-opd] error: either `--student-model <dir>` or `--smoke` is required.\n\
         See `arle train self-opd --help` for the full surface."
    );
    ExitCode::FAILURE
}

fn exit_from_result(result: Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[ARLE train] error: {err:#}");
            ExitCode::FAILURE
        }
    }
}
