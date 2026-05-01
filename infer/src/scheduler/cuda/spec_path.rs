use super::{ModelForward, Scheduler};

pub(super) struct SpecPath;

impl SpecPath {
    pub(super) fn draft_then_verify<M: ModelForward>(scheduler: &mut Scheduler<M>) {
        scheduler.step_spec_decode_launch_from_path();
    }
}
