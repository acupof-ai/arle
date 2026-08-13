#[cfg(feature = "cuda")]
use {
    anyhow::Result,
    std::{collections::VecDeque, path::PathBuf, sync::Arc},
    train::{
        cc_harness::{CcGroup, CcHarness},
        swe_dataset::SweTask,
    },
};

/// One task group's pending rollout (`--staleness` dial). Staleness 0 keeps
/// today's boot-ahead: sandboxes build in the background, the rollout itself
/// runs inline at collect (strictly on-policy). Staleness 1 runs the WHOLE
/// rollout on a background thread launched before the previous group's
/// train+merge, tagged with the policy version it launched under.
#[cfg(feature = "cuda")]
pub(super) enum PendingGroup {
    Booted(train::cc_harness::BootedGroup),
    Rolling {
        behavior_version: u64,
        handle: std::thread::JoinHandle<Result<CcGroup>>,
    },
}

/// The immutable inputs every group launch in a round shares.
#[cfg(feature = "cuda")]
pub(super) struct GroupLauncher<'a> {
    pub(super) tasks: &'a [(Arc<SweTask>, PathBuf)],
    pub(super) harness: &'a Arc<CcHarness>,
    pub(super) staleness: u8,
    pub(super) width: usize,
    pub(super) groups_per_update: usize,
}

#[cfg(feature = "cuda")]
impl GroupLauncher<'_> {
    fn launch(&self, i: usize, version: u64) -> PendingGroup {
        let (task, staged) = &self.tasks[i];
        let booted = self.harness.boot_group(task, staged, self.width);
        if self.staleness == 0 {
            return PendingGroup::Booted(booted);
        }
        let harness = Arc::clone(self.harness);
        PendingGroup::Rolling {
            behavior_version: version,
            handle: std::thread::spawn(move || harness.run_group(booted)),
        }
    }

    pub(super) fn top_up(
        &self,
        pending: &mut VecDeque<(usize, PendingGroup)>,
        launched: &mut usize,
        work: &[usize],
        version: u64,
    ) {
        while pending.len() < self.groups_per_update && *launched < work.len() {
            let i = work[*launched];
            pending.push_back((i, self.launch(i, version)));
            *launched += 1;
        }
    }
}
