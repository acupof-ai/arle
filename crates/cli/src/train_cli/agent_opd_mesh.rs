#[cfg(feature = "cuda")]
use {
    super::{
        agent_opd::build_value_critic,
        opd_engine::{
            load_agent_opd_serve_student, quiesce_and_release_engines, sync_and_restore_engines,
        },
    },
    crate::args::TrainAgentOpdArgs,
    anyhow::{Result, anyhow},
    std::path::PathBuf,
};

/// Rank-0 → follower stream for cp>1 agent-opd. Stochastic cc rollouts diverge
/// across mesh ranks, so rank 0 owns the harness + filtering and publishes
/// every update's batch plus the engine-lifecycle decisions around it;
/// followers serve rollouts and mirror the calls, keeping the writeback's cp
/// collectives on identical call sequences. Files live under the
/// coordinator-minted `ARLE_TRAIN_MESH_DIR`; publish is write-then-rename so a
/// reader never sees a partial file.
#[cfg(feature = "cuda")]
#[derive(serde::Serialize, serde::Deserialize)]
enum MeshMsg {
    Update {
        batch: Vec<train::update_strategy::ScoredTrajectory>,
        /// Mirror of the leader's quiesce + scratch/KV release before this
        /// update (skipped under staleness>0 with a group in flight).
        release_engines: bool,
    },
    /// End of one group's updates: re-merge LoRA when the leader did, then
    /// re-acquire the KV pool for the next rollout.
    GroupEnd { synced: bool },
}

/// Borrow twin of [`MeshMsg`] for the publish side — same serde shape, no
/// batch clone.
#[cfg(feature = "cuda")]
#[derive(serde::Serialize)]
pub(super) enum MeshMsgRef<'a> {
    Update {
        batch: &'a [train::update_strategy::ScoredTrajectory],
        release_engines: bool,
    },
    GroupEnd {
        synced: bool,
    },
}

#[cfg(feature = "cuda")]
pub(super) struct MeshUpdateChannel {
    pub(super) dir: PathBuf,
    seq: u64,
    /// Next consumed file the GC may delete (last cp rank only).
    gc_next: u64,
}

#[cfg(feature = "cuda")]
impl MeshUpdateChannel {
    pub(super) fn from_env() -> Result<Self> {
        let dir = std::env::var_os("ARLE_TRAIN_MESH_DIR").ok_or_else(|| {
            anyhow!("cp>1 agent-opd needs the mesh coordinator (ARLE_TRAIN_MESH_DIR unset)")
        })?;
        Ok(Self {
            dir: dir.into(),
            seq: 0,
            gc_next: 0,
        })
    }

    fn upd_path(&self, seq: u64) -> PathBuf {
        self.dir.join(format!("upd-{seq:08}.json"))
    }

    pub(super) fn publish<M: serde::Serialize>(&mut self, msg: &M) -> Result<()> {
        let tmp = self.dir.join(format!("upd-{:08}.tmp", self.seq));
        std::fs::write(&tmp, serde_json::to_vec(msg)?)?;
        std::fs::rename(&tmp, self.upd_path(self.seq))?;
        self.seq += 1;
        Ok(())
    }

    /// Delete every consumed file. Call ONLY right after an `Update` completed:
    /// its collective proves every rank has consumed all files below `seq`.
    /// (`GroupEnd` carries no collective, so it proves nothing — an eager
    /// delete there deadlocks a slower rank at cp>=3.)
    fn gc_consumed(&mut self) {
        while self.gc_next < self.seq {
            let _ = std::fs::remove_file(self.upd_path(self.gc_next));
            self.gc_next += 1;
        }
    }

    pub(super) fn finish(&self) -> Result<()> {
        std::fs::write(self.dir.join("end"), b"").map_err(Into::into)
    }

    /// Blocks until the next message (`Some`) or the end marker (`None`).
    fn recv(&mut self) -> Result<Option<MeshMsg>> {
        loop {
            match std::fs::read(self.upd_path(self.seq)) {
                Ok(bytes) => {
                    self.seq += 1;
                    return Ok(Some(serde_json::from_slice(&bytes)?));
                }
                Err(_) => {
                    if self.dir.join("end").exists() {
                        return Ok(None);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    }
}

/// cp rank > 0 of the cc-rollout lane: serve rollouts for rank 0's harness
/// (fleet endpoint) and mirror rank 0's update stream — including its engine
/// quiesce/release and LoRA re-merge decisions — until end-of-stream.
#[cfg(feature = "cuda")]
pub(super) fn run_agent_opd_cp_follower(
    args: &TrainAgentOpdArgs,
    lora: train::lora::LoraConfig,
    target_set: train::lora::LoraTargetSet,
    serve_port: u16,
) -> Result<()> {
    use autograd::optim::AdamW;

    let cp = train::context_parallel::CpContext::from_env();
    let update_preset = args.update_preset();

    let mut fleet = load_agent_opd_serve_student(args, lora, target_set, serve_port)?;
    let vocab = fleet.vocab;
    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let mut value_critic = build_value_critic(
        &update_preset,
        fleet.student.config().hidden_size,
        args.value_lr,
        &mut fleet.store,
    )?;
    let adapter_map = fleet.student.adapter_name_map();

    let mut rx = MeshUpdateChannel::from_env()?;
    // Rank 0 waits for this before its first session can hit our endpoint.
    std::fs::write(rx.dir.join(format!("serve-r{}.ready", cp.rank)), b"")?;
    let gc_owner = cp.rank + 1 == cp.size;
    while let Some(msg) = rx.recv()? {
        match msg {
            MeshMsg::Update {
                batch,
                release_engines,
            } => {
                if release_engines {
                    quiesce_and_release_engines(&fleet.infer_student)?;
                }
                let report = update_preset
                    .update(
                        &batch,
                        &fleet.student,
                        fleet.all_params.as_slice(),
                        fleet.trainable.as_slice(),
                        &mut optimizer,
                        value_critic.as_mut(),
                        vocab,
                        args.writeback_window,
                        &mut fleet.store,
                    )
                    .map_err(anyhow::Error::from)?;
                eprintln!(
                    "[agent-opd] follower update {}: trained={} loss={:.6}",
                    rx.seq - 1,
                    report.trained,
                    report.loss
                );
                if let Err(err) = fleet.store.backend().trim_memory_pool() {
                    eprintln!("[agent-opd] follower device-pool trim failed: {err}");
                }
                if gc_owner {
                    rx.gc_consumed();
                }
            }
            MeshMsg::GroupEnd { synced } => {
                sync_and_restore_engines(
                    &fleet.infer_student,
                    &mut fleet.store,
                    &adapter_map,
                    &fleet.student.param_name_map(),
                    lora,
                    synced,
                )?;
            }
        }
    }
    eprintln!("[agent-opd] cp follower rank {}: end of stream", cp.rank);
    fleet.serve_thread.shutdown()?;
    Ok(())
}
