# Plan — Train Runtime Architecture v1

**Status**: Draft · **Opened**: 2026-04-20 · **Driver**: systematize train stack for extensibility + CUDA readiness
**Relates**: [rust-agent-rl-single-node.md](./rust-agent-rl-single-node.md) · [cuda-kernel-crate-extraction.md](./cuda-kernel-crate-extraction.md)

---

## 1. Problem

5 training binaries (`pretrain`, `pretrain_qwen3`, `train_sft`, `train_grpo`, `train_multi_turn`) each duplicate the step loop: forward → loss → backward → `optimizer.step()` → `tape.zero_grad()`. Short-term pain shows up as:

- **AdamW is a concrete type, not a trait** — adding Lion/Muon/SGD means forking every binary.
- **No LR schedule** — all binaries run constant LR (`pretrain_qwen3:212` has a comment lamenting the missing warmup on resume).
- **No grad accumulation** — batch size ceilinged by VRAM, no way to simulate larger effective batch.
- **No optimizer-state checkpointing** — resume re-zeros AdamW moments.
- **No metrics sink** — stdout only, unstructured.
- **No shared checkpoint codec** — each binary rolls its own save path.
- **CUDA coming up** — `Backend` trait already supports Cuda device; AdamW currently calls `store.to_host(grad_id)` per param per step, which works for Metal (unified memory, cheap) but will be a PCIe bandwidth bottleneck on CUDA at scale.

Without a shared runtime, every backend × feature combination becomes 5 binary edits.

## 2. Guiding principles

1. **Extract behaviors as traits, keep concrete impls minimal.** Optimizer, LrSchedule, GradClip, MetricSink all become trait boundaries.
2. **Host-authoritative gradients stay the default.** `backend.rs:6` is explicit: "Host `Vec<f32>` stays authoritative; GPU backends upload, compute, download per call." Don't break this invariant — adding CUDA should *not* require device-resident tensors. Device-resident optim step lands later as an **additive** `Backend::optim_adamw_step` trait method with CPU fallback (same pattern as existing ops).
3. **TrainerLoop owns the loop; plugins are DI'd.** Binaries reduce to: build model + data closure + plugin config → `trainer.run(step_fn)`.
4. **No half-states** (`feedback_no_half_states.md`). Each phase commits a shippable subset wired end-to-end in ≥1 binary. No parallel "old loop" + "new trainer" drift.
5. **CUDA seams in place from day 1, CUDA impl later.** Trait surface + state codec must be CUDA-ready on the day Phase 2 lands, even if no CUDA code gets written.
6. **Don't pre-shape for hypothetical consumers** (`feedback_no_speculative_interface_shaping.md`). Only abstract what has ≥2 real call sites today. Optimizer = AdamW today, but 4+ binaries instantiate it → trait is justified. Lion/Muon stay un-implemented until asked.

## 3. Layered architecture

```
                     ┌──────────────────────────────────────────────┐
 Binaries            │ pretrain · pretrain_qwen3 · train_sft ·      │
 (thin composers)    │ train_grpo · train_multi_turn                │
                     └──────────────────────────────────────────────┘
                                         │
                                         ▼
 Training runtime    ┌──────────────────────────────────────────────┐
 (NEW)               │ Trainer<M, O, C, S>                          │
                     │   step loop · eval scheduler · ckpt codec    │
                     └──────────────────────────────────────────────┘
                          │      │       │       │        │
       ┌──────────────────┘      │       │       │        └────────┐
       ▼                         ▼       ▼       ▼                 ▼
 ┌──────────┐  ┌───────────┐  ┌──────┐ ┌──────┐ ┌────────────┐  ┌──────────┐
 │Optimizer │  │LrSchedule │  │Grad  │ │Grad  │ │MetricSink  │  │Checkpoint│
 │ trait    │  │ trait     │  │Accum │ │Clip  │ │ trait      │  │ Codec v2 │
 │ (AdamW)  │  │ (3 impls) │  │      │ │trait │ │(Null/Jsonl)│  │ (dir)    │
 └──────────┘  └───────────┘  └──────┘ └──────┘ └────────────┘  └──────────┘
       │                                     │
       ▼                                     ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ Module trait · TensorStore · Tape · Grads · Ops        (existing)   │
 └─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ Backend trait: Cpu · Metal · Cuda          (existing; additive)     │
 │   + future Backend::optim_adamw_step(...)  ← CUDA seam              │
 └─────────────────────────────────────────────────────────────────────┘
```

## 4. Trait surface

### 4.1 `Optimizer` (new, in `crates/autograd/src/optim.rs`)

```rust
pub trait Optimizer: Send {
    fn step(&mut self, store: &mut TensorStore, params: &[TensorId]) -> Result<()>;
    fn zero_grad(&mut self, store: &mut TensorStore, params: &[TensorId]);
    fn set_lr(&mut self, lr: f32);
    fn lr(&self) -> f32;

    /// Schema tag for on-disk state. e.g. `"adamw-v1"`. Used by
    /// CheckpointCodec to validate on import.
    fn state_schema(&self) -> &'static str;

    /// Export moments + scalars keyed by caller-supplied name.
    fn export_state(&self, names: &[(TensorId, String)]) -> OptimStateDoc;

    /// Restore moments; shape mismatch → Err; unknown names silently skipped.
    fn import_state(
        &mut self,
        doc: &OptimStateDoc,
        names: &[(TensorId, String)],
    ) -> Result<usize /* restored */>;
}
```

`AdamW` implements this (Phase 1's `AdamWState` / `export_state` / `import_state` already match the shape; Phase 2 just wraps them in the trait). Lion/Muon are future impls — no speculative scaffolding.

### 4.2 `LrSchedule` (✅ Phase 1 landed)

```rust
pub trait LrSchedule: Send + Sync {
    fn lr(&self, step: u64) -> f32;
    fn describe(&self) -> String;
}
```
Impls: `ConstantLr`, `LinearWarmup`, `CosineWithWarmup`. Parser: `parse_lr_schedule(spec, ...)`.

### 4.3 `GradAccumulator` (✅ Phase 1 landed)

Pure bookkeeper — `new(N)`, `loss_scale() = 1/N`, `observe_and_check_ready()`, `reset_after_step()`. No tensor/tape dependency.

### 4.4 `GradClip` (new, Phase 2, in `crates/train/src/grad_clip.rs`)

```rust
pub trait GradClip: Send {
    /// Clip gradients in-place; return pre-clip global norm for logging.
    fn clip(&mut self, store: &mut TensorStore, params: &[TensorId]) -> Result<f32>;
}

pub struct NoClip;
pub struct GlobalNorm { pub max_norm: f32 }
```
Port the existing `clip_grad_norm(...)` helper calls to `GlobalNorm`. Per-parameter value clipping is **not** first-class — can be added later if a real caller needs it.

### 4.5 `MetricSink` (✅ Phase 1 landed)

```rust
pub trait MetricSink: Send {
    fn emit(&mut self, sample: &MetricSample<'_>);
    fn flush(&mut self) {}
}
```
Impls: `NullSink`, `StdoutSink`, `JsonlSink`, `MultiSink`. Factory: `open_sink(jsonl_path, also_stdout)`.

## 5. CUDA extensibility contracts

Every new trait / codec MUST honor these so CUDA drops in without refactor:

| Surface | Rule | Why |
|---|---|---|
| `Optimizer::step` | Consumes gradients via `store` (host-authoritative today); backend-specific device-step is selected inside the impl via `backend.device()`. Trait signature **does not** expose device choice. | Current CPU/Metal path unchanged; CUDA impl adds `Backend::optim_adamw_step(...)` as additive trait method with CPU fallback (existing pattern, see `backend.rs:9`). |
| `OptimStateDoc` | Always `Vec<f32>` on host. Device-resident moments download on `export_state`, upload on `import_state`. | Checkpoints portable across devices (train on Metal, resume on CUDA, or vice versa). |
| `GradClip::clip` | Returns scalar host `f32` norm. Device-side global-norm reduction is an internal detail. | MetricSink always sees host scalars; no device sync leaks into logging. |
| `MetricSink::emit` | Expected to be ~microsecond — non-blocking from loop's PoV. JsonlSink buffered; Stdout fire-and-forget. | TrainerLoop calls `emit` only after device work is `eval`'d for the step. No mid-step device stalls. |
| `LrSchedule::lr` | Pure host f32; applied via `optimizer.set_lr(...)`. No device side. | Trivial to port. |
| `GradAccumulator` | Pure bookkeeper; no tensor import. | Device-agnostic by construction. |
| Checkpoint layout | Directory (HF-style), safetensors for moments. | HF interop, memory-mapped load, skip re-serialization on Metal↔CUDA moves. |

**Future additive `Backend` methods** (Phase 4+, not required now):
- `fn optim_adamw_step(&self, state, grads, params, hyper) -> Result<()>` with CPU default = current host loop.
- `fn global_grad_norm(&self, grads) -> Result<f32>` with CPU default = host sqrt(sum of squares).
- `fn scale_grads(&self, grads, scale: f32) -> Result<()>` for mixed-precision loss scaling.

None of these change Phase 2 / 3 work. They land when a CUDA perf ticket justifies them.

## 6. TrainerLoop

```rust
// crates/train/src/trainer.rs
pub struct TrainerConfig {
    pub total_steps: u64,
    pub grad_accum_steps: u64,
    pub log_every: u64,
    pub eval_every: Option<u64>,
    pub save_every: Option<u64>,
    pub save_dir: Option<PathBuf>,
    pub metrics_jsonl: Option<PathBuf>,
    pub resume_from: Option<PathBuf>,
}

pub struct Trainer<O: Optimizer, C: GradClip, S: LrSchedule> {
    optim: O,
    clip: C,
    schedule: S,
    accum: GradAccumulator,
    metrics: Box<dyn MetricSink>,
    cfg: TrainerConfig,
    step: u64,
}

pub struct StepCtx<'a> {
    pub step: u64,
    pub micro_idx: u64,          // 0..grad_accum_steps
    pub loss_scale: f32,          // = 1 / grad_accum_steps
    pub store: &'a mut TensorStore,
    pub tape: &'a mut Tape,
}

pub struct StepOutcome {
    pub loss: f32,                // post-scale; trainer reconstructs true loss
    pub token_count: u64,         // for tok/s metric
}

impl<O, C, S> Trainer<O, C, S> where O: Optimizer, C: GradClip, S: LrSchedule {
    pub fn run<F>(&mut self, params: Vec<TensorId>, mut step_fn: F) -> Result<()>
    where F: FnMut(&mut StepCtx<'_>) -> Result<StepOutcome>;

    pub fn run_with_eval<F, E>(
        &mut self,
        params: Vec<TensorId>,
        mut step_fn: F,
        mut eval_fn: E,
    ) -> Result<()>
    where
        F: FnMut(&mut StepCtx<'_>) -> Result<StepOutcome>,
        E: FnMut(&mut TensorStore, &mut Tape) -> Result<EvalResult>;
}
```

The binary stays responsible for:
- Building the model (weights in TensorStore)
- Constructing the data iterator
- The forward+loss closure (`step_fn`)

Trainer owns: LR schedule (per-step `optim.set_lr`), grad accumulation, backward-already-called sanity, clip norm, optim step, zero_grad, metrics emit, eval scheduling, save scheduling, resume-from-ckpt bookkeeping, RNG seed persistence.

**For GRPO / multi-turn** — the same `Trainer` works. Rollout + reward logic lives *inside* `step_fn` (it already does). The trainer only cares about: you gave me a loss, I run backward + step. GRPO's per-group advantage computation is orthogonal.

## 7. Checkpoint Codec v2 (directory layout)

Current `checkpoint.rs` uses single-file `LMCKP003` magic. Train_sft already writes HF-style directories (`step_N/model.safetensors + config.json + tokenizer.json`). Unify on directory:

```
step_000123/
  model.safetensors       # weights (f32 or bf16)
  optimizer.safetensors   # moments: each param gets two tensors "{name}.m", "{name}.v"
  trainer_state.json      # { step, schedule_name, schedule_params, accum_state, rng_seed, optim_schema }
  config.json             # model config (binary-specific)
  tokenizer.json          # tokenizer (if applicable)
```

**Why directory:**
- HF interop out of the box (safetensors + config.json + tokenizer.json matches HF convention).
- Moments as safetensors → memory-mapped load, device-transferable.
- `trainer_state.json` is small + human-readable for debugging resumes.
- Retires custom `LMCKP003` binary format (exists only for pretrain; migrate via compat reader for 1 release then drop).

**Resume**: Trainer reads `trainer_state.json`, validates `optim_schema` matches current `Optimizer::state_schema()`, imports moments + scalar state, jumps to `self.step = saved_step + 1`. Binary-side: model weights load is the binary's responsibility (it knows the architecture); trainer exposes a `resume_trainer_state(path) -> TrainerResumeDoc` helper.

## 8. Phase plan

### Phase 1 — Library primitives (parallel, in flight) — ✅ 3/4 green

- ✅ `LrSchedule` trait + 3 impls (`autograd/src/lr_schedule.rs`)
- ✅ `GradAccumulator` (`train/src/grad_accum.rs`)
- ✅ `MetricSink` (`train/src/metrics.rs`) — Null/Stdout/Jsonl/Multi
- 🟡 `AdamWState` export/import codec on the existing concrete `AdamW` (not trait'd yet)

### Phase 2 — Trait extraction + TrainerLoop skeleton + train_sft migration — ✅ landed 2026-04-20

- ✅ `Optimizer` trait in `crates/autograd/src/optim.rs`; `AdamW` implements it.
- ✅ `GradClip` trait + `NoClip` + `GlobalNorm` in `crates/train/src/grad_clip.rs`.
- ✅ `Trainer<O, C, S>` in `crates/train/src/trainer.rs` (incl. `run_with_hooks`, `resume_if_configured`, v2 codec, P1/P2/P3 from codex review 3d9125d/feae23b + P1 legacy compat from 3d9125d).
- ✅ CheckpointCodec v2 directory layout in `crates/train/src/checkpoint.rs`. `LMCKP003` reader retained for pretrain compat.
- ✅ **`train_sft.rs` migrated onto Trainer** (commits 44a7e19 + ad5568b + 49512b1). Binary ~250 LOC on the trainer, with `--lr-schedule`, `--warmup-steps`, `--min-lr`, `--grad-accum-steps`, `--metrics-jsonl`, `--resume-from` all wired. `--resume-from` roundtrips end-to-end: Trainer writes `trainer_state.json + optimizer.safetensors`, binary writes `model.safetensors` to the same `step_{:06}/` dir; resume overrides base weights from `<resume_from>/model.safetensors` before restoring optimizer state (fixes P1 flagged in codex review of ad5568b).
- ✅ Trainer step-level tests (15 tests in `crates/train/tests/test_trainer_loop.rs`) covering step counts, grad-accum, LR schedule wiring, metrics, save (incl. force-save on final step), eval-field omission, resume, legacy resume, force-emit on step 1 + final, hook firing, activation cleanup, full save→resume roundtrip.
- ⏳ End-to-end 2-step SFT smoke test — pending remote runner w/ Qwen3-0.6B weights.
- ⏳ Bench: train_sft throughput on Metal before/after — pending remote (stub in `docs/experience/wins/2026-04-20-wave3-train-sft-trainer-migration.md`).

### Phase 3 — Migrate remaining 4 binaries

- ✅ **`pretrain.rs` migrated** (commit 6bd0211 + fix ef24ca6 for `--grad-clip 0` panic). Binary ~415 LOC; uses `Trainer<AdamW, PretrainClip, ConstantLr>` with local enum wrapping `NoClip`/`GlobalNorm`. Save/resume deferred pending a Transformer safetensors codec (the hand-written loop also had no checkpoint support).
- ✅ **`pretrain_qwen3.rs` migrated** (commit bd5e277, preceded by `Trainer::run_with_eval_and_hooks` in 613ff3c and the tied/untied resume fix in 429efc3/8f2df76). Binary uses `Trainer<AdamW, PretrainClip, ConstantLr>` + `run_with_eval_and_hooks` — eval closure runs the held-out windows, `on_step_end` hook writes `step_<N>/{config.json, tokenizer.json, model.safetensors}` via the existing `save_checkpoint` helper. Trainer-side `save_every` / `save_dir` / `resume_from` stay `None`; trainer_state.json + optimizer.safetensors are deferred follow-up work. `resume_from_checkpoint` stays as a pre-Trainer weight-only load (with `tie_word_embeddings` guard + `load_into_strict`) and returns an absolute `start_step` that the hook reconstructs for save directory naming; metric-sample step numbers are relative to the Trainer's 0..total_steps counter. `--grad-clip 0/NaN/inf` warns + falls through to NoClip. New `--metrics-jsonl` flag.
- ✅ **`train_grpo.rs` SFT phase migrated** (commit 09c5c89 + fix 1a24db1). SFT warm-up runs through `Trainer<AdamW, GrpoClip, ConstantLr>` (local enum wrapping `NoClip`/`GlobalNorm` like `PretrainClip`). GRPO phase stays hand-written — rollout_group + ref_model + mid-step `mean_sampled_kl` do not fit the single `step_fn` shape cleanly. AdamW state flows across the SFT→GRPO boundary via `run_sft_phase → AdamWState → import_state` using the existing `Trainer::optim()` + `Optimizer::export_state`/`import_state` (no new public Trainer API needed, contra the original commit body). `CliError` now flows through an `ExitCode` wrapper that prints via `Display` instead of the default Debug format. New `--grad-clip`, `--no-grad-clip`, `--metrics-jsonl` flags; `GRAD_CLIP_NORM = 1.0` constant deleted. Follow-up: extend `--metrics-jsonl` to cover the GRPO phase (currently SFT-only); eventually migrate the GRPO phase once a GrpoTrainer/closure shape emerges from prototyping.
- ⏳ `train_multi_turn.rs` (~650 LOC) — similar two-phase shape to train_grpo (SFT warm-up + multi-turn rollout). Expected: migrate SFT phase only, leave multi-turn rollout hand-written. Not yet scoped in detail.
- Each binary lands as its own commit with a bench entry.
- Retire duplicated CLI arg handling; extend `cli_args.rs` with shared `trainer_args()` helper.

### Phase 4 — Eval + observability tightening

- Built-in metrics: `loss`, `lr`, `grad_norm`, `tok_per_sec`, `ms_per_step`, `alloc_mb` (if available).
- Perplexity derived from loss in metric pipeline.
- Held-out eval set support — `run_with_eval(...)` path. All metrics prefixed `eval.*`.
- Additive `Backend::global_grad_norm` with CPU default — opens device-side norm reduction seam without implementing it yet.

### Phase 5 — Scale features (gated; each a separate project)

- `DdpOptimizer<O>` wrapper (NCCL, CUDA+). Requires collective primitives crate first.
- `ActivationCheckpointPolicy` marker ops at Module boundary.
- `MixedPrecision<O>` wrapper (bf16 compute + f32 master weights + loss scaling).
- Additional optimizers: Lion, Muon — on demand.
- QLoRA — needs quant-aware ops; separate track (blocked on quant-backward work).

## 9. Success criteria

- **Phase 2 done when**: `train_sft --lr-schedule cosine-with-warmup --warmup-steps 100 --grad-accum-steps 4 --metrics-jsonl out.jsonl --resume-from step_50/` runs to completion, resumes correctly, writes JSONL, matches pre-refactor loss curve within bench noise.
- **Phase 3 done when**: all 5 binaries on Trainer; binary LOC reduced ≥40% in aggregate; every binary smoke-tested.
- **CUDA readiness done when**: `Backend::optim_adamw_step` trait method added with CPU default, `CudaBackend` overrides it, `train_sft --backend cuda` runs with ≥2× PCIe-bw reduction vs. host-authoritative step. (Gated — lands when CUDA weights bench drives the ask.)

## 10. Open questions (ckl decides)

1. **Checkpoint layout: directory (HF-style) or single-file (bump to `LMCKP004`)?** Proposed: **directory**. HF-interop + mmap + device-transfer. Downside: more files per checkpoint.
2. **Trainer vs RLTrainer?** Proposed: **one Trainer, GRPO logic lives in step_fn**. The GRPO-specific advantage calc is already per-step in the binary; trainer doesn't need to know.
3. **Retire `LMCKP003` immediately or keep reader for 1 release?** Proposed: **keep reader 1 release** so pretrain can cross the gap without orphaning checkpoints.
4. **Optimizer trait location: `autograd` or new `crates/train-runtime/`?** Proposed: **keep in `autograd`** alongside existing `AdamW`. Extracting a new crate is speculative until ≥2 optimizers exist.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Refactor churn causes bench regressions | Trainer is struct dispatch, not dyn; compiler devirtualizes. Bench train_sft before/after; target ±5%. |
| CUDA path turns out to need device-resident state | CUDA PoC as a standalone bench BEFORE committing to the current trait; if ≥10× PCIe cost, revisit trait to add `device()` query. Expected: current host-authoritative step is fine for ≤1B param models. |
| Directory ckpt breaks existing resume flows | Compat shim in checkpoint.rs reads both `LMCKP003` and directory layout for ≥1 release. |
| step_fn closure captures become awkward for GRPO rollout | Prototype GRPO migration in Phase 3 before declaring the shape frozen; willing to add `TrainerRl` subtype if single closure doesn't cut it. |

---

## 12. Tracked work

Phase 2 spans `>5 files` → new commits must cite this plan. Commits land on `main` per `feedback_commit_to_main.md`. Every runtime change under `crates/train/` or `crates/autograd/` that can affect numerics triggers a bench entry per `docs/bench-and-trace-spec.md` (training-loss-curve bench; guidellm is inference-side and does not apply here — document that exemption in commit bodies).
