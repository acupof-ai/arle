# Plan — Train Runtime Architecture v1

**Status**: Draft · **Opened**: 2026-04-20 · **Driver**: systematize train stack for extensibility + CUDA readiness
**Relates**: [rust-agent-rl-single-node.md](./rust-agent-rl-single-node.md) · [cuda-kernel-crate-extraction.md](./cuda-kernel-crate-extraction.md)

> **Current reality note**
> This runtime layer is model-family agnostic. The current train-side
> implementation already includes a generic Qwen-family control plane with
> Qwen3.5 as the optimized default. `pretrain` now dispatches across
> Qwen3 / Qwen3.5 families and resumes both weights and optimizer state,
> `train_sft` and `train_grpo` dispatch across Qwen3 / Qwen3.5 families,
> `train_grpo` and `train_multi_turn` now round-trip exact resume state on
> the active RL path, `train_multi_turn` exposes an explicit stepwise-GRPO vs
> sequence-level-GSPO objective switch,
> checkpoints are written as HF-style directories, and the shared Qwen3.5
> model path now supports hybrid linear-attn layers across scratch pretrain,
> LoRA/frozen eval, and RL on the local CPU + Metal path, while CUDA hybrid
> runtime acceptance remains pending. The target train-side
> model line is the Qwen3.5 architecture family.

---

## 1. Problem

The active training binaries (`pretrain`, `train_sft`, `train_grpo`, `train_multi_turn`) historically duplicated the step loop: forward → loss → backward → `optimizer.step()` → `tape.zero_grad()`. That factoring work is now mostly landed: the handwritten Transformer runtime is gone, the active train-side path already shares `Trainer<O, C, S>`, HF-style checkpoint dirs, exact-resume state, and a shared async observability sink. The remaining pain has narrowed to the parts that still resist the current generic runtime shape:

- **RL loops still sit partly outside `Trainer<O, C, S>`** — `train_grpo` and `train_multi_turn` still own rollout/reward/objective orchestration by hand because the current closure model is supervised-step shaped.
- **CLI/runtime composition is still hand-rolled per binary** — flags are not normalized across all train surfaces yet, and `clap` adoption is still open.
- **Infer-side unified `/v1/train/*` bridge is now a thin proxy** — current truth still remains the train-side server inside `crates/train`, but `infer` can now forward `/v1/train/status|events|stop|save` to that authority via `--train-control-url`.
- **Hybrid Qwen3.5 support is no longer trainer/runtime-blocked** — the shared train runtime now accepts hybrid scratch pretrain + RL on the local CPU + Metal path; remaining work is CUDA hybrid runtime acceptance and any further performance tuning, not runtime factoring.
- **CUDA device-resident optimizer / grad path is still a future seam** — current host-authoritative gradient flow is fine for correctness and local Metal, but not the final CUDA scaling story.

Without a shared runtime, every backend × feature combination becomes 5 binary edits.

## 2. Guiding principles

1. **Extract behaviors as traits, keep concrete impls minimal.** Optimizer, LrSchedule, GradClip, and MetricSink became trait boundaries; future abstractions must clear the same "≥2 real call sites" bar.
2. **Host-authoritative gradients stay the default.** `backend.rs:6` is explicit: "Host `Vec<f32>` stays authoritative; GPU backends upload, compute, download per call." Don't break this invariant — adding CUDA should *not* require device-resident tensors. Device-resident optim step lands later as an **additive** `Backend::optim_adamw_step` trait method with CPU fallback (same pattern as existing ops).
3. **TrainerLoop owns the loop; plugins are DI'd.** Binaries reduce to: build model + data closure + plugin config → `trainer.run(step_fn)`.
4. **No half-states** (`feedback_no_half_states.md`). Each phase commits a shippable subset wired end-to-end in ≥1 binary. No parallel "old loop" + "new trainer" drift.
5. **CUDA seams in place from day 1, CUDA impl later.** Trait surface + state codec must be CUDA-ready on the day Phase 2 lands, even if no CUDA code gets written.
6. **Don't pre-shape for hypothetical consumers** (`feedback_no_speculative_interface_shaping.md`). Only abstract what has ≥2 real call sites today. Optimizer = AdamW today, but 4+ binaries instantiate it → trait is justified. Lion/Muon stay un-implemented until asked.

## 3. Layered architecture

```
                     ┌──────────────────────────────────────────────┐
 Binaries            │ pretrain (Qwen-family) · train_sft ·         │
                     │ train_grpo · train_multi_turn                │
 (thin composers)    │                                              │
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

### 4.5 `MetricSink` (✅ Phase 1 landed; 2026-04-21 observability widening landed)

```rust
pub trait MetricSink: Send {
    fn emit(&mut self, sample: &MetricSample<'_>);
    fn event(&mut self, event: &TrainEvent<'_>) {}
    fn flush(&mut self) {}
}
```
Impls now include:

- `NullSink`
- `StdoutSink`
- `JsonlSink`
- `MultiSink`
- `SharedSink` (cloneable async worker-backed handle)
- `MlflowSink` (REST-backed remote adapter for run metadata, metrics, status,
  and checkpoint artifacts)
- `OtlpLogSink` (OTLP/HTTP structured-log adapter for vendor-neutral collectors)
- `WandbProcessSink` (optional sidecar adapter around the official W&B SDK; offline-first)

Current factories:

- `open_sink(jsonl_path, also_stdout)` → boxed `MetricSink`
- `open_shared_sink(jsonl_path, also_stdout)` → cloneable async handle for
  binaries that need to emit lifecycle/artifact events outside `Trainer`

`SharedSink` now uses a bounded queue with explicit overload policy: scalar
metrics are `try_send`ed and may drop with a warning counter under pressure,
while lifecycle/artifact events still block into the queue so `checkpoint` /
`run_end` cannot disappear silently.

The active train-side surface now also has `TrainEvent { kind, step, strings,
scalars, bools }` for lifecycle/artifact records (`run_start`,
`trainer_checkpoint`, `checkpoint`, `status`, `run_end`). JSONL output now
stores one event stream:

- metric records: `{"kind":"metric","phase":"train|eval|...", ...}`
- lifecycle/artifact records: `{"kind":"run_start"|... , ...}`

## 5. CUDA extensibility contracts

Every new trait / codec MUST honor these so CUDA drops in without refactor:

| Surface | Rule | Why |
|---|---|---|
| `Optimizer::step` | Consumes gradients via `store` (host-authoritative today); backend-specific device-step is selected inside the impl via `backend.device()`. Trait signature **does not** expose device choice. | Current CPU/Metal path unchanged; CUDA impl adds `Backend::optim_adamw_step(...)` as additive trait method with CPU fallback (existing pattern, see `backend.rs:9`). |
| `OptimStateDoc` | Always `Vec<f32>` on host. Device-resident moments download on `export_state`, upload on `import_state`. | Checkpoints portable across devices (train on Metal, resume on CUDA, or vice versa). |
| `GradClip::clip` | Returns scalar host `f32` norm. Device-side global-norm reduction is an internal detail. | MetricSink always sees host scalars; no device sync leaks into logging. |
| `MetricSink::emit` / `event` | Foreground path must stay non-blocking from the trainer's point of view. The active binaries now route sink I/O through `SharedSink`'s background worker. | TrainerLoop calls `emit` only after device work is `eval`'d for the step. Lifecycle/artifact events use the same async channel instead of a parallel blocking logger. |
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

Live trainer state already uses the v2 directory layout, and the active
Qwen3/Qwen3.5 train paths already write HF-style step directories. The
remaining checkpoint work is to keep every active entrypoint on that
directory contract:

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
- Keeps the active train path on one checkpoint truth instead of
  re-introducing a parallel single-file legacy codec.

**Resume**: Trainer reads `trainer_state.json`, validates `optim_schema` matches current `Optimizer::state_schema()`, imports moments + scalar state, jumps to `self.step = saved_step + 1`. Binary-side: model weights load is the binary's responsibility (it knows the architecture); trainer exposes a `resume_trainer_state(path) -> TrainerResumeDoc` helper.

## 8. Phase plan

### Phase 1 — Library primitives (parallel, in flight) — ✅ 3/4 green

- ✅ `LrSchedule` trait + 3 impls (`autograd/src/lr_schedule.rs`)
- ✅ `GradAccumulator` (`train/src/grad_accum.rs`)
- ✅ `MetricSink` (`train/src/metrics.rs`) — Null/Stdout/Jsonl/Multi
- ✅ 2026-04-21 observability widening — `MetricSample.phase`, `TrainEvent`,
  `SharedSink`, JSONL lifecycle/artifact records, and binary-level
  `run_start` / `checkpoint` / `run_end` wiring for `pretrain`,
  `train_sft`, `train_grpo`, `train_multi_turn`, and `eval_lm`
- 🟡 `AdamWState` export/import codec on the existing concrete `AdamW` (not trait'd yet)

### Phase 2 — Trait extraction + TrainerLoop skeleton + train_sft migration — ✅ landed 2026-04-20

- ✅ `Optimizer` trait in `crates/autograd/src/optim.rs`; `AdamW` implements it.
- ✅ `GradClip` trait + `NoClip` + `GlobalNorm` in `crates/train/src/grad_clip.rs`.
- ✅ `Trainer<O, C, S>` in `crates/train/src/trainer.rs` (incl. `run_with_hooks`, `resume_if_configured`, v2 codec, P1/P2/P3 from codex review 3d9125d/feae23b + P1 legacy compat from 3d9125d).
- ✅ CheckpointCodec v2 directory layout in `crates/train/src/checkpoint.rs`.
- ✅ **`train_sft.rs` migrated onto Trainer** (commits 44a7e19 + ad5568b + 49512b1). Binary ~250 LOC on the trainer, with `--lr-schedule`, `--warmup-steps`, `--min-lr`, `--grad-accum-steps`, `--metrics-jsonl`, `--resume-from` all wired. `--resume-from` roundtrips end-to-end: Trainer writes `trainer_state.json + optimizer.safetensors`, binary writes `model.safetensors` to the same `step_{:06}/` dir; resume overrides base weights from `<resume_from>/model.safetensors` before restoring optimizer state (fixes P1 flagged in codex review of ad5568b).
- ✅ Trainer step-level tests (15 tests in `crates/train/tests/test_trainer_loop.rs`) covering step counts, grad-accum, LR schedule wiring, metrics, save (incl. force-save on final step), eval-field omission, resume, legacy resume, force-emit on step 1 + final, hook firing, activation cleanup, full save→resume roundtrip.
- ✅ End-to-end 2-step SFT smoke test — CUDA remote validation landed on 2026-04-21 for the current Qwen-family train path (`train_sft -> eval_lm -> agent-infer -> resume` on `Qwen3-0.6B` and the generic dense/full-attn Qwen3.5-family flow). See `docs/experience/wins/2026-04-21-cuda-train-e2e-validation.md`.
- ⏳ Bench: train_sft throughput on Metal before/after — open follow-up; will land as a new wins/ entry when measured.

### Phase 3 — Migrate remaining 4 binaries

- ✅ **Legacy `pretrain` compatibility path retired from the active entrypoint set** (commit 6bd0211 + fix ef24ca6 for `--grad-clip 0` panic). The active train line is the dense/full-attn Qwen3.5-family path.
- ✅ **`pretrain` migrated onto the generic Qwen-family runtime** (source file `pretrain.rs`; commit bd5e277, followed by the 2026-04-20 family/resume tightening). Binary uses `Trainer<AdamW, PretrainClip, ConstantLr>` + `run_with_eval_and_hooks`, dispatches across Qwen3 / Qwen3.5 with Qwen3.5 as the default, writes HF-style `step_{:06}` dirs via the family checkpoint helpers, and now wires `save_every` / `save_dir` / `resume_from` into the trainer so `trainer_state.json + optimizer.safetensors` round-trip alongside model weights. Weight load still happens before `resume_if_configured`, but optimizer moments and step index now restore from the same checkpoint dir instead of resetting on resume. `--grad-clip 0/NaN/inf` still warns + falls through to NoClip. New `--model-family` + `--metrics-jsonl` flags.
- ✅ **`train_grpo.rs` SFT phase migrated** (commit 09c5c89 + fix 1a24db1, followed by the 2026-04-21 exact-resume follow-up). SFT warm-up runs through `Trainer<AdamW, GrpoClip, ConstantLr>` (local enum wrapping `NoClip`/`GlobalNorm` like `PretrainClip`). GRPO phase stays hand-written — rollout_group + ref_model + mid-step `mean_sampled_kl` do not fit the single `step_fn` shape cleanly. AdamW state flows across the SFT→GRPO boundary via `run_sft_phase → AdamWState → import_state` using the existing `Trainer::optim()` + `Optimizer::export_state`/`import_state` (no new public Trainer API needed, contra the original commit body). The RL path now also saves and resumes exact state: merged inference weights, current train weights, frozen reference-model weights, adapter snapshots, and trainer state. `CliError` now flows through an `ExitCode` wrapper that prints via `Display` instead of the default Debug format. New `--grad-clip`, `--no-grad-clip`, `--metrics-jsonl`, `--save-path`, `--save-every`, and `--resume-from` flags; `GRAD_CLIP_NORM = 1.0` constant deleted. Follow-up — ✅ extend `--metrics-jsonl` to cover the GRPO phase (landed 60f7183 + tests 2dd8607): added `JsonlSink::open_append` / `open_sink_append` factory so the GRPO loop extends the JSONL `run_sft_phase` already wrote, with step chained as `sft_steps + iter + 1`. Remaining follow-up: migrate the GRPO phase itself once a GrpoTrainer/closure shape emerges from prototyping.
- ✅ **`train_multi_turn.rs` now runs on the dense/full-attn Qwen3.5-family path** while keeping its GRPO rollout loop hand-written. It builds a `Qwen35Model`, saves HF-style step directories through the shared checkpoint helpers, writes merged inference weights plus exact-resume train artifacts (`train_model.safetensors`, `adapter_model.safetensors`, `trainer_state.json`, `optimizer.safetensors`), and supports `--resume-from` with deterministic seed-per-iter replay. It no longer depends on the handwritten Transformer runtime. It still does not fit `Trainer<O, C, S>`'s single-step closure shape cleanly, so the RL loop remains hand-written pending an RL-shaped trainer variant.
- Each binary lands as its own commit with a bench entry.
- Retire duplicated CLI arg handling; extend `cli_args.rs` with shared `trainer_args()` helper.

### Phase 4 — Eval + observability tightening

- ✅ Built-in metrics: `loss`, `lr`, `grad_norm`, `tok_per_sec`, `ms_per_step` land via Trainer emission (44a7e19 + follow-ups). `alloc_mb` deferred (requires a cross-backend RSS probe, separate track).
- ✅ Lifecycle + artifact events now flow through the same shared async sink as
  scalar metrics. The active binaries emit `run_start`, `checkpoint`, and
  `run_end`; `Trainer` emits `trainer_checkpoint`; `train_multi_turn` eval
  metrics also land through the same sink instead of `println!`.
- ✅ **Perplexity derived from loss in metric pipeline** — Trainer now emits `ppl = exp(loss)` on training samples and `eval_ppl = exp(eval_loss)` on eval samples, treating `loss` / `eval_loss` as token-mean cross-entropy in natural-log space (see `StepOutcome` / `EvalOutcome` metric contract in `crates/train/src/trainer.rs`). Non-CE callers get a numerically defined `ppl` field and should ignore it at the consumer. Sinks already null-fallback on non-finite f64 (JsonlSink) / render `inf` (StdoutSink), so cold-start overflow stays benign. Test `ppl_field_equals_exp_of_loss_field_in_metric_plumbing` pins the mechanical derivation (not semantic correctness) on both surfaces.
- ✅ Held-out eval set support via `run_with_eval(...)` / `run_with_eval_and_hooks(...)` landed in 613ff3c + bd5e277 (+ 813d4f6 leak/final-step fix). Eval fields are prefixed `eval_` (underscore, not dot — matches sink-kv convention: `eval_loss`, `eval_ppl`, `eval_tokens`).
- Remaining follow-up: widen the event schema into a true exporter contract
  (`TrainObservability` / OTLP / W&B / MLflow adapters) without changing the
  shared async sink shape now that the lifecycle/artifact stream exists.

### Phase 5 — Scale features (gated; each a separate project)

- `DdpOptimizer<O>` wrapper (NCCL, CUDA+). Requires collective primitives crate first.
- `ActivationCheckpointPolicy` marker ops at Module boundary.
- `MixedPrecision<O>` wrapper (bf16 compute + f32 master weights + loss scaling).
- Additional optimizers: Lion, Muon — on demand.
- QLoRA — needs quant-aware ops; separate track (blocked on quant-backward work).

## 9. Success criteria

- **Phase 2 done when**: `train_sft --lr-schedule cosine-with-warmup --warmup-steps 100 --grad-accum-steps 4 --metrics-jsonl out.jsonl --resume-from step_50/` runs to completion, resumes correctly, writes JSONL, matches pre-refactor loss curve within bench noise.
- **Phase 3 done when**: all active train entrypoints (`pretrain`, `train_sft`, `train_grpo`, `train_multi_turn`) run on the shared runtime surfaces, no dead legacy runtime code remains in `crates/train`, and every entrypoint is smoke-tested. This is now true for the local Qwen3.5 CPU + Metal path, including hybrid linear-attn; the remaining gap is CUDA hybrid runtime acceptance, not trainer/runtime plumbing.
- **CUDA readiness done when**: `Backend::optim_adamw_step` trait method added with CPU default, `CudaBackend` overrides it, `train_sft --backend cuda` runs with ≥2× PCIe-bw reduction vs. host-authoritative step. (Gated — lands when CUDA weights bench drives the ask.)

## 10. Open questions (ckl decides)

1. **Checkpoint layout**: directory (HF-style) is the current standard and is already used by the active train-side path (`train_sft` / `train_grpo` / `train_multi_turn`). Keep this as the canonical layout.
2. **Trainer vs RLTrainer?** Proposed: **one Trainer for supervised paths, with an RL-shaped variant only if rollout loops keep resisting the current closure model**. `train_grpo` and `train_multi_turn` are the decision point.
3. **Optimizer trait location: `autograd` or new `crates/train-runtime/`?** Proposed: **keep in `autograd`** alongside existing `AdamW`. Extracting a new crate is speculative until ≥2 optimizers exist.
4. **When to delete generic-but-unused abstractions?** Proposed: keep only abstractions with a current caller or an immediate integration plan; otherwise remove them rather than preserving dead code.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Refactor churn causes bench regressions | Trainer is struct dispatch, not dyn; compiler devirtualizes. Bench train_sft before/after; target ±5%. |
| CUDA path turns out to need device-resident state | CUDA PoC as a standalone bench BEFORE committing to the current trait; if ≥10× PCIe cost, revisit trait to add `device()` query. Expected: current host-authoritative step is fine for ≤1B param models. |
| Dead legacy code re-enters through “temporary” compatibility helpers | Delete compatibility code as complete slices and require every retained abstraction to have an active caller or immediate wiring plan. |
| step_fn closure captures become awkward for GRPO rollout | Prototype GRPO migration in Phase 3 before declaring the shape frozen; willing to add `TrainerRl` subtype if single closure doesn't cut it. |

---

## 12. Tracked work

Phase 2 spans `>5 files` → new commits must cite this plan. Commits land on `main` per `feedback_commit_to_main.md`. Every runtime change under `crates/train/` or `crates/autograd/` that can affect numerics triggers a bench entry per `docs/bench-and-trace-spec.md` (training-loss-curve bench; guidellm is inference-side and does not apply here — document that exemption in commit bodies).
