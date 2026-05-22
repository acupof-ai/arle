# 2026-05-22 — OPD 9B-teacher BF16 cycle: session wrap

> **Status:** cycle closed at 03:53 local; codex idle awaiting next-axis
> direction since tick 8 of the user's `/loop`. This wrap captures the
> ~2 h cooperative arc (Claude memory/observation + codex implementation +
> verification) so any agent picking up OPD 9B-teacher work has a single
> read-first entry. Cross-links to the dated `wins/`, `errors/`, and
> `plans/` entries that ship with the commits.

## Headline

Substrate win + hardware-blocked headline:

- **BF16 frozen-base autograd substrate shipped end-to-end** (autograd
  backend → embedding → matmul forward → matmul backward → Qwen3.5 LoRA
  loader), validated with 2 numerical CUDA tests passing and a wins
  entry. The substrate is general-purpose; it cuts student weight-bytes
  ~2× for any future LoRA-student run.
- **But same-card 9B GPTQModel-teacher → 0.8B LoRA student OPD on 16 GB
  is KILLED** — the strictest no-eval shape (`steps=1, rollout=1,
  prompt_max=1`) still hits **15871 MiB used / 73 MiB free** before
  `cuda alloc_zeros failed (mean_backward_device)`. Constraint is
  teacher + student + tape co-residency, not student weight storage.
- **Reframe**: the `ApiTeacher` / `MultiTeacher` track shipped earlier in
  this session (`c0a2975`, `0bfa852`, `13e70d2`) is no longer just
  "DX nice-to-have" — it's the **only practical 9B-teacher OPD path on
  16 GB** when the teacher runs out-of-card (different host, GPU, or
  process memory budget).

## Commit chronicle (13 commits this session)

### Loader correctness (1 commit)

| Commit | Axis |
|---|---|
| `4214b4d fix(qwen35): load BF16 linear-attn f32 tensors by dtype` | `infer/src/weight_loader.rs::load_tensor_1d_f32` was casting raw tensor bytes to f32 regardless of stored dtype, turning 32-element BF16 `A_log`/`dt_bias`/`norm.weight` into 16 garbage f32s. New `tensor_1d_to_f32` helper supports F32/BF16/F16. Unblocks Qwen3.5-9B GPTQModel linear-attn substage finiteness end-to-end. |

### Diagnostic instrumentation (1 commit)

| Commit | Axis |
|---|---|
| `424a4cf fix(cuda): include H2D allocation details in autograd upload errors` | `CudaBackend::upload_slice` now reports `shape={...} len={N} bytes={B} err={driver_err}` instead of "cuda htod copy failed". Instrument-first-then-fix discipline; needed to identify which exact tensor blew the 16 GB budget. |

### Documentation gates (3 commits)

| Commit | Axis |
|---|---|
| `306df1a docs(cuda): record GPTQModel full-logits gate after f32-load fix` | Full-vocab single-token parity vs HF BF16 reference: `arle_argmax == ref_argmax`, `top64_max_rel=12.4%`, `all_rmse_over_ref_rms=10.8%`. Within INT4-quant tolerance; teacher path licensed at the logits-API surface. |
| `6f2bd3a docs(opd): record GPTQModel 9B OPD memory kill` | First-pass OPD-memory KILL (before BF16 substrate). |
| `5875519 docs(opd): audit frozen BF16 LoRA student unblock` | Pre-implementation audit. Identified the autograd-only cut-point (BF16 weights live in backend, f32 grad/activation flow unchanged) — kept the substrate from blowing into a multi-module sprint. |

### Teacher DX (1 commit)

| Commit | Axis |
|---|---|
| `13e70d2 feat(opd): add API teacher config to infer-teacher bench` | `ApiTeacher` / `MultiTeacher` promoted from lib-available to example-runnable. Tests `cargo test -p train --lib teacher_infer` 5/5 green. |

### BF16 frozen-base substrate (4 commits)

| Commit | Axis |
|---|---|
| `1d51de3 feat(cuda): add autograd bf16 frozen matmul substrate` | `DeviceHandle::CudaBf16`, `import_local_bf16_as_f32`, cuBLAS BF16-RHS GEMM (`CUDA_R_16BF`). |
| `6b9e557 feat(cuda): add bf16 frozen embedding substrate` | Embedding lookup over BF16-stored vocab tables. |
| `6ab095a feat(cuda): add bf16 frozen matmul backward substrate` | Backward `matmul_bt` lhs-gradient over BF16 frozen RHS; 5/5 tests pass. |
| `6cfbfd1 feat(opd): load lora frozen base weights as cuda bf16` | Qwen3.5 LoRA-student loader stores large frozen base tensors as `CudaBf16` instead of expanding to f32. (`crates/train/src/qwen35_loader.rs +160`.) |

### License-or-kill (2 commits)

| Commit | Axis |
|---|---|
| `e59462f docs(opd): kill same-card 9b gptqmodel opd after bf16 student retry` | Smoke shape `steps=1, rollout=1, prompt_max=1, eval=0, no_cuda_graph` still OOMs at `mean_backward_device` with peak 15871/16384 MiB. Real evidence (`nvidia-smi-before/after.txt` + run logs in `bench-output/...bf16-student-smoke-noeval-memtrace/`). |
| `fc85461 docs(opd): refresh teacher api and 9b memory todo` | Plan-level reclassification: BF16 substrate landed, did not license same-card 9B path; same-GPU local API teacher does not solve co-residency. |

### Tranche C partial verify (1 commit)

| Commit | Axis |
|---|---|
| `6e11422 docs(infer): record 9b gptqmodel serve sanity recheck` | `arle serve` loads 9B GPTQModel and responds to `/v1/completions` with the loaded model id. **Output is still `!`** — model is loadable + responsive, but multi-token decode quality is a separate (low-priority) Tranche C question. |

### Hygiene (1 commit)

| Commit | Axis |
|---|---|
| `70e642b style(cuda): accept formatter churn on bf16 loader substrate` | Pure formatter follow-up on the three BF16-touched files. |

## Wins / Errors entries shipped

- `docs/experience/wins/2026-05-22-autograd-cuda-bf16-frozen-matmul-substrate.md` — BF16 frozen-base substrate (general-purpose, applicable to any LoRA-student run).
- `docs/experience/errors/2026-05-22-qwen35-9b-gptqmodel-08b-opd-memory-kill.md` — same-card 9B-teacher OPD memory KILL (covers both pre- and post-BF16-substrate retries, 8 KB).
- `docs/experience/errors/2026-05-22-qwen35-9b-gptqmodel-generation-f32load-fix-kill.md` — separate Tranche C question (multi-token `!` collapse despite finite substages and parity-passing single-token logits).

## Memory-budget arithmetic (why BF16 didn't license the headline)

- 9B INT4-GPTQ teacher forward ≈ 5–6 GB resident weights + KV/activations.
- 0.8B BF16 frozen student ≈ 1.6 GB weights + LoRA adapters + grads +
  AdamW state.
- Tape and backward activations dominate at non-trivial seq lens.
- 16 GB is oversubscribed; **no single weight-storage optimization closes
  the gap**.
- Right axes from here: separate teacher GPU/memory pool (remote
  `ApiTeacher`), or larger-memory GPU, or much smaller teacher.

## Next-axis options (codex idle awaiting user direction)

1. **Wire `ApiTeacher` into `arle train opd`** (Tranche D, low-medium
   risk): lift the example-level `--teacher-api-url` /
   `--teacher-config` flags into the canonical `arle train opd` CLI.
   Touches `crates/cli/src/train_cli.rs` which currently has parallel
   dirty paths — codex avoided that intentionally. **Needs user
   permission** before touching that file, or codex needs reassurance
   the parallel work has been reconciled.
2. **Smaller teacher** (e.g., 4B BF16) on the same card: would fit
   teacher+student+tape inside 16 GB. Tradeoff: loses the "distill from
   9B" headline. Requires no new code, just config + a bench.
3. **Tape-side memory optimization**: activation checkpointing for the
   student forward, paged grad allocator, mixed-precision LoRA optimizer
   state. Speculative; needs an audit before any commit.
4. **Tranche C deep-dive**: fix the multi-token `!` collapse in the
   serving path. Independent of OPD memory question. Lower priority per
   codex.

Codex's own closing line: *"下一条真正可行路线是远端/API teacher 或更低
内存 student/tape 路径"* — codex explicitly handed the next-axis choice
back to the user.

## Cross-links

- Prior cycle: [`2026-05-21-opd-cuda-cycle-wrap.md`](2026-05-21-opd-cuda-cycle-wrap.md)
- Earlier in the same week: [`2026-05-20-opd-cpu-perf-cycle-wrap.md`](2026-05-20-opd-cpu-perf-cycle-wrap.md)
- Teacher API plan: [`../plans/2026-05-21-opd-teacher-api-and-multiteacher-plan.md`](../plans/2026-05-21-opd-teacher-api-and-multiteacher-plan.md)
- OPD CUDA usage manual: [`2026-05-21-arle-opd-cuda-usage-manual.md`](2026-05-21-arle-opd-cuda-usage-manual.md)

## Post-cycle weight cleanup (2026-05-22)

Local 9B model weights deleted after this cycle wrapped — they served their
purpose as evidence (loader bug surface, parity reference, OOM target) but
are no longer load-bearing on 16 GB hardware. ~28 GB reclaimed:

- `~/.cache/modelscope/hub/Qwen/Qwen3___5-9B` (19 GB, BF16 base / parity ref)
- `~/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit` (11 GB, GPTQModel 4bit)
- `~/.cache/modelscope/hub/RedHatAI/Qwen3___5-9B-FP8-dynamic` (20 MB, killed)
- `~/.cache/modelscope/hub/Qwen/Qwen3___5-9B-Instruct` (empty)
- `~/.cache/huggingface/hub/models--mssfj--Qwen3.5-9B-GPTQ-INT4` (empty shell)
- `._____temp/` 9B subdirs (empty)
- All symlinks under `Qwen/` and `DavidWen2025/`

To retry 9B work later: re-download from ModelScope. Evidence numbers in
`docs/experience/errors/` + bench artifacts in `bench-output/...` are
preserved — the weights themselves are reproducible from upstream.

## Cooperative-cycle protocol notes (validated this session)

- **Audit-before-substrate works at fine granularity too**: the
  `5875519` BF16 audit doc kept a 4-commit substrate from blowing into
  a multi-module sprint by pinning the cut-point ahead of time.
- **Instrument-before-fix is mandatory** for memory-bound failures: the
  `424a4cf` diagnostic patch is what made the 15871 MiB number
  attributable. Without it the kill verdict would have been
  speculative.
- **License-or-kill applies even when the substrate is shippable**: the
  BF16 substrate is a real win, but it did not license the headline
  use-case. Both verdicts are valid simultaneously — wins entry stays,
  errors entry kills the specific 9B-on-16-GB path.
- **Goal "paused" is not the same as cycle finished**: codex kept
  working productively through "Goal paused" by treating the user's
  `/loop` directive as the active directive surface.
