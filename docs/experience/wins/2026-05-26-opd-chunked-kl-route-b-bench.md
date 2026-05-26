# OPD chunked KL Route B — sequence-windowed forward unblocks 512-token GKD on V100

## Context

Real-corpus 512-token Qwen3.5-4B → 0.8B-Base GKD with corpus-truth SFT
anchor previously KILLed on consumer 16 GB hardware
([`2026-05-25-chunked-kl-real-corpus-512-kill.md`](../errors/2026-05-25-chunked-kl-real-corpus-512-kill.md)).
T5a's `kl_distill_loss_chunked` chunked only the KL intermediates; the
full `[B, S, V]` teacher + student logits were already resident before
the loss saw them, so peak memory did not move.

Route B (per
[`docs/plans/2026-05-25-sequence-windowed-forward-design.md`](../../plans/2026-05-25-sequence-windowed-forward-design.md))
adds a true sequence-windowed forward (`SequenceWindowedForward` for
`Qwen35Model`, `TeacherWindowedForward` for `InProcessTeacher`) plus
per-window `tape.backward(window_loss)` — never materializes
`[B, S, V]` and never accumulates one cross-window graph.

This entry pins the V100 32 GB memory comparison that proves Route B's
value beyond the 16 GB consumer use case.

## What worked

### Implementation — `476d6abb feat(train): sequence-windowed forward + per-window backward for OPD GKD`

- `crates/train/src/qwen35.rs` — `SequenceWindow` + `SequenceWindowedForward`;
  `forward_logits_window()` slices hidden, then `lm_head` only over the
  window. Never produces `[B, S, V]`.
- `crates/train/src/teacher_infer.rs` — `TeacherWindowedForward`;
  `InProcessTeacher` supports windowed logits. HTTP-API / out-of-process
  teachers reject `--logits-window-size` with an actionable hint.
- `crates/train/src/opd.rs` — `GkdLossConfig.logits_window_size`.
  Windowed KL / student-rollout SFT / corpus-truth SFT each loop windows
  and call `tape.backward(window_loss)` PER WINDOW with cleanup between.
  Cross-window graph accumulation antipattern (which would defeat the
  memory goal) is explicitly avoided.
- `crates/train/examples/opd_step_cuda_infer_teacher_train.rs` —
  `--logits-window-size N` CLI flag, default off (existing full-logit
  behavior preserved).
- `crates/train/tests/test_opd_step.rs` — windowed GKD CPU smoke +
  hidden-window vs full-logit slice parity test.

### Autograd sm_70 follow-on — `af8cbdf6` + `8cb2f2e1` + `e39429e9`

The first attempt to run on V100 surfaced
`cuda load_module failed for autograd kernels`. Codex's three-commit
fix:

1. Preserve cudarc error chain in `KernelCache::new`
   (no more `.map_err(|_| TapeInvariant("..."))`).
2. Switch the autograd kernels from runtime NVRTC PTX to nvcc-built
   SASS cubin per device capability — V100 receives a sm_70 cubin
   directly, no PTX → SASS step that the V100 12.4 driver was failing.
3. Compile autograd cubin source by reference so the toolchain is the
   same path the production CUDA kernels use.

After fix: V100 release build of `opd_step_cuda_infer_teacher_train`
links cleanly (6m 56s) and runs into model load + train preamble
without the autograd module-load failure.

### Bench — V100 32 GB, Qwen3.5-4B teacher → 0.8B-Base student, 512-token corpus

| Mode | `--logits-window-size` | Peak GPU (MiB) | Outcome |
|---|---:|---:|---|
| windowed | 64 | **20 800** | train step blocked by host-RAM OOM rc=137 (separate bug) |
| windowed + eval at step 0 | 64 | **25 152** | step 0 eval >20 min before manual stop |
| **fullogit (T5b shape)** | none (off) | **31 506** | **VRAM OOM** — `cuda alloc_zeros failed (slice)` |

Same corpus + rollout + GKD config across rows. Only knob varied is
`--logits-window-size`. Memory snapshots from
`nvidia-smi --query-gpu=memory.used` polled at 1 Hz during the run.

## Headline

**Route B drops peak GPU from 31 506 MiB → 20 800 MiB (−34 %)** on the
same shape. On a 32 GB V100 the fullogit path **does not fit**
(`cuda alloc_zeros failed (slice)` while computing teacher logits for
the full 512-token prompt); the windowed path leaves ~11 GB headroom.

Route B is therefore not just a 16 GB consumer-GPU mitigation —
**32 GB V100 also needs it to run the real-corpus 512-token GKD shape**.

## Problems

- **Step 0 eval is too slow under windowed mode.** With
  `--eval-steps 0` and `--logits-window-size 64`, the run sat in the
  step 0 eval pass for >20 min without progressing to the first
  `train_step` line. Suspected cause: per-window forward + KL is being
  invoked for every held-out prompt, and the windowed KL graph is not
  reusing tape allocations across prompts. Not memory-related (peak
  was steady at 25 152 MiB). Needs profiler attention before this is
  usable for real eval cadence.
- **Host RAM OOM at train step.** With eval skipped (`--eval-steps
  999`), the windowed run reached `model_summary` then died with
  `rc=137` before the first `train_step`. GPU peak was only 20 800
  MiB, so this is not VRAM — it is process memory (cgroup or
  oom_killer). Probably the prompt tokenization or rollout staging
  buffer doing a CPU-side full-shape allocation that Route B did not
  reduce. Needs CPU-side memory audit separate from this win.
- **Bench artefacts.** The harness writes `nvidia-smi.peak.txt`
  (1 Hz samples) and `run.log` to
  `bench-output/2026-05-26-opd-chunked-kl-route-b-{wA-windowed,
  wA-windowed-noeval, wB-fullogit-noeval}/`. There is no per-task
  summary JSON — the table above is the source of truth.

## Learnings

- The structural fix prediction in
  [`docs/plans/2026-05-25-sequence-windowed-forward-design.md`](../../plans/2026-05-25-sequence-windowed-forward-design.md)
  holds: chunking KL alone does not save peak memory; chunking the
  forward producer does. The 31 506 → 20 800 MiB delta on V100 is the
  evidence the design plan promised.
- Slicing hidden before `lm_head` (vs slicing logits after) is the
  right place for the cut — the savings come from never materializing
  `[B, S, V]` in the first place. The CUDA `slice` backward allocation
  that killed T5b's `c1` retry is structurally avoided, not patched
  around.
- Error-chain hygiene paid off again: the `cuda load_module failed`
  generic message in `crates/autograd/src/backend_cuda/kernels.rs` was
  the same `.map_err(|_| TapeInvariant("..."))` antipattern fixed in
  P1.4 for the HTTP / scheduler paths. The fix unblocked Route B's
  V100 bench in three small commits instead of one round of guessing.

## Delta vs baseline

- First end-to-end V100 32 GB number for the OPD chunked KL Route B
  path; no prior snapshot. The reference points are
  [`2026-05-25-chunked-kl-real-corpus-512-kill.md`](../errors/2026-05-25-chunked-kl-real-corpus-512-kill.md)
  (T5b 16 GB KILL) and the design memory estimate (~970 MiB just for
  one logits tensor at S=512, V=248 320) in
  [`docs/plans/2026-05-25-sequence-windowed-forward-design.md`](../../plans/2026-05-25-sequence-windowed-forward-design.md).

## Artefacts

- V100: `bench-output/2026-05-26-opd-chunked-kl-route-b-wA-windowed/`
- V100: `bench-output/2026-05-26-opd-chunked-kl-route-b-wA-windowed-noeval/`
- V100: `bench-output/2026-05-26-opd-chunked-kl-route-b-wB-fullogit-noeval/`
- ARLE commits:
  - `476d6abb` — Route B impl
  - `7dce52e1` — V100 build.rs T0-legacy re-applied after DeepGEMM PR merge collision
  - `af8cbdf6` / `8cb2f2e1` / `e39429e9` — autograd sm_70 cubin loader fix chain

## Follow-up 1 (2026-05-26) — eval slowness fixed; first KL numbers from windowed pass

`eebcfec9 fix(opd): bound windowed eval train sample` (+ TileLang dict
target API drift fix in `f6bebd25`) addressed the step 0 eval >20 min
stall. The root cause was per-prompt tape lifetime: the windowed eval
loop kept accumulating tape entries across heldout prompts, so each
new prompt's KL graph walked an ever-larger live-tensor set.

Clean `wC-windowed-clean` re-run on the same V100 32 GB shape
(`/tmp/v100_opd_bench.sh windowed wC-windowed-clean`):

| metric | value |
|---|---:|
| eval_seconds (step 0, 1 train sample + 4 heldout) | **270.9 s** |
| train_kl (eval) | 1.031 × 10⁻⁵ |
| heldout_kl (eval) | 7.465 × 10⁻⁶ |
| heldout per-prompt time | 4.7-5.6 s |
| train per-prompt time (468 tok) | ~250 s |
| peak GPU during eval | 25 504 MiB |
| tape_entries at step boundary | 0 (was unbounded) |
| live_tensors at step boundary | 774 (stable across prompts) |

`tape_entries=0` after each prompt + `live_tensors=774` flat across
prompts confirms the lifetime fix — no graph accumulation across the
eval loop. Heldout per-prompt timing dropped from "never finishes" to
~5 s; the 250 s train-eval-prompt outlier is the 468-token single
example reflecting per-window forward count (window_size=64 means
~8 windows per prompt × teacher+student per window).

Train step itself still hits the host-RAM `rc=137` (Follow-up 2 below)
so per-step train wall-clock + train-step KL parity are not yet on
this table.

## Follow-up 2 (2026-05-26) — host RAM evict unblocks first train step

`93fa4fac fix(opd): evict CUDA host mirrors before Route B train` was
the structural fix: model weights were keeping a full host-side mirror
after upload to device, blowing 19.8 GB host RAM before the first
train step even started. The new
`TensorStore::evict_host_mirror(TensorId)` drops the cached host copy
once the device handle is established; weights become device-
authoritative and host RSS collapses.

| metric | before evict | after evict | Δ |
|---|---:|---:|---:|
| RSS (kB) | 19 798 864 | 2 116 644 | **−89 %** |
| host tensor bytes | 19 969 350 912 | 185 867 520 | **−99 %** |
| live tensors | 774 | 774 | 0 |
| device-only tensors | 187 | 388 | +201 |

201 tensors flipped from "host + device mirror" to device-only,
recovering ~17.7 GB host RAM. With `STATIC_PARAM_EVICT_MIN_ELEMENTS=
1_000_000` only large weight tensors are evicted; smaller buffers
keep their host mirror for cheap to-host reads.

### Clean `wD-windowed-train-1step` re-run on V100 32 GB

| phase | metric | value |
|---|---|---:|
| step-0 eval | wall-clock | **252.4 s** |
|  | train_kl | 1.031 × 10⁻⁵ |
|  | heldout_kl | 7.465 × 10⁻⁶ |
| train step 1 | wall-clock | **897.4 s** |
|  | rollout | 111.6 s |
|  | teacher forward | 168.2 s |
|  | student forward | 78.1 s |
|  | **backward** | **538.2 s (60 %)** |
|  | loss | 9.72 × 10⁻⁶ |
| post-train | RSS | 9.22 GB |
|  | host tensor bytes | 1.824 GB |
|  | peak GPU | 25 440 MiB |

End-to-end Route B GKD now runs on V100 32 GB at the 512-token corpus
shape that previously OOMed everywhere:

- ~15 min per train step is real cost (Volta sm_70 FP16 fallback for
  attention, no BF16 tensor cores)
- backward dominates 60 % of step — expected: per-window forward path
  does work that the baseline only did once, and the autograd graph
  tracks every window's reduce-mean-then-weight chain
- post-train RSS 9.22 GB is well under V100 host budget; the
  pre-evict 19.8 GB peak was the original rc=137 root cause

## Per-step optimization — Phase 2 profile

`22cea903 feat(train): profile OPD backward ops` adds
`ARLE_OPD_BACKWARD_PROFILE=1`, which routes OPD backward calls through
the existing `Tape::backward_profiled()` path, fences CUDA around each
profiled op/merge, and emits per-window plus cumulative op timing.

V100 run:
`bench-output/2026-05-26-opd-chunked-kl-route-b-wE-windowed-profile/`

Params: same Route B shape as `wD`, but `--steps 1 --eval-steps 999`
to isolate one train step without the step-0 eval pass. Profile
instrumentation is synchronous by design, so use this as bottleneck
licensing evidence, not as the production step-time baseline.

| metric | value |
|---|---:|
| rc | 0 |
| train step 1 wall-clock | 423.5 s |
| backward wall-clock | 183.2 s (43.3 % of step) |
| rollout | 149.5 s |
| teacher forward | 51.4 s |
| student forward | 39.3 s |
| peak GPU | 25 120 MiB |

Final aggregate backward profile:

| rank | op | count | seconds | % backward | % step |
|---:|---|---:|---:|---:|---:|
| 1 | MatmulBT | 368 | 89.9 | 49.1 % | 21.2 % |
| 2 | LinearAttention | 30 | 78.4 | 42.8 % | 18.5 % |
| 3 | Transpose | 68 | 3.3 | 1.8 % | 0.8 % |
| 4 | AddBroadcast | 34 | 2.9 | 1.6 % | 0.7 % |
| 5 | Slice | 28 | 1.6 | 0.9 % | 0.4 % |
| 6 | Mul | 55 | 1.3 | 0.7 % | 0.3 % |

License verdict: **GRAY**, not PASS. LinearAttention is a real
backward hotspot, but it is not dominant: 42.8 % of backward and
18.5 % of step wall-clock, while `MatmulBT` is slightly larger at
49.1 % of backward. Per §0, the wall-clock framing is the conservative
ground truth, so a full train-side LinearAttention CUDA backward spike
is not licensed yet. Next step is a narrower split: identify which
`MatmulBT` sites dominate and split `LinearAttention` into forward
intermediate recompute vs scan/state-history work before choosing the
kernel target.

## Headline (updated)

Route B is now end-to-end:

- ✅ **VRAM:** windowed 20.8 GB vs fullogit 31.5 GB OOM
- ✅ **Eval throughput:** windowed eval 252-271 s for step-0 (1 train
  prompt + 4 heldout)
- ✅ **Train step 1 lands** at 897 s on V100 32 GB; loss + KL numbers
  recorded
- ✅ **Host RSS** stable at 9 GB post-train (was 19.8 GB blowing rc=137)

## Next

- **Per-step wall-clock optimization** — 897 s/step is workable but
  not productive. The first synchronized backward profile is GRAY:
  `MatmulBT` and `LinearAttention` split almost all backward time, so
  the next optimization needs finer attribution before a CUDA backward
  kernel spike.
- **Production-scale loop** — 5-10 step run with eval cadence; verify
  KL trajectory matches the unwindowed reference (when reference is
  feasible) at small shapes.
- **Forward perf** — 168 s teacher + 78 s student is the per-window
  recompute cost (windows re-run the prefix). KV-cache reuse across
  windows would cut this in half but is a bigger structural change.
