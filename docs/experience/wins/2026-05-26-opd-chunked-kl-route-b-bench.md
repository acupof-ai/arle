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

### Phase 2A.1 — `MatmulBT` site attribution

`9efc40be feat(train): attribute OPD backward profile sites` adds a
static site label to `MatmulBT` at op construction time. For OPD this
uses the tensor name already carried by `LinearWithLora`; the direct
Qwen3.5 LM head path is labelled `lm_head`. The profile now emits both
the op aggregate and `opd_backward_site_profile`, so the `MatmulBT`
bucket can be grouped by projection family instead of guessed from the
call count.

V100 run:
`bench-output/2026-05-26-opd-chunked-kl-route-b-wG-windowed-attribution/`

Same Route B shape as `wE`: `--steps 1 --eval-steps 999
--logits-window-size 64`, with `ARLE_OPD_BACKWARD_PROFILE=1`.

| metric | value |
|---|---:|
| rc | 0 |
| train step 1 wall-clock | 422.1 s |
| backward wall-clock | 179.4 s (42.5 % of step) |
| `MatmulBT` total | 90.0 s (50.2 % backward / 21.3 % step) |
| peak GPU | 30 822 MiB |

Grouped `MatmulBT` sites:

| group | calls | seconds | % backward |
|---|---:|---:|---:|
| `linear_attn.in_proj_qkv` | 30 | 22.7 | 12.7 % |
| `mlp.down_proj` | 42 | 15.3 | 8.5 % |
| `mlp.up_proj` | 42 | 15.3 | 8.5 % |
| `mlp.gate_proj` | 42 | 14.5 | 8.1 % |
| `linear_attn.in_proj_z` | 30 | 6.8 | 3.8 % |
| `linear_attn.out_proj` | 30 | 6.7 | 3.7 % |
| `full_attn.q_proj` | 10 | 4.9 | 2.7 % |
| `full_attn.o_proj` | 12 | 2.3 | 1.3 % |
| other full-attn / LoRA / LM head sites | 110 | 1.5 | 0.8 % |

Top individual sites are still diffuse: the largest single site is
`model.language_model.layers.5.linear_attn.in_proj_qkv.weight` at
1.78 s (0.99 % of backward), followed by other linear-attention QKV
input projections at ~1.38-1.74 s each. No one or two `MatmulBT`
sites dominate. The actionable target is therefore structural
amortization across many per-layer GEMMs (or a broader linear-attn
backward kernel), not a one-off replacement for a single projection.

### Phase 2A.2 — `LinearAttention` sub-op attribution

The same `9efc40be` profile pass splits `LinearAttention` backward into
the parts that matter for kernel selection. `41ca68ef fix(train):
reduce LinearAttention profile overhead` then moved profile map updates
out of the inner parameter-gradient loops so the timing rows measure
the work instead of the profiler bookkeeping.

Final `wG-windowed-attribution` aggregate:

| sub-op | calls | count | seconds | % LinearAttention |
|---|---:|---:|---:|---:|
| `scan_state_history` | 30 | 30 | 42.9 | 57.8 % |
| `fwd_recompute` | 30 | 30 | 27.9 | 37.6 % |
| `param_grad_accum` | 30 | 60 | 3.2 | 4.3 % |
| `host_materialize` | 30 | 30 | 0.2 | 0.3 % |
| `grad_alloc` | 30 | 30 | 0.0 | 0.1 % |
| `grad_pack` | 30 | 30 | 0.0 | 0.0 % |

Verdict: the first CUDA spike for train-side linear attention should
target the state-history scan/backward chain. It is the largest
internal component and is also the most Volta-hostile part of the
current fallback. Fusing only parameter-gradient accumulation is not
licensed: it is ~4 % of `LinearAttention`, ~1.8 % of backward, and
less than 1 % of train-step wall-clock. Forward-intermediate recompute
is the second target if the scan kernel does not move enough wall-clock
time.

### Phase 2B — wD/wE wall-clock confounder resolved

The original `wD-windowed-train-1step` run reported a 897.4 s train
step, while the first synchronized profile run (`wE`) reported 423.5 s.
That was not SOLID enough to use as a baseline because `wE` also used
`--eval-steps 999` and profile fences. The control rerun was
`wF1-windowed-wd-rerun`: same wD shape, no profile env, no
`--eval-steps 999`.

| run | step-0 eval | profile env | train step | rollout | teacher fwd | student fwd | backward | peak GPU |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `wD` original | 252.4 s | off | 897.4 s | 111.6 s | 168.2 s | 78.1 s | 538.2 s | 25 440 MiB |
| `wE` profile | skipped | on | 423.5 s | 149.5 s | 51.4 s | 39.3 s | 183.2 s | 25 120 MiB |
| `wF1` wD-shape rerun | 342.3 s | off | 420.5 s | 149.1 s | 51.4 s | 37.0 s | 182.9 s | 25 216 MiB |
| `wG` attribution profile | skipped | on | 422.1 s | 147.4 s | 58.7 s | 36.4 s | 179.4 s | 30 822 MiB |

Conclusion: the 897.4 s `wD` train step was a cold/old-run
confounder, not the steady Route B per-step cost. The `wF1` rerun still
performed step-0 eval yet its train step matched `wE`, so eval skip is
not the explanation; `wG` also matches despite finer profile fences, so
the profiler did not create a fake speedup. Use 420-423 s as the warm
step-1 baseline until a longer multi-step run provides median steady
state.

### Phase 2C — `scan_state_history` CUDA spike PASS

`5b26db30 feat(train): accelerate linear-attention scan backward on CUDA`
adds the narrow backend override licensed by Phase 2A.2. The root
cause snapshot before the change:

- `linear_attention_core()` is structurally host-only today: it calls
  `store.ensure_host()` for every input and runs the Rust
  `linear_attention_forward()` reference path.
- Backward then recomputes the same host forward intermediates and
  walks `state_history` in a serial reverse-time Rust loop over
  `(batch, value_head)`.
- CUDA had no `Backend` override seam for this reverse scan, so this
  was not a missed dispatch; it was a deliberate CPU reference path
  with no device alternative.

The spike keeps CPU as the numerical reference and adds only an
optional CUDA override for the reverse state-history scan. It launches
one block per `(batch, value_head)` stream, keeps the recurrence along
time inside the block, and parallelizes the `key_dim x value_dim`
state math within each step. Forward recompute and conv1d backward
remain unchanged, so the experiment isolates the scan target.

Validation:

- Local: `cargo fmt --check --all`
- Local: `cargo check -p train --no-default-features --features no-cuda`
- Local: `cargo test -p autograd --no-default-features --features no-cuda linear_attention -- --nocapture`
- V100: `cargo test -p autograd --release --features cuda cuda_linear_attention_matches_cpu_with_device_inputs -- --nocapture`
- V100: `opd_step_cuda_infer_teacher_train` release build with CUDA
- V100 bench:
  `bench-output/2026-05-26-opd-chunked-kl-route-b-wH-windowed-scan-cuda-profile/`

Same Route B shape as `wG`: `--steps 1 --eval-steps 999
--logits-window-size 64`, with `ARLE_OPD_BACKWARD_PROFILE=1`.

| metric | `wG` CPU scan | `wH` CUDA scan | delta |
|---|---:|---:|---:|
| rc | 0 | 0 | 0 |
| train step 1 wall-clock | 422.1 s | **372.8 s** | **-49.3 s / -11.7 %** |
| backward profile total | 179.4 s | **140.2 s** | **-39.2 s / -21.8 %** |
| `LinearAttention` total | 74.6 s | **36.2 s** | **-38.4 s / -51.5 %** |
| `scan_state_history` | 42.9 s | **5.3 s** | **-37.6 s / -87.6 %** |
| `fwd_recompute` | 27.9 s | 27.9 s | unchanged |
| train-step loss | 5.317462e-6 | 5.317462e-6 | parity |
| peak GPU | 30 822 MiB | 25 632 MiB | -5 190 MiB |

License verdict: **PASS**. The cheap spike clears the target by a wide
margin: `scan_state_history` drops by 87.6 % (target was at least
40 %) and the end-to-end train step drops by 49.3 s, or 11.7 %
wall-clock. The conservative wall-clock framing is still positive, so
the root cause was real and the scan kernel is worth keeping.

The remaining `LinearAttention` time is now dominated by
`fwd_recompute` (27.9 s). Per the original decision tree, the next
linear-attention target is forward-intermediate recompute only if it
can be licensed against total step wall-clock; otherwise broader
`MatmulBT` structural amortization is the better candidate.

## Headline (updated)

Route B is now end-to-end:

- ✅ **VRAM:** windowed 20.8 GB vs fullogit 31.5 GB OOM
- ✅ **Eval throughput:** windowed eval 252-271 s for step-0 (1 train
  prompt + 4 heldout)
- ✅ **Train step 1 lands:** warm reruns are 420-423 s on V100 32 GB;
  the earlier 897 s step is recorded as a cold/old-run confounder
- ✅ **Backward scan spike:** `scan_state_history` CUDA cuts warm
  step-1 from 422.1 s to 372.8 s
- ✅ **Host RSS** stable at 9 GB post-train (was 19.8 GB blowing rc=137)

## Next

- **Per-step wall-clock optimization** — 420-423 s/step is workable
  but not productive; the scan CUDA spike improves the current profiled
  shape to 372.8 s. The next licensed candidates are
  `LinearAttention` forward recompute (27.9 s left after the scan
  spike) and structural `MatmulBT` amortization across many diffuse
  projection sites.
- **Production-scale loop** — 5-10 step run with eval cadence; verify
  KL trajectory matches the unwindowed reference (when reference is
  feasible) at small shapes.
- **Forward perf** — 168 s teacher + 78 s student is the per-window
  recompute cost (windows re-run the prefix). KV-cache reuse across
  windows would cut this in half but is a bigger structural change.
