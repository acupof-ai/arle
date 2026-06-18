# OPD memory/throughput infra gap

Date: 2026-06-18

Scope: Track 1 only, research-first. This pass did not change code, did not run
H20/pod jobs, and did not run training or benchmarks. Evidence below combines
existing ARLE research/wins entries, local source reads, and upstream public
docs. Source-survey conclusions are labeled as proposal, not shipped fact.

Existing ARLE research read first:

- `docs/research/2026-05-29-opd-memory-best-practice.md:11-31`,
  `:44-70`
- `docs/research/2026-05-28-opd-rollout-perf-208s-bottleneck.md:15-29`,
  `:37-84`
- `docs/research/2026-05-26-opd-route-b-perstep-perf-audit.md:8-28`,
  `:48-52`, `:104-109`

## Verdict

ARLE has already moved from the old resident-teacher Route B direction toward
the industry time-share pattern: infer rollout, teacher sleep/offload, and
student gradient checkpointing. For the current R1 35B-A3B-FP8 teacher to 4B
student shape, that is the right class of solution, not obvious reinvention.

The largest remaining infra gap is lower in the stack: OPD still materializes
dense teacher/student vocabulary logits for KL/JSD windows. The industry pattern
to copy is sparse teacher targets plus fused final-linear loss, not "more
offload" as the first move.

## Gap table

| 世界最佳做法 (引用来源) | 我们当前做法 (file:line) | gap | adopt 清单 (删什么/换什么/留什么) |
|---|---|---|---|
| vLLM sleep mode has level 1 weight offload plus KV discard, level 2 full weight+KV discard, and tagged wakeup for RLHF weight updates: <https://docs.vllm.ai/en/latest/features/sleep_mode/>. OpenRLHF Hybrid Engine sleeps vLLM during train and sleeps DeepSpeed during generation, with `--vllm.enable_sleep` / `--ds.enable_sleep`: <https://openrlhf.readthedocs.io/en/latest/hybrid_engine.html>. TRL documents server vs colocated vLLM and notes sleep mode for memory reduction: <https://docs.vllm.ai/en/latest/training/trl/>. | `crates/train/src/opd.rs:86-126` defines `EngineOffloadMode::{Off,All,Student,Teacher}` and `ARLE_OPD_ENGINE_OFFLOAD`; `crates/train/src/opd.rs:1360-1442` reloads/offloads teacher around KL target creation; `crates/train/src/opd.rs:2910-2966` handles student offload in the windowed path; `crates/infer-cuda/src/qwen35.rs:1798-1965` offloads CUDA weights to host and reloads them; `crates/infer-api/src/loaded.rs:440-500` exposes CUDA-only offload/reload. | Mechanism is the right family, but it is hidden behind env, Qwen35/CUDA-specific, and lacks explicit sleep levels/tags. `All`/`Student` are still exposed even though the current R1 fit evidence favors teacher-only. | Keep: teacher weight offload as the R1 memory profile. Replace: env-only control with an explicit train arg/profile named like sleep/time-share. Delete or demote: `All`/`Student` from canonical docs until they have measured value. |
| HybridFlow/verl treats rollout and training as different phases and can colocate actors/rollout engines with explicit dataflow: <https://verl.readthedocs.io/en/latest/hybrid_flow.html>. Verl async OPD passes teacher top-k logprobs/indices and computes KL in the rollout/logits processor path: <https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html>. | `examples/opd/run-math-r1-35b-to-4b.sh:49-75` sets the canonical R1 long-rollout shape with `LOGITS_WINDOW_SIZE=32`, `LAMBDA=0`, `ENGINE_OFFLOAD=teacher`, and `GRADIENT_CHECKPOINTING=1`; `examples/opd/run-math-r1-35b-to-4b.sh:157-210` wires those flags/envs into `arle train opd`. Existing fit evidence: `docs/experience/wins/2026-06-18-opd-r1-rollout1536-fit-gate.md:16-31`, `:37-59`. | The practical profile matches the time-share direction, but the older resident-teacher conclusion from the 4B memory research is stale for current 35B R1. The source default still parses unset offload as `Off` in `crates/train/src/opd.rs:108-124`, so the safe profile is not first-class. | Keep: 35B R1 as teacher-offload + grad-checkpointing. Replace: stale "resident if possible" guidance for this shape with "resident only for short fit A/B". Leave: 4B resident path as a small-model control, not the production default. |
| Best OPD/RLHF systems avoid dense teacher logits when only a distillation target is needed. Verl async OPD uses `teacher_topk_logps`, `teacher_topk_indices`, and attention masks, with actor-side KL from sparse teacher targets: <https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html>. | Pure-KL cached path runs full teacher forward over the rollout in `crates/train/src/opd.rs:2000-2018`, offloads teacher only after the dense teacher logits are cached in `crates/train/src/opd.rs:2024-2035`, slices per-window dense teacher logits in `crates/train/src/opd.rs:2095-2101`, and then computes student window logits in `crates/train/src/opd.rs:2124-2155`. Non-cached chunked path also materializes full teacher logits before slicing in `crates/train/src/opd.rs:1370-1397`. | Teacher memory is still `seq * vocab` dense logits at the target boundary. That is the opposite of the sparse top-k teacher-target pattern and is the main remaining memory-shape gap. | Replace: dense teacher target cache with top-k logprob/index targets for OPD pure-KL/JSD. Keep: dense teacher logits only as debug/exactness fallback. Delete later: hot-path dependence on `[seq, vocab]` teacher target tensors after parity and quality gates. |
| Liger fused final-linear loss avoids materializing large logits and computes gradients at the final linear + loss boundary; its docs claim up to 60% memory reduction and include `FusedLinearCrossEntropy`: <https://linkedin.github.io/Liger-Kernel/>. The fused CE source states the goal is avoiding the large `BT x V` logits tensor: <https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/src/liger_kernel/ops/fused_linear_cross_entropy.py>. Liger JSD implements generalized JSD/KL variants: <https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/src/liger_kernel/ops/jsd.py>. | ARLE already windows the loss, but still produces dense student logits: `crates/train/src/opd.rs:2048-2155` gets hidden states, projects `student.logits_from_hidden_window`, then calls `kl_distill_loss_for_config`; `crates/train/src/opd.rs:1446-1476` projects a KL window then calls `kl_distill_loss_chunked`. The loss implementation still does dense softmax/log-softmax chunks in `crates/train/src/loss.rs:37-195`. | Windowing is a partial workaround, not the best-practice end state. It reduces peak sequence length but still pays dense `[window, vocab]` materialization and generic autograd loss overhead. | Replace: OPD hot loss with one fused `lm_head + KL/JSD/CE` primitive. Keep: current chunked loss for tests and fallback. Delete later: generic dense `kl_distill_loss_chunked` from the hot OPD route once fused parity and quality pass. |
| FSDP activation checkpointing frees intermediates in forward and recomputes during backward; CPU activation offload can save more memory but may idle GPUs while waiting on CPU transfer: <https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/> and <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features-v2-pytorch-activation-offloading.html>. TRL GKD defaults `gradient_checkpointing=true` and has optional `activation_offloading`: <https://huggingface.co/docs/trl/en/gkd_trainer>. | `crates/train/src/qwen35_loader.rs:650-709` enables `ARLE_OPD_GRADIENT_CHECKPOINTING`; `crates/train/src/qwen35.rs:500-568` retains checkpoint inputs only for hidden states and trainable layer params; `crates/train/src/qwen35.rs:3200-3335` wraps transformer layers with checkpointing; `crates/autograd/src/ops/checkpoint.rs:1-57` disables tape during forward and recomputes on backward. Verification exists in `docs/experience/wins/2026-06-17-a3-qwen35-gradient-checkpointing.md:12-47`, with caution in `:61-63`. | This is close to standard activation checkpointing, but it is still env-only and lacks CPU activation offload. CPU offload is not the first gap to close because existing evidence says teacher-offload + checkpointing fits rollout1536, while backward compute dominates. | Keep: current checkpointing and equivalence gates. Replace: env-only toggle with explicit OPD profile flag. Defer: CPU activation offload until sparse targets/fused loss still leave a measured memory wall. |
| Once rollout is moved to an inference engine, training throughput work should target the remaining measured backward wall, not re-open rollout first. This matches the earlier ARLE Route B research conclusion that rollout had been the 208s bottleneck before the infer path: `docs/research/2026-05-28-opd-rollout-perf-208s-bottleneck.md:15-29`, `:77-82`. | Current measured entries show the wall moved: `docs/experience/wins/2026-06-18-opd-r1-rollout1536-fit-gate.md:37-43` has rollout done around 21s and backward around 481s for the fit gate; `docs/experience/wins/2026-06-18-opd-moe-input-grad-device.md:61-90` shows only a small rollout256 step win and remaining MoE grouped-linear/input-grad cost. | The next throughput bottleneck is not infer rollout. It is dense distill loss plus MoE backward/input-gradient work. This pass cannot claim speedup because no benchmark was run by instruction. | Keep: infer rollout path. Adopt next only after approval: one measured A/B at a time for fused distill loss, then MoE grouped input-gradient. Do not combine these with offload changes in one experiment. |

## Evidence quality

Evidence:

- Existing ARLE docs already measured the old rollout bottleneck and the later
  R1 fit gate.
- Local source lines show the actual offload, checkpointing, dense-teacher, and
  dense-student loss paths.
- Upstream docs show that sleep/time-share, sparse teacher targets, activation
  checkpointing, and fused final-linear loss are real industry patterns.

Hypothesis, not yet licensed:

- Sparse teacher targets will reduce ARLE OPD memory without hurting quality.
- A fused `lm_head + KL/JSD/CE` primitive will beat current windowed dense loss
  on wall-clock.
- CPU activation offload is lower ROI than sparse/fused loss for the current R1
  shape.

These hypotheses need controlled A/B later. This document is proposal only.

## Ranked adopt list

1. Canonicalize the R1 35B-to-4B profile as teacher sleep/offload plus student
   gradient checkpointing. Expose it as an explicit train profile/arg, not hidden
   env. Keep resident mode only as a small-shape control.
2. Replace dense teacher KL targets with sparse top-k logprob/index targets for
   OPD pure-KL/JSD. Keep dense teacher logits as debug fallback.
3. Add a fused final-linear distillation loss primitive modeled on Liger's fused
   linear CE/JSD direction, then retire dense `kl_distill_loss_chunked` from the
   hot route after parity and quality gates.
4. Keep current activation checkpointing. Defer CPU activation offload until the
   sparse/fused target path still shows a measured memory wall.
5. For throughput, measure fused distill loss first, then MoE grouped
   input-gradient. Do not combine either with offload-policy changes in the same
   license-or-kill experiment.

Stop condition: await approval before any implementation.
