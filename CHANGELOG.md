# Changelog

Progress record — one line per event (phase exit · default flip · verdict),
detail in the linked wins/errors entry. Oldest sections are condensed.

- [docs/stability-policy.md](docs/stability-policy.md)
- [docs/support-matrix.md](docs/support-matrix.md)

## [Unreleased]
- **FEAT — qwen4_exp batched prefill (`forward_prompt`): 512-token prompt at 31.0 tok/s (2.6× the 11.8 tok/s decode loop), prefill-then-decode BIT-EXACT at full scale — max rel 0.000e0 on logits AND every recurrent state (GDN S, conv rings, PLE ring, KV rows), argmax equal. Chunked path: seq-mode conv/GDN/PLE against the same resident state, batched flash + causal mask, chunk-wide HF↔GGUF perm maps, one MoE ids fence per (layer, chunk). The gate killed two real bugs: an in-place `add.comp` race (overlapping-workgroup kernel, alias-safe accum is the fix), and the coopmat GEMM dense lane's staged activations — bf16 (2^-8) AND f16 (2^-11) staging BOTH saturate to an argmax flip over 48 layers (drift crosses expert-selection boundaries in the 512-expert routers; `ARLE_VK_DISABLE_COOPMAT` ablation proved the GEMV fallback bit-exact, isolating the GEMM arm in one load). The ~2× GEMM lane (60.8 tok/s, `MmCmBf16` + a two-macro `TO_FLOAT_TYPE_B` vendor seam for f16 B) is therefore opt-in (`ARLE_QWEN4_PREFILL_GEMM=1`) behind a calibrated 20-second drift-envelope test. Measured bottleneck, accepted v1: the per-layer ids-fence GPU drain = per-token NVFP4 expert streaming, 87% of wall — grouped/indirect expert dispatch (S7) is the 100 tok/s lever, with a two-GEMM f16 residual split as the exact-fast-dense candidate** (2026-08-28; [wins](docs/experience/wins/2026-08-28-qwen4exp-batched-prefill.md))
- **PERF — qwen4_exp dense tier flips to W4A16 Q4_K by default: 79.0 → 55.5 ms/token (18 tok/s), zero activation quantization (the W4A8 q8_1-activation path that briefly existed was removed on sight — this model's contract is A16). Machinery: vendored specialized float-B `mul_mat_vec_q4_k.comp` (the generic shader cannot take K-quants), ggml-exact `quantize_q4_k_from_bf16` incl. `make_qkx2_quants`, 16-way threaded load quantize, per-family policy (640-wide shared-expert down auto-rides the free W8A16 Q8_0 rung). Quality, teacher-forced at identical prefixes: q4k 84.4% step-agreement vs near-lossless q8's 90.6% — a 2/32-position gap concentrated at razor-margin tokens; top1 drift 0.39 vs ~1.4 typical margins. `ARLE_QWEN4_DENSE=bf16|q4k|q8`. Also: llama.cpp-style embedded chat UI at GET / with `--open` and a one-click launcher, redesigned low-saturation; the free-running-agreement metric's compounding flaw caught and fixed (teacher forcing) before it could mislead the flip decision** (2026-08-28)
- **PERF — qwen4_exp decode 899 → 84.9 ms/token (10.6×, 11.8 tok/s) through seven parity-gated levers: resident linear attention (GDN state device-canonical, head map applied to activations not weights), device PLE (ring-vs-shift reconciled — the harness read 1.6e4 before the rotation fix), one-submit MoE tail, device full attention (an F32 gate AND an empty KV-plane list both kept 12 layers on host), the staged token loop (fences 339 → 50: MoE ids + PLE ring + logits), and verbatim-BF16 dense + hyper-connection tiers (load −45 s, 221k weights back to true value, logit moved TOWARD the checkpoint). Residual 0.5%; GPU busy ~67 ms of the 85 — the remaining wall is the bytes. The 1 GiB reserve guard fired on its own arc's additions and is 1.5 GiB with the arithmetic documented** (2026-08-27; [wins](docs/experience/wins/2026-08-27-qwen4exp-decode-10x.md))
- **PERF — qwen4_exp decode 899 → 266 ms/token (3.38×, same sitting): the per-stage profile (now a permanent env-gated test) showed 75.8% of the token was the HOST bf16 matvec at 9.76 GB/s (~338 ms of it spawning 4816 OS threads/token); the dense tier now runs on device via the new F16/BF16 GEMV (zero new GLSL — vendored `mul_mat_vec.comp` arms; lm_head 238 GB/s in one dispatch) with residency planned against the driver's 70.71 GiB budget and 3.076 GiB of expert stacks spilled to heap 0 (~2% read cost — an earlier 0.5% figure compared heap 1 to itself, corrected). Parity error profile unchanged digit-for-digit; hybrid logit moved 8e-4 CLOSER to the host oracle. Load regressed 223→309 s (scalar bf16→f16 re-encode; a Bf16 device format binding checkpoint bytes would remove it — 221k of 686M weights change value in the re-encode, so Bf16 is also the more faithful tier). Remaining, measured: host linattn recurrence 71 ms, submit+fence 56 ms, PLE 30 ms** (2026-08-27)
- **FEAT — Qwen3.8-Flash-Next (`qwen4_exp`, ~180B/A6B, 512-expert NVFP4 MoE + hyper-connections + 47.68 GiB n-gram PLE + hybrid linear/full attention) produces its first token on Vulkan / Strix Halo: ` Paris` top-1, full 48-layer forward, 21-stage device-vs-host parity ≤ 8.7e-5 (bf16-conv chain ≤ 1 ulp), mutations caught at rel 280 / 3.5e13. Five oracle-gated stages + a concurrent five-auditor adversarial audit; its three landmines (72 GiB slab plan vs the 70.71 GiB driver BUDGET — `VK_EXT_memory_budget`, not heap size; the per-family `1+w` norm fold; `norm_topk_prob` absent-defaults-true) all died before the forward ran. One wrong premise found: GGUF tiles K heads over V (`k = v % nk`) where HF repeat_interleaves (`k = v / 3`) — fixed by a load-time value-head permutation, kernel reused verbatim. Decode 0.68 s/token on the deliberately synchronous bring-up path; known levers: whole-token recording, device dense tier, residency replan (host-heap spill measured at 0.5% GPU-read cost)** (2026-08-27; [wins](docs/experience/wins/2026-08-27-qwen4exp-first-token.md))
- **FEAT — Qwen3.5-122B-A10B (`qwen35moe`, 63.65 GiB, 256 experts / 8 active) loads and generates on Vulkan / Strix Halo at ~12.5 tok/s decode (llama.cpp 23.41 on the same file). Five blockers, each found by loading and reading the error: split-GGUF (part 1 carries all KVs and ZERO tensors), MXFP4 block size (17 B/32), MXFP4 residency (the DequantF16 default planned 213 GiB against a 74.43 GiB heap — 90% of this checkpoint's elements are MXFP4), MXFP4 GEMV (the vendored shader already had the DATA_A_MXFP4 arms), and the MoE lane never resetting carried state at start_pos 0 — which killed the server on the SECOND request. MoE prefill is still the per-token loop; batched prefill and prefix reuse remain Qwen35-only** (2026-08-26; [wins](docs/experience/wins/2026-08-26-qwen35moe-122b-vulkan-bringup.md))
- **PERF — Vulkan prefill coopmat GEMM, 1.60x over `mul_mmq` (81.2 -> 129.9 tok/s, Qwen3.8-27B-Q4_K_M / Strix Halo; batched chunks had already replaced the per-token GEMV loop, TTFT 10.5-11.6x). llama.cpp's `l_warptile_mmq` is the *worst* of eleven tiles here (0.57x geomean, 0.20x at n=32) — the limit is per-subgroup accumulator VGPRs, not warp width, so cap `WM=WN=32` and buy tile area with more subgroups instead. Caveat recorded in the entry: the first pass measured this box in Armoury Crate Silent, where identical code reads 37.0 tok/s and the chunk-width curve goes FLAT — the ratio survived the re-measure, the structural conclusion drawn from the flatness did not** (2026-08-20; [wins](docs/experience/wins/2026-08-20-vulkan-coopmat-prefill-warptile.md))
- **PERF — Vulkan multi-turn turn-2 prefill 156.5 s → 3.2 s (49.7×): the lane's flat absolute-position KV can't be page-attached, so reuse keys on the tokens materialized into the device state (not `(slot, epoch)`, which turns over on the very requests reuse serves) and writes through the final sampled token** (2026-08-20; [wins](docs/experience/wins/2026-08-20-vulkan-resident-sequence-prefix-reuse.md))
- **FEAT — Qwen3.8-27B-Q4_K_M serves on Vulkan / Strix Halo at 9.4 tok/s, which is 59% of a hard 15.9 tok/s roofline; gemv is 92% of GPU time at 74% of DRAM peak, so weight prefetch is dead (3 layers = 774 MB vs ~32 MB on-chip) and only batching/MTP can move it** (2026-08-20; [wins](docs/experience/wins/2026-08-20-qwen38-27b-vulkan-strix-halo-roofline.md))

- **DSv4 slot budget — verdict: accepted. Prefill-transient reserve (`94a15d415`): the solve subtracts one prefill chunk's itemized working set (1352MB at TP=4) before planning slots.** Exact previously-OOMing config (FP8 TP=4, 128K total tokens, no request cap): pre-fix 18 slots + all-rank OOM at tick 4432; post-fix 17 slots, c=8 16/16 complete, 0 OOM, 0 preempts; a 3.3GB-free shared-GPU rank now rejects at boot instead of OOMing mid-serve. Same day: the FP8 checkpoint's 86 capture alloc nodes rooted to the dense-BF16 `wo_a` cuBLAS lane and fixed (`7fa06218a`, 0 alloc nodes both checkpoints), and the precision-matrix entry corrected — FP8 experts win c=1 (59.5 vs 44.4 tok/s; the 26.5 was census contamination). See `docs/experience/wins/2026-08-24-dsv4-budget-prefill-reserve.md`, `2026-08-24-dsv4-precision-matrix.md`.

- **GSPO `math-opd` lane — end-to-end on `Qwen3.8-27B-NVFP4` (single H20).** Boxed-answer grader + length-shaped reward (`1 - α·(len-len_min)/(len_max-len_min+ε)`, α=0.3) on the agent-opd GSPO loop. Three blockers fixed at root: RoPE clamp for the CC KV pool, an FP8-Marlin LoRA promote arm composing `marlin_fp8_to_e4m3` + `dequantize_fp8_block_scaled_to_bf16` (no new CUDA), and the `--lora-target-set` default flipped `all-linear` → `attention-qv` (all-linear's permanent BF16 promotion does not fit a 27B quantized student on one GPU). Smoke: 2 rounds, 8/8 groups `capped==0`, `is_ratio` ≈1.001, `clip_frac` <1, eval 0.875 → 0.875, both syncs clean. See `docs/experience/wins/2026-08-24-gspo-math-opd-smoke.md`.

- **GSPO `math-opd` run 1 — verdict: rejected (zero length compression).** 12 rounds, α=0.3 relative-within-group penalty. Eval accuracy 0.48→0.48 (noise), eval median pinned at 8192=max_tokens. Root cause: wrong samples got reward=0 regardless of length → no gradient on unsolved tasks (where the model rambles). Fix (`39c43d5d2`): absolute penalty on every sample — correct → `max(0, 1-α·len/L0)`, wrong → `-β·len/L0` (defaults α=0.5, L0=4096, β=0.05). See `docs/experience/errors/2026-08-24-gspo-math-opd-zero-length-gradient.md`.

- **Repo cleaning machinery adopted from deepseek-harness** — frozen wins/errors archive with sha256 manifest gate (`scripts/archive_experience.py`), residue sweeper (`scripts/clean_repo.py`), archive skill, all 22 CI actions SHA-pinned, cargo-machete nightly (first sweep removed 4 dead deps, incl. `infer-core` from both backends), loud lever-gate skips. See `docs/experience/wins/2026-08-24-repo-cleaning-machinery.md`.

- `Qwen3.8-27B-NVFP4` passes every probe that `ThinkingCap-Qwen3.6-27B-NVFP4` fails (tool calls included), at 52.3 tok/s with a 33%-acceptance MTP head — so NVFP4 training is viable and the corruption is a property of that one checkpoint, not the FP4 path.
- Note that the DSpark drafter RoPE fix (`7130f0b8b`) corrected a real dropped-config defect but did not move draft acceptance (13% before and after); the Qwen3.8 acceptance collapse is unexplained, see `docs/experience/errors/2026-08-23-dspark-rope-fix-was-not-the-cause.md`.
- Record the agent-OPD baseline for `Qwen3.8-27B-FP8` on the localized swe-smith corpus — 4/10 solved, 0.617 mean partial credit, every task control-validated; see `docs/experience/wins/2026-08-23-agent-opd-qwen38-baseline.md`.
- **DSv4 c=1 decode graph — default flip: armed by default (`1a48d179f`); `ARLE_DSV4_DECODE_GRAPH=0` selects the eager arm.** Matched A/B on `c1-graph-v26`, TP=4 on H20, 32k agent prompts: c=1 decode 40.8 → 44.2 tok/s (+8.3%), ITL p50 24.1 → 22.2 ms, p99 47.7 → 44.1 ms; eager arm proven at 0 captures. c=8/16 unchanged (gate is c=1-only), DSpark unchanged, MMLU 171/200 in both arms with 0 per-item diffs, needle envelope identical through 32768. Capture audit 0 alloc / 0 free / 0 host nodes, positive-control verified. An earlier +21.3% reading is superseded: the eager baseline itself improved 14.6% via concurrent Marlin and Markov-head work. Twelve replay defects fixed on the way. See `docs/experience/wins/2026-08-23-dsv4-c1-decode-graph.md`.
- Record that NVFP4 serving corrupts tool-call generation while FP8 on the same request is clean, which is what took every agent-OPD rollout to `edited=false`; cause unknown, see `docs/experience/errors/2026-08-23-nvfp4-tool-calls-corrupt.md`.
- **Verification harness — Metal gate, CI needle gate, concurrent arm, bench_compare fix.** `needle_gate.py --check` standalone exit-0/1 gate; `lever_gate.sh GATE_PROFILE=metal`; temp + concurrent arms integrated; `bench_compare.py` rewritten for v1 snapshots with COLLAPSE detection; Metal CI runs the needle ladder on the 0.8B model with `ARLE_METAL_AVAILABLE_RESERVE_MB`/`ARLE_METAL_RUNTIME_HEADROOM_MB` + `--system-reserve-bytes`/`--memory-budget-bytes` overrides (7 GiB GitHub runners cannot fit the 6 GiB reserve + 4 GiB headroom defaults). See `docs/experience/wins/2026-08-23-verification-harness-metal-gate.md`.
- **Rust toolchain bumped 1.95.0 → 1.98.0** (next trait solver on nightly, blog 2026-08-21; stable gets the `chunks_exact_to_as_chunks` clippy lint). 20+ byte-conversion sites converted to `as_chunks`; pod/Dockerfile pins bumped in lockstep. Three lint lanes exit 0.
- First agent-OPD rollouts on an NVFP4 27B reach the model: 16/16 `edited=false` with the tool path proven working, so the gap is sustaining tool use under a long agent context rather than emitting it; see `docs/experience/wins/2026-08-23-agent-opd-nvfp4-baseline.md`.
- **TP batched decode — default flip: the rows>1 `is_single` gate is gone (`ce190fc20`); TP decode batches go through the batched paged forward instead of B per-row forwards.** TP2 NVFP4-27B on H20: c≥2 ITL 1.55×–17.89×, aggregate 87 → 1,300 tok/s at c=32; c=1 wash; concurrent needle 54/54 exact. See `docs/experience/wins/2026-08-23-tp-batched-decode.md`.
- Emit Anthropic `thinking` content blocks only when the client enabled extended thinking. A chat template that reasons by default was sending them to every client, which aborts Claude Code's stream and took cc-harness agent rollouts to zero reward; see `docs/experience/errors/2026-08-23-anthropic-thinking-blocks-abort-claude-code.md`.
- **`--numa-pin` — characterized as single-rank-inert: the pin call is gated on `!cfg.is_single()` (`loader.rs:60`), so single-rank serves never pin; a same-binary A/B on Qwen3.6-35B-A3B-FP8 washed at c=1/16/32 (both arms unpinned).** Multi-rank evidence (2026-06-12, TP=8) already supports default ON; flag stays as the multi-rank opt-out. See `docs/experience/wins/2026-08-23-numa-pin-single-rank-inert.md`.
- **DeepGEMM min-routes mid-band — characterized, no flip: `--qwen35-deepgemm-min-routes 129` vs 1024 on Qwen3.6-35B-A3B-FP8, 1×H20. c=16 control identical (gate works); c=32 (R=256) wash (itl_p50 64.8 vs 66.6 ms); c=64 over KV capacity. The JIT/TMA small-band overhead persists into the FP8 contiguous mid-band; 1024 stays.** See `docs/experience/wins/2026-08-23-deepgemm-min-routes-mid-band.md`.
- **TP decode graph — default flip: the `model.tp.is_single()` gate is gone (`9b346ea1a`); whole-step decode graph now arms and replays under tensor parallelism.** NCCL/one-shot IPC all-reduces are stream-ordered and graph-capturable; eager fallback disarms on capture failure. TP2 NVFP4-27B on H20: c=1 ITL +9.9 %, c≥2 wash (GPU-bound ceiling), needle envelope identical to eager. See `docs/experience/wins/2026-08-22-tp-decode-graph-industry-path.md`.
- Close the LoRA merge loop for an NVFP4 base: `dequantize_fp4_marlin_to_bf16` plus FP8 slot setup and a `retired_marlin` keepalive, so `rubric-opd`/`agent-opd` can sync a trained LoRA back into an NVFP4 rollout engine. Verified on Qwen3.6-27B-NVFP4; see `docs/experience/wins/2026-08-22-nvfp4-lora-merge-loop.md`.
- **Flag deletion wave 3 follow-up — accepted: `examples/dsv4_resident_ab.rs` (601 lines) and the fused-WQKV override chain deleted (the axis was a proven +18.4 % winner with a preflight fallback); a dangling `set_dsv4_moe_contig_decode` doc comment removed.** See `docs/experience/wins/2026-08-22-flag-deletion-wave3.md`.
- **Flag deletion wave 3 — accepted: `--dspark-confidence-threshold`/`--mtp-adaptive`/`--mtp-min-accept`/`--dsv4-flashmla-decode` deleted with full chains (incl. the FlashMLA override API and the resident A/B example's scalar arm), `dsv4_decode_reuse_enabled()` shim removed; serve flags 53 → 49.** See `docs/experience/wins/2026-08-22-flag-deletion-wave3.md`.
- **Flag deletion wave 2 — accepted: `--qwen35-fa3`/`--qwen35-deepgemm`/`--qwen35-moe-decode-kernel` deleted (off-arms 2.76×/15.4×/10.8× worse), Metal warmup seam default fixed to match shipped false, 12 stale doc refs cleaned** (`5b880f2f8`). See `docs/experience/wins/2026-08-22-flag-deletion-wave2.md`.
- **FIX: decode graph was hardcoded OFF by the flag-deletion wave (`1864ddac5`) — serve mapping now hardcodes on (the off-arm costs −58.7 % TPOT); caught by the SOTA-defaults audit.** See `docs/experience/errors/2026-08-22-decode-graph-hardcoded-off-in-flag-deletion.md`.
- Gate the shared NVFP4 base against `marlin_fp4_gemm` rather than the group layout: the repack flushes lifted values under 2.0 to zero, so the two layouts only agree on inputs that avoid that step. Kernel verified correct under the new oracle; see `docs/experience/errors/2026-08-22-marlin-fp4-parity-wrong-oracle.md`.
- **`--dsv4-moe-transport` promoted from `ARLE_DSV4_MOE_TRANSPORT` env** (`cfcc5d4d9`) — the `--deepep-*` flags were inert without it; flag wins, env stays as fallback. Closes the codex P2.
- **Flag deletion wave — accepted: 10 proven A/B flags deleted (qwen35-decode-graph/-batched-decode/-gpu-router/-fa3-decode-splits, max-num-batched-tokens, dsv4-decode-reuse, metal-pipeline/-paged-kv-read, pool-model, extra_args), `--comm-backend` default reverted Auto→Nccl (unlicensed flip in `84c60dee5`; one-shot 51–53 vs NCCL 70–80), 5 env aliases merged, runtime-flags statics fixed** (`1864ddac5`). See `docs/experience/wins/2026-08-22-flag-deletion-wave.md`.
- **Batched DSpark on quantized KV — closed as rejected: op_timing pins the c≥8 loss on mixed-step scheduler stalls (57× 1.7 s vs 19× 1.97 s), and chunked-prefill 512/1024 does not fix it (c=4 −29 %, c=8 +5 % noise, c=16 wash).** Gate stays off; DSpark ships per-row at c=1. See `docs/experience/errors/2026-08-22-batched-dspark-quant-kv-verify-loses.md`.
- **Batched DSpark on quantized KV — rejected again with the verify-shape MMA kernel (`c177cda5b`): c=8 19.2 vs 21.8, c=16 11.2 vs 13.9 tok/s per-row; verify attention ruled out as the cause, gate restored (`1f36fea61`).** See `docs/experience/errors/2026-08-22-batched-dspark-quant-kv-verify-loses.md`.
- Read a shared NVFP4 base in its Marlin layout (`marlin_fp4_to_bf16`), so an OPD student can borrow the serving engine's frozen NVFP4 weights instead of holding a second copy (9 GB/rank vs FP8). Bit-exact against the group layout on sm_90; see `docs/experience/wins/2026-08-22-marlin-nvfp4-base-share-parity.md`.
- **OPD recompute chunk derives from the rank sequence — accepted.** Was a 4096
  constant. `chunk * rank_seq <= 2^30`, capped 16,384; `--opd-seq-chunk`
  overrides. Global 262,144 step: cp=4 692.9 → 618.4 s, cp=2 1,362.8 → 1,163.4 s,
  peak flat. `--fp8-native-gemm` stays opt-in (loss moves 0.23 %, six to ten
  times the envelope); the frozen-weight dequant cache measured as no effect.
  ([entry](docs/experience/wins/2026-08-22-opd-recompute-chunk-from-rank-sequence.md))

- **FA3 fp8 prefill attention from 64K tokens — accepted.** The quantized-pool
  prefill shim runs FA3's e4m3 kernels for prefill chunks over ≥64K KV (one
  descale per row/kv-head); 220K TTFT 129.7 → 108.2 s (−17 %), 32K and decode
  unchanged, needle 13/13 DET.
  ([entry](docs/experience/wins/2026-08-22-fa3-fp8-prefill-attention-long-context.md))

- **Global sequence 262,144 trains on 2 GPUs — accepted.** Sequence-parallel
  linear-attention core with cross-rank state carry (the all-to-all form's
  transient was O(global seq)), layer param grads parked on host, and the CP
  core collapsed into one tape entry. cp=2: 131,072 loss 3.034898 peak 64.6 GB,
  262,144 loss 1.561557 peak 85.7 GB; cp=4 262,144 loss 1.560897 peak 65.1 GB.
  Parity rungs bit-identical (4,096 9.857565 / 16,384 11.229959).
  ([entry](docs/experience/wins/2026-08-22-sp-linear-attention-core-262144-cp4.md))

- **MTP on the 32 K chain — measured, no flip.** Same binary, c=1/4/8/16 decode
  tok/s: no-spec 73.0/42.1/25.9/14.8, MTP d=2 84.3/42.0/24.8/14.0, d=4
  83.0/42.1/26.3/15.1; TTFT identical. +15 % at c=1 only (already the `auto`
  default); inert from c=4 until the verify step is batched. Closes the
  quantized-KV unification plan.
  ([plan](docs/plans/2026-08-22-quantized-kv-attention-unification.md))

- **CUDA drops Qwen3 dense and KIVI per-channel K.** `model_type=qwen3` fails at
  load; every quantized KV pool is per-(token, head) K+V on the tensor-core
  decode kernel. −7,340 lines incl. `decode_attention_quantized.cu`, the dense
  executor, the HD128 TileLang rows and the `--no-cuda-graph` flag.
  ([entry](docs/experience/wins/2026-08-22-delete-qwen3-dense-cuda-and-kivi.md))

- **Quantized paged attention on tensor cores — accepted; the only quantized
  decode path.** Phase 1 of the quantized-KV unification plan. Qwen3.8-27B-
  NVFP4, fp8 KV, 32 K prompts: per-request decode tok/s c=16 9.9–10.2 → 14.3
  (+40 %), c=32 5.9–6.0 → 8.9–9.0 (+50 %), c=1 wash; kernel 2.7–3.9×; needle 12/12 DET, eval
  177/200. Deletes the scalar kernel and the varlen fallback (−510 lines).
  ([entry](docs/experience/wins/2026-08-22-paged-attention-quantized-tensor-core.md))

- **Quantized paged attention dequantizes K/V once per GQA group — accepted.**
  One CTA serves the q-heads sharing a kv-head (group size scales with batch).
  Qwen3.8-27B-NVFP4, fp8 KV, 32 K prompts: per-request decode tok/s c=16 7.9 →
  10.0–10.2 (+27–29 %), c=32 4.5 → 6.0–6.1 (+33–36 %), c=1 wash; needle 12/12 DET, eval
  179/200 (unchanged).
  ([entry](docs/experience/wins/2026-08-21-paged-attention-quantized-gqa-shared-dequant.md))

- **`--spec-type auto` now resolves on the multiproc path.** The 0.5.8 default
  was lowered only in `serve_http`, which the multiproc coordinator never runs,
  so multi-GPU DSv4 workers loaded without the MTP head. Same-day cleanup also
  made `--mtp-draft-tokens` a loud error when `auto` finds no head, and deleted
  the closed-axis decode GEMM probe.
  ([entry](docs/experience/errors/2026-08-21-spec-auto-missing-on-multiproc.md))

- **W4AFP8 GEMV extended to M>1 for DSpark verify — rejected: 8.8% slower than
  CUTLASS grouped GEMM (42.4 vs 46.5 tok/s on the same long prompt), reverted in
  `9e21bcd99`; the same pass pins DSpark to c=1 (−32% at c=8, −47.7% at c=16 —
  the sequential draft tax scales with batch while verify savings do not).**
  ([entry](docs/experience/wins/2026-08-21-w4afp8-gemv-decode-lane.md))

## [0.5.8] - 2026-08-21

Two 4-bit checkpoint families now serve on CUDA, and the decode profile that
ranked their optimisation work turned out to have been taken at the wrong batch.

### NVFP4 serving — both families

- **DeepSeek-V4-Flash NVFP4 → W4AFP8, converted at load.** Routed MoE experts
  ship as E2M1 packed in I8 with F8_E8M0 per-1x32-block scales; the SGLang W4A8
  CUTLASS kernel wants signed INT4 with BF16 per-1x128-block scales. The
  conversion runs on GPU per expert, byte-neutral, and never retains the source,
  which is what keeps TP=2 inside 96 GB/GPU.
  ([TP=2](docs/experience/wins/2026-08-18-nvfp4-w4afp8-tp2-serve.md) ·
  [TP=4 decode](docs/experience/wins/2026-08-19-w4afp8-tp4-decode.md) ·
  [GEMV decode lane](docs/experience/wins/2026-08-21-w4afp8-gemv-decode-lane.md))
- **Qwen3.8-27B NVFP4, one resident layout.** Marlin `kFE2M1f` at decode; above
  the DeepGEMM prefill floor the same Marlin layout is widened to E4M3 in
  scratch for DeepGEMM's native FP8 MMA, 84 → 274 TFLOPS. Against
  Qwen3.6-27B-FP8 on one H20: 32K long-agent ITL +21.3% / +20.7% / +13.2% /
  +5.5% at c=1/4/8/16, resident 22.36 GB against 29.36, KV pool 1,779,114
  against 1,582,506.
  ([entry](docs/experience/wins/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md))

### Default flip

- **`--spec-type auto` is the default, and it is implemented.** It speculates
  whenever the checkpoint declares an MTP head (`mtp_num_hidden_layers`, which
  Qwen3.5 nests under `text_config`, or `num_nextn_predict_layers`; GLM ships 0).
  c=1 goes 20.50 → 11.94 ms per committed token at d=2; d=4 is not better. Above c=1 it is inert. Needle ladder exact x3 DET at
  four lengths, with engagement proven by a counter rather than by identical
  text — speculation is output-preserving under greedy, so the ladder alone
  cannot tell "ran and was correct" from "never ran".
  ([entry](docs/experience/wins/2026-08-21-spec-type-auto-default.md))

- **`--metal-warmup` defaults off: Metal cold start 0.62 → 0.35 s.**
  ([entry](docs/experience/wins/2026-08-20-cold-start-warmup-off-config-dedup.md))

### Decode

- **The decode profile was taken at the wrong batch, and it inverts.** Every
  lever had been ranked off Marlin 68.3% / attention 1.6%. At serving
  concurrency it is **attention 80.6% at c=16 and 82.7% at c=32**, Marlin 13.9%
  / 12.7% — attention scales with batch x context while the weight read does
  not, so their shares cross.
  ([entry](docs/experience/errors/2026-08-21-decode-profile-taken-at-the-wrong-batch.md))
- **Paged-attention KV row in one vector load.** A lane's 8 KV bytes are
  contiguous, so 48 `LDG.E.U8` become 4 `LDG.E.64`, and the KV scale leaves the
  per-element loop (`FMUL` 300 → 260). `cuobjdump` was the gate: it proved the compiler was not already
  vectorising, then caught two regressions in the fix before either reached a
  bench.
  ([entry](docs/experience/wins/2026-08-21-paged-attention-vector-load.md))

### Eval

- **NVFP4 matches same-base FP8, and the kernel work costs nothing.** 200 items,
  greedy, FP8 KV, one GPU back to back: NVFP4 with this release **179/200
  (89.5%)**, NVFP4 control 180/200 (90.0%), `Qwen/Qwen3.8-27B-FP8` **177/200
  (88.5%)**. At n=200 the binomial spread is ~4.2 items, so all three are
  indistinguishable — the paged-attention change is quality-neutral, and the
  4-bit checkpoint matches the 8-bit one on the same base. This closes the
  measurement debt of comparing Qwen3.8-NVFP4 against Qwen3.6-FP8, two different
  models. **Not a GSM8K result**: the items come from the repo's own
  `examples/opd/gsm8k-train.jsonl`, the TRAIN split; only the arm-to-arm
  difference on identical inputs carries.

### Verdicts


- **Metal decode — three directions ruled out on Qwen3.8-27B-MLX-2bit (M4 Pro,
  27.1 tok/s ceiling): fused GDR postprocessing within noise (21.5 → 21.6 tok/s,
  MLX async dispatch already hides the launch), the strong norm+gate-in-scan
  variant saves 0.009% of a step by traffic math, and the 2-bit matmul is at 83%
  of bandwidth.**
  ([entry](docs/experience/errors/2026-08-21-metal-decode-three-directions-ruled-out.md))
- **Conv-pair backward recompute — rejected: 6,720 MiB of tape tensors bought
  0 MiB residency (69,665 MiB before and after at local 65,536, bit-identical).
  Tensor bytes are not residency.**
  ([entry](docs/experience/errors/2026-08-20-tensor-bytes-are-not-residency.md))
- **`--kv-recall` — rejected and deleted (`3f826c204`): shrinking the attended
  set buys nothing while HBM is not the binding constraint, and the cost is paid
  regardless. L2/L3 keep only the lossless capacity path; the five correctness
  fixes stand as tiering plumbing.**
  ([entry](docs/experience/wins/2026-08-18-kv-recall-repaired-cp-and-selector.md))
- **DSv4 32K context cap — accepted (deleted): obsolete four days after the
  demand-paged joint (num_slots, pool_tokens) budget solve landed; `serve` passes
  `max_position_embeddings` through unmodified and starts at
  `max_prompt_tokens=1048576` (budget solver plans 4 slots on 4×H20, no crash).**
  ([entry](docs/experience/wins/2026-08-19-dsv4-context-cap-deleted.md))
- **Spec decode on NVFP4 — rejected for both paths: no spec 60.2 tok/s, DSpark
  16.8 (−72%), MTP 13.9 (−77%); the 6.5× single-token decode speedup makes the
  verify forward dominate, so speculation's arithmetic is inverted.**
  ([entry](docs/experience/wins/2026-08-19-nvfp4-marlin-tensorcore.md))
- **Per-channel FP8 on Marlin — accepted: the 145 GEMMs still on a scalar
  batched GEMV route to Marlin `kFE4M3fn` (already instantiated, no nvcc cost).
  +100.2% at c=16 (235.9 → 472.4 tok/s), NVFP4 now passes same-base FP8 through
  c=8 (+33.4% c=1 … +2.5% c=8); numerics 31/31, engagement proven by counter.**
  ([entry](docs/experience/wins/2026-08-19-marlin-fp8-per-channel.md))
- **Marlin blocks-per-SM search — accepted (`MARLIN_MAX_BLOCKS_PER_SM=5`, was
  pinned to 1): +4.5% decode, needle 6/6. It exposed two latent bugs
  (shared-mem budget lowered in place across chunks; lock buffer sized for one
  block per SM — an out-of-bounds write at bps=5), both fixed. The 32K-crash
  revert experiment was retracted as an unengaged arm — the revert arm never hit
  a partial prefix restore in 207 requests; cause not established, search stays.**
  ([entry](docs/experience/errors/2026-08-19-blocks-per-sm-search-two-latent-bugs.md))
- **Wave-2 dead-code deletion W8A16 27B champion-row A/B — accepted: ITL p50
  16.72 vs 16.70 ms, p99 +0.9%, TTFT −0.3%, all inside the noise floor; perf
  license granted for the one live-path touch (the unused `max_shared_mem`
  Marlin kernel arg).**
  ([entry](docs/experience/wins/2026-08-18-dead-code-deletion-wave2.md))
- **All-GPTQ W4A16 on V100 — rejected for non-expert weights: garbage output
  ("1+1=" → "iginigin_2222222"); the GPTQ kernel is correct for experts only.
  Workaround: dequantize non-expert weights to BF16 in checkpoint prep (+2 GB
  VRAM, 30 GB loaded, fits 32 GB).**
  ([entry](docs/experience/wins/2026-08-17-autoround-w4a16-v100.md))
  above M=32, not a replacement for Marlin.** Driven with one group as a dense
  GEMM it is 0.65x Marlin at M=1 and 0.90x at M=16, then 1.08x at M=32, 1.46x at
  M=48 and 1.90x at M=64. Not tile waste: at M=1 Marlin achieves 1,594 GB/s
  against the collective's 952 GB/s while reading more bytes, so wgmma and TMA do
  not help at one row. The same sweep relocated the lever — **Marlin costs the
  same at M=8 as at M=1** (0.0629 ms for 1, 4 and 8 rows), so the attack on its
  68.3% of decode is row count, which concurrency and speculative decode
  (`M = b*(d+1)`) both supply. Two latent crashes fixed on the way: the grouped
  GEMM aborted for any odd expert count on TMA descriptor alignment, then on the
  CUTLASS workspace start; even counts are unchanged.
  [entry](docs/experience/errors/2026-08-21-sm90-collective-loses-below-m32.md)
- **`gdr_decode_batch_kernel` — measured: latency-bound, not at a ceiling.** 13.0%
  of decode GPU time at 17.5-19.4% of compute peak and 20.2-22.5% of memory, IPC
  0.66, achieved occupancy 30.9% against an uncapped theoretical 100%. 61.6% of
  the stall is Long Scoreboard.
  [entry](docs/experience/wins/2026-08-21-gdr-decode-batch-is-latency-bound.md)

- **cp=2 131,072 — verdict: below the card by ~2–5 GB; the a2a linear-attention
  core is the ceiling.** Layer replays chunked end-to-end (linear projections,
  CP full attention over q tiles with ring FA3 tiled-q support, core over head
  groups): linear-layer replay 20.3 → 6.9 GB, full-attn 25.9 → ~4 GB, live peak
  75 GB of 97.5. Remaining deficit is allocator hoard around the core's
  O(global-seq) transient. Five stacked faults fixed on the way (NCCL teardown
  hang, pre-backward hoard, per-chunk trim, ring backward k-extents, flashqla
  geometry). Next step is a sequence-parallel core with cross-rank state carry.
  NVFP4 frozen base numerically validated (Δloss 0.012%), sharing unwired.
  [entry](docs/experience/errors/2026-08-21-cp2-131072-stacked-faults-and-the-a2a-core-ceiling.md)

## [0.5.7] - 2026-08-21

Seven days, 547 commits. Three threads: NVFP4 becomes a serving format worth
using, context-parallel training gets its ceiling back, and the CUDA operator
layer gets an organization.

**NVFP4 is now the smaller model, not just the smaller file.** A 23.4 GB 4-bit
checkpoint went from 39.3 GB resident — 10 GB *more* than the FP8 model it
competes with — to 22.4 GB, 7 GB *less*, by deriving DeepGEMM's prefill operand
from Marlin's resident layout instead of keeping both. Prefill moved off Marlin's
BF16 widening onto the FP8 tensor cores (84 → 274 TFLOPS), and two kernels on
that path were 4.0x and 3.7x off. Against Qwen3.6-27B-FP8 on the 32K agent
workload it now leads on ITL at every concurrency.

**CP training** recovered its 2-GPU sequence ceiling (114,688 → 131,072) and had
a 6.4x gradient error root-caused to one dropped `* 2` in a byte offset.

**~90 typed CUDA launchers** replaced hand-rolled FFI across seven phase exits.

- **REFACTOR (phase exit T5 + T8 closure) — CUDA operator organization complete: infer-cuda moe.rs split into qwen/dsv4/dsv4_deepep policy files; registry bound for qwen35/dsv4 MoE experts and dsv4 transport (15 semantic, 45 implementations); multi-model receipt (NVFP4 + two FP8 27B) identical counters/text, needle 12/12** (2026-08-21; `a8f6d8419`, [wins](docs/experience/wins/2026-08-20-cuda-operator-organization-t-series.md))
- **REFACTOR (phase exits T0–T4, T6, T7, T8-prep) — CUDA operator organization: ~90 typed launchers, infer-cuda raw FFI 217 sites → 0 (GDR fn-pointer table and peer-held moe.rs excepted), loader split 6,722 → 2,859+3,273+2,302, load-time quant storage validation, 12-operator registry, autograd NVRTC catalog+identity; remote receipt on H20: route counters and greedy text identical, needle 12/12 both binaries. T5 deferred (peer holds moe.rs)** (2026-08-20; `268c9d2fa`..`0367544ed`, [wins](docs/experience/wins/2026-08-20-cuda-operator-organization-t-series.md))
- **VERDICT (accept) — Qwen quant-linear dispatch consolidation (T2 tranche 1): one dispatcher, one route owner per weight family; all five remote gate classes pass on H20 — numerical parity, 5/5 route counters + identical completion, decode-graph capture, needle 12/12 ×2 families + lever PASS, 32K A/B within noise at c=1/4/8/16 (first c=1 baseline OOM was a foreign 22 GB resident; clean-GPU re-run 3/3, zero OOM), eval_harness 3/3** (2026-08-20; `b7432e52a`, [errors](docs/experience/errors/2026-08-20-quant-linear-dispatch-consolidation-pending-remote.md))
- **PERF — two prefill kernels on the NVFP4 path, 4.0x and 3.7x: non-GEMM overhead 838 → 303 ms (24.5% → 10.5% of the quantised path), and the 32K chain leads on ITL at every concurrency (+21.3% / +20.7% / +13.2% / +5.5% at c=1/4/8/16)** (2026-08-20; `905fc4fc2`, `ec5edf987`, `248639843`, `0879aa55b`, [wins](docs/experience/wins/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md), [rows](docs/baselines.md))
- **PERF — NVFP4 prefill on FP8 tensor cores: resident 39.35 → 22.36 GB (FP8 29.36), KV pool 1,302,407 → 1,779,114, 32K chain ITL +21.3/+18.2/+11.7/+3.9% at c=1/4/8/16** (2026-08-20; `a5df06c7c`, `30171f8be`, [wins](docs/experience/wins/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md), [rows](docs/baselines.md))
- **PERF — NVFP4 prefill on FP8 tensor cores: resident 39.35 → 22.36 GB (FP8 29.36), KV pool 1,302,407 → 1,779,114, 32K chain ITL +21.3/+18.2/+11.7/+3.9% at c=1/4/8/16, c=1 e2e −33.9% → parity** (2026-08-20; `a5df06c7c`, `30171f8be`, [wins](docs/experience/wins/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md), [rows](docs/baselines.md))
- **REFACTOR — NVFP4 single serving path: `fp4_route` dequant/GEMV variants, five dead W4A8 sidecar fields, two unreachable guards, one duplicated DeepGEMM launch removed; −458 lines; unsupported shape/SM tier now fails at load with the reason** (2026-08-20; `9f1987f25`, `0e2923dc0`)
- **VERDICT (accept) — Qwen speculative decoding honours the thinking budget: `--max-thinking-tokens 8` holds at 8 reasoning tokens on three prompts; the unlimited control runs 439/600/600. W8A16 lm_head routing in the same commit is unvalidated — no checkpoint on the box can reach it (both tie word embeddings to a BF16 tensor)** (2026-08-20; `834a87aed`, [errors](docs/experience/errors/2026-08-20-qwen-spec-budget-and-w8-lm-head.md))
- **PERF — 2-GPU CP training ceiling 114,688 → 131,072: linear-attention core gets its own checkpoint sub-group; CP transport frees as it consumes** (2026-08-20; `28a1a79ef`, `62b4927b8`, [wins](docs/experience/wins/2026-08-20-cp2-ceiling-114688-to-131072.md), [rows](docs/baselines.md))
- **PERF — Marlin no longer stores the model twice: freeing pre-repack bytes returns 18.7 GB, KV pool 281,577 → 790,603, 32K long-agent chain full recomputes 24 → 2, ITL ahead of FP8 at every point (+21.3% / +18.4% / +11.5% at c=1/4/8)** (2026-08-20; `a90d7ec50`, [wins](docs/experience/wins/2026-08-20-marlin-source-freed-18gb.md), [rows](docs/baselines.md))
- **FIX — Marlin fp32-reduce buffer sized for one block per SM while the grid is `sms × blocks_per_sm`: CUDA_ERROR_ILLEGAL_ADDRESS past ~512 tokens; root cause of the 33K prefill crash and of the MARLIN_MAX_BLOCKS_PER_SM=1 pin working** (2026-08-20; `75efc0142`, [errors](docs/experience/errors/2026-08-20-marlin-reduce-buffer-sized-for-one-block-per-sm.md))
- **VERDICT (accept) — #228 batched FlashMLA decode corruption fixed by `b4fec44b` (2026-06-15, indices reader pitch = writer pitch), verified on pod 2026-08-19: batch=4 byte_parity=true; batch=8 5/6 pass (1 failure = #229, separate non-FlashMLA bug). Issue closed.** (2026-08-19; [wins](docs/experience/wins/2026-08-19-batched-flashmla-decode-verified.md), [#228](https://github.com/cklxx/arle/issues/228), [#229](https://github.com/cklxx/arle/issues/229))
- **VERDICT (close — not reproducible) — #229 DSv4 concurrent-decode digit corruption: 40 `dsv4_parity` batch-decode trials (20 needle + 20 repeated-pattern, batch=8, TP=4) produced 0 failures; model experts are NVFP4, and the FP8 MoE kernel suspected in the #229 doc was replaced by the W4AFP8/NVFP4 path. Issue closed.** (2026-08-20; [errors](docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md), [#229](https://github.com/cklxx/arle/issues/229))
- **PERF — NVFP4 c=16 decode +51.7% (472.4 → 716.6 tok/s): the last four load sites had no Marlin repack, so 34% of FP8 GEMM calls stayed on the scalar GEMV** (2026-08-19; `1da4e0422`, [wins](docs/experience/wins/2026-08-19-nvfp4-marlin-remaining-load-sites.md), [rows](docs/baselines.md))
- **VERDICT (reject) — Marlin decode occupancy: three tunings, all reverted; warp occupancy 20.7% → 30.7% while throughput fell, the shared-memory over-request removal is a wash, and buying occupancy with registers spills** (2026-08-19; `7fc5e00c8`, [errors](docs/experience/errors/2026-08-19-marlin-decode-is-not-occupancy-limited.md))
- **FIX — CP ring FA3 pair offsets are bytes: one dropped `* 2` gave 6.4× wrong training gradients for two days; cp=2 grad_norm 14.01 → 2.15 vs cp=1's 2.20** (2026-08-19; `ad1192864`, [wins](docs/experience/wins/2026-08-19-cp-ring-fa3-byte-offset-fix.md), [errors](docs/experience/errors/2026-08-19-cp-training-gradients-regressed-and-the-gate-is-dead.md))
- **ERROR — CP training gradients 6.4× off single-card (grad_norm 14.01 vs 2.20, 27B cp=2 seq=32768); the 0.8B CP correctness arm has been unrunnable since FlashQLA went default-on 2026-08-05** (2026-08-19; [errors](docs/experience/errors/2026-08-19-cp-training-gradients-regressed-and-the-gate-is-dead.md), [rows](docs/baselines.md)) — bisect pending
- **BASELINE (re-anchor) — cp=4 training seq ceiling 229376; seq=131072 step is 17.5× the 2026-08-03 row (3100 s → 177.5 s)** (2026-08-19; `9c2c84675`, [wins](docs/experience/wins/2026-08-19-cp4-seq-ceiling-229376-and-17x-step.md), [rows](docs/baselines.md))
- **FIX — one prefill dequant arm firing at `M >= 2` cost NVFP4 5× aggregate throughput at c≥2, inverted both spec-decode paths, and crashed the server at 34K: 11.56 G FP8 params re-materialised to BF16 every step** (2026-08-19; [errors](docs/experience/errors/2026-08-19-fp8-dequant-arm-shadows-decode.md), [rows](docs/baselines.md)) — after-arm re-measure pending on the fixed binary
- **FIX — whole-slot park works under CP: 9,970/9,970 refusals → 390/390 round-trips, promote 130 ms @ 10K tokens, needle 48/48** (2026-08-19; `b2cc9b783`..`cb9a53373`, [wins](docs/experience/wins/2026-08-19-cp-slot-park-works-l2-l3-nonzero.md), [errors](docs/experience/errors/2026-08-19-cp-park-refused-so-l2-l3-never-written.md))
- **REFACTOR — dead-code deletion wave 4: −426 lines, non-CUDA crates, zero live paths touched** (2026-08-19; `aa10fccca`, [wins](docs/experience/wins/2026-08-19-dead-code-deletion-wave4.md))
- **FEAT — Qwen3.8-27B-NVFP4 mixed-precision inference: NVFP4 MLP + FP8 per-channel attention on H20** (2026-08-18; `33f4863c7`, [wins](docs/experience/wins/2026-08-18-qwen38-27b-nvfp4-inference.md))
- **FEAT — KV-recall × CP: shard-filtered recall under 2D parallelism; needle 21/21 TP=2 CP=2** (2026-08-18; `d76ac50ff`..`a29d24f5b`, [wins](docs/experience/wins/2026-08-18-kv-recall-cp-shard-filtered.md))
- **FEAT — MXFP4 W4A16 weights on Metal (opt-in); 9B pilot: affine ladder row retained (8K recall regression reproduced in stock mlx_lm)** (2026-08-18; `359d1b492`, [wins](docs/experience/wins/2026-08-18-mxfp4-metal-qwen35-9b.md))
- **FEAT — NVFP4 checkpoint loads + FP8 inference: DSv4-Flash-0731 TP=4 on H20** (2026-08-18; [wins](docs/experience/wins/2026-08-18-nvfp4-load-fp8-infer.md))
- **REFACTOR — dead-code deletion: −7,155 lines across 65 files, zero live paths touched** (2026-08-18; `1f3b0e62e`..`cd705eff3`, [wins](docs/experience/wins/2026-08-18-dead-code-deletion.md))
- **REFACTOR — dead-code deletion wave 2: −5,819 lines, CUDA C++/FFI + cross-crate, zero live paths touched** (2026-08-18; `63e52f55d`..`7d4731176`, [wins](docs/experience/wins/2026-08-18-dead-code-deletion-wave2.md))
- **FIX — CP prefill snapshotted blind-tail: skip `prefill_row_snapshotted` under 2D; needle ladder 21/21** (2026-08-18; `e767bd5ac`, [wins](docs/experience/wins/2026-08-18-cp-prefill-snapshotted-blind-tail-fix.md), [errors](docs/experience/errors/2026-08-18-cp-state-chain-pre-advance-recv.md))
- **FEAT — DP coordinator: least-in-flight multi-group routing** (2026-08-17; `806c4268a`, [wins](docs/experience/wins/2026-08-17-dp-coordinator.md))
- **PERF — TP/CP NCCL collectives onto comm_stream** (2026-08-17; `a59c6c661`, [wins](docs/experience/wins/2026-08-17-collectives-to-comm-stream.md))
- **PERF — DeepEP host stalls → on-device event ordering** (2026-08-17; `142b959d4`, [wins](docs/experience/wins/2026-08-17-deepep-host-stalls-event-ordering.md))
- **FEAT (accept) — 35B A3B AutoRound W4A16 runs on V100 (sm_70) via ARLE** (2026-08-17; [wins](docs/experience/wins/2026-08-17-autoround-w4a16-v100.md))
- **FEAT (accept) — CP T3.1: B2 CP decode head-sharding across the cp group** (2026-08-17; `807e6c0b4`, [wins](docs/experience/wins/2026-08-17-b2-cp-decode-head-sharding.md), [plan](docs/plans/2026-08-16-cp-ideal-state.md))
- **FEAT (accept) — CP T2: engine prefill context parallelism, replicated KV** (2026-08-16; [wins](docs/experience/wins/2026-08-16-cp-t2b-replicated-kv-prefill.md), [plan](docs/plans/2026-08-16-cp-ideal-state.md))
- **FIX — GDR prefill recurrent kernel: missing `__syncthreads()` smem race** (2026-08-16; `1f7948070`, [errors](docs/experience/errors/2026-08-16-gdr-prefill-smem-race.md))
- **FIX — windowed-GKD backward: residency bounded to one window; free-after-backward UAF** (2026-08-16; `2f90f7942`)
- **REFACTOR (accept) — CP T1: tape-free ring-attention core shared via cuda-kernels** (2026-08-16; `083e2e89a`, [wins](docs/experience/wins/2026-08-16-cp-t1-ring-core-extraction.md))
- **FIX (accept) — serve lifecycle: explicit memory budget wins; an engine cannot outlive its supervisor** (2026-08-16; [wins](docs/experience/wins/2026-08-16-serve-explicit-budget-and-parent-watchdog.md))
- **FIX (accept) — share-frozen-base: alias fused QKV/gate-up slices, no duplicate FP8 base** (2026-08-16; [wins](docs/experience/wins/2026-08-16-share-frozen-base-fused-slices.md))
- **FEAT (accept) — `--lora-merge-fp8`: 27B all-linear LoRA merge fits one GPU** (2026-08-16; `cd5d9afd1`, `ed945fde1`, [wins](docs/experience/wins/2026-08-16-lora-merge-requant-fp8.md))
- **FIX (accept) — OPD long-seq OOM: cached teacher hidden + O(n) student forward** (2026-08-16; `cd9784f6c`, `e96ee6a43`, [wins](docs/experience/wins/2026-08-16-opd-65536-longseq-oom-fix.md))
- **FIX (accept) — reasoning the model produced always reaches the client** (2026-08-15; [wins](docs/experience/wins/2026-08-15-openai-reasoning-content-lane.md))
- **VERDICT — agent-OPD parameter-update path executed on real claude rollouts** (2026-08-15; [wins](docs/experience/wins/2026-08-15-agent-opd-update-path-first-execution.md))
- **VERDICT (accept) — FP8 non-zero-delta merge verified over two rounds on one GPU** (2026-08-15; `d7d2366fe`, `d872cc37c`, `e14a4caf5`, `89b891905`, [wins](docs/experience/wins/2026-08-15-rubric-single-gpu-judge-residency.md))
- **VERDICT (resolve) — DSv4 first-token flip under concurrency = near-tied logit pair; no runtime defect** (2026-08-15; [wins](docs/experience/wins/2026-08-15-dsv4-first-token-flip-near-tied-pair.md), closes #202)
- **FIX (accept) — frozen-base ownership: one invariant, no per-site frees** (2026-08-15; `8c0ac637c`, `24202f656`, [wins](docs/experience/wins/2026-08-15-frozen-base-ownership-single-invariant.md), [review](docs/plans/2026-08-14-frozen-base-sharing-correctness.md))
- **PERF (accept) — w2s gates computed on device: s/step 3.614 → 2.742 (−24.1%)** (2026-08-14; `7b9b13393`, [bench](docs/experience/wins/2026-08-14-w2s-device-gates-and-chunked-regularizers.md))
- **VERDICT — agent-OPD runs end to end on one GPU through the real claude harness** (2026-08-14; smoke at `7b9b13393`, log `/host/aopd-smoke-0814.log`)
- **VERDICT — w2s 60-step e2e on 27B-FP8: confidence threshold is a near-switch (0.99 skips nothing, 0.9 skips 80% on GSM8K)** (2026-08-13; [bench](docs/experience/wins/2026-08-13-w2s-e2e-confidence-near-switch.md))
- **FIX (accept) — LoRA-targeted projections keep the trainer-owned base under frozen-base sharing** (2026-08-14; `7c4c9082f`, [design](docs/plans/2026-08-14-frozen-base-sharing-correctness.md))
- **FIX (accept) — OPD `--engine-offload student` step-1 NaN root-caused: frozen-base alias use-after-free** (2026-08-14; `a1a3fda92`, `ef486bd86`, `4b8b02f9f`, [bench](docs/experience/wins/2026-08-14-opd-offload-student-alias-uaf.md))
- **BASELINE (re-anchor) — Qwen3.6-27B-FP8 DSpark and DSv4-Flash-FP8 8xH20 DSpark re-measured at `fad8f4d5b`** (2026-08-14; [bench](docs/experience/wins/2026-08-14-sampling-penalties-verified-on-both-runtimes.md), [errors](docs/experience/errors/2026-08-14-raw-completion-continuation-flips-with-concurrency.md))

## [0.5.6] - 2026-08-14
- **FIX — single-GPU OPD: teacher pool sizing + engine-offload starvation** (2026-08-14; `f1f568d1a`, `c7f9c68ad`, [errors](docs/experience/errors/2026-08-14-opd-engine-offload-starves-autograd-forward.md))
- **PERF (accept) — OPD bf16 bridge event-ordered: 2.24× over the legacy sync; KV pool trim moved off the host** (2026-08-14; `196eb2bb1`, `49b469456`, `7fa81cf6d`, `35a773d52`, [bench](docs/experience/wins/2026-08-14-bf16-bridge-event-ordered.md))
- **FIX — `train w2s --save-every N`; VRAM on every step line** (2026-08-13; `e9116d3db`)
- **REFACTOR (accept) — five monolithic impl blocks holding 30-84% of their file** (2026-08-13; `d3b239ab7`, `07f2d0aaf`, `7eb1984f2`, `d982b7d50`, `8111138d9`, [method](docs/experience/wins/2026-08-13-orthogonal-axes-expanded-into-method-names.md))
- **REFACTOR (accept) — backend_cuda.rs 12874 → 2680 + 21 concept-named modules** (2026-08-13; `055726e9a`, `626b49e72`)
- **MEASURE — w2s step budget: the four KL terms are 46.6%, the student forward 12.8%** (2026-08-13; `18096ec7f`, [bench](docs/experience/wins/2026-08-13-w2s-step-budget-kl-terms-dominate.md))
- **REFACTOR (accept) — crates/train deletion, config layering, opd.rs split** (2026-08-13; `18096ec7f`, `79269266a`, `d048a0bce`)
- **FIX (accept) — w2s no longer round-trips the FP8 base through host** (2026-08-13; `62017ec8a`, `bc96d29ec`, `f77ca2eb5`, [errors](docs/experience/errors/2026-08-13-w2s-fp8-base-offload-roundtrip-was-lossy.md))
- **FIX (accept) — prefix-cache metrics report actual restored work** (2026-08-13; `c112b81de`, [bench](docs/experience/wins/2026-08-13-kv-prefix-metrics-and-oversubscription-slice.md))
- **FEATURE (accept) — FA3 quantized KV paths A+B for qwen35** (2026-08-13; `a3a769db1`, [bench](docs/experience/wins/2026-08-13-fa3-quant-paths.md))
- **FEATURE (accept) — DSpark spec decode with quantized KV; L2 tier demote/promote verified** (2026-08-13; `c04c700a7`, [bench](docs/experience/wins/2026-08-13-dspark-quant-kv.md))
- **FIX — HTTP sampling penalties validated at ingress; logit_bias survives the multiproc relay and the greedy fast path** (2026-08-13; `bd5e6f00a`, `13d39ea84`, `98caaaf25`)
- **INFRA — watchdog startup grace 120s→300s; one-off scripts pruned; conversion and quantization unified** (2026-08-13; `f10c6d9f3`, `ddb0ceccc`, `cbf33b667`, `abc6d70fe`)

## [0.5.5] - 2026-08-13
- **FEATURE (accept) — batched paged decode for FP8/INT8 KV pools** (2026-08-13; `ff33bdb77`)
- **REFACTOR (accept) — unify qwen35 FP8/INT8 KV on the NHD split-KV kernel; delete the dead TileLang FP8 path** (2026-08-13; `64be73980`)
- **BENCH — KV dtype comparison on H20, ThinkingCap-Qwen3.6-27B-FP8** (2026-08-13)

## [0.5.4] - 2026-08-12
- **FEATURE (accept) — INT8 KV cache support for Qwen3.5 paged attention** (2026-08-12; `b20859520`)
- **PERF (accept) — FP8 dequant GEMV floor lowered to M>=2: WMMA GEMM replaces cuBLAS for small batches** (2026-08-12; `b20859520`)

## [0.5.3] - 2026-08-11
- **PERF (accept) — DSv4 whole-slot KV tier serialization simplified: swap_out/swap_in persist only mutable fields; FP32 carry skipped via `fp32_carry_stale`** (2026-08-11; `3d499a4fb`)
- **FIX (accept) — DeepGEMM native build fixes for CUDA 12.9** (2026-08-11; `64ffa8dcf`)
- **VERDICT (reject) — FlashQLA `block_DV=32` improves wave count but fails numerical parity** (2026-08-10; `3582c881a`, `e2a837ff6`, [error](docs/experience/errors/2026-08-10-flashqla-block-dv32-numerical-kill.md))
- **VERDICT (reject and revert) — unmeasured CUDA split, fast-math, and GEMV changes regressed correctness** (2026-08-10; `17c60435e`, `9a6ca91ac9`, [error](docs/experience/errors/2026-08-10-unmeasured-cuda-micro-optimizations-regressed-correctness.md))
- **CALIBRATION — the anchor's `nsys` window over-states prefill kernel shares by 2.02×** (2026-08-09; `nsys` at `5cfe8494f`, [bench](docs/experience/wins/2026-08-09-pack-quantize-warp-per-block.md))
- **PERF (accept) — `pack_quantize` at 16 B loads: 5.13×, still bit-identical** (2026-08-09; `5cfe8494f`, [bench](docs/experience/wins/2026-08-09-pack-quantize-warp-per-block.md))
- **PERF (accept) — `pack_quantize` was instruction-bound; one warp per block gives 3.67× and −2.98% anchor wall** (2026-08-09; `554173b36`, [bench](docs/experience/wins/2026-08-09-pack-quantize-warp-per-block.md))
- **MODEL (supersede) — the anchor window is now an exact partition; prefill arithmetic is at the hardware floor** (2026-08-09; `70760bc09`, [bench](docs/experience/wins/2026-08-09-anchor-window-partitioned-exactly-prefill-arithmetic-is-finished.md))
- **FIX (root cause confirmed) — the Qwen3.6 trunk's final RMSNorm applied `w` instead of `(1+w)`; every eval on this model before today is a floor** (2026-08-08; `694245eec`, [entry](docs/experience/errors/2026-08-08-qwen36-final-norm-missing-offset.md))
- **VERDICT (close the lever) — the anchor's FP8 GEMM is 57.7% of all kernel time and runs at ~90% of FP8 peak** (2026-08-08; `70760bc09`, [bench](docs/experience/wins/2026-08-08-anchor-fp8-gemm-is-at-90-percent-of-peak.md))
- **PERF (accept) — DSpark draft attention was launched once per slot at 192 blocks; batching the slot axis gives ITL mean −10.4%** (2026-08-08; `3a8f99b1f`, [bench](docs/experience/wins/2026-08-08-dspark-draft-attention-slot-batched.md))
- **VERDICT (accept) — agent-opd rollout concurrency; production config is cp=4 × G=2** (2026-08-08; `7aef20557`, `f996e6826`, `5b1cd473d`, [bench](docs/experience/wins/2026-08-07-agent-opd-rollout-fleet.md))
- **BASELINE — decode re-anchored on a decode-shaped workload: draft attention is 30.5% of a tick; the prior anchor priced it at 4.3%** (2026-08-08; `nsys` at `70760bc09`, [bench](docs/experience/wins/2026-08-08-decode-shaped-reanchor-draft-attention-is-30pct.md))
- **VERDICT (reject the ranking, mechanism confirmed) — FA3 decode-verify is 29.2% of roofline and 0.39% of GPU time; the anchor is a prefill benchmark** (2026-08-08; `nsys` at `70760bc09`, [entry](docs/experience/errors/2026-08-08-anchor-is-a-prefill-benchmark-decode-levers-ranked-off-it.md))

## [0.5.2] - 2026-08-21
- **BASELINE — corrected Qwen3.6-27B DSpark anchor complete** (2026-08-10; runtime `9b38ba6c0`, runner `c98c4e0b2`, [bench](docs/experience/wins/2026-08-10-qwen36-27b-corrected-baseline.md))
- **FIX — benchmark warmup no longer primes a measured prefix** (2026-08-10; [error](docs/experience/errors/2026-08-10-benchmark-warmup-contaminated-cold-session.md))
- **FIX — DFlash draft norms restore Qwen3 plain-weight semantics** (2026-08-10; `9b38ba6c0`, [error](docs/experience/errors/2026-08-10-dflash-draft-norm-offset.md))
- **FIX — the canonical fixed-output benchmark now forces `ignore_eos=true`** (2026-08-10; [error](docs/experience/errors/2026-08-10-fixed-output-benchmark-allowed-early-eos.md))
- **PERF — V100 (sm_70) prefill: W4A16 dequant→FP16 GEMM + GDR/FA2 tuning** (`df77f7668`)

## [0.5.1] - 2026-08-07
- **VERDICT (accept, end-to-end null) — the prefix sidecar serialized 146.8 MiB per element; bulk copy is −9.5% on the operation and 0.9% of wall** (2026-08-07; `d626a1b03`, [bench](docs/experience/wins/2026-08-07-prefix-sidecar-serialize-bulk-copy.md))
- **FIX — agent-opd cp>1: rank 0 owns rollout, followers mirror the update stream** (`9da8ff777`)
- **VERDICT (confirmed) — agent-opd cp=2 fix validated; the cc-rollout training loop closes end-to-end under the new defaults** (2026-08-07/08, pod GPUs 4+5, [error entry](docs/experience/errors/2026-08-07-agent-opd-cp2-rollout-divergence-deadlock.md))

## [0.5.0] - 2026-08-07
- **VERDICT (accept) — the c≥4 DSpark decode regression is CLOSED; anchor re-anchored on `70760bc09`** (2026-08-07; [bench](docs/experience/wins/2026-08-07-dspark-rollback-replay-batched.md))
- **VERDICT (accept) — the DSpark verify linear core is batched; long-agent anchor re-anchored on `4933e1bf4`** (2026-08-07; `4933e1bf4`, `9119ebcbb`, [bench](docs/experience/wins/2026-08-07-dspark-verify-linear-core-batched.md))
- **DEFAULT FLIP + VERDICT (accept B / reject C) — `--checkpoint-reload-device` on by default; pinned checkpoint pool stays off** (2026-08-06; `5cec66ea3`..`d1870526f` + this flip, [bench](docs/experience/wins/2026-08-06-checkpoint-reload-and-pinned-offload.md))
- **WASH — reshape/rmsnorm backward heal is correct but a no-op for the profiled cost; `7da312d0d` kept** (2026-08-06; [error](docs/experience/errors/2026-08-06-healed-the-wrong-reshape-backward-not-recompute-forward.md))
- **VERDICT (reject) — OPD_SEQ_CHUNK 4096→8192 is a null; backward wall scales with total work on CPU** (2026-08-06; pod-only, [error](docs/experience/errors/2026-08-06-opd-chunk-knob-null-backward-is-total-work-cpu.md))
- **VERDICT (reject) — native FP8 training forward halves the GEMM cluster but moves the step wall 1%** (2026-08-06; `cafda607c`+`3c021aead`, [error](docs/experience/errors/2026-08-06-native-fp8-forward-optimized-the-wrong-17-percent.md))
- **FIX — REPL/OCR load caps slots at 1: Qwen3.5-9B now fits in 48 GB** (2026-08-06; [win](docs/experience/wins/2026-08-06-repl-single-slot-load.md))
- **FIX — CUDA serve auto-downloads HF model ids, matching Metal** (2026-08-06; [win](docs/experience/wins/2026-08-06-cuda-serve-auto-download.md))
- **DEFAULT FLIP — FlashQLA GDN chunkwise backward default-on: 80K training step 1.99×, backward 2.14×** (2026-08-05; `bb5561649`, [win](docs/experience/wins/2026-08-05-flashqla-gdn-backward-default-on-2x.md))
- **CHARACTERIZATION — an 80K training step is one kernel: GDN chunked-scan backward is 71%; FA3 is worth 3.54× at 80K, vs 2.17×** (2026-08-05; [win](docs/experience/wins/2026-08-05-80k-training-step-is-one-kernel.md))
- **DEFAULT FLIP — FA3 is the unconditional CP ring path; `ARLE_CP_RING_FA3` deleted** (2026-08-05; `15caff0d0`, [win](docs/experience/wins/2026-08-05-80k-training-step-is-one-kernel.md))
- **VERDICT — the prefill gap was a stub build: FlashQLA was never compiled into the pod binary; TTFT 31.08 → 25.01 s** (2026-08-05; `101d68b91`, `6e3f68fac`, [win](docs/experience/wins/2026-08-05-flashqla-was-never-compiled-into-the-pod-binary.md))
- **VERDICT — the decode step reaches parity with SGLang; the gap is now entirely prefill** (2026-08-04; `17fdb6aab`+`e1017b40d`, [budget](docs/experience/wins/2026-08-04-w8a16-decode-step-kernel-budget.md))
- **DEFAULT FLIP — FA3 decode split ceiling derived from the SM count: −11.2% decode step at batch 1** (2026-08-04; `574045dc1`+`53f9c5143`+`0e750fa18`, [win](docs/experience/wins/2026-08-04-fa3-decode-splits-fill-the-sms.md))
- **ACCEPT (perf) — FA3 replaces the scalar CP ring-attention kernels: 2.17× per training step; default OFF pending grad parity** (2026-08-04; `2fe12a2fe`+`df75a1da2`+`a15d3ec75`+`d293fcc74`, [win](docs/experience/wins/2026-08-04-fa3-ring-attention-2x.md))
- **ACCEPT — GDR chunk-prepare native CUDA: 289× per launch, −10% training fwd wall, losses bit-identical** (2026-08-03; `3d80dd473`, [win](docs/experience/wins/2026-08-03-gdr-prepare-native-289x.md))
- **PHASE EXIT — CP×DP mesh verified end-to-end; 131072 cp=4 runs clean; the training step is attributed** (2026-08-03; `4aa6e5e02`+`00e482f50`+`a644adab8`+`e57c59793`+`3cae75304`, [win](docs/experience/wins/2026-08-03-cpxdp-verified-and-training-step-attributed.md))
- **ACCEPT — per-token O(cached-pages) scan → O(1) counter: −6.0% decode ITL (cumulative −29.4%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-resident-page-scan-per-token.md))
- **ACCEPT — T6 GDN decode kernel: −2.8% decode ITL (cumulative −24.9%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t6-gdn-decode-kernel.md))
- **ACCEPT — T5b lm_head GEMV → cuBLASLt: −2.8% decode ITL (cumulative −22.7%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t5b-lmhead-cublas.md))
- **DEFAULT FLIP — `--qwen35-decode-graph` ON (serve + seam default)** (2026-08-03)
- **ACCEPT — T4 whole-step decode graph under paged KV: −7.9% decode ITL (cumulative −20.5%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t4-paged-decode-graph.md))
- **ACCEPT — T2 qkv + qkvz row-fusion: −2.5% decode ITL (cumulative −13.7%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t2-qkv-row-fusion.md))
- **ACCEPT — T5 small-M bf16 GEMV → cuBLAS: −5.1% decode ITL (cumulative −11.5%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t5-small-m-gemv-to-cublas.md))
- **ACCEPT — T3 in_proj_b+a row-fusion: −4.7% decode ITL (cumulative −6.7% with T1)** (2026-08-03; `4952f0df5`, [win](docs/experience/wins/2026-08-03-t3-in-proj-ba-fusion.md))
- **ACCEPT — T1 gate+up row-fusion: −2.1% decode ITL; Marlin fixed-grid correction re-ranks #196** (2026-08-03; `3e383c082`, [win](docs/experience/wins/2026-08-03-t1-gate-up-fusion.md))
- **VERDICT — W8A16 matched A/B vs SGLang: same kernel, same weights; the GEMM matches and SGLang decodes 1.57× faster — the gap sits in our runtime** (2026-08-02, [entry](docs/experience/wins/2026-08-02-w8a16-sglang-matched-ab.md))
- **ACCEPT — device-native cat: matched A/B verdict, strict win (−10.6 GB host RSS, ~5.6× faster)** (2026-08-02; `7276fa081`, [win](docs/experience/wins/2026-08-02-device-cat-ab-strict-win.md))
- **PHASE EXIT — W8A16 Marlin tensor-core GEMM: bf16-class decode at half the weight VRAM** (2026-08-02; `3ca42b44a`, [win](docs/experience/wins/2026-08-02-w8a16-marlin-tensorcore.md))
- **PHASE EXIT — real 27B 256K CP training runs end-to-end; the VRAM wall is measured at 94.2 GB/GPU (cp=2 fits)** (2026-08-02; `fd8e38e5c`, `b41b130e5`, `1734c69cc`, [win](docs/experience/wins/2026-08-02-linear-attn-cp-a2a-reorder-256k-runs.md))
- **ACCEPT — FA3 for batch==1 prefill (−4%) and the driver-context thread-lottery fix** (2026-08-02; `b0368426a`, [win](docs/experience/wins/2026-08-02-fa3-batch1-prefill-and-ctx-bind.md))
- **DEFAULT FLIP — `--qwen35-gdr-chunked` ON, licensed by the chat-format battery** (2026-08-02; `c2eb5de9e`)
- **VERDICT — the chunked-GDR GSM collapse adjudicated: bf16 drift in a margin-sensitive harness; kernels correct; chat-format quality identical** (2026-08-02; probe `aa03e0566`, [error](docs/experience/errors/2026-08-02-gdr-chunked-gsm-collapse-was-a-knife-edge-harness.md))
- **REVERT — `--qwen35-gdr-chunked` default back to OFF: GSM8K 11/100 vs 46/100** (2026-08-02; flip `2e2ab667c`, revert `715c37a0c`)
- **ACCEPT — FlashQLA chunked GDR generalized to head geometry and made real: 33K prefill −27%** (2026-08-02; `778fef873` + `5b851d193`, [win](docs/experience/wins/2026-08-02-flashqla-chunked-gdr-h48.md), [error](docs/experience/errors/2026-08-02-pod-b64-arg-truncation-stale-binary.md))
- **PHASE EXIT — the 27B step is profiled end to end; the backlog is re-ranked off measured share** (2026-08-01; [win](docs/experience/wins/2026-08-01-prefill-and-decode-step-budget.md), row in [docs/baselines.md](docs/baselines.md))
- **REJECT — `--qwen35-decode-graph` is a no-op under paged KV** (2026-08-01; not landed, [error](docs/experience/errors/2026-08-01-decode-graph-flag-is-a-noop-under-paged-kv.md))
- **ACCEPT — CP training now actually rings: fixed a `self.cp` split-brain, pod-verified FAIL→PASS** (2026-08-01; `3d9bc3717`, `b5ad2a136`, [error](docs/experience/errors/2026-08-01-cp-split-brain-forward-read-self-cp-not-arg.md))
- **REJECT — the draft attention is ALU-bound, but removing the IDIV only pays in a microbench** (2026-08-01; not landed, [error](docs/experience/errors/2026-08-01-draft-attention-idiv-win-is-microbench-only.md))
- **REJECT — the DSpark draft attention's per-key reduction axis** (2026-08-01; reverted in `aa4d2a6ec`, [error](docs/experience/errors/2026-08-01-draft-attention-reduction-axis-was-not-the-cost.md))
- **ACCEPT — ISO-Merger grafts one RL skill onto another, same-lineage, data-free** (2026-08-01; `aec0b17c7`, `a84cdfea9`, `a1940eee6`, [win](docs/experience/wins/2026-08-01-iso-merger-same-lineage-27b-graft.md))
- **PHASE EXIT — CP correctness core complete: ring full-attn + zigzag load-balance + linear-attn all-to-all-to-head, all CPU-gated** (2026-07-31; `8b3571973`, [win](docs/experience/wins/2026-07-31-cp-zigzag-seqshard-per-row-positions.md), [win](docs/experience/wins/2026-07-31-linear-attn-cp-all-to-all-to-head.md), [win](docs/experience/wins/2026-07-30-cp-ring-attention-and-all-to-all.md))
- **DEFAULT FLIP — DSpark static confidence truncation deleted; the head now drives the paper's goodput budget** (2026-07-30; [win](docs/experience/wins/2026-07-30-dspark-markov-confidence-batched.md))
- **DEFAULT FLIP — OPD seq-chunked recompute is unconditional; verdict still PENDING** (2026-07-30; `110632738`, `730ce7f31`, `bbd544f72`, `8970528d3`, [win](docs/experience/wins/2026-07-30-seq-chunk-bake-in-and-dparam-offload.md))
- **DEFAULT FLIP — `--dspark-conf-threshold` 0.5 → 0: the shipped default made spec decode slower than no spec decode** (2026-07-30; [win](docs/experience/wins/2026-07-30-dspark-markov-confidence-batched.md))
- **ACCEPT the batching, REJECT the confidence truncation — DSpark markov+confidence checkpoints now speculate at concurrency** (2026-07-30; `de58404b1`, `51985031d`, [win](docs/experience/wins/2026-07-30-dspark-markov-confidence-batched.md))
- **REJECT — data-free MoE expert merge (Qwen3.6-35B-A3B, 256→N)** (2026-07-30; [error](docs/experience/errors/2026-07-30-moe-expert-merge-collapse.md))
- **DEFAULT FLIP — `--spec-max-batch` 1 → 16: Qwen3.5/3.6 DSpark now speculates at concurrency** (2026-07-29; `6eada66df`, `7ceb39eb6`, [win](docs/experience/wins/2026-07-29-dspark-varlen-replay-c16-win.md), [win](docs/experience/wins/2026-07-29-dspark-batched-draft-across-slots.md))
- **ACCEPT — context-parallel N=2 OPD writeback runs end-to-end** (2026-07-29; `b8e2ad96b`, `f55c883a3`, [win](docs/experience/wins/2026-07-29-context-parallel-n2-writeback-works.md), [error](docs/experience/errors/2026-07-29-cp-nccl-wedge-is-hashmap-param-order.md))
- **ACCEPT — Qwen3.5/3.6 paged full attention: one launch per layer** (2026-07-28; `978c55e09`, `e628be4d3`, [win](docs/experience/wins/2026-07-28-fa3-one-launch-per-layer.md))
- **ACCEPT — Qwen3.5/3.6 converges onto the host-authoritative KV page mirror** (2026-07-28; `1fad68524`, [win](docs/experience/wins/2026-07-28-qwen35-host-authoritative-kv-mirror.md))
- **ACCEPT — training-system correctness program, Phases 1–5** (2026-07-27; commits `7bf66b90d`, `a48ebbc02`, `fb066003a`, `986d52d9e`, `2e67bd68e`)
- **CLOSE (Phase 7a — long agent writeback)** (2026-07-28; forward-peak win `e736c485a`, [win](docs/experience/wins/2026-07-28-opd-writeback-forward-peak-freed.md), [decomposition](docs/research/2026-07-27-opd-writeback-wall-decomposition.md))
- **REJECT (premise) — ISO near-isospectral premise fails on the DSpark head** (2026-07-28, `e7e33ff3b`; [errors](docs/experience/errors/2026-07-28-iso-premise-fails-on-dspark-head.md))
- **WITHDRAWN — "DSpark is net-negative once decode is fast"** (filed 2026-07-27 as a REJECT, `92175f3d5`; withdrawn 2026-07-28, `55bf627bc`)
- **ACCEPT — Qwen3.6-35B-A3B MoE is the faster serving target on 1×H20** (2026-07-28, `55bf627bc`; [champion row](docs/baselines.md))
- **ACCEPT — sm_90 paged decode attention routes to vendored FA3** (2026-07-27, `7a275d8ce` + `585e49337`; win: [2026-07-27-fa3-paged-decode-32k-2.76x](docs/experience/wins/2026-07-27-fa3-paged-decode-32k-2.76x.md))
- **ACCEPT — DSpark markov path batches by speculating its own chain** (2026-07-26, `ffc9ea652` + `0ade41244`; win: [2026-07-26-dspark-markov-chain-self-speculation](docs/experience/wins/2026-07-26-dspark-markov-chain-self-speculation.md))
- **ACCEPT — Agent RFT uses generation-time behavior probabilities** (2026-07-26; win: [2026-07-26-agent-rft-sidecar-denominator](docs/experience/wins/2026-07-26-agent-rft-sidecar-denominator.md))
- **DEFAULT FLIP — OPD carry GDN backward routes through the device chunked path** (2026-07-26, `d6ae52dc1` + `c4709d348`; bench: [2026-07-26-carry-gdn-device-reroute-tranche2](docs/experience/wins/2026-07-26-carry-gdn-device-reroute-tranche2.md))
- **REJECT (current form) — online markov-head self-RL cannot reach training scale; the markov path taxes 22.5%** (2026-07-26, `14669ec33`; bench: [2026-07-26-markov-head-online-selfrl-cannot-reach-scale](docs/experience/errors/2026-07-26-markov-head-online-selfrl-cannot-reach-scale.md))
- **FINDING — the DSpark draft is a good ranker and a bad argmax** (2026-07-26, `d420d894e`; bench: [2026-07-26-dspark-draft-is-a-good-ranker-bad-argmax](docs/experience/wins/2026-07-26-dspark-draft-is-a-good-ranker-bad-argmax.md))
- **AMEND — DSpark block size is a lever at concurrency** (2026-07-26; bench: [2026-07-26-dspark-block-size-is-a-lever-at-concurrency](docs/experience/wins/2026-07-26-dspark-block-size-is-a-lever-at-concurrency.md))
- **ACCEPT — one ragged-window launch per DSpark draft layer** (2026-07-26, `9a27eda4b`; bench: [2026-07-26-dspark-ragged-window-draft-attention](docs/experience/wins/2026-07-26-dspark-ragged-window-draft-attention.md))
- **ACCEPT (mechanism only, no serving delta) — one batched argmax per DSpark tick** (2026-07-26, `308c8b247`; bench: [2026-07-26-dspark-batched-argmax-tick](docs/experience/wins/2026-07-26-dspark-batched-argmax-tick.md))
- **WORKLOAD — bench workload is multi-turn agent sessions at the TraceLab medians** (2026-07-26, `08e1f10f8`; bench: [2026-07-26-long-agent-32k-is-the-workload](docs/experience/wins/2026-07-26-long-agent-32k-is-the-workload.md))
- **PHASE EXIT — spec-decode concurrency gate; three dispatch ladders → one `route_decode`** (2026-07-26, `69560ae55`; win: [2026-07-26-spec-decode-concurrency-gate](docs/experience/wins/2026-07-26-spec-decode-concurrency-gate.md))
- **DEFAULT — `spec_max_batch = 1`** (2026-07-26, `69560ae55`)
- **VERDICT — #128 DSpark accept-or-kill: KEEP as a c=1 feature; the 07-20 +63.8% vs 07-25 +5% gap was the dataset** (2026-07-26)
- **VERDICT — backward re-offload lifts the OPD-writeback device wall 24576→32768; 256K needs LA-chunk, beyond more offload** (2026-07-25, `e4be96108`; win: [2026-07-25-backward-reoffload-device-wall-24576-to-32768](docs/experience/wins/2026-07-25-backward-reoffload-device-wall-24576-to-32768.md))
- **REJECT — #127 "train a DSv4 draft head"; the trained head is public** (2026-07-25; docs/architecture-dsv4.md §7 corrected)
- **VERDICT — #160 device-fit park gate closed: backstop only, unreachable in practice** (2026-07-25, #160 closed; wins: [2026-07-24-dsv4-band-exhaustion-park-gate](docs/experience/wins/2026-07-24-dsv4-band-exhaustion-park-gate.md))
- **REJECT — "DSv4 cold boot is serialized on rank 0"** (2026-07-25, #181 closed not-planned)
- **DEFAULT FLIP — Qwen KV pool sizing: measured VRAM outranks the page floor** (2026-07-25, #178, `5c2931cd3`; wins: [2026-07-25-kv-pool-floor-yields-to-measured-vram](docs/experience/wins/2026-07-25-kv-pool-floor-yields-to-measured-vram.md))
- **DEFAULT FLIP — `--kv-disk` with a zero derived budget degrades to no-tier** (2026-07-25, #158, `59b86ee4c`)
- **VERDICT — DSpark V100 TP-lockstep stall: FIXED, measured** (2026-07-25, #168, `6c5553b45`; errors: [2026-07-21-dspark-v100-tp-lockstep-stall-kill](docs/experience/errors/2026-07-21-dspark-v100-tp-lockstep-stall-kill.md))
- **DEFAULT FLIP — writeback-offload threshold 4096 → 16384** (2026-07-24, #172; wins: [2026-07-24-writeback-offload-dial-back](docs/experience/wins/2026-07-24-writeback-offload-dial-back.md))
- **ACCEPT — FP8 quant loss on 27B: −0.25% PPL vs bf16** (2026-07-24, #174; wins: [2026-07-24-ppl-harness-fp8-matrix](docs/experience/wins/2026-07-24-ppl-harness-fp8-matrix.md))
- **REJECT — group-stagger admission for CC preamble prefix reuse** (2026-07-24, reverted in `2ab7883f1`; errors: [2026-07-24-group-stagger-premise-false](docs/experience/errors/2026-07-24-group-stagger-premise-false.md))
- **ACCEPT — agent-OPD sandbox staging outside the repo** (2026-07-24, `6bd40d663`+`b0a29443e`+`031c8c3f8`+`e21557fbc`)
- **ACCEPT — batched linear-attention CUDA device path** (2026-07-24, `ecc058b20` + `5f68d1f6e`)
- **ACCEPT — systematic review-fix sweep (26 findings); one relay regression fixed** (2026-07-24, `f0a635e02` + `837b89d39`)
- **DEFAULT FLIP — self-opd distill path fused → dense** (2026-07-24, `38bac08e6`)
- **REJECT — checkpoint-gate ×4 tightening reverted** (2026-07-24)
- **ACCEPT — agent-OPD rollout is idle-bound; concurrent mega-rollout GO** (2026-07-24)
- **ACCEPT — sm_120 FP8 MoE prefill: CUTLASS grouped GEMM (G2)** (2026-07-22)
### Removed (dead surface — `crates/train`, 2026-07-23)
- **train-crate systematic simplification, −4,134 LOC.**

## [0.4.0] - 2026-07-22
### Added
- **DSpark train sidecar** (`--dspark-train` / serve background trainer); batched verify (B>1); Qwen3.5/3.6 MTP speculative decode; agent-OPD cc-harness path; V100 (sm_70) serving substrate; ThinkingCap-27B-FP8; unified direct L3 storage (`kv-tier`); DSv4 local-NVMe cold load; qualified kernel artifact flow.
### Fixed
- **#167 Qwen3.6 temp>0 sampled-tail garbage; DSv4 extension-prompt prefix reuse; DSv4 plan-repair / HostPagedKvPool fatal at c32; SM-gate Qwen FP8 dense DeepGEMM to Hopper-only (`major == 9`); DSpark draft latent sliding-window.**
### Verdicts (selected)
- **2026-07-25 — bf16 tape Stage 1a (frozen prefix K/V) rejected: no VRAM win on Qwen3.6-27B.**
- **2026-07-21 — #167 closed: Qwen3.6 temp>0 sampled-tail garbage fixed (accept).**
- **2026-07-20 — DSpark train sidecar Phase 1 shipped (accept, end-to-end verified).**
- **2026-07-17 — DSv4 cold-boot #69 closed: fixed in code, disk-bound residual.**
- **2026-07-17 — DSv4 extension-prompt prefix reuse fixed (accept, wash).**
- **2026-07-17 — DSv4 prefill chunk default 128→2048 (default flip, accept).**
- **2026-07-17 — #164/#162 CLOSED (accept): c32 × 300 s oversubscription survival with real preemption (192 events, zero teardowns).**
- **2026-07-16 — adversarial review of the day's commits fixed 8 confirmed defects pre-deployment.**
- **2026-07-16 — DSv4 FP32 probe scratch hoisted off per-slot state (accept): per_slot 9618→338 MB, slot clamp 2→59.**
- **2026-07-16 — DSv4 FP32 prefill compressor grid-parallelized; serial probe kernel deleted (accept).**
- **2026-07-16 — DSv4 FP32 probe limited to prefill; unblocks DSpark (MTP) decode.**
- **2026-07-16 — DSv4 FP32 compressor extended to all compression boundaries.**
- **2026-07-16 — DSv4 FP32 compressor promoted to default.**
- **2026-07-15 — DSv4 long-context correctness blocked.**
- **2026-07-15 — DSv4 MegaMoE retained but correctness-blocked.**
- **2026-07-14 — DSv4 DSpark TP=4 concurrency licensed.**
- **2026-07-14 — DSv4 DSpark prompt router licensed for H20 TP=4.**
- **2026-07-14 — V100 (sm_70) prefill `cudaErrorNotSupported` fixed.**
- **2026-07-14 — DSv4 DSpark correctness PASS, opt-in unchanged.**

## [0.3.0] - 2026-07-12
### Added
- **DSpark block-draft speculative decoding** (`--spec-type dspark`); unified kernel set; content-addressed prebuilt kernel bundle; strategy-driven agent-OPD harness.
### Changed
- **2026-07-11 — DSv4 decode-region KV reuse default ON** (`6230d9d3d`)
### Eval

- **NVFP4 matches same-base FP8, and the kernel work costs nothing.** 200 items,
  greedy, FP8 KV, one GPU back to back: NVFP4 with this release **179/200
  (89.5%)**, NVFP4 control 180/200 (90.0%), `Qwen/Qwen3.8-27B-FP8` **177/200
  (88.5%)**. At n=200 the binomial spread is ~4.2 items, so all three are
  indistinguishable — the paged-attention change is quality-neutral, and the
  4-bit checkpoint matches the 8-bit one on the same base. This closes the
  measurement debt of comparing Qwen3.8-NVFP4 against Qwen3.6-FP8, two different
  models. **Not a GSM8K result**: the items come from the repo's own
  `examples/opd/gsm8k-train.jsonl`, the TRAIN split; only the arm-to-arm
  difference on identical inputs carries.

### Verdicts
- **2026-07-11 — DSpark draft-KV: cap full-layer at per-request ceiling** (Qwen3.6-27B, CUDA, `1ee72d809`)
- **2026-07-11 — DSpark/DFlash block-draft spec-decode: P1 LICENSED** (Qwen3.6-27B, CUDA)
- **2026-07-11 — DSv4 decode-region reuse: DEFAULT FLIPPED ON** (`--dsv4-decode-reuse`, was opt-in)
- **2026-07-11 — Agent-OPD round −30.1% wall (H20 GPU1 3-arm A/B), quality-neutral (`894be29fa`)**
- **2026-07-10 — DSv4 finish-write-through decode-region reuse: crash-fix gate PASS (opt-in `--dsv4-decode-reuse`), default flip pending perf**
- **2026-07-10 — DSpark-on-OPD default flip: quality-neutral LICENSED (opt-in), concurrency ≥4 DEFERRED**
- **2026-07-10 — DSv4 Route A prefix reuse "identity formula fix": REVERTED (`4ad32362e`)**
- **2026-07-10 — Qwen FP8 small-M dense GEMM: DeepGEMM from M=2 LICENSED; M=1 GEMV variants KILLED**
- **2026-07-10 — DSv4 KV-reuse Phases 2b+3b SHIPPED** (#154)
- **2026-07-10 — DSpark on the OPD rollout serve: wall-clock POSITIVE** (first e2e A/B, CC-as-harness, 16 real swe_smith tasks)
- **2026-07-10 — DSv4 prefix reuse RELICENSED (Phase 2a, content-keyed host-resident state pool)**
- **2026-07-10 — DSpark sampled (temp>0) spec decode LICENSED**
- **2026-07-10 — DSpark partial-ctx drafting (P2.5) LICENSED; sampling RNG cleared**
- **2026-07-10 — DSpark trained heads NO-LICENSE (z-lab backbone stays); P2 sampling verify KILLED as-is**
- **2026-07-10 — DSv4 Route A prefix reuse KILLED pending content-keyed redesign; warm-cache needle regression FIXED**
- **2026-07-10 — Qwen3.6 DSpark block draft LICENSED (short-ctx greedy)**
- **DSv4 decode-kernel levers #141/#142/#143 LICENSED (2026-07-04).**
- **Agent-OPD toy-corpus capability lane KILLED; harness + 12-round loop SHIPPED (2026-07-03).**
- **Phase 2 re-scoped; whole-step decode CUDA graph RE-KILLED (2026-06-21).**
### CUDA
- **Qwen3.6 serves on CUDA (2026-06-29); Qwen3.5-122B-A10B at TP4; GLM-5.2 (`glm_moe_dsa`, DSv4-DSA family) wired on the DSv4 path.**
### Metal
- **Qwen3.6 NextN/MTP spec decode shipped (2026-06-21)**
### Server
- **`/v1/chat/completions` now supports `stream=true`** (SSE `chat.completion.chunk` frames with `reasoning_content`/`content` deltas; closes the R5 tranche-2 deferral, #79)
### Repo
- **Renamed `agent-infer` → `arle`**

## [0.2.1] — 2026-06-15
> Consolidated section: tags `v0.1.5` (2026-05-02), `v0.2.0` and `v0.2.1`
> (both 2026-06-15) were cut without changelog sections. Everything below
> spans v0.1.4 → v0.2.1; per-tag artifacts live on GitHub Releases.
### Runtime rewrite — `infer-*` stack becomes the serving truth (2026-06-04)
- Breaking.
### Training surface — OPD-only (2026-05-18)
- Breaking.
### DSv4 perf campaign — adopt official kernels (2026-06-06 → 06-15)
- Official DSA indexer default-on: decode 124 ms → 26 ms flat @4096.
- FlashMLA `sparse_fwd` + FP8 DeepGEMM prefill default-on: 7.2 s → 3.48 s.
- Phase 0 debt closed 2026-06-10 (#56–#59).
- Seam-level KV-dtype dispatch `--kv-cache-dtype` (default bf16 unchanged); INT8/FP8 correctness LICENSED, opt-in pending a perf license (2026-06-12).
- Phase 1 batched-lane keystone closed (#61 2026-06-11, #60 2026-06-15): DSv4 B>1 decode takes the batched serving lane by default; residual c>1 throughput lever is DP-attn (#89).
### OPD train (CUDA) — new beta surface
- OPD mainline queue moved from experiment-only to operator-facing workflow; end-to-end OPD CUDA training stack landed on Qwen3-0.6B.
### Observability
- Low-overhead HTTP `request_trace` JSON summaries for streaming and buffered requests (TTFT, latency, throughput, KV/prefix-cache state, scheduler phase EMA).
- DSv4 Nsight trace artifacts committed under `docs/trace-artifacts/` (2026-05-14/15).
### CUDA
- DSv4 decode scratch reuse and uninitialized-allocation pass across attention, MoE, and compressor buffers (2026-05-15).
- DSv4 B=1 padded BF16 combine reduce-scatter default-on (`ARLE_DSV4_COMBINE_REDUCE_SCATTER`).
- **W4-hybrid prefill graph capture closes the 4k/c=4 gap — Tier 1 STRONG PROCEED** (`a56b7a9`/`c44788f` 2026-05-10; opt-in via `INFER_PREFILL_GRAPH=1` + `INFER_HYBRID_W4A8_PREFILL=1`, `35fc3cf`).
### Long-context (cross-backend)
- **RoPE scaling support** (YARN / Linear / NtkAware)
### Structured-output (xgrammar)
- `crates/xgrammar-sys` Rust safe wrapper over upstream `mlc-ai/xgrammar` v0.1.34, Phase 1 FFI scaffold (codex's #26).
### Metal
- Qwen3.5-0.8B MLX 4bit single-request step-driver: 305.5 tok/s mean / 304.7 p50 on M4 Pro 20c for `1024/256`.

> Older releases (0.1.x — pre-rewrite): see [CHANGELOG-history.md](CHANGELOG-history.md)
### 2026-08-04 — default flip: DSpark train sidecar `learning_rate` 1e-4 → 1e-3
  - See [errors/2026-08-03-dspark-online-sidecar-degrades-regardless-of-loss.md](docs/experience/errors/2026-08-03-dspark-online-sidecar-degrades-regardless-of-loss.md)
### 2026-08-04 — removed: the DSpark online train sidecar
  - See [errors/2026-08-04-dspark-bias-floor-model-was-wrong-twice.md](docs/experience/errors/2026-08-04-dspark-bias-floor-model-was-wrong-twice.md)
