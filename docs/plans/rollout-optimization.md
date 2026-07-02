# Agent-OPD rollout phase — measured decomposition and optimization plan

Date: 2026-07-02 · Status: measurement complete, optimizations NOT started
Scope: the ROLLOUT phase of `arle train agent-opd` (toy 1-task config) on 1×H20.

## Goal

Attribute every second of one agent-OPD round wall-clock to a code path, at
file:line granularity, and rank the optimization levers with estimated gain /
effort / kill-condition. Measurement + reading only; no code changed.

## Env / params (measured runs)

- 8×H20 pod, single clean GPU (run `rollprof3` on GPU 3, mem plateau 42.4 GB =
  this process alone; earlier samples `rolltrace1`/`rollprof1` on GPU 1 while it
  was clean — cross-checked, same shape). Binary: pod tree at `958536e9` +
  uncommitted LA-backward kernels, built 2026-07-02.
- Model `Qwen3.6-27B-FP8`: 64 layers = 48 linear-attn (GDN) + 16 full-attn, all
  dense FFN (no MoE hooks fired). `max_seq_len=16384`, KV pool 2048 pages BF16,
  `mem_fraction_static 0.2`, num_slots=2.
- Train config: `--task-limit 1 --rounds 3 --samples-per-prompt 1 --max-turns 6
  --max-tokens 256 --rollout-temperature 0.0 --writeback-window 512 --lora-rank
  16 --lora-target-set attention-qv`, share-frozen-base, spawner socket active.
- Instrumentation: `RUST_LOG=info` (ns timestamps), `ARLE_KVDRIFT_DEBUG=1`
  (per-row PREFILL / RESTORE-SIDECAR markers), 0.4 s sampler of GPU util/mem +
  last log line. A second run with `ARLE_QWEN35_PROFILE=1` (CUDA-event per-layer
  per-phase) provides the intra-prefill attribution (event `cuda_ms` sums match
  the unprofiled wall: 6.36 s profiled vs ~5.9 s unprofiled chunk).

## MEASURED — steady-state round segment table

ns-timestamped boundaries, round 1 of 3 (rounds 1 and 2 agree to <0.1 s).
Round wall = **18.71 s**, fully attributed (residual < 0.2 s):

| # | Segment | Wall (s) | Share | Evidence |
|---|---------|---------:|------:|----------|
| 1 | `ensure_kv_pool` + `boot_workdir` + overview(bash ls) + `reset_workdir` + session init | 0.07 | 0.4 % | pool-profile INFO → "Agent turn 1/6" INFO |
| 2 | Turn 1 prefill, 828-tok prompt (chunks 816 + 12) | ~6.2 | 33 % | PREFILL markers; chunk-816 alone ≈ 5.9 s at **GPU util 100 %** |
| 3 | Turn 1 decode (~22 tok tool-call) | ~0.7 | 4 % | turn-1 span 6.87 s minus prefill |
| 4 | Turn 2 = sidecar restore(816) HIT + prefill 74 + decode ~46 tok | 2.45 | 13 % | RESTORE-SIDECAR + PREFILL markers, decode ≈ 1.6 s |
| 5 | Turn 3 = sidecar restore(880) HIT + prefill 91 + decode ~35 tok | 2.17 | 12 % | same |
| 6 | `git diff` + pytest score + release inference scratch | 0.31 | 1.7 % | "Generated" INFO → "released full-attn KV pool" INFO |
| 7 | Writeback: release pool, forward 2.23, fused CE 0.004, backward 4.17, optimizer cleanup 0.31, trim + `sync_lora` 0.03 + tail | 6.85 | 37 % | `[masked-writeback] phase=` lines + pool-rebuild INFO |
|   | **Round total** | **18.71** | 100 % | |

Rollout proper (rows 1–6) = **11.86 s**; writeback = 6.85 s.

Derived rates (MEASURED):
- **Prefill ≈ 138 tok/s effective** (816 tok / 5.9 s), identical every round.
  Sub-turn tail chunks are equally slow per token (64-tok chunk ≈ 0.45 s).
- **Decode ≈ 27–36 tok/s** (B=1, no decode graph): ~103–122 LLM tokens/round
  (`total_targets=120–122`) in ~3.3 s across 3 turns.
- The autograd **writeback forward runs 1010 tok in 2.23 s = 453 tok/s — 3.3×
  faster than the inference engine's prefill** on the same trajectory. The
  "optimized" engine path is the anomaly, not the training path.
- GPU util: ~100 % through prefill/decode windows, 80–100 % through writeback;
  dips only at chunk/tool/score boundaries. The earlier "util 0 % for most of a
  round" report is **falsified** (likely sampled the wrong GPU or engine load).
- VRAM: rollout plateau 42.4 GB; writeback peak 85.4 GB; post-trim 36.1 GB.

## MEASURED — inside the 5.9 s prefill chunk (816 tok, per-layer CUDA events)

Top-level phases (children sum to parents; one chunk, profiled run):

| Phase | n(layers) | cuda_ms | Share |
|-------|----------:|--------:|------:|
| `qwen/dense_ffn` | 64 | 4348 | **68 %** |
| `qwen/linear_attention` (48 GDN layers) | 48 | 1572 | 25 % |
| ├─ `linear/in_proj` GEMM | 48 | 1040 | 16 % |
| ├─ `linear/out_proj` GEMM | 48 | 364 | 6 % |
| └─ `linear/gdr_recurrent` (serial scan) | 48 | **152** | 2.4 % |
| `qwen/full_attention` (16 layers) | 16 | 434 | 7 % |
| ├─ `full_paged/qkv_gemm` | 16 | 299 | 4.7 % |
| └─ `full_paged/attention` (FA3) | 16 | 3 | 0.05 % |
| norms/residuals/embedding | — | ~5 | <0.1 % |
| **Total** | | **6357** | 100 % |

The GEMMs dominate; the serial GDR scan — the previously suspected culprit — is
2.4 %. dense_ffn = **68 ms/layer at M=816**, absurd for H20.

### Root cause 1 (MEASURED): prefill GEMMs run the B=1 scalar-GEMV kernel

The profiled run shows **52,000 `qwen/fp8/gemv_batch` calls and zero
`qwen/fp8/dense_deepgemm` / `dense_dequant_bf16` calls** — every FP8 GEMM in
the rollout, prefill included, took the scalar warp-per-row decode GEMV.

Gate chain (`crates/infer-cuda/src/ops/quant_linear.rs`):
- `QWEN_FP8_DEEPGEMM_DENSE_MIN_M = 1024` (line 12);
- `fp8_deepgemm_dense_shape` requires `seq_len >= 1024` (line 206) → DeepGEMM
  declines M=816;
- the dequant→BF16-cuBLAS fallback uses the **same** MIN_M (line 380) → also
  declines;
- → `gemm_batch` (line 465) falls through to
  `gemv_fp8_block_scaled_batch_cuda`, whose own comment (lines 361-365) says
  *"Prefill must NEVER fall through to the scalar GEMV below — memory-bound
  per-token path (~20× slower at M=2048)"*.

DeepGEMM itself is available (7 `dense_deepgemm_warm` calls at startup, sm_90
gate passes, env default ON). The toy prompt is 827–828 tokens → the scheduler
(`chunked_prefill_size` default 2048, page-floor) emits one 816-token chunk —
just under the 1024 gate. **Production impact is not toy-only**: with prefix
restore working, every agent sub-turn prefills only the tool-result tail
(here 64–91 tokens; generally ≪1024), so *every* sub-turn prefill of a
multi-turn rollout takes the scalar path regardless of trajectory length.

## MEASURED — the prefix.rs:96 sidecar WARN root cause

Observed behavior (6 instrumented rounds across 2 runs, plus the 10-round run):

- **Within a round the restore HITs**: turn 2 restores at matched_len=816,
  turn 3 at matched_len=880, every round, no WARN — prefill resumes at
  start_pos 816/880. The within-round chaining (issue #85) works.
- **Cross-round reuse never happens, by design**: each round's
  `sync_lora_from_store` → `remerge_student_lora` →
  `engine.invalidate_prefix_cache()`
  (`crates/infer-api/src/serve_engine.rs:345`) empties the radix because
  q/v-LoRA changed and all cached KV is stale-epoch. The round-start ~828-token
  prompt therefore always fully re-prefills. This is correct, not a bug.
- **The WARN is intermittent, not every-round**: 2 occurrences in 10 rounds
  (rounds 7–8), zero in both 3-round runs, at matched_len=816/880 — i.e. a
  *within-round* sub-turn restore missing.
- Root cause of the miss: the sidecar store
  (`prefix_sidecar: HashMap<u64, Qwen35RecurrentSnapshot>`,
  `crates/infer-cuda/src/executor.rs:3357`) is capped at
  `RECURRENT_SIDECAR_CAP = 32` (line 3362) and on overflow evicts
  `keys().next()` — an **arbitrary HashMap victim, not LRU** (lines 3404-3407).
  Each sub-turn request inserts 2 keys (prefill-complete capture at
  `infer-core/src/lib.rs:956`, finish capture at lib.rs:1029) ≈ 6 keys/round;
  keys are FNV-1a hashes of the page-aligned token prefix, so token-identical
  rounds overwrite in place, but LoRA updates drift the greedy trajectory
  (seq_len 1009/1010/1011 across rounds; prompt token count 827/828 — constant
  3508 chars, the `git log -1 --oneline` commit hash in the repo-overview
  re-tokenizes differently each round since `boot_workdir` re-commits). Drifted
  rounds mint new keys; after ~6 drifted rounds the cap saturates and the
  arbitrary eviction can evict the *current* round's just-captured boundary key
  before the next sub-turn restores it → lookup miss → `Err` at
  executor.rs:3483 → WARN + full-recompute fallback (`prefix.rs:96-108`).
  Cost per miss: full re-prefill of the matched prefix (~6 s at today's 138
  tok/s; minutes at production 30K trajectories).
- Correctness note (READ, not yet observed failing):
  `invalidate_prefix_cache` clears the radix but **not** the executor-side
  `prefix_sidecar` snapshots. A restore only follows a radix match, and the
  fresh round re-captures over the same keys before its own restores — but if a
  round's capture is skipped (e.g. the D2H-failure skip at executor.rs:3419-3424)
  a same-key restore would serve **stale-epoch KV+recurrent state** silently.
  The invalidate should clear the sidecar map too.

## Attribution of the previously "unattributed ~3–4 s"

Fully accounted; there is no mystery segment. It was the sum of: prefill being
~6 s (assumed ~2–3 s), `optimizer_cleanup` 0.31–0.34 s, diff+pytest+scratch
release 0.31 s, pool release/trim ≈ 0.14 s, pool rebuild + boot 0.07 s — plus
one-time engine load (11.2 s: ~7 s weights + ~4 s autograd student) amortized
into earlier eyeball estimates.

## Corrected premises (brief → measured)

| Premise | Measured |
|---|---|
| rollout is ONE turn, ~750 generated tokens | 3 sub-turn engine requests; ~103–122 LLM tokens; response span 182 tok incl. tool results |
| prompt ~255 tokens | first-turn prompt 827–828 tokens (3508 chars) |
| WARN fires every round; prompt identical | WARN 2/10 rounds (sidecar-cap eviction); prompt drifts 1–2 tokens/round via the boot-commit hash |
| GPU util 0 % most of the round | util ≈100 % through prefill/decode, 80–100 % through writeback |

## Ranked optimizations

| # | Lever | Est. gain (toy round 18.7 s) | Effort | Kill-condition |
|---|-------|------------------------------|--------|----------------|
| 1 | **Fix the FP8 prefill M-gate**: route M below 1024 (down to ~64–128) to DeepGEMM dense — or at least to the dequant→BF16-cuBLAS path (`quant_linear.rs:12,206,380`) | prefill 7.7 s → ~1–1.5 s ⇒ round −30 % (~13 s); production: un-slows *every* sub-turn tail prefill | S (const + A/B rebuild) | same-binary A/B shows <1.5× on the 816 chunk, or needle-gate regression. HYPOTHESIS: exact small-M DeepGEMM efficiency unmeasured; the 20× claim is the in-code M=2048 number |
| 2 | **Multi-task batching** (production lever): N concurrent task rollouts through the continuous-batching engine (engine already multi-slot; rollout driver at `crates/train/src/agent_opd.rs:427-447` is strictly serial, one `run_turn` holds the engine lock) | ~N× rollout aggregate until GPU-bound — decode is host-orchestration-bound (~1074 launches/token, in-code note qwen35.rs:3306-3315), so batching amortizes the dominant cost | M (parallel sessions + per-task sandboxes already isolated; engine lock must move inside the step) | measured aggregate tok/s stops scaling at N=2 (would mean GPU-bound after lever 1) |
| 3 | **Whole-step decode graph** (`ARLE_QWEN35_DECODE_GRAPH`, exists, default OFF) | decode 3.3 s → ~2 s ⇒ round −6 %; bigger at production decode lengths | S (env flip + license) | needle gate fail, or ITL improvement <10 % (in-code prediction +30–75 %) |
| 4 | **Sidecar store hygiene**: LRU eviction (or per-slot keyed ring) instead of `keys().next()`, cap raised, and clear the map in `invalidate_prefix_cache` (correctness) | removes the intermittent ~6 s full-recompute (toy); prevents minutes-scale recompute + a silent stale-KV edge at production lengths | S | n/a (strict improvement); verify WARN count → 0 over ≥10 rounds |
| 5 | **Speculative decode / MTP**: Qwen3.5/3.6 MTP draft head exists (`qwen35.rs` `full_attention_into` draft-head path); DSv4 EAGLE infra in-repo | toy: ≤1–2 s (decode is small); production long-decode rollouts: decode dominates ⇒ meaningful | L (draft path validation + acceptance gate for hybrid arch) | acceptance rate too low at temperature-0 tool-call text, or spec-gate (≥2 prompts, self-consistency) fails |
| 6 | Sandbox/pytest cost | 0.38 s/round toy — no lever worth taking now; production repos (cp -a of large trees, pip setup, real suites) need re-measuring on real tasks | — | — |

Non-levers (measured): serial GDR prefill scan (2.4 % of chunk; the
`ARLE_QWEN35_GDR_CHUNKED` FlashQLA path would recover ≤0.15 s); sync_lora
(0.02–0.03 s); boot/reset (0.07 s). Writeback backward (4.2 s, 70 %
LinearAttention per the per-op profile) is a separate, already-explored lane —
the chunk-parallel LA backward landed at the 29.7 % kill threshold.

## Measured facts vs hypotheses

MEASURED (ns-timestamped logs, CUDA-event sums, GPU sampler; runs `rolltrace1`,
`rollprof1`, `rollprof3`, 2026-07-02):
- segment table above; 18.71 s round; prefill 138 tok/s; decode 27–36 tok/s;
  writeback forward 453 tok/s on the same trajectory;
- 52,000 scalar-GEMV calls, 0 DeepGEMM dense calls in the rollout;
- dense_ffn 68 ms/layer at M=816 (68 % of chunk);
- sidecar restore HIT in-round (start_pos 816/880 every round), WARN 2/10
  rounds at matched_len 816/880; radix invalidated every round at
  serve_engine.rs:345;
- prompt 3508 chars constant, 827/828 tokens across rounds.

HYPOTHESES (need their own license/kill):
- DeepGEMM (or dequant+cuBLAS) at M=64–816 recovers ~10–20× on those GEMMs —
  gate-read + in-code 20× claim, not yet A/B'd at these shapes;
- the commit-hash token drift is what varies the prompt (char-arithmetic exact:
  overview = 33 chars incl. 7-char hash; not byte-decoded);
- decode-graph +30–75 % (in-code analysis, unlicensed);
- batching scales ~linearly to N=4–8 (host-bound decode inference, unmeasured);
- turn-level decode token counts (22/46/35) estimated from chars/4 against
  `total_targets` — exact per-turn splits not logged.

## Repro

Pod launcher pattern (detached, per-GPU): sed the label/GPU/rounds of the
committed toy launcher; env `RUST_LOG=info ARLE_KVDRIFT_DEBUG=1` (+
`ARLE_QWEN35_PROFILE=1` for the per-layer run); 0.4 s sampler of
`nvidia-smi -i <gpu>` + `tail -1` of the run log. Labels used:
`rolltrace1` (3 rounds, kvdrift), `rollprof1` (1 round, per-layer profile),
`rollprof3` (3 rounds, kvdrift, clean GPU 3, current binary).
