---
title: 2026-05-10 next-session pickup state — quick orientation
date: 2026-05-10
type: research
status: session-end-checkpoint-for-next-pickup
---

# 2026-05-10 next-session pickup state — quick orientation

> **2026-05-10 later update**: this pickup state predates the Qwen3.5
> Medusa rollback audit. Any #28 Medusa pickup references should now be
> read as "Qwen3.5 recurrent rollback design first".

> One-page orientation for the next agent (codex reactivation OR
> fresh Claude session) to start from after this 16+ hour session.
> Read this first, then see `docs/index.md` for full context.

## §0 Read order on session start

1. This entry (one page, 3 minutes)
2. `docs/index.md` Last refreshed line (current state of all axes)
3. `de36538` retrospective (4 hallucinations + bilateral cooperative
   discipline working)
4. Skill v1.11.0 catalog: `.claude/skills/kernel-optimization/SKILL.md`
   — anti-patterns #29-32 are session-tested rules, not theoretical
5. Pickup queue (§3 below)

## §1 What's LANDED (today, 2026-05-10)

- **Phase 1 Substep 1.1 LICENSED** (codex `f86d0fd` + Claude
  `4f1b036` σ-tight n=2): TTFT -7.0%, ITL -3.2%, tok/s +2.1% on W4A16
  4k/c=4 vs 2026-05-08 baseline. Matches `e59beb5` -3-8% conservative.
- **PF8.1 + PF8.2 LANDED + smoke-verified**:
  - `940f49e` substrate (BF16→FP8 e4m3 quant + INT4 weight preprocess
    Apache 2.0 port, 181 LOC total, both feature gates clean)
  - `b628eca` PF8.1 runtime smoke PASS (max rel err 5.99% < 12.5%
    FP8 floor)
  - `451d094` PF8.2 runtime smoke PASS (caught 5th hallucination
    BY the smoke itself — bit-pack arithmetic memory-recall error)
- **PF8.4 dispatch wiring LANDED** (`db063ff`, +38 LOC): opt-in
  `INFER_MARLIN_W4_FP8_PREFILL=1` env var, bail at call site pending
  PF8.3 GEMM kernel.
- **🎉 PF8.3 Strategy A' substrate LANDED `11763ba`** (codex commit
  `feat(cuda): PF8.3 W4 FP8 Marlin prefill substrate`, 12 files
  +3936/-13 LOC: marlin_pf8/{core/scalar_type.hpp 58 + dequant.h 611
  + kernel.h 45 + marlin.cuh 176 + marlin_dtypes.cuh 151 + marlin_mma.h
  271 + marlin_template.h 2081} + marlin_w4_fp8_kernel.cu 253 + ffi/gemm.rs
  +20 + tensor.rs +55 + linear.rs +142 + wins entry 86). Substrate
  END-TO-END VALIDATED:
  Full validation trace:
  - `077b600` compile smoke PASS
  - `a0758e7` Strategy A' validation
  - cargo check PASS 3m51s
  - cargo clippy PASS 3m49s
  - greedy_consistency PASS 4.33s on `infer/models/Qwen3-4B-W4-hybrid-zpfix`
  - e2e PASS on hybrid checkpoint
  - **codex review caught 3 REAL bugs (FIXED)** per `ace3cbe`:
    parallel-M loop off-by-N (HIGH untriggered shape), max_par/lock
    workspace contract (HIGH possibly related Task #43), graph capture
    vs PF8 scratch (MEDIUM perf)
  - cargo check + clippy + tests RE-PASS post-fix
  - codex review FINAL pass running THIS tick (post-fix verify)
  - mma confirmed `m16n8k32` (line 99 of marlin_pf8/marlin_mma.h, k=32 path)
- **PF8.5 prep tooling COMPLETE** (Claude this session):
  - `3fa5e74` eval_ppl_pf83.py — PPL Δ% gate adapter
  - `84d61eb` bench_pf83_ab.sh — e2e A/B wrapper
  - `c382fba` pf83_license_sequence.sh — orchestrator (+ `bf47413`
    fix: hybrid-default INFER_TEST_W4A8_MODEL_PATH for Step 1, +
    `a6cf5ac` --dry-run flag with 5/5 OK pre-flights verified)
  - `e99e5a5` defaults to hybrid checkpoint
  - License gates per `aebd4a5`: TTFT Δ ≥ -8% σ<5% n=3 + greedy
    PASS + PPL Δ% ≤ +1.0% wikitext
- **codex review pattern empirical validation** (`ace3cbe`): 3 bug catches
  on 1 diff in 27 min review = high amortized value. Skill v1.12.0+
  candidate documented (strengthen #29 OR add #33: for non-trivial
  substrate codex review is gate, not formality).
- **codex review WEDGE → RESOLVED** (`b34de92` + `647210e` errors +
  recovery confirmed): PID 1867396 was alive 51m+ (State=S,
  futex_wait, 0:10 CPU) blocked on OpenAI API via localhost:7897 proxy.
  Diagnostic ruled out infrastructure (proxy HTTP 400 in 2ms, OpenAI API
  HTTP 401 in 393ms, other codex sessions fine). PushNotification
  dispatched recovery (`kill -TERM 1867396`). RESOLVED: PID 1867396 GONE
  + codex resumed pre-commit chain (standalone nvcc smoke PASS + cargo
  check re-PASS 3m49s + cargo clippy re-PASS 3m51s + targeted greedy/e2e
  re-running). Skill v1.12.0+ candidate #34 documented: timeout(1)
  insufficient for SIGTERM-catching subprocesses; use --kill-after for
  hard enforcement (or kill -9 PID directly).
- **PF8.3 substrate post-wedge ADDITIONAL FIXES** (codex narration THIS
  session): beyond the 3 review-caught bugs (ace3cbe parallel-M loop +
  max_par/lock workspace + graph capture vs PF8 scratch), codex applied
  "scale dtype 修正和 lazy sidecar gate" during post-review iteration.
  These additional fixes will appear in codex's substrate commit (still
  pending; cargo test in flight PID 1877276 started 07:05). Watch
  codex's eventual wins entry for full fix list.
- **PF8.5 ergonomics protection** (Claude this session — extends `c382fba`):
  - `bf47413` Step 1 INFER_TEST_W4A8_MODEL_PATH=hybrid (anti-pattern #29)
  - `e99e5a5` defaults to hybrid checkpoint
  - `a6cf5ac` --dry-run flag (5/5 OK pre-flights verified)
  - `9bb3843` bench_pf83_ab.sh RUST_MIN_STACK=8388608 (Task #43 protect)
  - `c6ccd24` stale-binary warning (compares mtime mar..._kernel.cu vs target/release/infer)
  - `172c311` PATH=$REPO/.venv/bin:$PATH (M_e gauntlet 2026-05-07 pattern,
    bench v2 root cause: guidellm not on PATH)
  - `45579c0` add INFER_HYBRID_W4A8_PREFILL=1 to all 3 entry points
    (bench v2 root cause: hybrid model loader requires both this AND
    INFER_MARLIN_W4_FP8_PREFILL — anti-pattern #29 manifested 2nd time)
- **PF8.5 bench v3 RUNNING + LIVE NUMBERS** (started 07:19, ~3min in
  at 07:22): GPU 100%, server PID 1907144 at 90.9% CPU.
  - engine_ttft_us=75000.0 = **75ms TTFT** baseline INT8 reference
  - engine_itl_p50_us=10000.0 = **10ms ITL p50**
  - batch_occupancy=0.65 = 65%, kv_util=64.1%
  - requests=326, tokens_out=41351 (rate=4 phase active, transitioning
    to rate=8)
  - step_phase: prefill 7.3ms / decode 185us / total 7.6ms
  - ETA completion ~07:29 (4 streams × 60s × 2 A/B runs + overhead)
  - License threshold per a66d99a: treatment FP8 TTFT ≤ 75 × 0.92 =
    **69ms** for -8% LICENSE; > 75 × 1.03 = 77.3ms for KILL
  - Logs: /tmp/claude-pf85-bench-v3.log + bench-output/2026-05-10-pf83-baseline-int8-run2/
- **#34 RESOLVED** (`df37a68`): `arle data download` + `arle model
  download` CLI surfaces. Data download VERIFIED working on current
  binary (`8735361` Medusa Phase 1.A pickup chain survey). Model
  download source landed but binary stale (2026-05-08 vs 2026-05-10
  source) — rebuild needed for Phase 2-3 model fetches.
- **Machete framing canonical disambiguation** (`aa9f72e`): 5+ user
  reissuances of literal "Machete W4 移植" formally mapped to Path
  B-Phase2' (PF8 chain Strategy A'). Future ticks won't re-litigate
  the Hopper-only blocker.
- **Skill v1.11.0 LANDED** (`b551bea`): canonicalized 4 anti-patterns
  (#29-32) from session retrospective. Now load-bearing for future
  sessions.
- **#36 KILLED** (`9bbc441`): PrefixAware Layer 2 — substrate works
  but op-point fails (warm p95 +17%, cold p95 +114%, starvation
  4.56→8.33×). QueueBound stays default; opt-in CLI retained.
- **#40 Tier 1 wins LANDED** (`c44788f`): -92.5% engine TTFT (this
  session sealed via cooperative codex+Claude chain).

## §2 What's KILLED (with reasoning)

- **Path B-Phase2' Phase 0 P0.A** (`67f18b9` codex + `61c9666` Claude
  architectural synthesis): cutlass FP8 GEMM smoke decode 1.86× <
  2× kill threshold. **W4 decode HBM-bound on weights; FP8 mma is
  wrong lever**. User's "-20-40% ITL via FP8" is **structurally
  infeasible** on sm_89. Same memory-bound ceiling explains why
  Machete (Hopper) wouldn't help on sm_89 even if backportable.
- **Substep 1.2 atomic_add** (in design, `0d63a52`): raw grep proves
  W4A16 `marlin_kernel.cu` has only `int* locks` (no
  `max_par × 64 × n` reduce buffer). W4A8 alt deferred to prefill-only
  FP8 axis.
- **Machete sm_89 backport** (`e65a096` 5-pt convergent evidence):
  `arch::Sm90` hardcoded throughout (collective_builder + mainloop +
  generate.py + Readme + 2026-05-09 prior survey all confirm
  Hopper-only). Default Path B-Phase2' (W4+FP8) tried instead —
  also KILLED for ITL but prefill-only TTFT axis is viable (PF8 chain).
- **M_spec classical external draft** (`#27` closed at -73%/-46%
  tok/s on 4k random text per M_spec plan).

## §3 Pickup queue (priority order)

### POST-COOPERATIVE-LOOP state (this session, 2026-05-10 EOD+~14hr — current)

Since H8-VERDICT below, ~8 Claude commits + codex Task #35 in flight:

**Claude-authored chain (8 commits, 2026-05-10 ~08:30-09:30)**:
- `0be7220` SKILL `kernel-optimization` v1.12.0 — anti-pattern #33
  (codex review value-add load-bearing) + #34 (greedy single-PASS not
  sufficient, pair with sustained-load) + #34b (bench 0-success →
  CHECK SERVER LOG FIRST)
- `05e2135` `M_pf83_h1prime_static_scratch.md` (264 LOC design,
  codex-pickup-ready, bench v11 gated): PF8Scratch struct + State
  integration + 110 LOC delta + 10-step checklist; eliminates per-call
  alloc fragmentation under sustained PF8 load
- `40a9184` `docs/index.md` Last-refreshed line refreshed to EOD+580
  reflecting H8-DISPROVEN + SKILL v1.12.0 + H1' design state
- `9c9d699` `2026-05-10-pf83-framing-trap-rule6-case-study.md` (153
  LOC) — codifies PF8.3 chain as §0 SOLID rule-6 case study parallel
  to 2026-05-08 NVTX framing trap (aggregate failure-rate framing ≠
  per-request functional ground truth)
- `868e147` `scripts/pf83_bench_health.sh` (118 LOC) — operationalizes
  SKILL #34b: single-shot 3-line verdict for any guidellm bench-output
  + server log pair (exit codes branch caller to debug-kernel vs
  debug-tool vs proceed-license)
- `e3e1ab5` `2026-05-10-w4a8-vs-bf16-accuracy-regression-observed.md`
  (133 LOC) — surfaced via codex Task #35 verification: full
  greedy_consistency binary fails W4A8-vs-BF16 with 84.4% diff
  (>>> 25% lenient assert at greedy_consistency.rs:365). Created
  Task #48. Bisect candidates: `35fc3cf` `c44788f` `09ae5a5`. NOT
  blocking Task #35 (codex correctly used targeted test).
- `60f114f` `2026-05-10-matched-control-escape-hatch-discipline.md`
  (132 LOC) — captures codex's `INFER_PREFILL_WARMUP=0` design
  decision in Task #35 as evidence point for future SKILL revision;
  explicit decision NOT to sediment yet (single-occurrence, n=2-3
  needed); §4 watch-list for n+1 evidence

**Codex-authored work in flight (Task #35, ~19min as of last capture)**:
- 5 code files modified: `infer/src/model.rs`, `qwen3/forward.rs`,
  `qwen35/forward.rs`, `deepseek/state.rs`, `scheduler/cuda/core/warmup.rs`
- `docs/environment.md` documenting `INFER_PREFILL_WARMUP=0`
- Pass 3 cap=8 prefill warmup added on top of existing Pass 1 + Pass 2
- `reset_for_warmup_clear` State trait hook + 3 impls per `58b0ac1`
  reconciliation
- Default-on with explicit escape hatch for matched-control A/B
- CUDA release build PASS 1m 03s (initial) + 6.96s (incremental)
- clippy `-D warnings` PASS
- Targeted `test_greedy_solo_vs_concurrent` PASS 19.11s
- Full greedy binary FAILED on EXISTING W4A8-vs-BF16 84.4% regression
  (Task #48, NOT blocking #35)
- About to start cold-start A/B bench (single-binary, baseline =
  `INFER_PREFILL_WARMUP=0`, treatment = default)
- **Bench IN FLIGHT** as of 2026-05-10 EOD+~14.5hr: server PID 1968247
  serving on port 8000 (12.6 GiB GPU, 100% util), baseline arm 1/6
  running. 6-round A/B plan = baseline×3 + treatment×3. First-attempt
  bench script had a bash-array-in-subshell bug (only printed env, no
  server start); codex recovered + caught a self-pkill misadventure
  (pkill -f command-string matched the running shell command itself
  → killed own shell). Now using port-level cleanup instead of
  command-string match. Discipline pattern: codex caught both bugs
  via post-action verification, not via pre-action review — argues
  for SKILL #33 "codex review --uncommitted" more aggressive use on
  bench scripts too, not just code substrate.
- Then will write wins entry per bench-and-trace-spec template

**New tasks**:
- Task #47 created: PF8.3 H1' static-scratch refactor (gated on bench
  v11 license)
- Task #48 created: Audit + bisect W4A8-vs-BF16 84.4% accuracy
  regression (NOT blocking, opt-in path only)
- Task #46 closed: H8 fix (DISPROVEN, root cause = H1' load-dependent)

**Cooperative discipline preserved throughout**: every Claude commit
used explicit-path `git add` (never `git add .` or `git add -A`),
codex's 6 WIP files untouched per skill #30. Codex independently
demonstrated SKILL v1.12.0 #34 internalization (planned sustained-load
smoke without explicit Claude prompt) + matched-control escape-hatch
discipline (per `60f114f` evidence point).

**Next session pickup priority (unchanged from POST-H8-VERDICT)**:
1. User runs bench v11 manually per recipe in POST-H8-VERDICT below
2. If LICENSE → codex picks up Task #47 H1' refactor
3. If KILL → close Task #44 + pivot Task #28 Medusa
4. Independent of bench v11: codex completes Task #35 + commits + writes wins entry

### POST-H8-VERDICT state (this session, 2026-05-10 EOD+~10hr)

**🎯 H8 DISPROVEN** (`57c37b5`): cargo build with diagnostic patch
PASS 4m 48s, then `pf83_h8_verify.sh` showed:
- ✅ Server with PF8 enabled + 2 curl returned valid Chinese+English text
- ❌ "cleared pre-existing CUDA error" diagnostic NEVER fired
- ✅ 0 gemm_w4_fp8_marlin_cuda failed log lines

**KEY INSIGHT**: PF8 path WORKS at conc=1. PF8.3 KILL is **LOAD-DEPENDENT**,
not dispatch-broken. **H1' (per-call alloc fragmentation)** is now the
confirmed root cause direction.

**PF8.5 license decision NOW POSSIBLE at conc=1**:
- v3 baseline INT8 conc=1: 53.6ms TTFT mdn, 6.8ms ITL, 1.1 req/s, 697 tok/s
- License threshold: TTFT ≤ 49.3ms (Δ ≥ -8% per a66d99a §2)
- Bench v11/v12 attempts blocked by Claude session sleep limits — **USER
  MUST RUN MANUALLY** in terminal:

```bash
cd /home/ckl/projects/arle
mkdir -p bench-output/2026-05-10-pf83-treatment-conc1-FINAL
PATH=$PWD/.venv/bin:$PATH

RUST_MIN_STACK=33554432 INFER_HYBRID_W4A8_PREFILL=1 INFER_MARLIN_W4_FP8_PREFILL=1 \
  target/release/infer --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix --port 8000 \
  > /tmp/pf83-FINAL-treatment.log 2>&1 &
sleep 30  # wait warmup

guidellm benchmark run \
    --target http://127.0.0.1:8000 \
    --model infer/models/Qwen3-4B-W4-hybrid-zpfix \
    --processor infer/models/Qwen3-4B-W4-hybrid-zpfix \
    --profile concurrent --rate "1" --max-seconds 60 --warmup 5 \
    --random-seed 20260416 \
    --data 'prompt_tokens=512,prompt_tokens_stdev=1,prompt_tokens_min=512,prompt_tokens_max=512,output_tokens=128,output_tokens_stdev=1,output_tokens_min=128,output_tokens_max=128' \
    --output-dir /home/ckl/projects/arle/bench-output/2026-05-10-pf83-treatment-conc1-FINAL \
    --backend openai_http \
    --backend-kwargs '{"validate_backend": "/v1/models", "request_format": "/v1/completions"}' \
    --disable-console-interactive \
    --outputs json --outputs csv --outputs html

pkill -f "target/release/infer.*--port 8000"
```

Compare TTFT mdn from output → if ≤ 49.3ms → LICENSE → codex H1'
static-scratch refactor for conc≥2.

### POST-PF8.3 KILL state (this session, 2026-05-10 EOD+~9hr)

**🚫 PF8.3 RUNTIME KILL** (`0cde63d`) — substrate landed (`11763ba`) but
gemm_w4_fp8_marlin_cuda fails code 2 (cudaErrorMemoryAllocation) on
101380/101380 requests under sustained load. greedy_consistency PASS
was conc=1 false-positive (per skill #29 strengthening).

**8 hypotheses ranked** (`2472e8a` + `cd7732a` + `c9abe8e`):
- H8 HIGHEST: sticky cudaGetLastError surfaces prior-kernel error
- H1': cudarc allocator first-call overhead
- H2: sm_89 100 KB smem exceeded by Hopper-tuned tile variants
- H6: ctx.ordinal/stream context mismatch
- H7: FFI -1 args (mostly DISPROVEN by auto-detect at line 168-176)

**H8 fix path ready** (`1b3f76c` + Task #46): 1-line cudaGetLastError
clear + diagnostic at marlin_w4_fp8_kernel.cu line 138, ~40 min codex
to verify+apply+bench v11+pair-with-greedy-consistency-conc-2-4.

**Medusa Phase 1.A pivot BLOCKED** (`ad14636`): HF_TOKEN not set,
lmsys-chat-1m gated. User auth setup OR non-gated alternative needed.

**5 user options pending** decision (per ad14636 §4):
1. HF auth setup → Medusa Phase 1.A
2. Non-gated alt dataset (humaneval) → Medusa Phase 1.A
3. PF8.3 kernel fix per H8 (Task #46)
4. #35 cap=8 prefill warmup (per 58b0ac1 reconciliation)
5. Session pause + summary (~80 commits, 9+ hours)



### Codex's natural pickup (highest leverage)

**PF8.3 FP8 marlin GEMM kernel** (~800-1200 LOC, 1-2 days codex) —
**STATUS: codex briefed via tmux paste-buffer THIS tick + Working
(2s)**. Brief: `/tmp/codex_brief_pf83.txt`. Strategy B selected
(single-template mirror, NOT verbatim port — m16n8k16→m16n8k32 mma
shape mismatch per `259277c`).
- Brief in `a66d99a` §1 + scope analysis in `259277c`
- Dispatch wiring already landed (`db063ff`); just plug kernel call
  into bail site at `infer/src/ops/linear.rs:1966+`
- Reuses cutlass sm_89 FP8 template from P0.A spike (per `d5a6679`
  unstick: `GemmUniversalWithAbsMax` + `arch::Sm89` +
  `LinearCombinationGenericWithScalingAndAbsMax`)
- KEY: shape mismatch m16n8k16 → m16n8k32 (k dim doubles, inner-loop
  changes substantially) — NOT a verbatim port (per `259277c`)
- License gate: TTFT p50 Δ ≥ -8% σ < 5% n=3 (per `a66d99a` §2)
- Strategy A (verbatim cascade marlin_template.h ~2000-3000 LOC) or
  Strategy B (single-template mirror marlin_w4a8_kernel.cu ~800-1200 LOC)
- Recommended: Strategy B for this scope

After PF8.3 lands: **PF8.5** = end-to-end TTFT bench A/B
(W4+INT8 baseline vs W4+FP8 prefill treatment).

### Long-term ITL win path (P0 hypothesis, blocked by training cost)

**#28 Medusa scaffold** (~500 LOC + 1 week training, `a66d99a` §5
P4): only remaining hypothesis for -50%+ ITL on sm_89 W4 decode per
`61c9666` architectural analysis. UNPROVEN until executed. Now
unblocked via `df37a68` #34 CLI surface.

### Research / planning (Claude-doable)

- **W3/W2 quantization research** (P3 in `09ae5a5`): direct weight
  footprint reduction for ITL ceiling. -25-50% ITL ceiling per quant
  level. No immediate impl path; needs PPL gate methodology.
- **#36 PrefixAware revisit**: 3 follow-up paths documented in
  `9bbc441` (cold_headroom sweep / session_id workload / c=32).
  None P0 since #40 already delivered single-stream gap closure.

## §4 Open decisions awaiting user

1. **PF8.3 strategy**: A (verbatim cascade) vs B (single-template
   mirror). Recommended B per `259277c` Strategy C analysis.
2. **#28 Medusa investment**: 1-week training cost + UNPROVEN
   acceptance rate. Worth it given M_spec classical KILL evidence?
3. **Machete name disambiguation** (still open per `e65a096`):
   user reissued "Machete W4 移植" 4+ times despite Hopper-only
   evidence. Default = Path B-Phase2' (W4+FP8 sm_89 native). If user
   means literal Machete sm_89 backport: 1800-3300 LOC + multi-week
   + KILL near-certain.

## §5 Anti-pattern reminders (skill v1.11.0+)

Load-bearing for next session:

- **#28**: tool-output-vs-peer-claim → re-run + raw quote in same response
- **#29**: default test fixtures may be known-broken (load-bearing
  THIS session: codex caught greedy_consistency PASS on W4A8
  checkpoint silently kept PF8 path INACTIVE before re-running on
  hybrid; saved false-license risk per `da45380`+`473081d`)
- **#30**: git status BEFORE commit (not just before add) in cooperative
  session — preserved THIS session by explicit-path `git add` for
  every Claude commit, leaving codex's untracked-modified files
  untouched
- **#31**: ANY ARLE surface claim needs raw evidence in same response
  (extends #28 beyond contesting peer; covers CLI flags, kernel
  internals, HTTP routes, baseline checkpoint match, model variants,
  bit-pack arithmetic, mma instruction shapes, model file locations,
  binary build dates)
- **#32**: peer "Waiting >5min" warrants direct ps/log/curl verify
  (4b30c15 33min wedge evidence). Codex's narrated-progress
  Working state is NOT a wedge even at 33+ min if narration
  shows command transitions.

**7 hallucinations** sedimented this session (escalated from 5 in
earlier checkpoint):

| # | Tick | Hallucination | Reality | Caught by |
|---|------|---------------|---------|-----------|
| 1 | `0f4d0ae` | --max-waiting-requests CLI flag exists | Never | codex |
| 2 | `43bda9c` | W4A16 reduce buffer max_par×64×n | W4A8 has it | codex |
| 3 | `4b30c15` | ARLE has /health endpoint | /healthz+/readyz | self via grep |
| 4 | `5bf0e20` | baseline-B5 vs newdequant comparable | different variants | self via cat |
| 5 | `451d094` | bit-pack 0x76543210→0xFEDCBA98 | →0x89ABCDEF (LSB→MSB) | empirical smoke |
| 6 | `818b4e0` | FP8 mma uniformly m16n8k32 | BOTH k=16 + k=32 | raw grep on vllm marlin_mma.h |
| 7 | `473081d` | hybrid checkpoint missing locally | EXISTS at infer/models/ | codex run + raw ls |

Common-mode pattern: confident claim about ARLE/upstream surface
based on internal recall instead of raw verification. Even
"deterministic computation" (#5 bit-pack) and "cited evidence" (#7
CARGO_MANIFEST_DIR convention not followed) can be hallucinated.

**Mitigation**: Step-by-step derivation OR empirical smoke OR raw
cross-check are the only reliable defense. Cite evidence only after
following it through to verification.

## §6 Session productivity summary

Claude commits today: ~30+ across substantial scope. Codex idle
~18 ticks since `f86d0fd` Phase 1.1 wins entry. Bilateral cooperative
discipline established + working when both agents engaged. Solo
Claude productive but reaching diminishing returns without new user
direction OR codex reactivation.

**Recommended next user action**: pick from §4 open decisions OR
let loop continue self-driving (Claude will keep producing PF8 chain
incremental progress, plus auxiliary research entries).

## §7 Cross-references (start here)

- `docs/index.md` — full Last refreshed line covers all axes
- `de36538` — session retrospective with 4 hallucinations chain
- `b551bea` — skill v1.11.0 canonical anti-patterns
- `a66d99a` — NEW prefill-only FP8 directive (PF8.1-5 substep plan)
- `259277c` — PF8.3 scope analysis (shape mismatch finding)
- `61c9666` — architectural P0.A KILL synthesis (FP8 wrong lever for
  decode)
- `e65a096` — Machete sm_89 BLOCKER 5-point convergent evidence
- `09ae5a5` — strategic priority revision
