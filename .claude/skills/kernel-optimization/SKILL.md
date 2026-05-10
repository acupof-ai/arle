---
name: kernel-optimization
description: Use this skill when the user asks to optimize, tune, speed up, or improve the performance of a GPU/CPU kernel, operator (op), attention path, GEMM call, decode/prefill path, quantization op, scheduler hot path, or any "make this faster" / "reduce ITL/TTFT" / "lower memory" / "拉满 utilization" / "调 kernel" / "优化算子" request. Captures the methodology — formula-predict → measure binding constraint → single-variable A/B with matched controls → combinational A/B when interactions suspected → tradeoff explicit (no tradeoff = not at extremes) → license-or-kill — and an industry-reference catalog (FlashAttention, cutlass, Marlin, SGLang, vLLM, TileLang, ncu/nsys methodology) so each attempt is grounded, not hand-waved.
version: 1.15.0
---

# kernel-optimization

Methodology for grounded GPU/operator optimization in ARLE. Built from
hard-won evidence in this repo:

- 4 P0 paths killed in 2026-Q1/Q2 (M_pf-gemm autotune / M_pf-fuse / M_b.2.2 split-KV / M_pf-graph Phase 0)
- 3 independent evidence pieces converged on "kernel time isn't binding at sm_89 4k longctx" (master §3.3 R1, Phase 0 -0.8% TTFT KILL `8b4a03b`, E2 BN=32 +2.5% regression `f76ccc4`)
- §0 SOLID rule 6 (`847a132`): **framing 多角度交叉,wall-clock is ground truth** — NVTX-window 55.7% framing was the Phase 0v2 trap

The skill exists because **without methodology, kernel sweeps repeat ±2% noise** and burn weeks on incremental wins that aren't on the binding-constraint axis.

---

## Mantra (read every time)

1. **Predict with formula, not vibes.** Hardware constants × workload constants → predicted Δ%. No formula = hand-wave.
2. **Measure binding constraint first, sweep tunables second.** ncu / nsys / wall-clock — not arithmetic alone, not NVTX window framing.
3. **Single A/B variable, matched controls.** Multi-variable change = can't attribute.
4. **Combinational A/B for known interactions.** 2×2 grid catches "BLOCK_M × NUM_STAGES" smem coupling.
5. **Tradeoff explicit. No tradeoff named = not at extremes yet.** Per user 2026-05-08 directive: "因为没有取舍大概率是当前方向都没有做好做到极致".
6. **License-or-kill with σ < 5% across n≥3.** Single-run win + tight σ ≠ same as repeated win.

---

## When to use

- "优化 X 算子" / "make Y faster" / "speed up Z kernel" / "拉满 GPU utilization"
- "调 BLOCK_M / NUM_STAGES / NUM_THREADS" / "tune attention tile" / "tune GEMM algo"
- "为啥 ARLE 比 SGLang 慢 X%" / "close the gap to <competitor>"
- "FP8 / W4 / quantization" / "speculative decoding" / "graph capture" → quant + spec + graph all need this skill's methodology
- "尝试 piecewise prefill / cutlass / Marlin / FlashAttention" — applying an industry pattern
- After a KILL — "为啥 small improvement 不算大改进的累积" → re-examine via this skill

Don't use for: pure correctness work (bug fixes), API design, refactor without perf goal, build-system changes.

---

## Workflow

Run these phases in order. Skipping a phase is the recipe for a kill (4 prior P0 KILLs proved it).

### Phase 1 — State the optimization target explicitly

Pick one (no "make it faster all-around" — that's vibes):

| Target | Metric | Where measured |
|---|---|---|
| **Latency** | TTFT p50/p99, ITL p50/p99, TPOT | Client (guidellm) + server (`/v1/stats`) |
| **Throughput** | tok/s, req/s, total token/s | guidellm sweep |
| **Memory** | KV cache size, weight size, peak VRAM | nvidia-smi + `/v1/stats` |
| **Occupancy** | warps/SM, active blocks/SM | ncu (proxy, never the final goal) |
| **Compute utilization** | TFLOPS achieved / TFLOPS peak | ncu tensor pipe pct |
| **Memory bandwidth** | GB/s achieved / 672 GB/s (4070 Ti SUPER) | ncu HBM pct |

**Acid test**: if pre-optimization you can't say "I want to move <metric> from <X> to <Y> within <Z> minutes wall-clock", you don't have a target. Stop.

### Phase 2 — Hardware constraint sheet

Pull the SM-specific limits BEFORE writing code.

```bash
# Verify GPU + SM
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Get smem/SM, reg/SM, threads/SM via ncu device query
ncu --query-metrics-collection device | head -50
```

Reference table (MUST update if running on different hardware):

| SM | Smem/SM | Reg/SM | Threads/SM | BF16 TFLOPS | FP8 TFLOPS | Notes |
|---|---:|---:|---:|---:|---:|---|
| 80 (A100) | 164 KB | 64 K | 2048 | 312 | — | first BF16 |
| 86 (RTX 30xx) | 100 KB | 64 K | 1536 | varies | — | consumer Ampere |
| **89 (RTX 4070 Ti SUPER / 4080S / 4090)** | **100 KB** | **64 K** | **1536** | **88.5** | **706** | **ARLE primary**, native FP8 mma |
| 90 (H100) | 228 KB | 64 K | 2048 | 989 | 1979 | TMA + WGMMA |
| 100 (B100/B200) | TBD | TBD | TBD | TBD | TBD | native FP4 mma |

**Key hardware traps caught in this repo**:

- **Hopper-default kernels on Ada underperform.** TileLang HD128 carries `BLOCK_M=64, BLOCK_N=64, NUM_STAGES=2` with comment "tuned during the H100 spike". sm_89 has 100 KB smem/SM (vs Hopper 228 KB) — these defaults push smem to ~96 KB/CTA = occupancy 1 block/SM ceiling. **First revisit target on consumer cards.**
- **cuBLASLt FP8 on Ada requires TN layout.** NN returns `CUBLAS_STATUS_NOT_SUPPORTED`. Verified `/tmp/fp8_smoke.cu` (M_quant §9).
- **cuBLASLt heuristic dispatch ≠ cutlass direct mma.** cuBLASLt FP8 hit ~24% of 8× theoretical on sm_89; cutlass direct may hit higher. License Phase 0 W8A8 on cutlass smoke, NOT cuBLASLt smoke.
- **sm_89 has no native FP4 mma.** NVFP4 is sm_100+ only — emulated FP4 on Ada is slower than W4A16 Marlin.

### Phase 3 — Find the binding constraint (NOT skippable)

The single most common kernel-optimization bug in this repo: sweeping tile parameters without first proving kernel time is the binding fraction of latency. Three KILL entries learned this.

```bash
# nsys — system-level launch density + dispatch overhead
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/profile_nsys_guidellm.sh <label> --concurrencies 4 --max-seconds 60 \
  --data 'prompt_tokens=4096,...,output_tokens=256,...'

# ncu — kernel-internal occupancy / smem / reg / mem-bound vs compute-bound
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/profile_ncu_guidellm.sh <label> --bench <existing-bench-dir> \
  --family attention --set full --launch-skip 5 --launch-count 5

# Internal counters (see bench-and-trace-spec §3) — every wins entry must cite these
```

**Wall-clock is ground truth** (§0 SOLID rule 6, `847a132`):

| Profiler view | Risk |
|---|---|
| NVTX-windowed % of frame | **Framing trap** — Phase 0v2 license attempt cited 55.7% NVTX dispatch but wall-clock was 0.32% |
| ncu single-launch metrics | Doesn't account for queue depth / multi-CTA SM share |
| nsys `cudaLaunchKernel` host time | Closer to ground truth, still needs / step time normalization |
| **bench-anchor wall-clock TTFT/ITL Δ%** | **Ground truth — every other view cross-checks against this** |

**Binding-constraint license matrix**:

| If profiler shows | Binding is | Optimize |
|---|---|---|
| ncu HBM > 80% peak on kernel | Memory-bandwidth | Quantize weights (W4/W8), reduce tensor footprint, fuse |
| ncu tensor pipe < 40% peak + occupancy > 50% | Compute-suboptimal | Bigger fragment (BLOCK_M=128), better mma path (cutlass vs cuBLASLt heuristic) |
| ncu occupancy ≤ 1 block/SM + smem > 90 KB | Smem-bound | Reduce stages, smaller BLOCK_N, persistent kernel |
| nsys launches/sec > 200 + dispatch > 30% step time | Launch-overhead | Graph capture (multi-key), fuse adjacent ops |
| Internal `prefill_queue` peak > 1 + slots full | Scheduler | chunk policy, admission tuning, bigger num_slots |
| All above < binding threshold | **Already at extremes** | KILL the experiment, pivot axis (quant / spec / algorithmic restructure) |

**The Phase 1.5 escape hatch**: if Phase 3 says you're already extreme on every axis at this hardware, the optimization request is a magnitude-axis question, not a tile-axis question. Pivot to quantization / speculative decoding / different SM target. Don't burn weeks on tile sweeps.

### Phase 4 — Predict with formula

Before any A/B run, write:

```
predicted_delta = f(hardware_constants, workload_constants, tunable)

example (E2 BN=32 attempt):
  smem_per_cta(BN=64) = (BM*HD + BN*HD + BN*HD) * 2B * STAGES = 96 KB
  smem_per_cta(BN=32) = 48 KB
  predicted_occupancy_gain = floor(100KB/48KB) - floor(100KB/96KB) = 2 - 1 = +1 block/SM
  predicted_throughput_gain = +50% (if smem-bound, otherwise much less)
  predicted_iteration_cost = BN halved doubles outer KV loop iters → +N% sync overhead
  net_predicted = max(throughput_gain - iteration_cost, 0)
```

**Without explicit formula, no run.** Hand-wave optimization burns GPU time and confidence.

The formula must show MAGNITUDE not just sign. "BN=32 should be faster because more occupancy" is too vague. "BN=32 frees +1 block/SM but doubles outer-loop iter count from 64 to 128" is the kind of formula that catches the actual outcome (the latter dominated).

### Phase 5 — Single-variable A/B with matched controls

ONE thing changes. All else stays.

```bash
# Baseline (commit hash recorded)
git rev-parse --short HEAD > /tmp/baseline-sha
scripts/bench_guidellm.sh <name>-baseline --concurrencies 4 --max-seconds 120 \
  --warmup 10 --data '<exact-spec>'

# Treatment (single variable change)
# Edit ONE thing (kernel constant, env var, dispatch path), build incremental
scripts/bench_guidellm.sh <name>-<treatment> --concurrencies 4 --max-seconds 120 \
  --warmup 10 --data '<exact-spec>'
```

**Matched-control checklist** (every miss = uncontrolled comparison):

- [ ] Same model + weights path
- [ ] Same KV cache dtype (BF16 vs FP8 ≠ matched! Phase 0 KILL contaminated by this)
- [ ] Same `--num-slots` / `--max-seq-len` / scheduler args
- [ ] Same data spec (prompt_tokens / output_tokens distribution)
- [ ] Same concurrency
- [ ] Same warmup + duration
- [ ] No other GPU process running (single-card serial)
- [ ] σ across n≥3 runs < 5% (ground truth precision)

**The "isolation motive" trap (anti-pattern #8 forward direction)**:

The most common matched-control violation in this repo is committed
*forward*: forcing one variable to a non-production setting because it
"isolates" the variable under test. Example pattern:

> "I want to A/B Marlin W4A16 vs BF16 weight without confounding KV
> quantization, so I'll force `--kv-cache-dtype bf16` on both arms."

This LOOKS like good methodology — same KV dtype both arms, isolated
variable. But the **production-default arm becomes a synthetic baseline**
(`19.27 ms ITL` was measured at production-default auto-FP8 KV; forcing
BF16 KV on the comparison arm makes the comparison unmatched).

Two real instances in this repo, opposite directions:

1. Phase 0 KILL `8b4a03b` (codex): forced BF16 KV → -0.8% TTFT artifact (wrong direction)
2. Round 1-3 Marlin `8e73dad` (Claude): forced BF16 KV → 1.06× ITL artifact (right direction, wrong baseline anyway)

Round 1 was self-corrected at `2853551` once codex's matched bench at
`f6f3af3` (production-default Marlin, **no** `--kv-cache-dtype`
override) showed actual 1.64× ITL with -39% Δ.

**Rule**: when the comparison includes a "baseline" measurement that
already exists in the project (e.g. `786a20a` ARLE pre-Phase 0 was at
production-default KV format), the treatment arm MUST run at the same
production-default unless you re-run the baseline yourself with the
new forced setting. "Isolation" is achieved by re-baselining, not by
forcing the treatment arm into a non-production config.

If you find yourself reaching for `--kv-cache-dtype bf16` (or any other
production-override flag) to "isolate", STOP. Either:
- Run a fresh baseline with the same forced setting (matched, but
  diverges from production reality), OR
- Use the production-default both arms (matched, reflects production).

Never compare forced-treatment vs production-default-baseline.

**Statistical sanity**:

- Single run = noise sample. n≥3 minimum.
- σ > 10% → workload not stable, increase `--max-seconds` or `--warmup`, or check for thermal throttling.
- Δ < 2× σ → not a real win. Treat as noise.
- Δ > 10% with σ < 5% → license trigger.

### Phase 6 — Combinational A/B for known interactions

If two variables have known coupling (smem budget = f(BLOCK_M, BLOCK_N, NUM_STAGES)), single-variable sweep misses interaction effects. Run a 2×2 (or 3×3 if budget allows) grid:

| | NUM_STAGES=2 | NUM_STAGES=3 |
|---|---|---|
| BLOCK_N=32 | A | B |
| BLOCK_N=64 | C (baseline) | D |

Interactions to catch:

- BLOCK_M × NUM_STAGES (smem coupling)
- BLOCK_M × NUM_THREADS (warp tile distribution; TileLang 0.1.9 fails with `warp_col_tiles ≤ 8`)
- KV format × graph capture (Phase 0 KILL was BF16-forced ≠ production auto-FP8)
- Tile size × workload shape (4k vs 8k vs 32k may favor different tiles)

### Phase 7 — Tradeoff explicit (the user-mandated step)

**Per user 2026-05-08**: "no tradeoff named = not at extremes yet". Every winning A/B candidate must enumerate what was sacrificed:

| Tradeoff axis | Question to answer |
|---|---|
| **LOC complexity** | How many new state machines / branches / FFI calls added? |
| **Hardware specificity** | Does this break on sm_80 / sm_90 / sm_100? |
| **Compiler/runtime version** | Does this break on TileLang 0.2.x or cutlass 4.x? |
| **Maintainability** | Will this need re-tune after every model size change? |
| **Numerical correctness** | Has the precision margin been verified (greedy_consistency)? |
| **Generality** | Does this win at one shape regress at another? (Multi-shape A/B mandatory) |
| **Scheduling impact** | Does this change admission policy / introduce serialization? Phase 0 KILL had this. |
| **Memory budget** | Is the win consuming smem/regs/VRAM headroom needed elsewhere? |

If all axes return "no sacrifice" — the optimization is fictional. **No free lunch.** Either the baseline was suboptimal (worth re-examining the prior measurement) or the win is measurement noise.

Real example tradeoffs:

- Graph capture wins TTFT but adds capture/replay state, multi-key cache LOC, and FP8 graph plumbing → tradeoff: LOC + scheduling complexity for dispatch reduction.
- W4A16 Marlin halves decode bandwidth but requires GPTQ checkpoint conversion + Marlin-specific weight repack → tradeoff: weight-prep workflow + accuracy ≤ 0.5 PPL loss.
- Cutlass FP8 direct mma may beat cuBLASLt heuristic but is sm_89-specific kernel choice with TN-only layout constraint → tradeoff: kernel surface area + portability.
- Phase 0 single-bucket graph saved 3.8 ms launches but serialized prefill admission via envelope clamp → tradeoff: launch-overhead win for admission throughput loss (net -0.8% TTFT, KILLED).

### Phase 8 — License-or-kill

Use the project's canonical thresholds where they exist:

| Plan | License threshold | Kill threshold |
|---|---|---|
| M_pf-graph (Phase 0/v2) | TTFT p50 Δ ≥ +10% | < +5%, or any ITL/tok-s regression |
| M_quant (Phase 0 W8A8) | TTFT Δ ≥ 5× (theoretical 7.9×) | < 2× → re-verify implementation; e2e garbage / greedy diff > 5% → KILL |
| M_quant Phase 0 v2 (cutlass FP8) | speedup ≥ 6× → ✅ proceed; 3-6× → ⚠ lower ROI; < 3× → ❌ FP8 path KILL, pivot W4A16 |
| Generic kernel re-tune | TTFT/ITL Δ ≥ 10% with σ < 5% across n=3 | < 5% within noise band |

Always write a wins or errors entry. **Both outcomes accumulate knowledge.** A KILL entry that names the framing trap (NVTX vs wall-clock) is worth as much as a win.

---

## Industry pattern catalog

Apply these by **identifying the binding constraint first** (Phase 3), THEN selecting the matching pattern. Don't apply patterns by reputation.

### Attention path

- **FlashAttention 2/3** (Tri Dao). Online softmax tile, warp-specialization (FA3), persistent kernel.
  - Reference: <https://arxiv.org/abs/2307.08691>
  - Pattern: SRAM-resident KV tile, m_i/l_i online aggregation, no quadratic memory.
  - When: prefill or decode where attention is HBM-bound (almost always for long context).

- **PagedAttention** (vLLM, Kwon et al.).
  - Reference: <https://arxiv.org/abs/2309.06180>
  - Pattern: KV cache in fixed-size pages, page table indirection, eliminates fragmentation.
  - ARLE has this (`crates/cuda-kernels/csrc/kv/`).

- **TileLang HD128 paged attention** (this repo).
  - File: `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`
  - Pattern: TileLang DSL → AOT cubin per (q-heads, kv-heads, SM). Online softmax + paged KV.
  - Constraint: TileLang 0.1.9 codegen rigid on `BLOCK_M / NUM_THREADS` (`warp_col_tiles > 8` rule, fragment layout assert).

### GEMM path

- **cutlass** (NVIDIA library). `device_gemm_universal`, FP8 / FP4 / mixed-precision paths.
  - Reference: <https://github.com/NVIDIA/cutlass>
  - Pattern: Hierarchical tile (CTA / warp / mma). Pick over cuBLASLt when heuristic dispatch is suboptimal (24% utilization on sm_89 → cutlass smoke verification).

- **cuBLASLt** (NVIDIA closed-source). Convenient API + heuristic algo selection.
  - Trap: heuristic may pick suboptimal algo for non-canonical shapes. Always cross-check vs cutlass on sweeps.
  - Trap: FP8 on Ada requires TN layout (NN unsupported).

- **Marlin W4A16** (Elias Frantar, IST Austria).
  - Reference: <https://arxiv.org/abs/2408.11743>
  - Pattern: 4-bit weight unpacking + fused FP16 GEMM. Single-source weight bandwidth path.
  - When: decode is HBM-bound on weight read AND model is small enough that weight quant accuracy holds (Qwen3-4B works, 70B+ may need SmoothQuant + AWQ).
  - ARLE: `crates/cuda-kernels/csrc/gemm/marlin_*.cu` already lands.

- **MergedColumnParallelLinear / QKVParallelLinear** (vLLM/SGLang).
  - Pattern: fuse Q+K+V projections into one GEMM with split output. Saves 2 launch overheads + improves cache reuse.
  - Trap: ARLE M_pf-fuse Phase 0 (gate-up fusion) hit cuBLASLt heuristic regression at large N — caught by Phase 5 single-variable A/B before merge.

### Quantization

- **SmoothQuant** (MIT-Han-Lab).
  - Reference: <https://arxiv.org/abs/2211.10438>
  - Pattern: weight smoothing factor migrates outlier magnitude from activations into weights → makes activation quant easier. Use BEFORE per-tensor activation quant.

- **AWQ** (Lin et al.).
  - Reference: <https://arxiv.org/abs/2306.00978>
  - Pattern: protect salient weight channels (top 1%) from quantization based on activation magnitude. Higher fidelity than naive RTN.

- **NVFP4** (Blackwell native).
  - Reference: NVIDIA Blackwell whitepaper.
  - Pattern: E2M1 FP4 weight + per-block FP8 scale + FP8 activation. **sm_100+ native; sm_89 emulated is slower than W4 Marlin.**

### Scheduling / dispatch

- **PiecewiseCudaGraphRunner** (SGLang).
  - Reference: SGLang source `python/sglang/srt/model_executor/cuda_graph_runner.py`
  - Pattern: 42 token-count buckets, capture full prefill layer loop. Trades capture-time for replay launch-overhead reduction.
  - Phase 0 KILL lesson: **single-bucket + tail eager fallback** does NOT match this pattern. Multi-key cache + tail handling are required to reproduce.

- **Continuous batching** (vLLM, SGLang, ARLE).
  - Pattern: dynamic request arrival, mixed prefill+decode in same step.
  - Trap: ARLE M_b.3 G1 segment-aware mixed batch was DEFERRED (per master §7.7) — non-binding constraint.

- **Speculative decoding (Medusa/EAGLE)**.
  - References: <https://arxiv.org/abs/2401.10774> (Medusa), <https://arxiv.org/abs/2401.15077> (EAGLE).
  - Pattern: cheap draft model proposes K tokens, expensive target verifies in parallel. Tok/s × 2-3 if acceptance ≥ 70%.
  - Trap: requires draft + verification kernel + acceptance loop; **trains a model** (Medusa heads) → data + training risk.

### Profiling methodology

- **NVIDIA Nsight Compute** (ncu) — kernel-internal metrics.
  - Reference: <https://docs.nvidia.com/nsight-compute/>
  - Trap (this repo, 2026-05-08): ncu 2026.1.1.0 dropped `--attach-pid`; project's `scripts/profile_ncu_guidellm.sh` wrapper needs migration to `--mode=attach --hostname` semantics.
  - Pattern: `--set full` for one-time deep dive; `--set basic` for sweep iterations; `--launch-skip N` to past warmup.

- **NVIDIA Nsight Systems** (nsys) — system-level launch timeline.
  - Reference: <https://docs.nvidia.com/nsight-systems/>
  - Pattern: capture range via `cudaProfilerStart/Stop` (ARLE has signal handler per master §4.2 M_nsys P0). Window the bench warmup out, profile peak load only.

- **Internal counters** (bench-and-trace-spec §3).
  - File: `docs/bench-and-trace-spec.md` §3
  - Pattern: every wins entry MUST cite `/v1/stats` snapshots before/during/after. External tools see client-side; internal counters explain WHY.

---

## Anti-patterns (caught in this repo)

Each anti-pattern has a project commit/entry where it was paid for.

1. **No formula prediction → hand-wave optimization**
   - Caught by: `f76ccc4` E2 BN=32 (predicted occupancy gain ignored loop-iter cost)
   - Fix: Phase 4 mandatory.

2. **Multi-variable change → can't attribute**
   - Caught by: M_b.2.2 split-KV opt-in changed both kernel + path + format simultaneously, regression couldn't be bisected.
   - Fix: Phase 5 single-variable rule.

3. **NVTX window framing without wall-clock cross-check (framing trap)**
   - Caught by: `847a132` SOLID rule 6, Phase 0v2.B license attempted with 55.7% NVTX framing while wall-clock was 0.32%.
   - Fix: Phase 3 wall-clock-is-ground-truth rule.

4. **Hopper defaults on Ada (consumer card kernel mismatch)**
   - Caught by: TileLang HD128 `BLOCK_M=64, BLOCK_N=64, NUM_STAGES=2` docstring "tuned during the H100 spike" → 96 KB smem on 100 KB Ada.
   - Fix: Phase 2 hardware constraint sheet.

5. **Tile parameter sweep without binding-constraint evidence**
   - Caught by: E1+E2 attempts (3 independent evidence pieces converged on dispatch-binding before single tile sweep moved TTFT >10%).
   - Fix: Phase 3 must precede Phase 5.

6. **License on "capture exists" not "capture reused"**
   - Caught by: Phase 0 KILL — graph captured 30 keys but recaptured constantly (single-key cache invalidation).
   - Fix: Phase 8 license thresholds reference REAL workload, not synthetic.

7. **cuBLASLt heuristic ≠ cutlass direct mma**
   - Caught by: `/tmp/fp8_smoke.cu` 1.88× cuBLASLt utilization, cutlass smoke pending.
   - Fix: When heuristic-dispatch GEMM lib < expected, sanity-check via direct kernel before declaring hardware ceiling.

8. **Production default ≠ A/B baseline (matched-control violation)**
   - Caught by: Phase 0 forced BF16 KV (graph compat) compared against production auto-FP8 baseline → contaminated -0.8% TTFT comparison.
   - **Caught again forward direction** (`2853551` self-correction): Round 1-3 Marlin forced BF16 KV against an FP8-KV-implicit baseline citation → 1.06× ITL artifact while production-default Marlin (`f6f3af3`) actually delivers 1.64× ITL. Same trap, opposite direction. The "isolation motive" disguises the violation when the forced setting feels like good methodology.
   - Fix: Phase 5 matched-control checklist + new "isolation motive trap" callout (v1.2.0).

9. **No σ → noise reported as win**
   - Caught by: every wins entry now cites σ across n≥3.
   - Fix: Phase 5 statistical sanity.

10. **No tradeoff named → not at extremes**
    - User directive 2026-05-08.
    - Fix: Phase 7 tradeoff axis enumeration.

11. **Same-typedef-name across BF16 vs FP16 kernels masks conversion overhead**
    - Caught by: ARLE Marlin path — `ffi::Half` typedef refers to `__nv_bfloat16`
      in BF16-native kernels (e.g. `turboquant_weight_gemv.cu`) but the literal
      IEEE-754 FP16 in Marlin (`marlin_kernel.cu`). Same name, different
      hardware semantics; per-call cost differs by 2 elementwise launches
      (`bf16_to_fp16_cuda` + `fp16_to_bf16_cuda`).
    - Caught by: `b3f22ea` Round 4 prep survey resolving the typedef.
    - Fix: When integrating a third-party W4/W8 kernel into a BF16 stack,
      grep the kernel's `.cu` for `__half` vs `__nv_bfloat16` literals BEFORE
      assuming the FFI typedef is uniform. If the kernel is FP16-internal,
      bake conversion cost into Phase 4 formula.

12. **Single-kernel choice ≠ optimal at all batch sizes (decode vs prefill duality)**
    - Hypothesis was: ARLE Marlin used at all batch>1 (`linear.rs:65-93`); decode
      (M≤8) is launch-overhead-bound where Marlin's 6-launch wrapper hurts,
      while prefill (M=2048) is compute-bound where Marlin's tensor cores win.
      Hypothesis predicted hybrid dispatch (small-batch → `W4A16BatchGemv`,
      large-batch → Marlin) would improve decode ITL.
    - Initial caught by: Round 4 prep `b3f22ea` matched-contrast launch density (6 vs 1).
    - **HARDENED v1.3.0** (R4 #6 KILL `4571082`): hypothesis **REFUTED by data**.
      Implementing hybrid dispatch (`MARLIN_DECODE_BATCH_THRESHOLD=8`) at
      `f00ff8b` produced **+60.7% ITL regression** at batch=4 decode (18.9 ms
      vs Arm B Marlin all-batch 11.76 ms). Greedy 2/2 PASSED (correctness
      preserved); σ ITL 0.06 ms (real signal, not noise).
    - **Root cause** (post-R4 #6): launch overhead is the cost of *amortizing
      tensor-core compute*. The cost is real, but the **benefit is even larger**.
      W4A16BatchGemv (CUDA-core GEMV, no tensor mma) at batch=4 is +61% slower
      than Marlin's multi-launch tensor-core path despite single-launch
      dispatch. Marlin's tensor-core throughput dominates the 5-launch
      overhead at decode batch ≥ 2 on sm_89.
    - Fix: Phase 7 tradeoff axis "Tensor-core advantage at small batch" must be
      a hypothesis-under-test, not an assumption. **Test the dual-kernel
      hypothesis with formula + bench BEFORE landing dispatch changes**;
      assume tensor-core dominance until empirically proven otherwise. The
      decode-vs-prefill duality applies to **kernels without tensor cores**
      (e.g., norms, RoPE, sampling) but **NOT** to compute-heavy ops with
      tensor cores (Marlin W4 GEMM, FA-3 attention, cutlass GEMM).
    - **Magnitude direction wrong**: Phase 4 formula predicted 1.23-1.47× ITL
      improvement; actual 0.62× = +60% regression. Skill enforced σ-tight
      single-arm KILL hard; methodology preserved future axis preservation.

13. **NULL result is real elimination, not skill failure**
    - Caught by: Round 2 (alloc_zeros skip) + Round 3 (variant swap) — both
      Δ < 0.5%, σ < 5%, hypothesis cleanly eliminated.
    - Without methodology, NULL is read as "something didn't work, drop the
      axis"; with methodology, NULL is read as "this hypothesis is dead, here
      are the surviving N candidates ranked by next-test cost".
    - Fix: Phase 8 errors entry must list surviving hypotheses with cost
      ranking. Cumulative table across rounds shows hypothesis space narrowing,
      preserving the axis until the actual binding cause is isolated.

14. **Upstream-data parser silent corruption masks "almost-working" kernel/pack**
    - Caught by: 5593865 (2026-05-08) — `scripts/convert_gptq.py` decoded
      AutoGPTQ qzeros without `+1` correction. AutoGPTQ stores
      `zero_point - 1` (so qzeros=7 means actual zero=8) but ARLE parser
      treated 7 as the actual zero → every weight off by 1 quantization
      unit → ~14% systematic bias per element → cumulative through 36
      layers → wrong logits (silent for W4A16 = "marginal accuracy 1.06×",
      catastrophic for W4A8 = "all-`!` garbage").
    - Today's chain: 4 hours debugging on quantize_qwen3_w4a8.py + 9
      hypothesis iterations (H3, H3b, H3c, H4, perm correction, MAGIC_NUM
      bound, GPTQ-aware mode, clamp ≤16) — most stemmed from THIS
      upstream parser bug, not from internal pack/kernel issues.
    - Audit `01ace86` checked pack + kernel + FFI + loader byte-by-byte
      against PR #31, but TREATED upstream GPTQ checkpoint as trusted.
      That trust was misplaced for ~1 year (silent W4A16 marginal).
    - Fix: when investigating "checkpoint produces wrong output", AUDIT
      THE UPSTREAM PARSER FIRST. Dump qweight/qzeros/scales raw values,
      compare to AutoGPTQ source spec EXACTLY (zero-1 convention,
      sym/asym, scale magnitude, g_idx interpretation, bit-extraction
      order, sign extension). The parser is the FIRST suspect for "looks
      slightly off" symptoms — internal kernel/pack iteration is the
      LAST.
    - These hidden contracts don't appear in kernel source or pack
      source — they're entirely in the parser → kernel expectation chain.

15. **"Warm-server" implicit dependency trap**
    - Caught by: `19d12c2` cap=8 override 100% turn success vs `bwa4piqqx`
      cap=8 default 76% on fresh-build server (same nominal config).
      Override case had been preceded by prior benches that warmed
      CUDA Graph cache for batches 5-8; default fresh-build did not.
    - Without methodology, single-run LICENSE based on warm-server
      reads as production-ready when it's actually conditional on
      cache state.
    - Fix: production-readiness benches MUST start from cold
      `cargo clean && cargo build` build state OR explicitly document
      warm-state assumption with deployment guidance. For
      CUDA-Graph-related benches, log warmup output and note any
      cache-dependent behavior.
    - Generalizes: any bench whose result varies between cold and
      warm process states needs both verified before LICENSE.

16. **Implicit-coupling-via-shared-default trap**
    - Caught by: `12300c5` bumped `Some(4) → Some(8)` in `forward.rs:316`
      but `core/warmup.rs:47` had hardcoded `max=4` independently.
      Single-line config flip broke implicit two-place coupling →
      production regression on fresh-server cold-start.
    - Without methodology, "1 LOC change" reads as low-risk when
      it's actually multi-site coupling rewrite.
    - Fix: future config-change PR commit body must include grep
      evidence dump:
      ```bash
      $ grep -rn 'OLD_VALUE' infer/src/ crates/cuda-kernels/src/
      file1.rs:N: this PR changes
      file2.rs:M: ← also needs OLD_VALUE → NEW_VALUE flip (coupling)
      ```
      This forces author to AUDIT all coupling sites before single-line
      change merges. Codex review process should require this template.

17. **Bimodal failure distribution masks single-run LICENSE**
    - Caught by: `a0a3f42` cap=8 6-run dataset showed 67% normal mode
      (76-92% turn success) + 33% degraded mode (56% turn success,
      byte-identical 23424 tokens out across multiple occurrences).
      `8281047` LICENSE was based on single normal-mode run.
    - Without methodology, single-run LICENSE assumes unimodal distribution
      and is systematically optimistic when bimodal exists.
    - Fix: multi-run sampling characterizes DISTRIBUTION, not single
      "true" value. Deployment confidence must account for mode
      probability:
      - N=1: point estimate (can be normal or degraded outlier)
      - N=3: detect bimodal vs unimodal
      - N=10+: full distribution shape + confidence interval
    - For binary-outcome thresholds (turn success ≥ 95%), N=3 minimum
      across run positions to characterize whether stable or progressive
      degradation. Single-run with σ-tight metrics is necessary but NOT
      sufficient.
    - Distribution-shape rule: if runs split into modes, document mode
      probability in LICENSE entry. Production confidence band =
      `mode_prob × mode_value` summed across modes, not just single
      best-case.
    - **Workload-shape refinement**(`063da81`): bimodal modes can be
      WORKLOAD-SHAPE-SPECIFIC, not config-global. cap=8 default is
      bimodal at W4 c=8 8K-prompt burst (stress shape) but CLEAN HARD-
      LICENSED at W3 c=4 short-multiturn (low-pressure). Production
      deployment guidance must distinguish bimodal-affected vs clean
      shapes — single global "production caveat" oversimplifies.
      Workload classification matters: shape × cap × concurrency ×
      prompt-length all factor into bimodal trigger.

18. **Phase 0 substrate audit before scoping new wiring**
    - Caught by: `1217375` A1 audit — codex's original B3 Step 2 plan
      (`c0ddd4f`) added `Arc<RwLock<RadixCache>>` field to
      `SchedulerHandle`, violating backend isolation. Code-grep revealed
      `runtime/admission.rs:187/193/741` ALREADY production-wires
      `lookup_or_stage` returning matched_tokens — exactly what Step 2
      needs. Refined architecture: integrate at CUDA-runtime admission
      (NOT HTTP-layer SchedulerHandle).
    - Without methodology, "new wiring" plans assume from-scratch
      implementation when adjacent code paths already provide the data.
      Wrong-layer plumbing violates backend isolation OR duplicates
      existing work.
    - Fix: Phase 0 reconnaissance ALWAYS audits the CLOSEST production-
      wired layer that touches the dependency before scoping. For
      "needs RadixCache" features, check:
      (1) Is RadixCache already accessed in production scheduler path?
      (2) Where exactly (HTTP handle, scheduler core, runtime admission)?
      (3) What does that path return that we can reuse?
    - Generalizes to ALL substrate dependencies (KV pool, allocator,
      tokenizer, kernel cache, paged buffers): inventory existing
      production wiring BEFORE designing new field/method/cross-layer
      access. Saved 70-100 LOC (30% scope reduction) + 0.5d wall-time
      on B3 Step 2 by routing through existing `lookup_or_stage`
      instead of duplicating at SchedulerHandle level.
    - Companion to anti-pattern #14 (upstream parser audit) and #16
      (implicit-coupling-via-shared-default): all three share the
      "audit existing before scoping new" methodology core.

19. **Dispatch directive paths must be verified at write-time**
    - Caught by: `8935851` (fixed `codex-pickup-queue-2026-05-09-eod.md`
      broken link in docs/index.md `14116c1`) + `de8b4dc` (fixed
      `infer/src/model/weight_loader.rs:514` stale path — actual file
      is `infer/src/weight_loader.rs:514` top-level not in model/).
      Both stale paths shipped tonight in artifacts intended for
      cross-session codex dispatch.
    - Without methodology, "dispatch directives" (paste-buffer-ready
      briefs, pickup queue file refs, doc index links) accumulate stale
      paths as the codebase moves around. Codex picking these up
      tomorrow hits "file not found" → wasted context-rebuild time
      OR worse, confused implementation against wrong file.
    - Fix: when writing a dispatch directive or pickup queue with file
      paths, ALWAYS run `ls <path>` or `grep <symbol> <file>` to verify
      the path exists at write-time. For symbol references (line
      numbers, function names), grep the file at directive-write time
      AND at pre-dispatch time (paths can drift between writing and
      firing). Bake the verification into the artifact: add a
      "verified YYYY-MM-DD" note next to each path so future
      consumers (cron-fired Claude, codex) see when freshness was
      last checked.
    - Special case for cross-session artifacts (pickup queues,
      EOD anchors): paths persist across days/weeks while the
      codebase moves. The artifact's *value* is sub-minute dispatch;
      stale paths erase that value entirely. Verify-at-write +
      verify-at-dispatch is the cheap insurance that preserves it.
    - Companion to anti-pattern #18 (Phase 0 substrate audit): #18
      is about substrate-dependency reuse before scoping new code;
      #19 is about file-path freshness in artifacts that outlive
      the session that wrote them. Both share "audit before trust"
      discipline.

20. **Phase 0 root-cause hypothesis inheritance trap**
    - Caught by: `c076aae` (audit-of-audit on `1fdd763`). Phase 0
      source audit verified file paths/LOC/comment claims (4/4 SOLID)
      but inherited the upstream "cold prefill GEMM = 33% degraded
      path" hypothesis without re-evidencing it. The c20b1ce attribution
      chain that followed turned out to be NO-OP (see #22).
    - Without methodology, Phase 0 substrate audits feel "complete"
      because the immediate file-existence/path/LOC claims verify,
      but the causal chain that motivates the work is itself a
      hypothesis that wasn't re-checked at audit time.
    - Fix: Phase 0 audit must restate the **root-cause chain** being
      implemented and explicitly mark which links are evidenced vs
      hypothesis. License-or-kill gates inserted appropriately at
      hypothesis links. "Audit's audit" catches the inheritance trap.
    - Companion to anti-pattern #25 (hypothesis-context vs
      implementation-context mismatch): both are forms of unstated
      assumption propagating into substrate work.

21. **Recipe-itself audit gap (recipe artifacts un-dry-run-audited)**
    - Caught by: `b55bfcd` (block-as-rvalue scoping fix on `2fafa9e`
      Phase 1.A nvtx recipe — would have failed compile) + `af44efa`
      (nsys-target fix on same recipe — would have profiled python
      bench client instead of Rust server, yielding empty NVTX data).
      **Two empirical evidence points** showing recipes need their
      own audit before pickup application.
    - Without methodology, recipe-style briefs (copy-paste-ready code
      diffs, shell commands, step-by-step procedures) inherit the
      hypothesis-vs-evidence trap as any other prescription. Writing
      a recipe ≠ having a working recipe.
    - Fix: post-recipe-write audit checklist:
      (a) Syntax correctness (does diff compile? does shell parse?)
      (b) Scoping correctness (bindings reach later uses? guards drop right?)
      (c) Tool/file existence (script exists? expected interface?)
      (d) Side effects / data flow (does profiling target match data source?)
      (e) For shell recipes: `--help` / `--dry-run` audit before pickup.

22. **Twin-commit fix attribution trap**
    - Caught by: `919c0fb` (silent-fail discovery) + `8d91d20` (NO-OP
      finding) + `3fea979` (Layer-7 closure: `12300c5` was the actual
      fix, `c20b1ce` is dead cosmetic). 7-layer SOLID gap chain on
      c20b1ce attribution. Three wins entries had to be annotated with
      corrected attribution (`655accf` + `9bc4729`).
    - Without methodology, when two commits co-ship as "fix the issue",
      default attribution to BOTH causes future readers to repeat the
      no-op fix in similar situations. One may be the real fix, other
      may be no-op cosmetic OR config-no-op (NO-OP when num_slots ≥
      prefill_cap config) OR silent-fail (silent-break when slot-out-
      of-bounds).
    - Fix: revert each in turn, measure individual contribution.
      License criteria 3-level escalation:
      (1) Code logic correct
      (2) Effect measurable in target environment
      (3) Attribution validated by controlled A/B (Layer-8: fix
          confounding variables, e.g. num_slots constant across cells)

23. **Truncated-output partial-view trap**
    - Caught by: `156d2c2` (false-alarm self-audit). Cargo test output
      shows multiple test-binary results; tail-only view of "0 passed
      M filtered out" can mislead because it's from one specific
      binary that doesn't contain the target test. Truncated middle
      contains the actual lib-tests pass.
    - Without methodology, hasty alarm raised on partial output wastes
      cycles. Codex was correctly interpreting full 165-line output;
      Claude flagged based on tail only.
    - Fix: before raising concern from truncated tool output, verify
      (a) target's gating compiles under given flags, (b) required
      runtime dependencies (paths/env/feature flags) satisfied,
      (c) cross-reference middle of truncated output for actual
      target binary's run result, (d) understand multi-binary semantics
      (cargo test runs N binaries, each filters independently).

24. **Cell-collapse blindness in N-cell A/B design**
    - Caught by: `1ccb448` (post-P0.2 cell-collapse finding). When
      designing 4-cell A/B for c20b1ce attribution kill, post-P0.2
      revert of c20b1ce permanently changed substrate state, making
      cells (a) and (d) identical on current main. Without the audit,
      tomorrow's pickup would have written redundant cell (a) recipe
      and run duplicate experiment.
    - Without methodology, cell INDEPENDENCE under current substrate
      state may break when substrate changes between A/B design time
      and execution time.
    - Fix: after each substrate landing, re-derive each cell's required
      edits and check for identity overlaps. If 2 cells differ only
      in dimensions current main has already normalized, they collapse
      to single experiment. Same applies to cell (b) which may become
      non-reproducible if substrate change is permanent.

25. **Hypothesis-context vs implementation-context mismatch (the bench-only blindspot)**
    - Caught by: `fe9ea8a` (preliminary KILL bench) + `3b9cc06`
      (refined batch∈2..=8 gate ALSO killed). Both Claude's Phase 0
      audit (`6ade2d4`) AND Codex's audit-of-audit (`5bb99d7`)
      verified syntactic correctness ("override condition compiles
      type-safely"). Both missed semantic context check ("in which
      batch contexts does override actually fire?"). Override fired
      in PREFILL (batch=4096) where W4A16BatchGemv loses to Marlin
      tensor-core, opposite of decode-target hypothesis. Bench
      showed +37% ITL regression.
    - **Critical methodology lesson: bidirectional code-level audit
      can share mental-model blindspot. Empirical bench is the truly
      orthogonal SOLID layer.** Audit chains examine the same claim
      space; only running the actual workload reveals semantic
      context-mismatches.
    - Without methodology, hypothesis target context (e.g. "decode
      M ≤ 8") and implementation firing context (e.g. "batch > 1
      includes prefill") drift apart, override leaks into untested
      contexts where cost-tradeoffs invert.
    - Fix:
      (a) Phase 0.5 context-semantics check after Phase 0 syntax +
          audit-of-audit syntax-of-syntax. What conditions does the
          implementation fire under? Are those conditions a SUBSET of
          the hypothesis target context? If broader, gate more narrowly
          OR explicitly accept broader scope with separate per-context
          evidence.
      (b) For high-stakes axis selections (LICENSE bench, strategic
          axis), include a smoke-bench step BEFORE declaring audit
          complete, not after. Saves audit-blindspot-grounded LANDS
          that bench reveals as KILLED post-substrate.

26. **Smoke-test small-shape success ≠ production-shape success (capture-key combinatorics)**
    - Caught by: `a7a8b94` #37 Path B v1 KILL (2026-05-10). Path B device-memory
      `start_pos` smoke at shapes (page_indices_len=4, prefix_token_rows=3,
      paged_tokens=8) showed clean cache hits and dropped multi-key churn.
      But at production 4k/c=4 (page_indices_len up to 64+, prefix_token_rows
      up to 128+, varying per request), **388 unique capture keys appeared
      across a 60s window with 0% reuse** — exactly the same outcome as the
      Path A multi-key cache that had been killed two days earlier.
    - The smoke shapes accidentally collapsed the variation that production
      actually exhibits: a handful of fixed dim values can hit a 1-key cache
      trivially; a continuous distribution of dim values blows the key space
      out by orders of magnitude.
    - Fix: Smoke benches for **cache-hit-rate** claims must enumerate the
      production distribution of cache-key dims, not pick fixed test points.
      Either (a) sample real production traces for the dim distribution, or
      (b) parameterize the smoke over the full expected dim range and report
      key-cardinality + reuse-rate, not just functional correctness. **Cache
      claims need cardinality evidence, not hit/miss-on-one-shape evidence.**
    - Companion to anti-pattern #6 ("license on capture exists not capture
      reused"): same family of error — the shape under test must match the
      shape under license.

27. **Bucketing without scalar-capture sync (semantic cache miss disguised as functional cache hit)**
    - Caught by: `a56b7a9`/`c44788f` #40 Path B.2 wins (2026-05-10), specifically
      Codex's "second-order bucketing" insight beyond Claude's brief.
    - Setup: Path B.2 added bucketing to the cache key (`page_indices_len`
      rounded to 64, `prefix_token_rows_len` rounded to 128). Without the
      second-order step, bucketed-key collisions still produce semantic
      cache MISSES because the captured TileLang launch parameters
      (`total_pages`, `prefix_token_count`) were baked from the FIRST
      capture's exact dim, not the bucket capacity. Replay with a different
      dim within the same bucket fed an outdated scalar into the captured
      kernel — silent functional bug or wasted re-capture.
    - Fix: When bucketing a cache key, **every captured scalar that depends
      on the bucketed dim must use the bucket capacity, not the
      first-capture exact value**. Pad input vecs to bucket capacity, zero-fill
      the slack, assert capacity invariants at allocation site. Audit the
      capture site for "every place this dim flows in" — kernel scalars,
      grid dims, mask sizes, allocation sizes.
    - Evidence of magnitude: without second-order sync, Path B v1 had 388
      unique capture keys at 4k/c=4. With bucketing alone (first-order),
      keys collapse but kernels would replay with stale scalars. With
      second-order sync (Path B.2), 7 unique keys + 98.5% reuse + engine
      TTFT -92.5% + +632% throughput.
    - Methodology lesson: **the "obvious" first-order fix is usually only
      half the win** when the cache key is a derived shape and downstream
      capture absorbs other shape-derived values. Audit the full data flow
      from key → capture, not just the key itself.

28. **Hallucinated tool output overrides peer-agent investigation**
    - Caught by: `ee2c5b0` (2026-05-10) — Claude challenged codex's
      conclusion that `--max-waiting-requests` CLI flag does not exist
      in `infer/src/main.rs`. Claude cited a "grep result" claiming the
      flag existed at line 133. Codex (rightly) trusted Claude's
      "correction" and used `--cold-headroom 253` workaround instead.
      Two ticks later Claude audited codex's errors entry, re-ran the
      verification grep, and **direct re-verify proved codex CORRECT
      the whole time**: line 133 is `scheduler_mixed_policy`, no
      `max_waiting_requests` field anywhere, `git log -S` returns empty
      (string never existed in main.rs history).
    - Root cause: Claude trusted internal model recall of prior bash
      output over what tool actually returned. Either misread or
      pattern-completed expected lines from `--admission-policy` +
      `--cold-headroom` to "infer" `--max-waiting-requests` exists.
      Built a "correction" of peer agent on fabricated evidence.
    - **Real-world cost**: Two ticks of cooperative work proceeded with
      the wrong assumption. Claude then briefed codex on Layer 2 again
      with the same hallucinated flag, would have caused server start
      to fail with clap conflict if not caught by audit-of-audit.
    - Companion to anti-pattern #25 (hypothesis-context vs
      implementation-context mismatch — both about agent context drift)
      but distinct: #25 is "audit-chain shared blindspot", #28 is
      "agent fabricates evidence to override peer's correct
      conclusion". Empirical bench (#25's lesson) doesn't catch #28
      because the bench command itself is built on the fabrication.
    - Fix: When "correcting" a peer agent's claim about file contents,
      the correction MUST include a re-run of the verification command
      in the SAME response, with the literal raw output quoted —
      NOT a summary of memory. Stale memory of prior tool output is
      hypothesis, not evidence (per
      `feedback_first_principle_solid_or_deeper.md`). Tie-breaker for
      conflicting evidence (peer investigation vs your recall): a
      fresh tool invocation showing raw output that both agents can
      examine.
    - Also: the SUPERSEDED notice on the prior research entry is
      load-bearing — without it, future readers replay the same
      fabrication. Always update the originating doc when a later
      audit invalidates its claim.

29. **Default test fixtures may be known-broken — verify before relying on PASS/FAIL**
    - Caught by: `eb2b4b6` (2026-05-10) — codex's #36 PrefixAware
      Layer 2 greedy_consistency check. The default W4A8 test model
      at `infer/tests/greedy_consistency.rs:30` (`Qwen3-4B-W4A8-marlin`)
      is the naive max-scale checkpoint that's been known-broken
      since #25 W4A8 accuracy fix introduced the GPTQ-calibrated
      variant. Codex caught + overrode via
      `INFER_TEST_W4A8_MODEL_PATH=Qwen3-4B-GPTQ-W4A8-marlin` before
      relying on the gate.
    - Pattern: when running existing tests as a license/kill gate
      for a substrate change, the test's default fixture (model,
      dataset, config) may be a known-broken artifact retained for
      historical reasons. Test PASS doesn't necessarily mean
      substrate works; FAIL doesn't necessarily mean substrate broke.
    - Mitigation: before relying on a test's verdict, grep the test
      source for fixture defaults + cross-reference against project
      status (recent errors entries about that fixture). When in
      doubt, override via env var to use the production-canonical
      fixture.
    - Companion to #28 (verify raw output not memory recall): both
      are about VERIFYING the substrate of a claim before trusting
      it. #28 is "verify the file content"; #29 is "verify the test
      fixture matches what production uses".
    - **Evidence accretion (n=4 as of 2026-05-10 EOD+880)**:
      - n=1 `eb2b4b6` original codex W4A8 fixture override
      - n=2 codex's Task #48 (`8d1caad`) independent rediscovery via
        `git log -S 'test_w4a8_vs_bf16_token_diff'` — found 81b6481
        documenting "W4A8 substrate produces 100% garbage output";
        codex tightened gate from 25% → 1% AND changed default to
        qzeros-fixed `Qwen3-4B-GPTQ-W4A8-zpfix`
      - n=3 Claude's `be133f8` audit found same broken default
        constant duplicated in BOTH `e2e.rs:21` AND
        `greedy_consistency.rs:30` — "broken defaults may be
        DUPLICATED across test files via copy-paste constants"
      - n=4 `b956f3a` Claude itself committed the anti-pattern by
        substituting W4-hybrid-zpfix model into test designed for
        W4A8-marlin-class fixtures → 100% diff was test/fixture
        mismatch, not real bug. **Strengthens to "test-fixture
        compatibility check is the responsibility of whoever invokes
        the test, not just whoever wrote the test"**.
    - Universalized rule: applies to (a) test authors, (b) test
      consumers, (c) anyone substituting fixtures via env var
      override. Same pattern at all three layers.

30. **Commit-time worktree race in cooperative session — `git status` BEFORE commit, not just before add**
    - Caught by: `0d63a52` + `994a294` recovery + `ca09db0` discipline
      demonstration (2026-05-10). Claude's `09ae5a5` commit accidentally
      bundled codex's Substep 1.1 implementation (3 files +
      `marlin_dequant.cuh` + `marlin_kernel.cu` mod + wins entry) with
      Claude's unrelated `docs(research)` REVISION research entry.
    - Failure mode: between Claude's `git add docs/research/<my-file>.md`
      (1 file staged) and Claude's `git commit -m "..."`, codex's
      parallel `git add` (likely `git add -A`) staged its WIP files.
      Claude's `git commit` then captured the UNION of staged files
      (4 files, not 1). Claude's commit message described only the
      research entry, but the diff included codex's substantial
      Substep 1.1 substrate.
    - Result: codex's Phase 1.1 implementation shipped under Claude's
      "docs(research)" commit attribution. Required follow-up
      `0d63a52` errors entry + `994a294` build-restore (because the
      bundled rename `.h → .cuh` left `marlin_kernel.cu` with stale
      include).
    - Rule: `git status --short` BEFORE `git commit` (not just before
      `git add`). The window between `git add` and `git commit` is
      when codex's parallel staging can race in. Recipe:
      ```bash
      git status --short          # check 1 (before add — the usual)
      git add docs/my-file.md     # add my file
      git status --short          # check 2 (race window check) ← KEY
      git diff --cached --stat    # confirm staged set is what you intend
      git commit -m "..."         # safe scope
      ```
      `994a294` build-restore + `ca09db0` doc-sync demonstrated this
      discipline correctly (status BEFORE commit, single file confirmed
      via `--cached --stat`).
    - Companion to memory rule
      `feedback_git_status_before_commit_in_cooperative.md` (which
      already covered "before commit" but in different framing — now
      the rule is sharpened with the race-window evidence).

31. **ARLE surface claims need raw evidence in same response, even when not contesting peer**
    - Caught by: `d387b03` (2026-05-10, 4th hallucination this session) +
      `c3bb82b` (3rd hallucination this session). Original #28 rule said
      "verify raw output when contradicting peer", but Claude made
      multiple hallucinated claims about ARLE's surface WITHOUT contradicting
      anyone — Claude just confidently stated false claims based on
      memory recall.
      - 3rd hallucination: `4b30c15` claimed ARLE has `/health` endpoint
        in unstick brief → reality is `/healthz` + `/readyz` (k8s convention,
        verified `router.rs:68-69`)
      - 4th hallucination: `5bf0e20` claimed 2026-05-09 baseline-B5 was
        comparable to newdequant-r1 for Phase 1.1 Δ% computation → reality
        is different checkpoint variants (zpfix vs sym-g128, verified via
        raw `cat command.txt`)
    - Pattern: Claude's confident claim about ARLE/bench surface (CLI
      flags, file structure, kernel internals, HTTP routes, baseline
      checkpoint match, model variant) based on internal recall of
      "how things usually work" — without grepping the actual code/files.
      Each claim plausible but ARLE-specific reality differs.
    - Strengthened rule (extends #28): ANY claim about ARLE's surface
      MUST be backed by raw `grep`/`Read`/`cat` output IN THE SAME
      RESPONSE making the claim. Generic conventions don't apply —
      ARLE's implementation may differ. This applies to:
      - CLI flag existence + argument types
      - File/module structure + function signatures
      - Kernel internals (which kernel has what buffer, which arch tag)
      - HTTP route endpoints + serialization formats
      - Baseline checkpoint match (which model variant each bench used)
      - Scheduler config defaults (max_waiting, cold_headroom, etc.)
    - Failure mode is silent: hallucinated claims often cause peer
      agent to do unnecessary work (codex spent 8 min searching for
      a CLI flag that didn't exist; codex initially used wrong endpoint
      for readiness probe). Recovery cost averages 1-2 ticks per
      hallucination.

32. **Peer agent "Waiting >5min" with no observable progress warrants direct process-state verify**
    - Caught by: `4b30c15` (2026-05-10) — codex was Working on a
      `for i in $(seq 1 120); do curl -fsS .../v1/models; sleep 2; done`
      poll loop for 33+ minutes after the nohup'd server died
      immediately at startup. Server PID 1810426 left no log output
      (0-byte log file) + curl connection refused, but codex's tmux
      timer kept incrementing without making progress.
    - Pattern: when peer agent's terminal shows "Waiting for
      background terminal X minutes" with no observable forward
      progress (no new commits, no log growth, no GPU activity
      indicating the work is happening), don't trust the timer.
      Directly verify the underlying process state.
    - Mitigation: if peer "Waiting >5min", run as part of next-tick
      3-state scan:
      - `ps -p <PID>` → process alive?
      - `ls -la <log-file>` → log file growing?
      - `curl <expected-endpoint>` → service responding?
      If process is dead, send unstick brief proactively. This tick
      pattern recovered ~33min of codex bandwidth that would have
      been wedged indefinitely.
    - Companion to anti-pattern #28 + #31: all three are about
      VERIFYING substrate before trusting peer-agent state. #28 is
      "peer investigation might be wrong, but verify before
      correcting"; #31 is "your own claim about ARLE surface needs
      raw evidence"; #32 is "peer's progress timer needs raw
      evidence too".

13. **Codex review value-add IS load-bearing for non-trivial substrate
    (NOT formality)** — anti-pattern #33 v1.12.0
    - Caught by: `ace3cbe` 2026-05-10 PF8.3 substrate session — codex
      review caught 3 REAL bugs that build + clippy + greedy_consistency
      + e2e all PASSED:
      1. Parallel-M launch loop off-by-N (HIGH severity, untriggered
         by current test fixtures = anti-pattern #29 territory)
      2. max_par/lock workspace contract underrun (HIGH severity,
         likely manifests as Task #43 stack overflow under sustained load)
      3. Hybrid W4 graph capture vs PF8 scratch interaction (MEDIUM,
         performance not correctness)
    - All 3 require contextual understanding (loop accounting cross-line,
      cross-language workspace contract, cross-feature interaction)
      that linters/tests cannot provide.
    - Empirical evidence: 3 bugs / 1 diff / 27 min review = high
      amortized value
    - Strengthened rule: when build + clippy + tests all pass on a
      non-trivial diff (≥3 files OR FFI boundaries OR cross-feature
      interactions), run `codex review --uncommitted` BEFORE commit.
      Skip review only for ≤3-file mechanical changes.
    - Companion to #29: tests passing ≠ code correct; codex review
      provides another verification layer that catches what tests
      miss.

14. **greedy_consistency single-request PASS is NECESSARY but NOT
    SUFFICIENT for new GEMM kernel substrate** — anti-pattern #34 v1.12.0
    - Caught by: `0cde63d` 2026-05-10 PF8.3 RUNTIME KILL — kernel
      passed greedy_consistency at conc=1 (4.33s) and e2e PASSED, but
      ran 100% failure rate (101380/101380 with code 2) under
      sustained conc=4+ bench load. Confirmed by `57c37b5` H8 verify:
      kernel STILL works at conc=1 but fragments under sustained load.
    - Root cause: greedy_consistency runs single requests with small
      batches; failure modes specific to sustained load (allocator
      fragmentation, OOM under high concurrency, kernel resource
      exhaustion under burst) DON'T MANIFEST at conc=1.
    - Strengthened rule: PAIR greedy_consistency PASS with sustained-
      load bench (≥30s, multiple concurrencies 1+2+4) BEFORE declaring
      license. Substrate validation is incomplete without conc≥2
      sustained-load proof.
    - **Sub-rule (#34b)**: when bench reports "0 successful requests"
      OR "all-zero latency table", CHECK SERVER LOG FIRST before
      debugging bench tool. v3-v10 PF8.5 attempts wasted 30+ min
      on guidellm CLI quirks (PATH, --backend-kwargs, --outputs html,
      absolute path, pre-mkdir) when the actual issue was 100%
      kernel failure surfaced via server log line 627 (`prefill
      batch failed: gemm_w4_fp8_marlin_cuda failed with code 2`).
      First diagnostic step: `grep -c "failed with code" /tmp/<server>.log`
    - Companion to #29 + #33: #29 is "default test fixtures may be
      broken"; #33 is "tests passing ≠ code correct (codex review
      catches more)"; #34 is "tests passing AT one shape ≠ tests
      passing at all shapes (sustained-load bench catches more)".

15. **Warmup target shape budget must clamp to (effective workload
    shape budget × hardware headroom)** — anti-pattern #38 v1.13.0
    - Caught by: `b4a3c38` 2026-05-10 Task #35 cap=8 prefill warmup
      §6.8 + `182d67b` §6.13 + codex Task #35 commit `a2ad788`. Two
      independent evidence points reached n=2 graduation threshold:
      1. **n=1**: codex implementing Pass 3 cap=8 prefill warmup
         discovered `max_seq_len=512` (test config) vs
         `chunked_prefill_size=4096` (warmup target) → Pass 3
         attempted to warm shapes the test window could never reach.
         Fix: clamp warmup token cap to `effective_max_seq_len`.
      2. **n=2**: same Task #35 implementation, B=8 × 2048 tokens/row
         exceeds 16GB VRAM budget → Marlin scratch OOMs at maximum
         shape. Substrate gracefully falls back to 1024 tokens/row
         (good defensive design, no crash) but the warmup work for
         the impossible 2048 shape was wasted.
    - **Generalization**: warmup-based optimizations (graph capture,
      kernel JIT, allocator pre-warming) target a SET of shapes. The
      shape set must be:
      - **Reachable by the actual workload** (otherwise warmup is
        dead work — the larger shapes will never be hit).
      - **Within hardware budget** (otherwise warmup OOMs and either
        crashes OR triggers fallback that wastes the warmup).
    - **Detection rule**: at warmup-target-set declaration, compute
      max shape cost = (max_m × max_k × dtype_bytes + lockstep
      buffers). Compare to (model_max_seq_len) AND (free_VRAM_bytes
      × headroom_ratio, e.g. 0.7). If max shape exceeds either,
      either:
      - **(a) clamp**: reduce target set to fit constraints (per #38
        n=1 fix — `max_seq_len` cap)
      - **(b) graceful fallback**: detect OOM at runtime + adapt
        (per #38 n=2 fix — Marlin scratch OOM → 1024 tokens/row)
    - Both ARLE Pass 3 evidence cases independently arrived at one
      of these patterns (a) and (b) respectively. Future warmup
      substrate should adopt one explicitly, not rely on accident.
    - Companion to #34 (necessary-not-sufficient bench) + #37
      (multi-shape bench discipline): #34 is "single bench shape
      doesn't validate kernel"; #37 is "single bench shape doesn't
      validate substrate"; #38 is "single warmup shape budget
      doesn't validate against hardware constraints". All three are
      "single-X is necessary but not sufficient" patterns at
      different abstraction levels.

16. **Static code audit (grep + dispatch-trace) is hypothesis-grade
    evidence; behavioral A/B is ground truth — both required before
    designing or hypothesizing about substrate** — anti-pattern #36
    v1.14.0
    - Caught by:
      - **n=1**: `2cc608a` 2026-05-10 H1' design REVISION discovered
        `MarlinScratch` struct + `_with_scratch` variants ALREADY
        EXISTED in `linear.rs` (saved 40 LOC by reusing existing
        pattern). Cure: grep for variants before designing.
      - **n=2 (INVERSE direction)**: `e8b6b31` 2026-05-10 Task #43
        hypothesis OVERTURNED by behavioral A/B — Claude's `1ba06f0`
        dispatch-audit predicted Arm A (env=on) HEALTHY + Arm B
        (env=off) KILL; reality showed Arm A KILL with 36 OOM
        failures + Arm B HEALTHY. Static dispatch trace (linear.rs
        :2064-2095 + qwen3/forward.rs:312-313) was directionally
        wrong; root cause turned out to be persistent scratch +
        graph resources competing with KV cache, opposite of the
        per-call alloc fragmentation hypothesis.
    - **Generalization**: static code audit (grep + dispatch trace)
      shows STRUCTURE but not MEMORY/PERFORMANCE BEHAVIOR. Hypotheses
      derived from static audit need behavioral verification before
      designing or planning around them.
    - **Detection rule**: when proposing a substrate change OR
      hypothesizing root cause from code reading alone, run a cheap
      A/B (60s + 2 servers) FIRST to verify direction. If A/B
      contradicts the audit-derived hypothesis, the audit missed a
      load-bearing factor (memory budget, graph capture cost,
      timing, etc.). Don't commit to design or plan based on audit
      alone.
    - **Examples of audit-only-insufficient signals** (extend list as
      n+1 evidence accumulates):
      - Memory pressure interactions (Task #43 case)
      - Existing pattern duplication (H1' design case)
    - Companion to §0 SOLID rule 1 ("推断 ≠ SOLID") and rule 3
      ("混淆变量必须隔离"): #36 is the practical implementation —
      grep gives you the hypothesis, A/B gives you the evidence.
      Both are needed; either alone leads to either over-engineered
      designs (audit alone, ignoring existing patterns) or
      directionally wrong fixes (audit alone, ignoring memory
      behavior).

17. **Tasks closed `root cause TBD` need canary-grade regression test
    or tightened acceptance gate** — anti-pattern #35 v1.15.0
    - Caught by:
      - **n=1**: `e3e1ab5` 2026-05-10 W4A8-vs-BF16 84.4% diff flagged.
        Task #25 (W4A8 accuracy fix) was closed `root cause TBD` with
        the existing greedy_consistency gate at 25% — lenient enough
        that 84.4% slipped past unnoticed at closure. The real
        substrate-broken state was rediscovered ~weeks later when
        Claude's PF8 work loaded the same fixture default.
      - **n=2**: `81b6481` 2026-05-08 errors entry documents
        "W4A8 substrate produces 100% garbage output" — the broken
        state WAS already known and documented in errors/ TWO DAYS
        BEFORE Task #48 dispatch, but the 25% gate was loose enough
        that the test still passed, masking the canary signal.
        Documentation alone (without test-gate enforcement) does
        not prevent decay; the temporal gap (2 days between known-
        broken documentation and downstream Claude work loading the
        same fixture) underscores the silent-decay danger.
      - **n=3**: `8d1caad` codex Task #48 fix bundle TIGHTENED the
        gate from 25% → 1% AND swapped default fixture to
        qzeros-fixed checkpoint. The 1% gate IS the canary that
        would have caught Task #25's decay at original closure
        time. Per `b956f3a` Claude research: "codex's fix is
        stronger than I credited" — the gate tightening is the
        load-bearing canary mechanism, not just the fixture swap.
    - **Generalization**: when a task closes `root cause TBD`, the
      closure is OK only if accompanied by ONE of:
      - **(a) Tightened regression gate** at a threshold that would
        flag any substrate decay (Task #48 bundle: 25% → 1%).
      - **(b) Pinned numerical bench reference** future re-runs
        compare against (e.g. wins entry with σ-tight numbers).
      - **(c) Explicit "intentionally loose" annotation** with a
        named kill-condition + planned graduation date.
      Without (a)/(b)/(c), root-cause-TBD closures silently decay
      into substrate bugs that resurface as confusing failures
      months later when downstream work touches the same code path.
    - **Detection rule**: any commit that closes a task without
      explicit root cause MUST update either a test (preferred,
      enforces forever) OR a wins entry (enforces during next
      bench cycle) OR a documented error budget. Reviewers should
      block PRs that close issues without one of (a)/(b)/(c).
    - **Why this matters**: the deepest evidence here is that the
      broken state was BOTH documented in errors/ AND already in
      the test suite — but the test gate was loose enough that
      the test passed. Documentation without enforcement is
      necessary but not sufficient. The canary must be a
      machine-checked threshold, not a written claim.
    - Companion to #29 (default test fixtures may be broken) +
      #34 (tests passing ≠ code correct under load): #29 is
      "fixtures decay"; #34 is "load-shape coverage decays"; #35
      is "acceptance-threshold decay" — all three are forms of
      silent substrate-state decay that machine-checked canaries
      catch when written-claim documentation does not.

---

## Quick reference (cheat sheet)

```
Goal              ┐
Hardware sheet    │ Phase 1-2: target + constraints
                  │
Profile binding   ┐ Phase 3: ncu / nsys / bench wall-clock
                  │ Skip = repeat ±2% noise
                  │
Formula predict   ┐ Phase 4: hardware × workload → predicted Δ%
                  │ No formula = no run
                  │
Single A/B        ┐ Phase 5: 1 var, matched controls, σ < 5% n=3
Combo A/B         ┘
                  │
Tradeoff explicit ┐ Phase 7: every win names what was sacrificed
                  │ No tradeoff = noise / not extreme
                  │
License or KILL   ┘ Phase 8: σ-confident + tradeoff named, OR
                    documented errors entry with framing
```

ARLE-specific quick paths:

```bash
# Hardware: 4070 Ti SUPER (sm_89, 100 KB smem/SM, 706 TFLOPS FP8, 88.5 BF16)

# Wall-clock baseline
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh <label> --concurrencies 4 --max-seconds 120 \
  --warmup 10 --data 'prompt_tokens=4096,...,output_tokens=256,...'

# Kernel-internal
PATH=.venv/bin:$PATH scripts/profile_ncu_guidellm.sh <label> \
  --bench bench-output/<dir> --family attention --set full

# System-level dispatch
PATH=.venv/bin:$PATH scripts/profile_nsys_guidellm.sh <label> \
  --concurrencies 4 --max-seconds 60 --data '...'

# TileLang JIT smoke (no GPU, kernel-codegen sanity)
.venv/bin/python scripts/tilelang_jit_smoke.py

# Verify clean kernel changes via existing smoke + cargo
cargo build --release --features cuda 2>&1 | tail -8
cargo test --release --features cuda --test e2e
cargo test --release --features cuda --test greedy_consistency
```

---

## Related

- `CLAUDE.md` §Benchmarks (every runtime change → bench entry)
- `docs/bench-and-trace-spec.md` (§3 internal counters, §6 auto-iterate, §7 protocol rules)
- `docs/projects/2026-05-07-arle-master-strategy.md` §0.1 主战场 3 axis (agent + 量化 + 投机)
- `docs/plans/M_quant-fp8-w4-magnitude-path.md` (formula-driven quantization plan)
- `docs/experience/errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md` (Phase 0 KILL — single-bucket + envelope clamp + BF16 contamination)
- `docs/experience/errors/2026-05-08-e2-prefill-bn32-failed-kernel-time-not-binding.md` (E1/E2 KILL — kernel time not binding at sm_89 4k, TileLang 0.1.9 codegen rigidity)
- `847a132` AGENTS.md §0 SOLID rule 6 (framing 多角度交叉, wall-clock ground truth)
- `aa15bea` AGENTS.md §0 第一原则 SOLID
- Industry papers: FlashAttention (2307.08691), PagedAttention (2309.06180), Marlin (2408.11743), SmoothQuant (2211.10438), AWQ (2306.00978), Medusa (2401.10774), EAGLE (2401.15077)
- Cutlass: <https://github.com/NVIDIA/cutlass>
- TileLang: <https://github.com/tile-ai/tilelang>
- Triton (alternative DSL): <https://github.com/openai/triton>
- ncu reference: <https://docs.nvidia.com/nsight-compute/>
- nsys reference: <https://docs.nvidia.com/nsight-systems/>

---

## Recent skill version history

| Version | Date | Anti-patterns | Source commits |
|---|---|---:|---|
| v1.0.0 | 2026-04-XX | 8(initial) | initial creation |
| v1.1.0 | 2026-05-XX | 11 | added #9-11 |
| v1.2.0 | 2026-05-XX | 12 | added #12 (decode-vs-prefill duality) |
| v1.3.0 | 2026-05-XX | 13 | `faffcb0` added #13 (NULL elimination) |
| **v1.4.0** | **2026-05-08** | **14** | **`6c627c4` added #14 (upstream-data parser silent corruption per `5593865` qzeros bug)** |
| **v1.5.0** | **2026-05-08** | **17** | **`f05ea3a` added #15-17 from cap=8 chain** |
| **v1.5.1** | **2026-05-08** | **17(refined)** | **`9f65b4d` #17 workload-shape refinement per `063da81`** |
| **v1.6.0** | **2026-05-09** | **18** | **`125f795` added #18 Phase 0 substrate audit per `1217375` A1 audit + B3 Step 2 -30% scope** |
| **v1.7.0** | **2026-05-09** | **19** | **`c768b70` added #19 dispatch directive path verification per `8935851` index.md broken link + `de8b4dc` pickup queue stale path** |
| **v1.8.0** | **2026-05-09** | **25** | **(this commit) batch-added #20-25 from c20b1ce attribution + R4#6 KILL + recipe audit chain. Anti-pattern theme: "audit at every prescription layer including recipes themselves"; key lesson: empirical bench is truly orthogonal SOLID layer that catches what bidirectional code audits both miss (#25 evidence). Sources: `c076aae` #20 / `b55bfcd`+`af44efa` #21 (2 evidence points) / `919c0fb`+`8d91d20`+`3fea979` #22 / `156d2c2` #23 / `1ccb448` #24 / `fe9ea8a`+`3b9cc06` #25** |
| **v1.9.0** | **2026-05-10** | **27** | **(this commit) added #26-27 from #37 Path B v1 KILL → #40 Path B.2 wins chain. Theme: "cache-hit-rate claims need cardinality evidence, and bucketing fixes need second-order scalar-capture sync". Sources: `a7a8b94` #26 (Path B v1 388-key churn at 4k production despite shape-(4,3,8) smoke success) / `a56b7a9`+`c44788f` #27 (Codex's second-order bucketing insight beyond Claude brief: bucketed key + captured scalars baked at first-capture dim = semantic miss; bucketed key + captured scalars baked at bucket capacity = 98.5% reuse, engine TTFT -92.5%). Compound learning: the same Phase B family of optimization required two distinct anti-pattern lessons, one per KILL→WIN cycle.** |
| **v1.10.0** | **2026-05-10** | **28** | **(this commit) added #28 from `ee2c5b0` SOLID-critical hallucination chain. Theme: "agent fabrication overrides peer's correct conclusion when memory of prior tool output is trusted over fresh verification". Source: Claude challenged codex's correct claim that `--max-waiting-requests` CLI flag does not exist, cited fabricated grep evidence, codex (rightly) trusted the "correction" and used `--cold-headroom 253` workaround. Two ticks later audit-of-audit re-ran verification → direct evidence proved codex correct from start (`git log -S` shows string never existed in main.rs). Lesson distinct from #25 ("audit-chain shared blindspot"): #28 is "agent fabricates evidence", and empirical bench doesn't catch it because the bench command itself is built on the fabrication. Fix: when correcting peer agent file-content claim, MUST re-run verification in SAME response and quote raw output literally, NOT summarize memory.** |
| **v1.11.0** | **2026-05-10** | **32** | **(this commit) batch-added #29-32 from same-day cooperative discipline session. Theme: "verify substrate of EVERY claim, not just contested ones". Evidence chain: 4 hallucinations sedimented in single session (`0f4d0ae` CLI flag, `43bda9c` reduce buffer, `4b30c15` /health endpoint, `5bf0e20` baseline mismatch) + cooperative race in `0d63a52`/`994a294` recovery + 33min wedged poll in `4b30c15`. Sources: `eb2b4b6` #29 (default test fixture broken since #25, codex correctly overrode via env var) / `0d63a52`+`994a294`+`ca09db0` #30 (commit-time worktree race; status BEFORE commit not just before add) / `c3bb82b`+`d387b03` #31 (ARLE surface claims need raw evidence even when not contesting peer; 4 hallucination pattern caught by self-audit) / `4b30c15` #32 (peer "Waiting >5min" warrants direct ps/log/curl verify; recovered ~33min of codex bandwidth). Cumulative compound learning: `de36538` retrospective + `940f49e` self-implementation by Claude (PF8.1+2) demonstrated discipline working — cooperative pipeline recovers from individual mis-claims when each agent applies raw-evidence-required rule.** |
| **v1.12.0** | **2026-05-10** | **34** | **(this commit) added #33+#34 from PF8.3 substrate session evidence. Theme: "code-correct ≠ runtime-correct under load". Evidence chain: codex review caught 3 real bugs that all formal gates passed (`ace3cbe` parallel-M loop + max_par/lock workspace + graph capture interaction); PF8.3 RUNTIME KILL with 101380/101380 failures despite greedy_consistency PASS at conc=1 (`0cde63d` + `57c37b5` H8 verify). Sources: `ace3cbe` #33 (codex review IS load-bearing for non-trivial substrate, NOT formality; 3 bugs/27min review = high amortized value; required when build+clippy+tests pass on FFI/cross-feature/parallel logic diffs) / `0cde63d`+`57c37b5` #34 (greedy single-request PASS NECESSARY but NOT SUFFICIENT; pair with sustained-load bench at conc 1+2+4; sub-rule #34b: bench 0-success → CHECK SERVER LOG FIRST, wasted 30+min on guidellm CLI quirks when real cause was kernel 100% failure visible in /tmp/<server>.log). Cumulative compound learning: 7 hallucinations across this session + 3 codex-review bug catches + 1 RUNTIME KILL exposed by sustained load = code-correctness gates and runtime-correctness gates are SEPARATE concerns; both required for license-grade substrate.** |
| **v1.13.0** | **2026-05-10** | **35** | **(this commit) graduated #38 from candidate to canonical anti-pattern after n=2 evidence threshold reached in same Task #35 cap=8 prefill warmup implementation cycle (per `b4a3c38` §6.8 + `182d67b` §6.13 + codex commit `a2ad788`). Theme: "warmup target shape budget must clamp to (effective workload shape × hardware headroom)". Evidence chain: same Task #35 implementation independently discovered both failure modes — n=1 max_seq_len=512 vs chunked_prefill_size=4096 mismatch (Pass 3 warming unreachable shapes; codex applied cap fix); n=2 B=8 × 2048 tokens/row exceeds 16GB VRAM → Marlin scratch OOM (substrate gracefully falls back to 1024 tokens/row). Both n=1 and n=2 are within ONE substrate development cycle but with INDEPENDENT failure mechanisms (config-vs-config alignment vs hardware-vs-shape alignment) — this satisfies n=2 distinct-mechanism evidence threshold. Generalization: warmup-based optimizations target shape sets; the set must be (a) reachable by actual workload AND (b) within hardware budget. Detection rule + (a) clamp / (b) graceful fallback patterns documented. Companion to #34 (single-bench-shape) + #37 (multi-shape bench discipline) — all three are "single-X is necessary but not sufficient" patterns at different abstraction levels.** |
| **v1.15.0** | **2026-05-10** | **37** | **(this commit) graduated #35 (root-cause-TBD canary) from candidate to canonical after n=3 evidence reached. Theme: "Tasks closed `root cause TBD` decay into substrate bugs without machine-checked acceptance gates". Evidence chain: n=1 `e3e1ab5` Task #25 W4A8 closed root-cause-TBD with lenient 25% gate — 84.4% diff slipped past unnoticed; n=2 `81b6481` errors entry already documented "W4A8 substrate produces 100% garbage" but documentation alone (without test-gate enforcement) didn't prevent decay; n=3 `8d1caad` codex Task #48 fix TIGHTENED gate from 25% → 1% — the 1% gate IS the canary that would have caught Task #25's decay at closure (per `b956f3a` Claude research note). Generalization: closing root-cause-TBD requires (a) tightened gate / (b) pinned bench reference / (c) explicit "intentionally loose" annotation with named kill-condition. Documentation without enforcement is necessary but not sufficient. Companion to #29 (fixture decay) + #34 (load-shape coverage decay) — all three are silent substrate-state decay forms that machine-checked canaries catch when written-claim documentation does not.** |
| **v1.14.0** | **2026-05-10** | **36** | **(this commit) graduated #36 from candidate to canonical after n=2 evidence reached including INVERSE-direction case. Theme: "static code audit is hypothesis-grade evidence; behavioral A/B is ground truth — both required". Evidence chain: n=1 `2cc608a` H1' design REVISION (MarlinScratch already existed in linear.rs, grep for variants saved 40 LOC); n=2 INVERSE `e8b6b31` Task #43 hypothesis OVERTURNED by behavioral A/B (Claude's `1ba06f0` dispatch-audit predicted scratch path safer; reality showed scratch path KILLS with 36 OOM failures, eager fallback HEALTHY — opposite causal direction). The INVERSE n=2 case is especially load-bearing: static audit was directionally wrong, not just incomplete. Cure: cheap behavioral A/B FIRST before designing/planning around audit-derived hypotheses. Companion to §0 SOLID rule 1 (推断 ≠ SOLID) — #36 is the practical implementation: grep gives the hypothesis, A/B gives the evidence; either alone leads to either over-engineered designs (audit ignoring existing patterns) or directionally wrong fixes (audit ignoring memory/timing behavior).** |

Cumulative compound learning pattern:single-day cap=8 chain produced
3 anti-patterns(#15-17)+ 1 refinement via 6+ verification ticks。Each
verification added empirical evidence that compounded into rule
sophistication。Skill rules accumulate via empirical evidence,not
upfront design。

**v1.8.0 batch trigger evidence**:6 anti-patterns emerged from 2 parallel
audit cycles in single 24h cron-loop session(c20b1ce 30-stage main +
R4#6 7-stage orthogonal)。Both cycles closed via empirical evidence
(c20b1ce attribution corrected via 7-layer chain;R4#6 KILLED via 2
benches)。Anti-pattern #25 itself has 2 audit evidence points(`fe9ea8a`
preliminary + `3b9cc06` refined-gate-also-fails),demonstrating "bench
is truly orthogonal SOLID layer" empirically。

For future maintainers:when adding new anti-patterns,reference the
specific source commit + research entry that triggered the rule。
This preserves evidence trail and prevents rule drift。
