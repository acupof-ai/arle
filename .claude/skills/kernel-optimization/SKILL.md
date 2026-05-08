---
name: kernel-optimization
description: Use this skill when the user asks to optimize, tune, speed up, or improve the performance of a GPU/CPU kernel, operator (op), attention path, GEMM call, decode/prefill path, quantization op, scheduler hot path, or any "make this faster" / "reduce ITL/TTFT" / "lower memory" / "拉满 utilization" / "调 kernel" / "优化算子" request. Captures the methodology — formula-predict → measure binding constraint → single-variable A/B with matched controls → combinational A/B when interactions suspected → tradeoff explicit (no tradeoff = not at extremes) → license-or-kill — and an industry-reference catalog (FlashAttention, cutlass, Marlin, SGLang, vLLM, TileLang, ncu/nsys methodology) so each attempt is grounded, not hand-waved.
version: 1.3.0
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
