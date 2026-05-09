# Ops-layer optimization roadmap post-P1.0/P1.2

> Per nsys evidence(`aaf0b55`)prefill 97% dominant + P1.0 LANDED `9773904`
> -31.5% TTFT,this brief identifies **next-priority ops-layer axes ranked
> by alignment with empirically-dominant phase + per-LOC ROI**。
>
> Codex currently on P1.2(W4A8 graph capture hoist)。This roadmap = **what
> to dispatch next after P1.2 lands**。

## Substrate gap audit(grep-verified)

| Op file / kernel | State | Wired? | Gap |
|---|---|---|---|
| `linear.rs::fused_mlp_into` BF16 | ✅ 1-launch fused | ✅ wired | none |
| `linear.rs::fused_mlp_into` quantized | ❌ **4-launch fallback** | ✅ wired | gate gemv + up gemv + silu_mul + down gemv,no fusion |
| `csrc/attention/decode_attention_quantized.cu` | ✅ kernel exists | ❌ **0 callers in Rust** | dead substrate |
| `csrc/attention/decode_attention_turboquant.cu` | ✅ kernel exists | ❌ **0 callers in Rust** | dead substrate |
| `csrc/attention/decode_attention_varlen_fp8.cu` | ✅ kernel exists | ❌ **0 callers in Rust** | dead substrate |
| `tilelang_batch_decode_paged_hd128_fp8` cubin | ✅ AOT-built `(32,8)` | ❌ **0 callers in Rust** | M_quant Phase 0 substrate dead |
| `tilelang_batch_decode/prefill_paged_hd64` | ✅ AOT-built `(16,1)` | ❌ **0 callers in Rust** | historical small-model substrate |

→ **3 dead attention kernels + 2 dead TileLang families + 1 fusion gap** in current ops layer。

## Priority ranking(by prefill-dominance alignment + per-LOC ROI)

### P1.3 — Quantized fused_mlp (HIGHEST prefill-aligned)

**Current**:`linear.rs:749-773` for quantized weights:
```rust
gemv(gate_proj, x, act);          // launch 1
gemv(up_proj, x, &mut up_out);    // launch 2
silu_mul_cuda(act, up_out, act);  // launch 3
gemv(down_proj, act_ref, out);    // launch 4
```

**Hypothesis**:fuse(W4 gate-up batched dequant-gemv + silu_mul fused output + W4 down dequant-gemv)→ **~2 launches**(gate-up combined + down)or even 1。

**Targets**:
- W4A16 Marlin path:exploit Marlin's gate+up combinable structure(both same input x)
- W4A8 path:hybrid path benefits since P1.0 prefill IS W4A8

**ROI estimate**:prefill MLP is ~30-40% of prefill compute(TTFT)。Saving 2-3 launch overheads per layer × 32 layers = significant accumulation。Per nsys evidence prefill 97% dominant,this directly attacks the dominant phase。

**Effort**:200-400 LOC(new fused W4 gate-up GEMV + silu_mul + W4 down GEMV path),~1-2 days

**License criterion**:TTFT prefill -5% to -15% improvement vs P1.0 baseline (1632ms)。

**Anti-pattern #25 prevention**:gate behind `INFER_QUANT_FUSED_MLP=1` env var initially,explicit prefill-only path(decode-batched fallback to current 4-launch),verify per-batch-size vs cuBLAS-style baseline。

### P1.4 — FP8 decode attention wire (decode bandwidth axis)

**Current**:`tilelang_batch_decode_paged_hd128_fp8_q32_kv8_run_cuda` cubin compiled but no caller in `infer/src/ops/attention.rs`。

**Hypothesis**:wire FP8 decode for FP8 KV-cache config(`--kv-cache-dtype fp8`)→ halve decode bandwidth + improve sustained tok/s。

**Targets**:Q heads=32 / KV heads=8 currently compiled config,covers Qwen3-4B GQA shape exactly。

**ROI estimate**:**less aligned with prefill 97% dominant evidence**(targets decode bandwidth not prefill compute)。But trivial wiring cost = high ROI per LOC even with smaller gain。

**Effort**:**~50-80 LOC**(add dispatch arm in `attention.rs` for FP8 decode + select cubin via `ffi::tilelang_batch_decode_paged_hd128_fp8_q32_kv8_run_cuda`)。~0.5d。

**License criterion**:decode tok/s preserved or +10% improvement;ITL ≤ +5% regression budget。No correctness regression(greedy_consistency PASS in FP8 KV mode)。

**Why NOT promoted to P1.0/P1.2 priority**:demoted by nsys evidence(decode 3% of active GPU per `aaf0b55`)。Only worth doing because cubin already exists and wiring is trivial。

### P1.5 — Quantized attention TileLang unification (long-term cleanup)

**Current**:3 dead kernels in `csrc/attention/`:
- `decode_attention_quantized.cu`
- `decode_attention_turboquant.cu`
- `decode_attention_varlen_fp8.cu`

→ Per "no half-states" rule,either revive(wire)or remove(cleanup)。

**Hypothesis A(revive)**:re-implement quantized decode attention via TileLang DSL,unify W4A8 KV + W4 weight + BF16 attention into single kernel family。Aligns with TileLang analysis Phase B/C strategy(state-space + MLA + quant attention layer)。

**Hypothesis B(cleanup)**:simply delete dead kernels,document KILL evidence,reduce maintenance burden。

**ROI**:hypothesis A could enable KV W4A8 #33 task at lower LOC cost(reuse TileLang prefill/decode infrastructure rather than write custom CUDA)。But KV W4A8 was demoted post-nsys evidence。

**Effort**:Hypothesis A = 800-1500 LOC + 1-2 weeks。Hypothesis B = 100-200 LOC delete + cleanup audit。

**Recommendation**:**Hypothesis B(cleanup)deferred to #24 substrate cleanup observation 2026-05-14**。Add these 3 files to cleanup候选 list。Hypothesis A reconsidered post-P1.2 if prefill-dominant axis saturates。

## Recommended sequence post-P1.2 LAND

| Phase | Item | Effort | Prefill-aligned? | Decision criterion |
|-------|------|-------:|-----------------:|--------------------|
| P1.3 | **Quantized fused_mlp**(gate-up combine + W4 down)| 200-400 LOC | ✅ HIGHEST | prefill TTFT -5% → license,KILL otherwise |
| P1.4 | **FP8 decode attention wire** | 50-80 LOC | ⚠ decode axis | trivial-LOC ROI even with small gain |
| P1.5 | **Cleanup dead attention kernels** | 100-200 LOC delete | n/a | per #24 observation 2026-05-14 |

**P1.3 is the clear next dispatch** post-P1.2 LAND:
- Aligned with nsys-evidenced prefill 97% dominance
- Targets the same axis P1.0 already proved valuable on(prefill compute)
- Builds on hybrid W4A8 substrate from P1.0(W4A8 prefill path is now the right home for fused MLP)
- Per-LOC ROI high(200-400 LOC for ~5-15% additional TTFT reduction)

## What's NOT recommended

- **❌ KV W4A8 #33**:per nsys decode 0.9% of active GPU,5-10 day commitment unwarranted
- **❌ Medusa #28**:targets per-token decode latency 3% of active GPU,15-25 day commitment risks no world-#1 needle move
- **❌ TileLang full ops migration**:per my earlier analysis,Marlin/cuBLAS/elementwise/sampling do NOT benefit;keep TileLang scope = attention + future MLA + state-space ops
- **❌ Speculative decoding axes**:already KILLED 4× per `2026-05-08-spec-decode-*` errors entries

## Cross-references

- `aaf0b55` nsys decomposition evidence(prefill 97% dominant)
- `9773904` P1.0 LANDED Hybrid Phase 2(-31.5% TTFT)
- `/tmp/codex-brief-p1.2-graph-capture-hoist.txt` P1.2 directive(in flight)
- `linear.rs:749-773` quantized fused_mlp fallback path(P1.3 target)
- `csrc/attention/*_quantized.cu` + `*_turboquant.cu` + `*_varlen_fp8.cu` dead substrates(P1.5 cleanup)
- `crates/cuda-kernels/src/ffi/attention.rs:703-735` FP8 decode FFI declared but uncalled(P1.4 wire)
- Skill v1.8.0 anti-patterns #20-25(applied throughout per audit discipline)
- §0 first principle(CLAUDE.md):empirical evidence > hypothesis

## Status

Ops-layer optimization roadmap post-P1.0/P1.2 codified。**P1.3 quantized fused_mlp is the empirically-strongest next dispatch**(prefill-aligned + reuses P1.0 hybrid substrate)。

Codex finishes P1.2 → audit-of-audit → if LANDED + ITL closes → P1.3 directive ready to paste-buffer。

§0 in action:**every axis pre-justified by nsys evidence + per-LOC ROI**,not hypothesis-grade。Anti-pattern #25 prevention applied:each candidate has explicit context-target subset gating(env var + phase + alignment check pattern from P1.0).
