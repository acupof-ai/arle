# Session summary EOD+189 — loop pause checkpoint

> 2026-05-10 — 用户 direction:"完成这次后先写个总结暂停所有的 loop吧"。
> 本 entry capture 此 cron-driven autonomous session 的最终 state,作为
> resume reference。**321 commits since 2026-05-08**,multi-axis multi-day
> autonomous progress。

## 战略 outcome — 力争世界第一 inference runtime

### LANDED milestones(performance)

| Axis | Commit | Impact | Notes |
|------|--------|--------|-------|
| 🎉 P1.0 hybrid W4A8 prefill dispatch | `9773904` | TTFT **-31.5%** env-gated `INFER_HYBRID_W4A8_PREFILL=1` | Substrate-existing,首先 ship 大头 |
| 🎉 P1.2 W4A8 Marlin scratch hoist for graph capture | `ca0673b` | ITL p50 **-13%** vs P1.0;P1.0 ITL tradeoff RESOLVED;out tok/s +17.5% | Decode-side hoist pattern |
| 🎉 #24 W4 prefill scratch hoist for opt-in graph capture | `35fc3cf` | Mirror P1.2 decode pattern → prefill side | Multi-key bucket cache prerequisite,#37 bench pending |
| 🎉 B3 Step 2 PrefixAwareAdmission | `b85929b` | Multi-tenant TTFT **-24.2%**(σ/mean=4.5%) | Default queue-bound preserved,opt-in `--admission-policy=prefix-aware` |
| 🎉 RoPE YARN scaling Phase 1+2 全部 LANDED | `e30bffe`/`0185f42`/`3027210`/`53e069e`/`d5f67b4`/`cb80829`/`da53d81` | 7 incremental commits "Phase X step Y" discipline | Long-ctx unblocker |
| 🎉 P0.0 Phase 1.B SGLang re-verify | `0969480` | Evidence-grade ARLE-vs-SGLang same-machine bench | Decision input for next strategic axis |

### KILLed hypotheses(empirical evidence)

3 consecutive ops-layer KILLs with distinct failure modes:

| KILL | Commit | Failure mode | Anti-pattern |
|------|--------|--------------|--------------|
| 🚫 P1.3 quantized fused_mlp | `edacfe7` | TTFT +7.3% regression — launch reduction at saturated kernel(cuBLAS autotune already optimal)| #25 production gate |
| 🚫 P1.4 TileLang FP8 decode wire | `51dd5b2` | Output garbage but greedy PASS — substrate semantic mismatch(scale layout / FP8 cast / dequant)| **#26 candidate same-output-but-garbage** |
| 🚫 P1.6 QKV projection packing | `4d5f870` | Flat -0.1% TTFT + r3 server failure(43 ok / 88 failed)— packed weight 缩 KV pool → c=4 4k prefix-cache pressure | **#27 candidate memory-cost-shadow** |

Plus W4A8 多 KILL + #38 mixed policy KILL — codex 自主 audit-stage KILL discipline strong。

### Strategic findings

- **ARLE 4k prefill +76.6% lag vs SGLang**(1639.3ms vs 928.4ms)— 真 weakness 暴露(despite P1.0 -31.5% improvement,绝对值仍差)
- **ARLE decode-dominant -64% to -71% lead vs SGLang**:c=1 13.2ms vs 36.1ms,c=4 32.6ms vs 111.0ms
- 用户 direction:**"算了混合用吧 DeepSeekv4也是混合用的"** — hybrid TileLang+custom+Marlin 是 industry-standard,**不 push 全面 TileLang 迁移**
- TileLang migration audit `9373aa2`:14.5K LOC custom CUDA(38 files),Tier 4 W4 Marlin asm 高风险 hand-tuned tensor-core(parity 未 proven)— hybrid validated by user

## In-flight work(暂停时 state)

### #37 multi-key bucket prefill graph cache
- #24 hard prerequisite LANDED `35fc3cf`
- Multi-key impl 完成(per `56b6355` re-scope)
- **Bench-only remaining** — wins entry template ready `1168381`
- Resume action:run scripts/bench_guidellm.sh + populate template + commit

### M_rope-yarn-scaling Phase 3
- Phase 1+2 complete(7 commits LANDED)
- Phase 3 CUDA-side bench plan ready `8466202`(no Mac needed)
- Phase 1c integration tests ready `894ae9e`
- Resume action:Phase 3 CUDA-side bench execution

### xgrammar-sys FFI(P1 parallel axis)
- 新 crate `crates/xgrammar-sys/` 启动
- `Cargo.toml` workspace member 已 add
- C API headers in flight:`arle_xgrammar_matcher_*`(fill_next_token_bitmask / accept_token / is_terminated)
- Per `e5a8378` strategic ranking:不阻塞 P0,structured-output latency feature value
- Resume action:complete C++ → Rust FFI binding + Rust integration

### Open backlog(deferred)

- **#21** TRT-LLM bench(deferred)
- **#24 substrate cleanup** observation 2026-05-14 due — mechanical cleanup ready per `4394899`
- **#32** M_spec Medusa(P2,wall-clock 2-3w + α prediction-only risk)
- **#33** M_quant KV W4A8(demoted post-nsys evidence)
- **Chunked prefill**:codex `e5a8378` strategic ranking 提议 P0,但 user "hybrid" 决策后 deprioritized

## Anti-pattern catalogue 进展

- ✅ #25 production-scale gate(P1.3 + P1.4 + P1.6 evidence,validated)
- 📋 #26 candidate same-output-but-garbage(P1.4 catch,need 2nd instance)
- 📋 #27 candidate memory-cost-shadow(P1.6 catch,need 2nd instance)
- Skill v1.9.0 等 #26+#27 second instance trigger codification

## 量化 progress

- **321 commits since 2026-05-08**
- ~107 commits this session(EOD+1 → EOD+189)
- 5 wins entries + multiple errors entries committed
- 4 docs/research/ entries(strategic synthesis / anti-pattern candidate / TileLang audit / cleanup audit)
- 5 plans/ entries(pickup queue refresh / Phase 0v3 validation / Phase 3 bench / TileLang audit)

## Resume conditions(when loop resumes)

1. **Read this entry first** for full state snapshot
2. **Read** `docs/index.md` Last refreshed line for canonical state pointer
3. **Read** `docs/plans/codex-pickup-queue-2026-05-09.md` for pickup ranking
4. **Check codex pane state**(tmux 0:0)
5. Tactical priorities by ROI:
   - (a) #37 bench A/B run + wins entry → license/kill verdict
   - (b) RoPE YARN Phase 3 CUDA bench
   - (c) xgrammar-sys FFI completion(if user resumes structured-output focus)
   - (d) #24 substrate cleanup post-2026-05-14 observation period

## Cross-references

All 6 canonical surfaces fresh:
- `docs/index.md` `b1062d7`
- Pickup queue `731573e`
- Strategic synthesis `8047072`(3 ops-layer KILLs analysis)
- Anti-pattern #26 research `2778dc8`
- Cleanup audit prep `4394899`
- TileLang migration audit `9373aa2`
- Test inline doc warning `c41198d`(greedy_consistency anti-pattern #26 reach radius)

## Loop pause directive

Per 用户 EOD+189:**"完成这次后先写个总结暂停所有的 loop吧"**

- Cron loop:用户 will stop their cron timer
- Memory:此 entry 是 final state checkpoint
- Codex 0:0 仍 Working on xgrammar-sys FFI — user to decide 是否 stop / let finish
- 任意 future cron tick fires before user stops:respond minimal-overhead memory-only,no new work dispatch
- Resume:user explicitly signals new direction → read this entry + canonical surfaces → reactivate

## §0 SOLID self-assessment

- ✅ Evidence-grade(P0.0 Phase 1.B same-machine N=3 paired SGLang re-verify)
- ✅ License-or-kill discipline(3 KILLs all reverted clean,errors entries committed)
- ✅ Anti-pattern catalogue (2 new candidates evidence captured for future codification)
- ✅ Hybrid strategy validated(user "DeepSeek V4 也是混合用的" reference)
- ⚠ #37 throughput bench 未 run — license/kill verdict pending

## Final state metrics

**Codebase health**:
- worktree clean ✓
- 6 canonical surfaces fresh ✓
- in-flight axes(xgrammar-sys / Phase 3 bench / #37 bench)清晰 documented
- Resume cost:最小 — 任意 surface trace-able 回 state evidence
