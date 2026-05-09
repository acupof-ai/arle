# Anti-pattern #26 candidate — same-output-but-garbage

> 2026-05-09 EOD+149 — production-scale failure mode 发现于 P1.4 TileLang
> FP8 decode wire KILL(`51dd5b2`)。Skill v1.9.0 candidate(目前 1 个 catch,
> 需要 2-3 corroborating instances 才 codify)。

## Background

`greedy_consistency` test(`infer/tests/greedy_consistency.rs`)是 ARLE 既有
correctness gate,设计目的:catch B=1 vs B=N batched-decode divergence(2026-04
之前的 Track A 主要 bug surface)。

测试 invariant:
```
solo_output(prompt) == concurrent_output(prompt, batch)
```

具体 assertion:同一 prompt 在 B=1 单独跑 vs 与其他 prompts 并发 B=N 跑,
**输出 token sequence 必须严格相等**(greedy argmax stability)。

**注:greedy_consistency 不检查输出 quality**,只检查 solo vs concurrent 一致性。

## P1.4 三阶段 evidence

P1.4 wire TileLang FP8 decode cubin(env-gated `INFER_FP8_KV_DECODE=1`)对照 3
模式:

| Mode | greedy_solo_vs_concurrent | 输出 quality |
|------|---------------------------|-------------|
| BF16 KV default | ✅ PASS | ✅ 正常文本 |
| FP8 KV custom kernel | ✅ PASS | ✅ 正常文本 |
| **FP8 KV + INFER_FP8_KV_DECODE=1**(TileLang) | **✅ PASS** | **❌ 重复/乱码** |

第 3 模式是 anti-pattern #26 production-scale catch:**solo 与 concurrent
输出严格一致(greedy_consistency PASS),但两者一致地输出 garbage**(repeating
tokens,word salad)。

## Failure mode anatomy

`greedy_consistency` 是 **invariant under "both-wrong-same-way"**:
- 当 solo 和 concurrent 都执行同一 broken kernel
- 同样 broken 的 numerics 在两条 branch 都重现
- argmax 在 broken numerics 下产生**一致的** token sequence
- Test PASS — 因为 solo == concurrent
- 但 token sequence 本身是 garbage

具体到 P1.4 case:
- TileLang FP8 cubin 与 ARLE existing FP8 KV cache 的 scale layout / FP8 cast /
  dequant 语义不对齐
- Cubin 读 FP8 KV 时 dequant 错误 → attention output 错误
- 错误是**确定性**的(same broken kernel 同样的错)
- Solo 和 concurrent 都用同一 broken kernel → 输出相同 garbage
- greedy_consistency invariant 满足 → PASS

## 与 anti-pattern #25 关系

**Anti-pattern #25**(production-scale gate):equivalence test / microbench /
unit smoke 不能替代 production bench。Production bench(end-to-end serving
+ real workload)是 ultimate truth gate。

**Anti-pattern #26 candidate**:greedy_consistency(production-style integration
test)也不是 ultimate truth gate。它有 specific blind spot — 当 broken
kernel 是 deterministic broken,solo == concurrent invariant 仍满足。

**层级关系**:
- Unit / smoke / microbench → catch syntactic + small-scale numerical bugs
- Equivalence test → catch cross-path numerical drift
- greedy_consistency → catch B=1 vs B=N batched-decode divergence
- **(GAP)**Output quality test → catch deterministic-broken-but-consistent garbage
- Production bench → catch perf regression(but may miss output quality if model still emits valid-shape tokens)

P1.4 暴露了 "greedy_consistency → production bench" 之间的 quality blind spot。

## Mitigation 提议

**Option A — Output quality assertion**:
- Pair greedy_consistency 与 output-quality 检查
- 例如:assert output != all-same-token / output 包含某 reference subword /
  perplexity vs reference < threshold
- Effort:~50 LOC adding to existing `greedy_consistency.rs`
- Risk:reference output 可能 model-version-sensitive,需 model-specific 维护

**Option B — Golden output snapshot**:
- 对 fixed seed + fixed prompt 保存 golden output snapshot
- 任何 kernel/model 改动后 verify snapshot 不变
- Effort:~100 LOC + test_data/ snapshot files
- Risk:snapshot drift 不可避免(GPU determinism 难保证),需 tolerance band

**Option C — Perplexity gate**:
- 跑 small reference dataset(WikiText valid 100 samples),compute perplexity
- Assert perplexity < threshold(threshold from BF16 reference run)
- Effort:~150-200 LOC + reference perplexity baseline
- Risk:GPU expense per test run(每次 ~30s-1min)

**推荐**:Option A 作为 immediate gate(low LOC + low maintenance),
Option C 作为 release gate(higher quality bar,less frequent)。

## Codification criteria

Anti-pattern #26 候选,**未 codify into Skill v1.9.0**:
- 当前 1 个 production-scale catch(P1.4)
- Skill 通常需要 2-3 corroborating instances 才 codify(per Skill methodology)
- 等待 next instance:可能在 future quantization wire / new attention kernel
  wire / FP8 path 出现

监控触发条件:
- 任何新 attention kernel wire(尤其 quant + new precision)
- 任何 KV-cache format change
- 任何 dequant kernel 替换

如果再出现 greedy_consistency PASS but bench/output quality FAIL → 第 2 catch,
此时 codify into Skill v1.9.0 anti-pattern #26 + add Option A mitigation 到
canonical test infrastructure。

## Cross-references

- `51dd5b2` P1.4 KILL commit
- `docs/experience/errors/2026-05-09-p1.4-fp8-decode-tilelang-killed-output-degeneration.md`
  (P1.4 errors entry — primary evidence)
- `infer/tests/greedy_consistency.rs`(test file with the invariant)
- `2e21da1` ops-layer roadmap(P1.4 directive context)
- Anti-pattern #25 catalogue(`6ab2293` Skill v1.8.0 commit)
- `edacfe7` P1.3 KILL(P1.3 KILL was a different pattern — TTFT regression catches via
  production bench,not greedy_consistency blind spot)

## Status

**Evidence-grade**:1 production-scale catch(P1.4)。
**SOLID-grade**:80% — gap 是 codification timing(等 2-3 instances)。
**Action**:此 entry 作为 reference doc。如出现第 2 instance,trigger Skill
v1.9.0 update + canonical test infra change(Option A mitigation)。
