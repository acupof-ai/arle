# c20b1ce was NO-OP in production,not silent-fail — 919c0fb深化

> Per `919c0fb` 5-layer SOLID gap chain finding。**深一步**:919c0fb 正确
> 识别 incoherent code path,但 empirical impact analysis 缺一步 — 在
> production-default config(num_slots=prefill_cap=8)下,c20b1ce 是 **NO-OP**
> not silent-fail。改进归因因此完全错误。

## 919c0fb's claim

`c20b1ce` extended `let max_bs = num_slots.min(256)` →
`num_slots.max(prefill_cap).min(256)`。

919c0fb finding:
- If `max_bs > num_slots`(i.e. `prefill_cap > num_slots`),warmup loop's
  `alloc_tokens(slot, 1)` for slot ≥ num_slots fails silently → break
  'warmup → warmup actually covers only 0..num_slots
- 5-layer SOLID gap chain documented

## 深一步 — empirical impact triage

Production-default config(per `27fd5de validation`,entry
`2026-05-08-w3-c4-cap8-default-clean-100pct-tt-improved.md` line 23):
- `--num-slots 8`(server CLI)
- Qwen3-4B `max_concurrent_prefill_requests = Some(8)`

→ `num_slots = 8`,`prefill_cap = 8`
→ `max_bs = num_slots.max(prefill_cap).min(256) = max(8, 8).min(256) = 8`
→ Pre-c20b1ce: `max_bs = num_slots.min(256) = 8`
→ Post-c20b1ce: `max_bs = 8`

→ **SAME max_bs**!c20b1ce 在 production-default config 下是 **NO-OP**,
not silent-fail。

## Why this matters more than 919c0fb states

919c0fb 提出 c20b1ce 是 incoherent fix → "bimodal regression root cause
STILL unverified"。那是对的,但更 fundamental:

如果 c20b1ce 是 NO-OP,**76→92→100% turn success 改进必然来自 OTHER 代码改变**:
1. `12300c5` cap=8 default flip(单独可解释一些)
2. `19d12c2` related changes
3. Statistical variance smoothing(N 增加平滑 bimodal)
4. 其他 simultaneous changes(`infer_serve` runtime updates etc.)
5. Workload-shape interactions(prompt/token distribution)
6. Pure noise

→ **919c0fb 的 strategic conclusion 加倍 SOLID**:
- bimodal root cause unverified (919c0fb 已说)
- BUT ALSO bimodal MITIGATION attribution to c20b1ce 完全错(本 brief 加)
- → 76→100% improvement 可能来自 12300c5 alone,or 自然 variance,or
  其他原因 — 必须 controlled A/B 隔离

## 6-layer SOLID gap chain(extended from 919c0fb's 5)

| Layer | Claim | SOLID Verdict |
|-------|-------|---------------|
| 1 | c20b1ce extends max_bs to prefill_cap | code change is real,但 EFFECT depends on config |
| 2 | "Closes db20d34 H4 root cause" | INCOHERENT(per 919c0fb)|
| 3 | "76→92→100% turn success post-c20b1ce" | REAL improvement,但 ATTRIBUTION 错(本 brief)|
| 4 | "1fdd763 Phase 0 audit confirms decode warmed 0..8" | INHERITED gap layer 2 + new layer:didn't check num_slots vs prefill_cap relation |
| 5 | "c076aae audit-of-audit caught hypothesis-inheritance" | 部分对,但 focused on prefill GEMM routing,not the no-op realization |
| 6(NEW) | **In production num_slots=prefill_cap=8,c20b1ce is NO-OP** | **THIS BRIEF** |

## Implication for empirical claims

Documents that need re-attribution:
- `2026-05-08-warmup-fix-c20b1ce-verified-92pct-turn-success.md`:claims
  "Warmup fix c20b1ce empirically validated at cold-start" — invalid
  attribution per layer 6
- `2026-05-08-w3-c4-cap8-default-clean-100pct-tt-improved.md`:claims
  cap=8 default + c20b1ce gives 100% — c20b1ce contribution = 0 per
  num_slots=prefill_cap empirics
- `2026-05-08-cap8-chain-final-synthesis.md`:synthesis claiming bimodal
  closed via c20b1ce — based on attribution that's now invalid

These wins entries' OBSERVED data is real(N runs of bench numbers),but
the CAUSAL ATTRIBUTION to c20b1ce is unsupported。

→ Need controlled A/B:
- A:revert c20b1ce → run bench → measure
- B:keep c20b1ce → run bench → measure
- If A == B → c20b1ce contribution = 0(预期 per layer 6 analysis)
- If A < B → c20b1ce contribution real(matters for non-default configs
  where num_slots ≠ prefill_cap)

## Why 919c0fb still matters(not contradicted by this brief)

919c0fb's `5-layer SOLID gap chain` and "P0.3 directive update needed"
are STILL valid:
- Decode paths warmed for 0..num_slots only(true regardless of c20b1ce)
- P0.3 prefill warmup pass scope unchanged(prompt-length GEMM not
  decode batch-size GEMM)
- Anti-pattern #22 candidate "incoherent-fix masked by silent failure
  path" is real — even if it didn't trigger in production-default,it
  COULD trigger in custom configs(num_slots < prefill_cap)

This brief **extends** 919c0fb,doesn't contradict。Both findings hold:
1. c20b1ce code is incoherent(919c0fb)
2. c20b1ce empirical impact in production-default = 0(this brief)

→ Both feed P0.0 Phase 1 evidence decomposition priority。

## Skill v1.8.0 anti-pattern #22 refinement

`919c0fb` proposes:
> Anti-pattern #22 candidate:"Incoherent-fix masked by silent failure path"

**Refinement per layer 6**:
> Anti-pattern #22 expanded:"Incoherent-fix masked by silent failure
> path **OR by config coincidence rendering it NO-OP**"。Both forms
> cause 'fix shipped + downstream stability' attribution错误。Verify
> empirical impact via controlled A/B,not commit-message read。

This is broader than 919c0fb's framing because:
- Silent fail = code logic incoherent + runtime error suppressed
- NO-OP = code logic plausible but environment makes it identity transformation
- Both produce same wrong attribution outcome
- Both require same fix:**A/B controlled experiment vs trust commit message**

## §0 first principle escalation

919c0fb 已 escalated:"every fix claim itself must be license-or-kill
verified BY TRYING IT"。

**This brief further escalates**:license-or-kill must include
**empirical impact in target environment**,not just code-level
correctness。c20b1ce 通过 code review(改了正确的事:让 max_bs
respect prefill_cap)BUT 没有 empirical impact 在 production-default。

→ License criteria refinement:
1. Code logic correct(919c0fb's level)
2. Effect measurable in target environment(this brief's level)
3. Attribution validated by controlled A/B(meta-level)

Three levels,三个 license gate,each escalates §0 rigor。

## Cross-references

- `919c0fb` original 5-layer SOLID gap finding
- `c20b1ce` code change(num_slots.max(prefill_cap).min(256))
- `12300c5` cap=8 default flip(co-shipping with c20b1ce)
- `27fd5de` validation(--num-slots 8)
- Production wins entries citing c20b1ce as fix:
  - `wins/2026-05-08-warmup-fix-c20b1ce-verified-92pct-turn-success.md`
  - `wins/2026-05-08-w3-c4-cap8-default-clean-100pct-tt-improved.md`
- 6 research entries citing the bimodal-c20b1ce chain
- §0 first principle:CLAUDE.md "求真务实,追求极致"

## Status

Layer-6 SOLID gap codified。Compounds with 919c0fb to make P0.0 Phase 1
evidence decomposition非常 critical:

**Actionable**:
- Pre-Phase-1.A:run controlled A/B with c20b1ce reverted vs kept,
  fixed num_slots=8 prefill_cap=8 production-default
- If A == B → c20b1ce 是 NO-OP confirmed,P0.3 prefill warmup needs
  fresh root-cause hypothesis
- If A ≠ B → c20b1ce DOES affect prod somehow (maybe via subtle
  side-effect),investigate

**Anti-pattern #22 expanded** for skill v1.8.0:silent-fail OR config-
no-op,both produce wrong attribution。Both need controlled A/B to
license。

**Bidirectional audit cycle now 13 commits**:
1-9: prior audit cycle
10:  b85929b LANDS B3 Step 2
11:  b55bfcd recipe scoping fix
12:  153fd93 anti-pattern #21 codify
13:  919c0fb c20b1ce incoherent finding
14:  (this brief) c20b1ce no-op深化

Compounding rigor produces both **wrong claims caught** and **better
methodology codification**。
