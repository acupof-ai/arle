# 2026-05-08 EOD+67 — Final state anchor for resumption

> Single-pane reference for tomorrow's pickup。Captures production-ready
> state,3-axis status,concrete pickup queue,decisions awaiting user,
> cumulative methodology rules from today's 67-tick cron+codex chain。

## Production-ready state(2026-05-08 EOD)

### Axis 1 — agent workload(master §2.1 真 agent battlefield)
- W3 c=4 short multiturn:✅ **LICENSED HARD**(`063da81` 384/384,**TTFT -45%**)
- W3 c=16 short multiturn:✅ LICENSED HARD(`27fd5de` 384/384)
- W4 c=8 8K agent burst:⚠ CONDITIONAL(bimodal 76-92%/56%,fix `56dbd1c` 1.5d codex)
- W4 c=16+ 8K:❌ blocked by KV memory(#33 KV W4A8 paired axis)

**Hot-path code on main**:`b708e00` admission deadlock fix + `12300c5` cap=8 + `c20b1ce` warmup decode-only。

### Axis 3 — weight quantization(master §1.2.1.A)
- BF16:✅ baseline
- W4A16 GPTQ-zpfix:✅ LICENSED post `2a3a6f0` qzeros `+1` fix(**ITL +54%**,11.73 ms)
- W4A8 GPTQ-Marlin re-pack:✅ prefill LICENSED(**TTFT -36%**,1632 ms),decode deferred
- FP8 weight:🔴 KILL(cuBLASLt smoke 1.88×,cutlass v2 deferred)
- TurboQuant:✅ production(legacy)

**Key fix**:`2a3a6f0` 1-line `+1` to `convert_gptq.py` qzeros decode resolved BOTH W4A16 marginal accuracy AND W4A8 garbage simultaneously。

### Axis 2 — speculative decoding(master §7.4)
- Classical:DEAD(4-KILL evidence per `aa00c6a`)
- Medusa Phase 0:✅ done(`afdddec`,scope -38%,8-9 days codex)
- Phase 1.A.1 alpaca smoke:🔧 wget workaround unblocks(`4b5bb91`)
- Phase 1.B `arle train medusa` CLI:⏳ pending codex pickup
- HF Hub library bug `da68b98`:workaround sustainable,fix P3

## Concrete pickup queue(priority order)

### P0 — Hybrid Phase 1b loader patch
- Plan:`6be30ce` directive,scope `9dc32d6` correction = **155-175 LOC across 2 files**(0.75-1 day codex)
- Phase 0 reconnaissance done(`1959a21`)+ Phase 1a checkpoint merge(`b6502f7`)
- Unblocks W4A16/W4A8 hybrid prefill-decode routing(-14% E2E predicted)

### P0' — M_warmup prefill pass
- Directive:`56dbd1c`,**1-1.5 days codex**(~150 LOC + N=3 validation)
- Closes cap=8 bimodal investigation(predicted 100% deterministic)
- Reuses existing `forward_prefill_batch_with_pool`

### P1 — B3 PrefixAwareAdmission
- Source:`a1965ab` SGLang multi-tenant 2× gap root cause
- ~350 LOC(200 policy + 50 wiring + 100 tests)
- Closes -50% TTFT on multi-tenant axis(157 ms target)

### P1 — KV W4A8(#33)
- Plan:`M_quant-kv-w4a8.md`(`1e713de`)
- 5-10 days new kernel `decode_attention_w4_a_fp8`
- Unblocks c=16 hybrid memory budget(per `1959a21` Phase 0)
- 4× KV pool capacity vs BF16,2× vs FP8/INT8

### P1' — Medusa Phase 1.B
- Phase 1.A directive:`b4ae33f`(dataset chosen `tatsu-lab/alpaca` smoke + `WizardLM_evol_instruct_70k` production)
- Phase 1.A.3 wget workaround:`4b5bb91`(52k samples / 2.3M tokens)
- Phase 1.B blocker:`arle train medusa` CLI doesn't exist yet(~150 LOC)
- 8-10 days codex per `afdddec`

## Decisions awaiting user

- D-Hybrid:**P0 Hybrid Phase 1b GREEN-LIGHT?**(per `9754aca` plan recommendation)
- D-Medusa:**start Phase 1.B impl now or wait for hybrid land?**
- D-WarmupFix:**P0' priority confirmation**(closes cap=8 bimodal,1.5d codex)
- D-B3:**P1 priority alongside hybrid**?(SGLang 2× gap closure)
- D-KV:**P1 alongside hybrid + Medusa**?(memory + c=16 unblock)

## Cumulative methodology rules(skill v1.3.0 → v1.5.1,17 anti-patterns)

Today's burst added 4 new anti-patterns:
- **#14** Upstream parser silent corruption(`6c627c4`,GPTQ qzeros `+1`)
- **#15** Warm-server implicit dependency trap(`f05ea3a`)
- **#16** Implicit-coupling-via-shared-default trap(`1f70059` grep evidence by example)
- **#17** Bimodal failure distribution masks single-run LICENSE(`f05ea3a`,refined `9f65b4d` workload-shape)

Plus methodology insights pending skill #18 candidate:
- Hypothesis priority bump without controlled experiment trap(`f7da3e1`)
- Bimodal investigation:go granular per-failure-event vs aggregate metrics(`641e9bf`)

## Cumulative loop value(2026-05-08 ~24h)

- **70+ commits** across 3 axes + methodology
- **5 production-ready features**:cap=8 W3 LICENSED,W4A16 +54%,W4A8 prefill -36%,scheduler deadlock fix,GPTQ qzeros fix
- **5+ codex+claude collaboration validated cycles**(30-min fix-validation tight loops)
- **0.5 day saved retrospectively**:had grep-evidence rule been applied earlier,cap=8 chain would have taken 1 tick instead of 7

## File reference index

### Plans(P0/P1 ready for codex pickup)
- `docs/plans/M_quant-w4a16-w4a8-hybrid-prefill-decode.md`(P0 hybrid)
- `docs/plans/M_quant-hybrid-phase1b-loader-directive.md`(P0 §2.1)
- `docs/plans/M_warmup-prefill-pass-directive.md`(P0' bimodal fix)
- `docs/plans/M_medusa-phase1a-dataset-directive.md`(P1' Medusa)
- `docs/plans/M_quant-autogptq-marlin-integration.md`(axis 3)
- `docs/plans/M_quant-kv-w4a8.md`(P1 KV)
- `docs/plans/codex-pickup-queue-2026-05-08.md`(meta)

### Master strategy
- `docs/projects/2026-05-07-arle-master-strategy.md`(updated `15f8964`)

### Methodology(skill catalog)
- `~/.claude/skills/kernel-opt/`(v1.5.1,17 anti-patterns)

### Open errors
- `docs/experience/errors/2026-05-08-w3-c16-arle-deadlock.md`(resolved by `b708e00`)
- `docs/experience/errors/2026-05-08-gptq-qzeros-off-by-one-broke-w4a8-source.md`(resolved by `2a3a6f0`)

## Status

**Codex idle since EOD+43**(`2a3a6f0` 47m work session,~3+ hours ago in cron-tick time)。Pickup queue prioritized,all directives concrete enough for direct execution。

PushNotifications sent at:
- EOD+37(W3/W4 admission unblock + W4A8 GPTQ landed)
- EOD+43(W4A8 greedy gate PASS + 9-iter chain closed)
- EOD+51(D1+D4 cap=8 single-line flip)

Memory file `codex_session_state.md` rolling per-tick。This anchor doc
provides stable reference for resumption。Tomorrow Claude / next codex
can read this single doc + memory tail for full context。

## Cross-references

- All 70+ commits referenced inline。
- Master strategy: [`docs/projects/2026-05-07-arle-master-strategy.md`](../projects/2026-05-07-arle-master-strategy.md)
- Codex pickup queue: [`docs/plans/codex-pickup-queue-2026-05-08.md`](../plans/codex-pickup-queue-2026-05-08.md)
- Today's wins entries:`docs/experience/wins/` 2026-05-08-*
- Today's errors entries:`docs/experience/errors/` 2026-05-08-*
- Today's research entries:`docs/research/` 2026-05-08-*(40+ briefs)
