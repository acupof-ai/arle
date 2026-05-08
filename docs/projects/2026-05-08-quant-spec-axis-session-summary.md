# 2026-05-08 quant + spec axis session — 14-commit consolidation

> Single-page reference for future agents / session restart. Today's work
> spans master strategy §6.1 5-cap moat capability axes 1 (W4 quant),
> 4 (spec-decode), and 5 (xgrammar plan).

## Production state at end-of-session

| Axis | Capability | Status | Production default | Reference |
|---|---|---|---|---|
| W4A16 weight | LICENSED | ✅ ITL 1.64× vs BF16 | **YES** (recommended decode) | `f6f3af3` |
| W4A8 weight | substrate landed but BROKEN | ⚠ "fast garbage" 100% token diff | NO (axis open) | `81b6481` |
| Spec-decode classical (4k random text) | KILL'd both setups | ❌ α=7-19% sub-license | NO (workload-dead) | `5f26675`, `3ac5f4d` |
| xgrammar FFI | plan only | ⏳ codex own ~400-600 LOC | NO (P1 future) | `3864751` |

## 14-commit timeline (chronological)

1. **`f6f3af3`** docs(wins): M_quant W4A16 Marlin bench on sm89 — **license fired** ITL 11.76 ms (1.64× vs BF16)
2. **`2853551`** docs(errors): R1-3 baseline correction — production Marlin actually 1.64×, not 1.06× (anti-pattern #8 forward-direction self-correction)
3. **`4571082`** Revert R4 #6 + errors entry — hybrid dispatch KILL HARD (+60.7% ITL regression at batch=4 decode; tensor-core dominance refutes hybrid)
4. **`d09480b`** Skill v1.2.0 → v1.3.0 — anti-pattern #12 hardened by R4 #6 KILL evidence (decode-vs-prefill duality NOT universal for tensor-core ops)
5. **`e61d26e`** W4A8 substrate LAND (mixed: TTFT -36% ✅, ITL +63% eager-mode penalty ⚠ — later reframed as "fast garbage")
6. **`62e75ee`** W4A8 graph capture hoist plan — codex own, ~200-400 LOC, gated on accuracy fix
7. **`81b6481`** W4A8 garbage gate — `test_w4a8_vs_bf16_token_diff` fails 100% (BF16: "Paris..." / W4A8: ".........11.1.11111111 baudaskan1...")
8. **`e20f24c`** Claude 5-hypothesis ranking for W4A8 bug
9. **`b65c8c6`** Codex H2 RULED OUT (s3 dtype consistent FP16 end-to-end, byte-level verification)
10. **`88dfafc`** Claude H4+H5 RULED OUT (kernel `dequant_per_group` SUB=0x64086408 = correct -8 offset; activation_quant.cu 59 LOC verified correct)
11. **`5f26675`** self-spec K=5 sparse-KV KILL — α≈7%, -73% tok/s
12. **`3864751`** xgrammar FFI scaffold plan — master §7.5 P1.2
13. **`e3ca4d8`** Codex H3 mechanism — INT8 vs FP16 mma fragment layout (70% confidence; W4A16 FP16 perms 32 bytes/thread vs W4A8 INT8 perms 16 bytes/thread)
14. **`3ac5f4d`** external draft Qwen3-0.6B K=5 KILL — α≈19%, -46% tok/s; second independent KILL evidence

## W4A8 hypothesis chain (5 → 1 narrowing)

| H | Description | Status | Source |
|---|---|---|---|
| H1 | quantize script scale-perm ordering | algebra OK, unit-test pending | `e20f24c` |
| H2 | s3 FP16/BF16 mismatch | RULED OUT | `b65c8c6` (codex) |
| **H3** | get_perms vs INT8 mma fragment layout | **PRIME (70%)** | `e3ca4d8` (codex) |
| H4 | int4 - 8 offset missing | RULED OUT | `88dfafc` (Claude SUB magic decode) |
| H5 | activation INT8 scale wrong | RULED OUT | `88dfafc` (Claude 59 LOC read) |

**Codex next step**: cherry-pick PR #31 INT8 perms verbatim; replace `/tmp/quantize_qwen3_w4a8.py::get_perms()` with INT8-layout version.

## Spec-decode axis status

**4k random text 4-conc workload** — CLOSED (workload-dead):

| Setup | α est | tok/s ratio | Verdict |
|---|---:|---:|---|
| self-spec K=5 sparse-KV | ~0.069 | 0.270 | KILL |
| ext-draft Qwen3-0.6B K=5 | ~0.187 | 0.535 | KILL |

**Open axis paths**:
- Agent W3/W4 structured workload (master §2.1 production shape) — untested; gated on master §7.1 P0.0 W3/W4 bench harness
- Long-ctx 32k+ self-spec — sparse-KV designed-for regime; untested
- Medusa multi-head — promotion above classical Leviathan recommended based on 2 KILL evidence

## Skill v1.3.0 anti-patterns codified

The session validated three anti-patterns:

- **#8 forward-direction**: production-default ≠ A/B baseline (R1 caught after `2853551` self-correction; isolation-motive callout in Phase 5)
- **#12 hardened**: decode-vs-prefill duality NOT universal — applies to non-tensor-core ops; Marlin W4 + cutlass GEMM stay on tensor-core path. R4 #6 KILL evidence at `4571082`.
- **#13 NULL = real elimination**: 4-tick W4A8 hypothesis chain narrowed 5 → 1 via cumulative ruled-out commits; spec-decode 2 KILLs at 4k closed workload while preserving axis for other shapes.

## Methodology validation

This session demonstrates **skill anti-pattern accumulation** working:
- 5 hypotheses → 1 prime suspect over 4 ticks
- 2 KILL'd workloads → axis preserved for other shapes
- Self-correction (R1 baseline mismatch) → skill v1.3.0 hardening
- W4A8 substrate "fast garbage" caught BY new test added this session — without test, false LAND would have shipped

Per skill rule #6 (License-or-kill σ < 5%): all KILLs this session had σ < 5% with single-arm conclusive.

## Open task tracker (end-of-session)

| Task | Status | Owner | Description |
|---|---|---|---|
| #24 | pending (blocked on #25) | codex | W4A8 graph capture hoist (~200-400 LOC) |
| #25 | pending | codex | W4A8 accuracy fix (cherry-pick PR #31 INT8 perms per `e3ca4d8`) |
| #26 | pending | codex | M_xgrammar FFI scaffold (~400-600 LOC, plan `3864751`) |
| #27 | completed (KILL) | Claude | M_spec ext-draft Qwen3-0.6B K=5 — α=19% sub-license |

## Cross-references (canonical)

- Master strategy §0.1 三 axis: agent + 量化 + 投机 (codebase truth)
- Master §6.1 5-cap moat: capabilities 1 (W4 quant), 4 (spec), 5 (xgrammar)
- M_quant master plan: `docs/plans/M_quant-fp8-w4-magnitude-path.md`
- M_quant W4A16 Marlin license: `docs/experience/wins/2026-05-08-m_quant-w4a16-marlin-bench.md` (`f6f3af3`)
- M_quant R4 #6 hybrid dispatch (KILL'd): `docs/plans/M_quant-marlin-round4-hybrid-dispatch.md`
- M_quant W4A8 production bench: `docs/plans/M_quant-w4a8-prod-bench.md` (`db573c5`)
- M_quant W4A8 graph capture hoist: `docs/plans/M_quant-w4a8-graph-capture-hoist.md` (`62e75ee`)
- M_quant KV W4A8 (codex orthogonal axis): `docs/plans/M_quant-kv-w4a8.md` (`1e713de`)
- M_spec classical bench plan: `docs/plans/M_spec-decode-classical-bench-first.md` (`5a3ff50`)
- M_xgrammar FFI plan: `docs/plans/M_xgrammar-ffi-scaffold.md` (`3864751`)
- Skill v1.3.0: `.claude/skills/kernel-optimization/SKILL.md` (`d09480b`)

## Next-session priority queue

1. **Codex pickup #25** (W4A8 accuracy via PR #31 perms cherry-pick) — unblocks #24 and W4A8 default-on flip
2. **Master §7.1 P0.0 W3/W4 bench harness** — gating for spec-decode axis re-test on production shape
3. **Codex pickup #26** (xgrammar FFI substrate)
4. **Long-ctx 32k spec-decode test** — sparse-KV designed-for regime
5. **DSV4 spec V4 / HD64 wiring** — codex own per master §7.2

## Rule for next session

- **Trust the hypothesis chain**: 4-tick narrowing from 5 to 1 is real progress, not "stuck on the bug". Codex H3 mechanism + perms cherry-pick is the unblock.
- **Don't reflexive-KILL on workload-specific failures**: 4k random text spec-decode dead does NOT close the spec axis. W3/W4 + long-ctx + Medusa remain.
- **W4A16 Marlin remains production decode default** until W4A8 fix; do not attempt premature default-on flip.
- **Apply skill v1.3.0 Phase 5 isolation-motive callout**: any `--kv-cache-dtype` override triggers matched-control double-check.
