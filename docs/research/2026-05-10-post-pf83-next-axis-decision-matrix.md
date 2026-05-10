---
title: Post-PF8.3 next-axis decision matrix — concrete pickup paths for both PF8.5 LICENSE and KILL branches
date: 2026-05-10
type: research
status: forward-looking-decision-matrix-for-next-agent
---

# Post-PF8.3 next-axis decision matrix — concrete pickup paths for both PF8.5 LICENSE and KILL branches

> **2026-05-10 later update**: KILL-branch Medusa pickup references are
> superseded for Qwen3.5. The active next step is recurrent rollback
> design/prototype before Medusa heads/training.

> Written THIS tick while codex Working 35m 48s on PF8.3 substrate
> commit pre-checks (cargo check PASSED + clippy + targeted CUDA
> tests serial). Once codex commits substrate + PF8.5 e2e bench
> runs, the next-axis decision is binary. This entry documents
> concrete pickup paths for both branches so the next agent (codex
> reactivation OR fresh Claude) can act without re-litigating.

## §0 The decision point

Once codex's PF8.3 substrate commits + PF8.5 license sequence runs
(`scripts/pf83_license_sequence.sh` + license-or-kill review):

```
PF8.5 result
├── LICENSE (TTFT Δ ≥ -8% σ<5% n=3 + greedy PASS + PPL Δ% ≤ +1.0%)
│   └── PF8 chain DONE → pivot to next P0 axis
│
└── KILL (TTFT Δ < -3% OR ITL regression OR PPL Δ% > +5% OR greedy FAIL)
    └── PF8 chain CLOSED → errors entry → pivot to next P0 axis
```

In BOTH branches the next P0 axis is the same: **#28 Medusa**
(per `61c9666` architectural analysis, only remaining ITL win path
on sm_89 W4 decode). LICENSE adds prefill TTFT improvement on top
but doesn't address decode ITL.

## §1 LICENSE branch concrete pickup

### Step 1: Land PF8.3 wins entry consolidation

Codex's commit will include `docs/experience/wins/2026-05-10-pf83-w4-fp8-marlin-substrate.md`
(currently untracked). After codex commits, Claude/codex should:

- Verify entry follows the PF8.5 license sequence outcome
- Add Δ% measurements from `scripts/pf83_license_sequence.sh` output
- Cross-reference `aebd4a5` license matrix
- Update `docs/index.md` Last refreshed line with PF8.5 LICENSE result

### Step 2: Update PF8.4 dispatch from opt-in to default

Currently `INFER_MARLIN_W4_FP8_PREFILL=1` is opt-in (per
`db063ff`). After license decision:

- If σ<5% n=3 confirmed: flip default to enabled in
  `infer/src/ops/linear.rs:255+` `marlin_w4_fp8_prefill_enabled()`
- Add bench entry showing default-flip is safe per CLAUDE.md
  §Benchmarks "feature-flag default flips" require bench
- Estimate: ~10 LOC change + 1 bench run

### Step 3: Pivot to #28 Medusa Phase 1.A

Per `8735361` survey:
```bash
arle data download --repo lmsys/lmsys-chat-1m --file data.jsonl
```

Claude can trigger this directly (4 GB download, network-bound,
non-conflicting with other work). Updates Task #28 to in_progress
with concrete first artifact.

### Step 4: Plan Medusa Phase 1.B handoff to codex

Phase 1.B = train 4 Medusa heads × 1 week. Codex own. Claude
prepares:
- Training config skeleton (extends existing `crates/train/`
  infrastructure)
- Hand-off brief with dataset path, target model, training
  hyperparameters per `M_medusa-required-path.md`

## §2 KILL branch concrete pickup

### Step 1: Errors entry documenting KILL specifics

Critical: name WHICH gate failed and WHY. Possible failure modes:

| Gate | Failure mode | Root cause hypothesis |
|------|-------------|----------------------|
| greedy_consistency | output divergence | numerical: FP8 acts precision insufficient for hybrid kernel |
| PPL Δ% > +5% | accuracy break | per-channel act scales not preserved through prefill |
| TTFT Δ% < -3% | no improvement | bandwidth-bound: act-quant wastes time vs INT8 baseline |
| TTFT regression | slower than baseline | extra act-quant kernel + dispatch overhead exceed FP8 mma savings |
| ITL regression | decode side-effect | dispatch path leaked into decode (shouldn't happen per linear.rs:83 phase guard) |

Each failure mode points to different root-cause investigation.
Errors entry must specify which one.

### Step 2: PF8 chain rollback (NOT delete substrate)

Substrate stays in tree (vendored vLLM Apache-2.0, no obligation to
remove). Disable dispatch via:
- Default `INFER_MARLIN_W4_FP8_PREFILL=0` (already the default)
- Document in `docs/support-matrix.md` that PF8 path is opt-in and
  KILLed at op-point
- Keep PF8.4 dispatch enum + bail for future revisit

### Step 3: Pivot to #28 Medusa (same as LICENSE branch step 3)

PF8 chain not delivering doesn't change the architectural reality:
**Medusa is the only remaining ITL win path on sm_89 W4** per
`61c9666`. KILL just removes one prefill optimization — decode ITL
ceiling is unchanged.

## §3 Quantization research alternatives (P3 in 09ae5a5 priority)

If user wants to defer Medusa training (1 week cost):

### W3 quantization

- Direct weight footprint reduction: 4→3 bit = -25% memory
- Per `09ae5a5` revised priority: P3 (-25-50% ITL ceiling per quant level)
- Existing W3 substrate: TBD (need source survey)
- PPL gate methodology: same as PF8.3 PPL gate (eval_ppl.py KV-format
  axis adapts to quant axis)
- Effort: ~1 week scaffold + bench

### W2 quantization

- Even more aggressive: 4→2 bit = -50% memory
- Risk: PPL Δ likely large; needs SmoothQuant + AWQ-class methods
- Effort: ~2 weeks (research + impl + bench)
- Less ROI than Medusa per 61c9666 architectural analysis

### NVFP4 (sm_89 emulated)

- Per skill v1.11.0 §2 hardware traps: "sm_89 has no native FP4 mma —
  emulated FP4 on Ada is slower than W4 Marlin"
- KILL near-certain on sm_89; revisit on sm_100+
- NOT recommended

## §4 Scheduling/dispatch follow-ups (P2)

If quant axis exhausted + Medusa training too expensive:

### #35 cap=8 prefill warmup fix

- Plan exists: `docs/plans/M_warmup-prefill-pass-directive.md`
- Scope: ~100-150 LOC, 1 day
- Closes 76-92%/56% bimodal cap=8 issue per `641e9bf`
- Ready for codex pickup

### #30 Hybrid W4A16/W4A8 dispatch Phase 2-3

- Phase 1 LANDED via task #42 (`marlin_dequant.cuh`)
- Phase 2-3 substrate work TBD
- Lower P than #28 Medusa per current priority

### #43 Server stack overflow fix

- Bug logged this session: server crashes under sustained W4A16
  4k-token bench load
- Blocks long-running benches (PF8.5 n=3 sequence affected if bug
  triggers)
- Codex/Claude should investigate as separate workstream

## §5 Anti-pattern reminders (carry forward)

Per `b551bea` skill v1.11.0+ canonical:

- **#28**: tool-vs-peer-claim → re-run + raw quote in same response
- **#29**: default test fixtures may be broken (load-bearing this
  session — codex's catch saved false-license risk per
  `da45380`+`473081d`)
- **#30**: git status BEFORE commit (preserve cooperative isolation)
- **#31**: ANY ARLE/upstream surface claim needs raw evidence in
  same response (covers CLI flags, kernel internals, HTTP routes,
  baseline checkpoint match, model variants, bit-pack arithmetic,
  mma instruction shapes, model file locations, binary build dates)
- **#32**: peer "Waiting >5min" warrants direct ps/log/curl verify
  EXCEPT when narration shows command transitions (codex's narrated
  Working state at 35m+ this tick is not a wedge)

## §6 Cross-references

- `aebd4a5` PF8.3 PPL gate methodology (license matrix authoritative)
- `a66d99a` NEW prefill-only FP8 directive (PF8 chain definition)
- `61c9666` architectural analysis (FP8 wrong lever for decode, Medusa
  only path)
- `8735361` Medusa Phase 1.A pickup chain survey
- `aa9f72e` Machete framing canonical disambiguation
- `b551bea` skill v1.11.0+ anti-patterns
- `2c736d0` next-session pickup state (most recent state checkpoint)
- `M_medusa-required-path.md` Medusa Phase 1-3 plan
- Task #28 Medusa scaffold (codex own, blocked on training)
- Task #30 Hybrid W4A16/W4A8 substrate (P2 fallback)
- Task #35 cap=8 prefill warmup (small scope)
- Task #43 server stack overflow (bug fix)
- Task #44 PF8 chain in_progress

## §7 Status

Forward-looking decision matrix complete. BOTH PF8.5 LICENSE and
KILL branches converge on #28 Medusa as next P0 axis. Quantization
research (W3/W2) is P3 alternative if Medusa training cost
unacceptable. Scheduling/dispatch follow-ups (#35, #30, #43) are P2
parallel workstreams.

Next agent picking up after codex's PF8.3 commit can:
1. Read this entry (~5 min orientation)
2. Run `scripts/pf83_license_sequence.sh` (PF8.5 sequence)
3. Pick LICENSE or KILL branch from this entry
4. Execute concrete next-step from matched section

Per skill v1.11.0+ #28+#31: every claim grounded in cross-references
to existing entries (no new assertions about ARLE state — only
synthesis of prior raw-evidence-grounded findings).
