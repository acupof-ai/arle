---
title: PF8.5 license blocker CANCELLED — hybrid W4 marlin checkpoint EXISTS at infer/models/ (Claude hallucination #7 — wrong dir checked)
date: 2026-05-10
type: research
status: pf8.5-blocker-cancelled-supersedes-da45380
---

# PF8.5 license blocker CANCELLED — hybrid W4 marlin checkpoint EXISTS at infer/models/ (Claude hallucination #7 — wrong dir checked)

> Codex this tick re-ran greedy_consistency with
> `INFER_TEST_W4A8_MODEL_PATH=/home/ckl/projects/arle/infer/models/Qwen3-4B-W4-hybrid-zpfix`
> + `INFER_MARLIN_W4_FP8_PREFILL=1` → **PASSED in 4.33s**. Independent
> Claude verification THIS tick: the hybrid checkpoint EXISTS at
> `infer/models/Qwen3-4B-W4-hybrid-zpfix/` (4.5 GB, `quant_type:
> marlin_w4_hybrid`, created 2026-05-08).
>
> **My da45380 entry's "PF8.5 license blocker — hybrid checkpoint
> missing" claim is WRONG.** This is **hallucination #7** this
> session: I checked top-level `models/` (returned empty for hybrid)
> but didn't check `infer/models/` (CARGO_MANIFEST_DIR convention).

## §0 Direct evidence (raw verification THIS tick)

### Hybrid checkpoint exists (raw ls)

```bash
$ ls -la /home/ckl/projects/arle/infer/models/Qwen3-4B-W4-hybrid-zpfix/
drwxr-xr-x 1 ckl ckl   290  5月 8日 18:57 .
-rw-r--r-- 1 ckl ckl  1633  5月 8日 18:57 config.json
-rw-r--r-- 1 ckl ckl 4529756920  5月 8日 18:57 model.safetensors
-rw-r--r-- 1 ckl ckl       707  5月 8日 10:42 added_tokens.json
... (full tokenizer + config files)
```

4.5 GB safetensors + complete config + tokenizer. Created 2026-05-08
(2 days before this session).

### config.json confirms hybrid quant type

```bash
$ grep "quant" /home/ckl/projects/arle/infer/models/Qwen3-4B-W4-hybrid-zpfix/config.json
    "quant_type": "marlin_w4_hybrid",
```

Matches `weight_loader.rs:539` mapping that triggers
`from_hybrid_w4_marlin` loader path → `hybrid_w4_fp8_qweight`
sidecar populated → PF8 dispatch eligible.

### codex's greedy_consistency PF8+hybrid run

Per tmux capture THIS tick:

```
INFER_TEST_W4A8_MODEL_PATH=/home/ckl/projects/arle/infer/models/Qwen3-4B-W4-hybrid-zpfix \
  cargo test ... greedy_consistency test_greedy_w4a8_marlin_optional ...

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 4.33s
```

**PASSED with hybrid checkpoint loaded** (4.33s, faster than the
7.04s W4A8-only run because hybrid loader does PF8.2 preprocess at
load → less work per step OR same work with cached kernel).

Codex's narration: "PF8+hybrid env 组合下 targeted greedy 已通过，
且没有再触发 hybrid gate failure。为了避免'测试通过但没走新分支'的
假阳性，我会再从代码和必要的轻量日志入口确认实际 dispatch 条件，不做
长 bench。"

Codex is NOW auditing whether dispatch was actually triggered, per
anti-pattern #29 discipline (test passing ≠ path exercised).

## §1 Hallucination #7 this session

| # | Tick | Claim | Reality | Caught by |
|---|------|-------|---------|-----------|
| 1 | `0f4d0ae` | --max-waiting-requests CLI flag exists | Never existed | codex |
| 2 | `43bda9c` | W4A16 has max_par×64×n reduce buffer | W4A8 has it not W4A16 | codex |
| 3 | `4b30c15` | ARLE has /health endpoint | /healthz+/readyz only | self via router.rs grep |
| 4 | `5bf0e20` | 2026-05-09 baseline-B5 comparable to newdequant | Different checkpoint variants | self via raw command.txt |
| 5 | `451d094` | bit-pack `0x76543210 → 0xFEDCBA98` | Actually `→ 0x89ABCDEF` (LSB→MSB) | empirical smoke run |
| 6 | `818b4e0` | FP8 mma is uniformly m16n8k32 | Has BOTH m16n8k16 AND m16n8k32 | raw grep on vllm marlin_mma.h |
| 7 | **THIS TICK** | **No hybrid Qwen3 checkpoint locally** | **EXISTS at `infer/models/Qwen3-4B-W4-hybrid-zpfix`** | **codex's actual test run + raw ls verification** |

### Common-mode pattern strengthening

I checked `ls /home/ckl/projects/arle/models/` (top-level) — empty
for hybrid. Concluded "no hybrid checkpoint locally". DID NOT check
`infer/models/` despite knowing the test convention `CARGO_MANIFEST_DIR
+ "/models/..."` makes the canonical path `infer/models/...` (not
top-level).

The greedy_consistency.rs:30 const I quoted in da45380 §2 LITERALLY
said:

```rust
const W4A8_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B-W4A8-marlin");
```

CARGO_MANIFEST_DIR for `infer/tests/greedy_consistency.rs` is
`infer/` — so the path expands to `infer/models/Qwen3-4B-W4A8-marlin`,
NOT top-level `models/`. I had the evidence right in the entry I
wrote and STILL checked the wrong directory.

This is an even more egregious version of #6 (claim contradicts
evidence I myself just cited). Per skill v1.11.0+ #28+#31
strengthening: **citing evidence is not the same as following it**.
Apply the conventions you document.

## §2 Updated PF8.5 status (correcting da45380)

| Phase | Status | Evidence |
|-------|--------|----------|
| PF8.1 act quant | LANDED + smoke PASS | `940f49e` + `b628eca` |
| PF8.2 weight preprocess | LANDED + smoke PASS | `940f49e` + `451d094` |
| PF8.3 GEMM substrate | COMPILE+CHECK+CLIPPY PASS | codex untracked marlin_pf8/ + marlin_w4_fp8_kernel.cu |
| PF8.3 FFI integration | DONE (untracked) | gemm.rs + tensor.rs + linear.rs codex diffs |
| PF8.3 hybrid loader | DONE (auto-PF8.2 at load) | tensor.rs:869-887 |
| PF8.3 greedy_consistency | **PASSED on hybrid checkpoint** (4.33s) | THIS TICK codex run |
| PF8.3 dispatch audit | IN PROGRESS (codex Working 20m) | per anti-pattern #29 discipline |
| PF8.4 dispatch enum + env | LANDED (opt-in stub) | `db063ff` |
| PF8.5 prep tooling | LANDED | `3fa5e74` + `84d61eb` + `c382fba` |
| PF8.5 e2e bench | READY (use INFER_TEST_W4A8_MODEL_PATH=infer/models/Qwen3-4B-W4-hybrid-zpfix) | this entry |
| **Task #45 converter** | **CANCELLED** (hybrid checkpoint already exists) | this entry |

## §3 Updated PF8.5 invocation

For codex/Claude to run PF8.5 license sequence:

```bash
# Use the hybrid checkpoint (NOT the W4A8-only one)
export INFER_TEST_W4A8_MODEL_PATH=/home/ckl/projects/arle/infer/models/Qwen3-4B-W4-hybrid-zpfix
export MODEL=/home/ckl/projects/arle/infer/models/Qwen3-4B-W4-hybrid-zpfix

# Run full license sequence
scripts/pf83_license_sequence.sh           # full preset
# or
scripts/pf83_license_sequence.sh --quick   # ~2-min triage
```

The 3 scripts I landed THIS session (3fa5e74 eval_ppl_pf83.py,
84d61eb bench_pf83_ab.sh, c382fba pf83_license_sequence.sh) all
take MODEL env var or default `models/Qwen3-4B-W4A8-marlin` — should
be updated to reference `infer/models/Qwen3-4B-W4-hybrid-zpfix` for
PF8.5 to actually exercise the new path.

This is a follow-up improvement: amend the 3 scripts to default to
hybrid checkpoint location OR document the explicit env var
requirement. NOT a blocker — just an ergonomics improvement.

## §4 Cross-references

- `da45380` (SUPERSEDED — claimed hybrid missing, was wrong)
- Task #45 — CANCELLED (no converter needed)
- `b628eca` (PF8.1 runtime smoke PASS — pattern of empirical verification I should have applied to my own claim)
- `451d094` (anti-pattern #28 strengthening — empirical smoke caught hallucination)
- `b551bea` (skill v1.11.0+ #28-#32 anti-patterns)
- `infer/models/Qwen3-4B-W4-hybrid-zpfix/config.json` (`"quant_type": "marlin_w4_hybrid"`)
- `infer/tests/greedy_consistency.rs:30` (CARGO_MANIFEST_DIR convention)

## §5 Status

PF8.5 license blocker CANCELLED. Hybrid checkpoint exists locally,
2 days old. Codex's PF8.3 work has full path to license decision
without converter prep work.

Hallucination #7 catalogued — common-mode of "cite evidence then
apply wrong conventions". Skill v1.11.0+ #28+#31 strengthened
implicitly: when you cite a path convention, follow it before
checking adjacent locations.

Codex's exemplary behavior (find checkpoint + re-run + audit
dispatch per anti-pattern #29) demonstrates correct discipline
under same uncertainty Claude failed under.

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(infer/models/ ls + config.json grep + codex tmux capture, all
THIS tick).
