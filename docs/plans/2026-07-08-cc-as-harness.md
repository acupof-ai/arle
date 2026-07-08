# Claude Code as the agentic harness

> Status: Active — proposed 2026-07-08. Replaces the `terminus` agent for both
> capability eval and agentic-OPD rollouts.

**Verdict.** Retire `terminus`. Drive our Qwen3.6-27B (via ARLE serve's existing
Anthropic Messages endpoint) with the `claude` CLI running headless. Claude
Code's native structured tool-use replaces terminus's brittle text-format parser,
which is what masks the model's true capability today. Zero shim: CC speaks the
Anthropic API, `crates/infer-server/src/anthropic.rs` serves it, done.

## Why — the measured problem

The pass rate is not a capability read; it is a harness-noise read. On the best
real run (agentic-OPD round 2, 13 tasks ×3 = 39 rows, 20 resolved = 51.3%; 57.6%
over the post-security-deletion set), **14 of 19 unresolved rows are harness/infra,
only 5 are genuine capability**:

| failure_mode | what it is | CC harness fixes? |
|---|---|---|
| `fatal_llm_parse_error` (largest bucket) | terminus's custom text/XML tool-call parser rejects the model's output | **Yes** — CC uses structured Anthropic tool-use → OpenAI `tool_calls`, no regex parse |
| `parse_error` | same family | **Yes** |
| `unknown_agent_error` | the terminus research-code agent crashed | **Likely** — CC is a hardened production CLI |
| `agent_timeout` | agent burned the wall clock (the turns-to-first-edit wall) | **Partial** — CC's loop is more turn-efficient; the model's own latency is unchanged |
| `test_timeout` | the task's tests are slow / infra | No — independent of the agent |
| `unset` | genuine capability fail (tests ran, failed) | No — it's the model, but a clean harness lets true capability show |

CC directly removes the three largest buckets. `fatal_llm_parse_error` alone is
the single biggest capability-masking loss across every run.

## Architecture — zero shim

```
claude (headless) ──ANTHROPIC_BASE_URL──▶ ARLE serve (anthropic.rs) ──▶ Qwen3.6-27B-FP8
        │  native structured tool-use       │  tools/tool_use/tool_result/stream    │
        └── session.jsonl ──arle train cc-convert──▶ masked-CE training records
```

`anthropic.rs` already maps Anthropic `tools`/`tool_use`/`tool_result`/
`tool_choice`/`stream`/`thinking` to the internal OpenAI request
(`to_chat_request`, `fan_out_blocks`). CC points straight at it.

## Wiring

### 1. Serve — exposes both `/v1` and the Anthropic endpoint
```bash
CUDA_VISIBLE_DEVICES=0 arle serve --model-path /host/Qwen3.6-27B-FP8 \
  --bind 0.0.0.0 --port 8000 --max-running-requests 4 \
  [--lora-adapters <opd_adapter> --lora-alpha 32]
```
`--max-running-requests >= 2` is required — CC fires concurrent requests.

### 2. CC self-hosted, offline config — the crux
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=dummy                  # bypass real Anthropic auth
export ANTHROPIC_MODEL=Qwen3.6-27B-FP8              # main model = ours
export ANTHROPIC_SMALL_FAST_MODEL=Qwen3.6-27B-FP8  # CC's helper calls stay on-box, no egress
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_TELEMETRY=1 DISABLE_AUTOUPDATER=1 DISABLE_ERROR_REPORTING=1
# one-time: ~/.claude/settings.json = {"hasCompletedOnboarding": true}  (skip headless onboarding)
```

### 3. Per-task headless call — inside the sandbox, no egress, no prompts
```bash
cd $workdir
claude -p "$INSTRUCTION" \
  --model "$ANTHROPIC_MODEL" --max-turns 40 \
  --allowedTools Bash Read Write Edit Grep Glob \
  --dangerously-skip-permissions \
  --output-format stream-json --verbose > $workdir/.cc_session.jsonl
```
`stream-json` emits the full message stream (incl. `tool_use`/`tool_result`) —
exactly what `cc-convert` parses.

### 4. One agent, two scorers (task set differs, agent identical)
- **SWE-Pro** (OPD substrate): `scripts/cc_swe_baseline.py` already IS this —
  boot the staged tree → `claude` headless → `sandbox.rs::score_workdir`
  (non-empty `git diff` ∧ `test_patch` + `pytest <fail_to_pass>` exit 0).
- **terminal-bench-core** (68 benign tasks, general eval): `tb run --agent
  claude-code -k api_base=… --model …` — tb's built-in claude-code adapter keeps
  tb's per-task test scoring. If the pinned tb version lacks the adapter, a ~20-line
  `AbstractInstalledAgent` subclass that shells the call in §3 covers it.

### 5. OPD loop — the smooth finish
```
serve(LoRA) → cc_swe_baseline over the train set → collect passing .cc_session.jsonl
            → arle train cc-convert → masked-CE → new LoRA → repeat
```
`cc-convert` (offline entry referenced at `anthropic.rs:245`) replaces
`terminus_to_records.py`. The whole agentic-OPD loop moves from research-code
terminus to the production CC loop — the harness stops being a confound in the
capability-lift measurement.

## Retire (deletion-style)
`terminus` agent invocations, `scripts/terminus_to_records.py`, the litellm
`-a terminus -m openai/…` path in `terminal_bench_eval.sh`, and the terminus
calls in `tbench_*.sh` — all replaced by `--agent claude-code` or
`cc_swe_baseline.py`.

## The one knob to watch (a property, not a gate)
Whether the model's tool calls parse cleanly at the `anthropic.rs` → OpenAI
mapping. If Qwen occasionally emits malformed structure, CC **retries
automatically** (native production-loop behavior terminus lacks) — itself an
advantage over terminus. If it genuinely breaks, add grammar-constrained tool
decode on the ARLE side; CC's retry likely absorbs it first.

## Already built vs. missing
- **Built (~80%):** the `anthropic.rs` Messages endpoint, `cc_swe_baseline.py`,
  `arle train cc-convert`, staged SWE-Pro substrate (`/host/swe_run2`).
- **Missing:** (a) verify the CC ↔ ARLE-anthropic path end-to-end on one task
  (structured `tool_use` in `session.jsonl` + a scored result); (b) confirm the
  pinned tb version's built-in `claude-code` agent (else the 20-line adapter).

## References
- `crates/infer-server/src/anthropic.rs` — Messages endpoint, tool mapping.
- `scripts/cc_swe_baseline.py` — CC-as-harness SWE driver + scorer.
- `docs/plans/2026-07-03-agentic-opd-27b-capability-curve.md` — phase-2 OPD context.
- Memory: `reference_cc_as_harness_pod_recipe`.
