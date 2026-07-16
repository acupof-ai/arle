# Claude Code as the agentic harness

> Status: Active — proposed 2026-07-08, end-to-end verified on the pod the same
> day (§Verified). Replaces the `terminus` agent for both capability eval and
> agentic-OPD rollouts.

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
  --dump-messages-dir /host/cc_dumps \
  [--lora-adapters <opd_adapter> --lora-alpha 32]
```
`--max-running-requests >= 2` is required — CC fires concurrent requests.
`--dump-messages-dir` writes each raw `/v1/messages` body to disk — that server-side
dump is what `cc-convert` consumes for training records (not CC's own session file).

### 2. CC self-hosted, offline config — the crux
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_API_KEY=dummy                     # any value; startup never validates format
export ANTHROPIC_MODEL=Qwen3.6-27B-FP8             # main model = ours
export ANTHROPIC_SMALL_FAST_MODEL=Qwen3.6-27B-FP8  # CC's helper calls stay on-box, no egress
export IS_SANDBOX=1                                # MANDATORY on this root container — see below
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_TELEMETRY=1 DISABLE_AUTOUPDATER=1 DISABLE_ERROR_REPORTING=1
```
**`IS_SANDBOX=1` is load-bearing.** The pod container runs as root (uid 0), and
`--dangerously-skip-permissions` refuses to run under root/sudo — `IS_SANDBOX=1`
is what lets `bypassPermissions` engage. Without it the whole path is red.
Headless startup needs no onboarding/login shim (verified CC 2.1.204). A
non-200 from the endpoint triggers a 10-retry exponential backoff (~38s total) —
a broken serve stalls for minutes rather than failing fast.

### 3. Per-task headless call — inside the sandbox, no egress, no prompts
```bash
cd $workdir
timeout ${TASK_WALL_SECS:-1200} claude -p "$INSTRUCTION" \
  --model "$ANTHROPIC_MODEL" \
  --allowedTools Bash Read Write Edit Grep Glob \
  --dangerously-skip-permissions \
  --output-format stream-json --verbose > $sessions_dir/$task_id.jsonl
```
The session transcript goes **outside** `$workdir` — writing it into the repo
pollutes the `git diff` the scorer keys on. `--allowedTools` is an allowlist;
`WebFetch`/`WebSearch`/`Task` must stay off — the sandbox is offline, and one web
call stalls on the 10-retry ~38s backoff. `timeout` replaces the missing
`--max-turns`; size it against the ~15s ttft (hard tasks legitimately take 20+
turns — the turns-to-first-edit wall).
CC 2.1.204 has **no `--max-turns` flag** — cap via the prompt or a wall-clock
timeout on the process, not a turn count. `stream-json` gives a local transcript
for debugging/scoring; training records come from the **server-side**
`--dump-messages-dir` (§1), not this file.

### 4. One agent, two scorers (task set differs, agent identical)
- **SWE-Pro** (OPD substrate): the cc harness inside `arle train agent-opd`
  (`crates/train/src/cc_harness.rs`; formerly `scripts/cc_swe_baseline.py`,
  deleted 2026-07-16 P3) IS this — boot the staged tree → `claude` headless →
  `sandbox.rs::score_workdir` (non-empty `git diff` ∧ `test_patch` +
  `pytest <fail_to_pass>` exit 0).
- **terminal-bench-core** (68 benign tasks, general eval): the built-in
  `claude-code` agent does **not** work as-is — its `_env` forwards only
  `ANTHROPIC_API_KEY` + `ANTHROPIC_MODEL` (no `ANTHROPIC_BASE_URL`, no
  `IS_SANDBOX`, and the command omits `--dangerously-skip-permissions`), verified
  against `terminal_bench/agents/installed_agents/claude_code/claude_code_agent.py`.
  A **required** ~20-line `AbstractInstalledAgent` subclass overrides `_env`
  (add `ANTHROPIC_BASE_URL` + `IS_SANDBOX=1`) and the command builder (add
  skip-permissions), keeping tb's per-task test scoring. Note the tb agent
  installs `claude` via nvm+npm **inside each task container**, which then also
  needs npm egress + the same base_url/IS_SANDBOX injection.

### 5. OPD loop — the smooth finish
```
serve(LoRA, --dump-messages-dir D) → cc rollout over the train set
   → filter D to passing tasks → cc-convert (in-memory) → masked-CE
   → new LoRA → repeat
```
Now one command: `arle train agent-opd` runs this loop in-process
(2026-07-16 plan P2). `cc-convert` reads the server-side message dump and
replaces `terminus_to_records.py`. The whole agentic-OPD loop moves from
research-code terminus to the production CC loop — the harness stops being a
confound in the capability-lift measurement.

## Retire (deletion-style)
`terminus` agent invocations, `scripts/terminus_to_records.py`, the litellm
`-a terminus -m openai/…` path in `terminal_bench_eval.sh`, and the terminus
calls in `tbench_*.sh` — all replaced by `--agent claude-code` or the
`arle train agent-opd` cc harness.

## End-to-end refinements (found in the chain audit)

**Correctness:**
- **Training-record capture must be serial per task.** `cc-convert` attributes
  dumps to attempts by **time window** (`cc_convert.rs:26` `CcWindow`,
  fullest-messages dump in `[t_start,t_end)`); the dump filename is
  `<epoch_ms>_<seq>` with no task/session id (`coordinator.rs:42`). Concurrent
  tasks overlap in time → windows cross-contaminate. So: **eval lane runs
  concurrent** (needs only pass/fail, no cc-convert); **the OPD training-record
  lane runs one CC at a time**, snapshotting `t_start/t_end` around each call and
  passing labelled windows. Intra-session concurrency (count_tokens etc.) is
  safe — the "fullest dump" heuristic ignores the small requests.
- **Session transcript outside `$workdir`** and **web/subagent tools off** — see §3.

**Performance:**
- **RadixCache the ~20k-token CC system prompt.** It is constant across every
  task and every turn — the single biggest throughput lever. Verify the serve
  prefix-cache hits it; if not, each task re-prefills 20k tokens for nothing.

**Control:**
- **Greedy eval.** The held-out pass-rate needs temp=0 for reproducibility, but
  CC exposes no sampling params — force it serve-side (an eval mode) rather than
  through CC.

**Design choice (not a bug):** distillation is conditioned on CC's 20k system
prompt, so the student learns behaviour given that prompt. Consistent only if
production serving also goes through the CC harness; otherwise it is train/serve
skew to account for.

## The one knob to watch (a property, not a gate)
Whether the model's tool calls parse cleanly at the `anthropic.rs` → OpenAI
mapping. If Qwen occasionally emits malformed structure, CC **retries
automatically** (native production-loop behavior terminus lacks) — itself an
advantage over terminus. If it genuinely breaks, add grammar-constrained tool
decode on the ARLE side; CC's retry likely absorbs it first.

## Verified end-to-end on the pod (2026-07-08)

Full adversarial smoke — install → headless startup → live tool-use round-trip →
dump → cc-convert — **passes** on the H20 box with CC 2.1.204. The load-bearing
assumption holds: **Qwen3.6-27B emits a clean structured `tool_use`** (a `Write`
with `file_path`+`content`), not garbage — so terminus's parse noise really was
the harness, not the model.

| Hop | Result |
|---|---|
| Install `claude` (node v22 + `npm i -g @anthropic-ai/claude-code`) | WORKS — direct npmjs.org egress 200 (socks proxy was down, not needed) |
| Headless startup offline, dummy token | WORKS — no onboarding/auth block |
| CC → `/v1/messages` connect | WORKS |
| `/v1/messages/count_tokens` | WORKS (HTTP 200) |
| Qwen `tool_use` parse (load-bearing) | **WORKS — clean `Write` call** |
| `tool_result` round-trip → `end_turn` | WORKS — `hello.txt` == `banana` |
| server dump → `arle train cc-convert` | WORKS — 1 record (`prompt=3234 masked=43`) |
| tb built-in `claude-code` agent | BREAKS — no base_url/IS_SANDBOX (adapter required) |

**Two non-obvious fixes (folded into §2–§4):** `IS_SANDBOX=1` (root-container
permission block) and dropping `--max-turns` (no such flag). `cc-convert
--tokenizer` takes the `tokenizer.json` **file**, not the model dir. CC's
~20k-token system prompt makes first-token ~15s — budget serve concurrency.

**Residual:** installing full `terminal-bench` on the pod is painful (PyPI egress
~58 kB/s, litellm timed out mid-pull) — pre-stage wheels or `tn push`. Only
needed for the tb-core lane; the SWE-Pro OPD lane (`arle train agent-opd`) needs
no tb install.

Pod artifacts: `/host/npm-global/bin/claude` (2.1.204),
`/host/cc_smoke/{serve.log,dumps/,records.jsonl}`.

## References
- `crates/infer-server/src/anthropic.rs` — Messages endpoint, SSE encoder, tool mapping.
- `crates/infer-server/src/coordinator.rs:220-225` — routes `/v1/messages`,
  `/v1/messages/count_tokens`, `/v1/models`; `--dump-messages-dir` at :41-56.
- `crates/cli/src/train_cli.rs:87` — `arle train cc-convert` (dump → records).
- `crates/train/src/cc_harness.rs` — CC-as-harness SWE driver + scorer
  (ported from `scripts/cc_swe_baseline.py`, deleted 2026-07-16 P3).
- `docs/plans/2026-07-03-agentic-opd-27b-capability-curve.md` — phase-2 OPD context.
- Memory: `reference_cc_as_harness_pod_recipe`.
