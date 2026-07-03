# CC-as-harness closes end-to-end on the pod: 27B edits real ansible repos (2/3) where the in-house harness got 0 edits

## Goal

First end-to-end run of the Claude-Code-as-harness lane on real hardware:
Qwen3.6-27B-FP8 served on one H20 behind the new Anthropic `/v1/messages`
adapter, the standalone `claude` Linux binary (offline-fed: npm platform
package `@anthropic-ai/claude-code-linux-x64` packed locally, tn-pushed — no
node, no pod network), `scripts/cc_swe_baseline.py` over the 3 staged real
SWE-Pro ansible eval instances from the 06-27..30 campaign.

## Results (n=3 probe — infrastructure verdict, not a capability claim)

| instance | edited | turns | wall | corrected verdict (py3.11 venv, PYTHONPATH=lib:test, negative-control validated) |
|---|---|---|---|---|
| ansible-0ea40e0 | no | 3 | 34 s | FAIL — no-edit bail (abstention) |
| ansible-12734fa | **yes** | 10 | 393 s | FAIL — edited the wrong site (`plugins/filter/core.py`; RepresenterError persists) |
| ansible-5e36960 | **yes** | 6 | 651 s | FAIL — right area, broke expected API (5 failed, 2 passed) |

**Corrected baseline = genuine 0/3** (clean of harness artifacts): one
abstention + two plausible-but-wrong fixes. Edits 2/3.

- **The whole chain works**: adapter under real CC load (zero 4xx/5xx in the
  serve log, 4 slots absorbed CC's concurrent requests), 250 MB native binary
  runs under ELKEID, scoring drives pytest in the CC-edited workdirs.
- **The harness thesis is visible at n=3**: the SAME 27B that never emitted a
  single edit across 30-turn rollouts in the in-house agent loop
  ([errors/2026-06-29](../errors/2026-06-29-agent-opd-accept-wall-is-no-edit-exploration-not-wrong-dir.md))
  edited 2/3 real repos under Claude Code's scaffolding. Harness quality is a
  capability multiplier; pass@1 = 0/3 says the fixes still aren't correct —
  exactly the gap OPD distillation is for.
- Three scorer/env findings:
  1. `cc_swe_baseline.py` must pass `PYTHONPATH=lib:test` for ansible units
     (the Rust path always had `--pythonpath`; the driver didn't) and honor
     `before_repo_set_cmd` (missing `resolvelib` dep).
  2. These ansible-core 2.12/2.14 trees cannot even COLLECT on the pod's only
     Python (3.12: `_AnsiblePathHookFinder has no attribute find_spec`) —
     pod-side scoring needs a pinned ≤3.11 env; the 06-2x campaign's eval
     numbers on these instances were fail-by-error, not fail-by-test.
  3. Adapter: non-streaming responses can leak a stray `</think>` into the
     text block (streaming drops reasoning; non-streaming must strip it too).
- Serving perf lever for CC workloads: the radix prefix HITS CC's ~17K-token
  system prompt but the hybrid recurrent sidecar is missing
  (`recurrent sidecar restore failed → full recompute`, matched_len up to
  17,232) — every turn re-prefills ~17K tokens; TTFT 82 s. Fixing sidecar
  save/restore for the prefix path is the top lever.

## Learnings

- **Corpus staging gate for phase 2**: an instance belongs in the corpus only
  if, on the target pod env, the base tree's hidden tests FAIL (not ERROR)
  and the gold patch PASSES — the same self-check the synthetic generator
  runs. SWE-Pro instances pinned to old interpreter eras must be filtered.
- The offline-feed recipe for CC on an air-gapped pod: `npm pack
  @anthropic-ai/claude-code-linux-x64` locally → push tgz → extract → the
  `claude` ELF is standalone (the wrapper npm package is a 20 KB stub whose
  postinstall just copies this binary).
- CC-side env that mattered: `--max-running-requests 4` on the serve
  (concurrency), `IS_SANDBOX=1 DISABLE_AUTOUPDATER=1
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` on the client.

Raw: pod `/host/cc_baseline_results.jsonl`, `/host/cc_baseline.log`,
`/host/cc_serve.log`; workdirs `/host/cc_baseline_work/`.
