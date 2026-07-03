# CC-as-harness closes end-to-end on the pod: 27B edits real ansible repos (2/3) where the in-house harness got 0 edits

## Goal

First end-to-end run of the Claude-Code-as-harness lane on real hardware:
Qwen3.6-27B-FP8 served on one H20 behind the new Anthropic `/v1/messages`
adapter, the standalone `claude` Linux binary (offline-fed: npm platform
package `@anthropic-ai/claude-code-linux-x64` packed locally, tn-pushed — no
node, no pod network), `scripts/cc_swe_baseline.py` over the 3 staged real
SWE-Pro ansible eval instances from the 06-27..30 campaign.

## Results (n=3 probe — infrastructure verdict, not a capability claim)

| instance | edited | turns | wall | outcome |
|---|---|---|---|---|
| ansible-0ea40e0 | no | 3 | 34 s | no-edit bail |
| ansible-12734fa | **yes** | 10 | 393 s | UNSCOREABLE on this pod (see below) |
| ansible-5e36960 | **yes** | 6 | 651 s | genuine FAIL (5 failed with correct PYTHONPATH) |

- **The whole chain works**: adapter under real CC load (zero 4xx/5xx in the
  serve log, 4 slots absorbed CC's concurrent requests), 250 MB native binary
  runs under ELKEID, scoring drives pytest in the CC-edited workdirs.
- **The harness thesis is visible at n=3**: the SAME 27B that never emitted a
  single edit across 30-turn rollouts in the in-house agent loop
  ([errors/2026-06-29](../errors/2026-06-29-agent-opd-accept-wall-is-no-edit-exploration-not-wrong-dir.md))
  edited 2/3 real repos under Claude Code's scaffolding. Harness quality is a
  capability multiplier; pass@1 = 0/3 says the fixes still aren't correct —
  exactly the gap OPD distillation is for.
- Two scorer findings, both environmental:
  1. `cc_swe_baseline.py` must pass `PYTHONPATH=lib:test` for ansible units
     (the Rust path always had `--pythonpath`; the driver didn't).
  2. ansible-12734fa's hidden tests cannot even COLLECT on the pod's Python
     3.12 (`_AnsiblePathHookFinder has no attribute find_spec` — ansible-core
     2.12-era loader vs py3.12). The instance is unscoreable in this env; the
     06-2x campaign's eval numbers on it were fail-by-error, not fail-by-test.

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
