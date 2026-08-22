# DSv4/MTP guidellm canonical bench — SSE usage fix VERIFIED, sweep still capacity-blocked at TP=4 — 2026-07-06

> Status: **partial PASS.** The commissioned SSE `stream_options.include_usage`
> fix (`e16c89968`) is confirmed working end-to-end on the real CUDA multiproc
> coordinator path — this is the exact bug that blocked every prior canonical
> guidellm attempt. Two more environment-layer blockers were found and fixed en
> route (guidellm CLI version drift; a transformers/checkpoint config-load
> bug). But the canonical **sweep** profile still cannot complete at TP=4 — a
> genuine, reproduced (round 2) capacity ceiling: only **1** concurrent 4096-token request fits in the KV
> pool at the max `--max-total-tokens` the MTP startup gate allows. A bounded
> c=1 exploration run (non-canonical) produced real, guidellm-computed
> numbers. GPU1 was occupied the entire session (same foreign tenant as
> rounds 1-3), so TP=8 was not available to clear the ceiling.

## SLO-shape probed?  N

The canonical sweep (M≥4096 prefill, batch≥4) did not complete — see Problems.
The supplementary c=1 run used M=4096 prefill but batch=1, so it does not
clear the SLO-shape bar either. Deferred, not a KILL claim.

## Roofline check

Not computed — no compute/memory-bound op isolated this session; this is a
serving-pipeline verification task, not a kernel-level optimization.

## Goal

1. Verify the `e16c89968` SSE usage fix holds on the real CUDA multiproc
   coordinator (not just the Metal in-process path already verified locally).
2. Get the canonical `scripts/bench_guidellm.sh` sweep to complete for
   DeepSeek-V4-Flash-FP8 + MTP, now that the usage-blocker is gone.

## Hypothesis

The SSE usage fix should unblock guidellm's streaming preflight
unconditionally (backend-agnostic bug, per round 2's diagnosis) and the
canonical sweep should then run to completion.

## GPUs used

**GPU1 occupied the entire session** (51,373 MiB, same standing foreign
`Qwen3.6-27B-FP8` terminal-bench serve, PID 1198140, confirmed alive
throughout, untouched) — checked live via `nvidia-smi` at task start and
periodically; never cleared. Per the task brief's explicit fallback rule,
used **TP=4/EP=4 on GPUs 4,5,6,7** (confirmed free throughout), matching
rounds 2-3's proven-good topology for direct comparability. TP=8 (the
canonical production topology) was not available this session.

## Pod state

- `scripts/pod.sh sync` → pod tree `e56adab0 docs(train): plan OPD Metal
  backend, ...` (descendant of the commissioned `e16c89968`; confirmed via
  `git -C /host/arle-build log --oneline -1` showing both `e56adab0` HEAD and
  `e16c8996` in history).
- Build: `cargo build --release --features cuda,nccl,deepep --bin arle` →
  `BUILD_EXIT=0` (compiled 6 crates, 49.67s).

## Command (server, final working config)

```bash
INFER_CUDA_DEVICES=4,5,6,7 INFER_TP_SIZE=4 \
  ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_INCREMENTAL_KV=1 NCCL_DEBUG=WARN \
  ./target/release/arle serve --backend cuda \
    --model-path /host/DeepSeek-V4-Flash-FP8 --bind 0.0.0.0 --port 18198 \
    --spec-type mtp --max-total-tokens 12288 --max-prompt-tokens 12000
```

Canonical bench (unmodified wrapper, attempted):

```bash
scripts/bench_guidellm.sh dsv4-mtp-tp4 --target http://localhost:18197 \
  --model DeepSeek-V4-Flash-FP8 --processor /root/dsv4-processor-fix
```

Supplementary c=1 exploration (non-canonical, ran to completion):

```bash
scripts/bench_guidellm.sh dsv4-mtp-tp4-c1 --target http://localhost:18198 \
  --model DeepSeek-V4-Flash-FP8 --processor /root/dsv4-processor-fix \
  --concurrencies 1 --max-seconds 45
```

## Environment

- **Backend:** CUDA, H20 ×4 of 8 (GPUs 4-7, 97871 MiB/card), CUDA 12.9.
- **Model:** DeepSeek-V4-Flash-FP8, `/host/DeepSeek-V4-Flash-FP8`.
- **Commit:** pod HEAD `e56adab0` (descendant of commissioned `e16c89968`).
- **Feature set:** `cargo build --release --features cuda,nccl,deepep --bin arle`.
- **Non-default flags:** `--spec-type mtp` (`mtp_draft_tokens=2`,
  `mtp_draft_topk=1`); `INFER_TP_SIZE=4`; `INFER_CUDA_DEVICES=4,5,6,7`;
  `--max-total-tokens 12288 --max-prompt-tokens 12000` (see Problems for why
  16384 and 8192 both fail — 12288 is the empirically-found working point).
- **Profiling state:** OFF.
- **Server launch:** verified engine-ready — `[multiproc-coord] all 4 worker
  engines ready; opening HTTP`, `serving OpenAI v1 on http://0.0.0.0:18198`.
  `CUDA engine: executor clamped slots 256 -> 45`.
- **guidellm:** pod had `0.7.0` installed (incompatible CLI shape, same
  regression round 2 found); reinstalled `guidellm[recommended]==0.6.0` per
  `requirements-bench.txt`'s existing (correct) pin.

## Canonical params (locked, attempted but sweep did not finish)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256` (+ stdev/min/max clamps)
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`

## Decode-check (before any bench)

`curl /v1/chat/completions`, `temperature=0`:
- "What is the capital of France? Answer in one word." → `"Paris"` (16 prompt
  / 26 completion tokens).
- "Count from 1 to 20." → exact correct sequence, 12 prompt / 96 completion
  tokens.

**MTP active, confirmed via server log**: `[dsv4-mtp] depth=2 topk=1
draft_rows=2 verify_rows=3 accepted=2 ... accept_total=73 reject_total=29` —
71.6% acceptance rate; `/v1/stats` after the two smoke requests:
`generated_tokens=122` over `steps=53` (~2.3 tok/step, consistent with
`draft_tokens=2` actively landing multi-token steps, not a silent no-op).

## SSE usage-fix verification (the core commissioned check) — PASS

Raw `curl -sN` against the live CUDA multiproc coordinator (not Metal):

```
data: {"choices":[{"delta":{"reasoning_content":"The user just said \"Say hi\". This","role":"assistant"}, ...}],..., "usage":null}

data: {"choices":[{"delta":{},"finish_reason":"length",...}],..., "usage":null}

data: {"choices":[],"created":1783322844,"id":"chatcmpl-...","model":"DeepSeek-V4-Flash-FP8","object":"chat.completion.chunk","usage":{"completion_tokens":10,"prompt_tokens":6,"total_tokens":16}}

data: [DONE]
```

**Confirmed: the trailing `"choices":[]` chunk with populated `usage` appears
correctly before `[DONE]`, on the real CUDA multiproc coordinator relay path**
(`coordinator.rs`), matching the behavior already verified locally on Metal.
This is the exact fix that blocked round 2's guidellm preflight
(`probe_streaming_completions` requires a populated `usage` chunk) — **the
core commissioned bug is fixed and now verified on both backends.**

## Problems

Three additional, independent blockers were found and worked around/resolved
this session, none of which are the SSE-usage bug (which is fixed):

### 1. guidellm CLI version drift (reproduces round 2's finding)

Pod had bare `pip install guidellm` resolve to `0.7.0` (incompatible CLI:
`guidellm benchmark run` vs the wrapper's expected shape), despite
`requirements-bench.txt` correctly pinning `guidellm[recommended]==0.6.0`.
Reinstalled the pin explicitly; `guidellm benchmark run --help` then matched
the wrapper's expected flags. **Not a new bug** — same drift round 2 already
documented; the fix is to actually run `pip install -r requirements-bench.txt`
(or equivalent) as part of pod setup, not rely on a prior install surviving.

### 2. NEW — transformers 5.6.0 cannot load this checkpoint's config/tokenizer via AutoConfig/AutoTokenizer

`model_type: "deepseek_v4"` is unregistered in transformers 5.6.0's
`CONFIG_MAPPING`. Both `AutoConfig.from_pretrained(...)` and
`AutoTokenizer.from_pretrained(...)` (with or without `trust_remote_code`)
crash identically:

```
File ".../transformers/modeling_rope_utils.py", line 758, in standardize_rope_params
    self.rope_parameters.setdefault("original_max_position_embeddings", self.max_position_embeddings)
AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'
```

Root cause, read at source: the generic `PreTrainedConfig` fallback (used
because `deepseek_v4` isn't registered) declares a `rope_parameters` field but
does **not** declare `max_position_embeddings` as a dataclass field (that's
only declared on real per-model subclasses). The config.json's legacy-style
`rope_scaling`/`rope_theta` keys trigger `__post_init__`'s
`convert_rope_params_to_dict` → `standardize_rope_params`, which
unconditionally references `self.max_position_embeddings` — an attribute that
was never set because it isn't a declared field on the fallback class. This
is a genuine transformers-library gap for any unregistered `model_type` whose
config carries legacy rope keys — orthogonal to ARLE, guidellm, or today's SSE
fix. It would have blocked round 2 too, had round 2 gotten far enough (it was
blocked earlier, by the SSE bug).

**Workaround (guidellm-invocation only, does not touch the checkpoint or
ARLE code):** `deepseek_v3` **is** registered in this transformers version and
is architecturally the same family (per
`project_glm52_is_dsv4_family.md`/prior docs). Built a scratch directory
(`/root/dsv4-processor-fix`, pod-local, not committed) containing a copy of
`tokenizer.json` + `tokenizer_config.json` + `config.json` with `model_type`
patched to `"deepseek_v3"`. Verified `AutoConfig`/`AutoTokenizer.from_pretrained`
then load cleanly (`max_position_embeddings=1048576` correctly read). Pointed
guidellm's `--processor` at this directory instead of
`/host/DeepSeek-V4-Flash-FP8` directly. This only affects guidellm's own
tokenizer/token-counting; ARLE's own DSv4 loader reads `config.json` directly,
never through HF `transformers`, so this workaround has zero effect on the
actual served model.

### 3. STILL BLOCKING — TP=4+MTP capacity ceiling makes the canonical sweep impractical (confirms round 2)

With the two blockers above fixed, the canonical sweep launched and appeared
to "hang" — `requests_completed` frozen and `queue_depth` pinned near 509 for
20+ minutes. Root-caused via `py-spy dump` + server log grep (not assumed):
**not a hang** — `grep -c "admission reject" run-dsv4mtptp4c.log` → **67,708**
admission rejects at `--max-total-tokens 8192` (the first value tried,
matching round 2's MTP fallback value). Raised to `--max-total-tokens 12288`
(the max value the MTP startup gate tolerates short of outright rejecting —
**16384 hard-rejects at startup** with MTP: `DSv4 KV budget rejected startup:
the shared FlashMLA pool's per-layer remainder (0MB) cannot hold even one
slot's band at max_seq_len 16384`, whereas the same 16384 worked fine
*without* MTP per the prior `2026-07-06-dsv4-flashmla-budget-reconciliation-verified.md`
— MTP's extra draft-head layer costs ~1.6GB more weight footprint at TP=4,
consuming exactly the margin that made 16384 boot before). At 12288, the pool
has 96 free pages; **one** 4096-token prompt consumes ~50 of them
(`kv_free_pages: 96→46`), leaving too few for a second concurrent request —
`active_requests` stayed pinned at 1 throughout, confirmed via repeated
`/v1/stats` polling, even as `queue_depth` (server-side admitted backlog)
climbed past 500. At ~6.7s/request-serial, draining that backlog would have
taken **~55 minutes** — killed the run rather than let it burn that time
(`kill -- -<pgid>`, confirmed via `nvidia-smi` GPUs 4-7 back to 0 MiB after).

This is not a new discovery — it is the **exact same capacity ceiling round 2
already documented and root-caused** ("TP=4/EP=4 cannot currently sustain the
canonical sweep's concurrency range at the canonical 4096-token prompt
shape... a genuine capacity ceiling of the halved-GPU config"), now
re-confirmed to still hold with MTP's extra weight overhead making it
slightly *worse* (12288 max vs round 2's 8192-with-crashes; the crash bug
itself was separately fixed by the FlashMLA budget-reconciliation commit,
so this session got a clean *admission-queue backlog* instead of a crash —
progress, but the fundamental single-concurrency ceiling remains).

**The only way to clear this ceiling is TP=8** (2× the free post-weights VRAM
budget) — not available this session (GPU1 occupied throughout).

### 4. NEW finding — DSv4 CUDA multiproc SSE stream is not token-granular

Decoded directly from the raw curl trace (not inferred): a 256-token
completion produces only **`Stream Iter Per Req` median = 4** SSE chunks
total (confirmed via the guidellm CSV column), and **TTFT (median 8092ms) ≈
full request latency (median 8.1s)** — i.e. content is delivered in ~2 large
batched chunks near the end of generation, not incrementally per token. This
is visible directly in the raw curl trace from the usage-fix check: a
10-token completion sent its *entire* `reasoning_content` as **one** delta
chunk, only splitting into `finish_reason` + usage-trailer chunks after.
**This is a distinct bug from the usage-populate fix** (which correctly adds
one *additional* trailing chunk; the pre-existing content-delivery chunking
was already coarse). Not root-caused to a specific coordinator.rs line this
session (out of scope for a bench-execution task) — flagging for a follow-up
investigation, since it invalidates TTFT as a true first-token metric for
this server today.

## Results — supplementary c=1 exploration (non-canonical; NOT the SLO gate)

Ran to completion (`--concurrencies 1 --max-seconds 45`, real guidellm 0.6.0
tool output, not hand-rolled):

| rate (req/s) | TTFT p50 (ms) | TTFT p99¹ (ms) | ITL p50 (ms) | ITL p99¹ (ms) | req/s actual |
| --- | --- | --- | --- | --- | --- |
| concurrent c=1 | 8091.8 | 8492.0 | 0.121 | 0.163 | 0.1 (6 reqs / 45s) |

¹ guidellm reported these as its p95 columns (its console table only prints
p95, not p99 — the raw CSV's percentile array top value, used above, is its
highest reported percentile point, functionally its p95/near-max).

Additional columns from the raw table:
- Request latency: median 8.1s, p95 8.5s.
- TPOT: median 31.6ms, p95 33.2ms (~31.6 decode tok/s single-stream).
- 6 successful, 0 errored, 0 incomplete requests in the 45s window (bounded
  correctly — `elapsed_time: 49.0s` including drain of the last in-flight
  request past the nominal cutoff).
- Input/output token shape confirmed exactly on-target: median input 4097
  tok, median output 256 tok, median total 4353 tok (matches the canonical
  `prompt_tokens=4096,output_tokens=256` spec).

## Results — service-side KV / scheduler metrics (c=1 run)

| metric | before | after |
|---|---:|---:|
| `active_requests` | 0 | 1 |
| `queue_depth` | 0 | 0 |
| `kv_free_pages` | 96 | 46 |
| `steps` | 6 | 828 |
| `requests_completed` | 1 | 7 |
| `host_demoted_pages` | 22 | 110 |

## Results — request accounting

| metric | value |
|---|---:|
| completed input tokens (median) | 4097 |
| completed output tokens (median) | 256 |
| errored requests | 0 |
| incomplete requests | 0 |

## Learnings

- **The exact SSE usage-populate fix commissioned today (`e16c89968`) is
  confirmed correct on the CUDA multiproc coordinator relay path, not just
  the previously-verified Metal in-process path.** Round 2's diagnosed root
  cause (`coordinator.rs` never passing `Some(usage)`, `stream_options` never
  read) is resolved; the trailing `"choices":[]` + populated `usage` chunk
  now appears exactly as vLLM/SGLang do.
- **A "hung" sweep should be root-caused via the server's own admission log
  before assuming a client-side hang** — `grep -c "admission reject"` found
  67,708 hits instantly; `py-spy dump` on the client confirmed it was
  correctly idle (not busy-looping), and `/v1/stats` polling showed the real
  mechanism (backlog queued server-side, draining at the true concurrency
  ceiling). Chasing "why is the client stuck" without checking the server
  log would have wasted much more time.
- **MTP's extra draft-head layer meaningfully tightens the TP=4 KV budget** —
  the same `--max-total-tokens` value that boots clean *without* MTP
  (16384, per the separately-verified FlashMLA-budget-reconciliation fix)
  hard-rejects *with* MTP at TP=4; the safe ceiling drops to ~12288.
- **Fixing one canonical-bench blocker reliably surfaces the next one** —
  today's chain was usage-populate (fixed) → guidellm CLI pin drift (fixed,
  recurring) → transformers/checkpoint config-load bug (new, worked around)
  → TP=4 capacity ceiling (pre-existing, reconfirmed) → SSE chunking
  granularity (new, flagged). Each layer needed its own root-cause check
  before concluding "blocked" — none of them were the same bug wearing a
  different face.

## Δ vs baseline

- **Baseline:** the 2026-07-06 TP=4 attempt (blocked entirely by the SSE-usage
  bug; zero guidellm-computed numbers).
  This session is the first to produce a real, tool-computed guidellm table
  for DSv4+MTP, even though it's the non-canonical c=1 shape rather than the
  full sweep.

| metric | baseline (round 2) | now (c=1, non-canonical) | Δ |
|---|---|---|---|
| guidellm preflight | blocked (usage never populated) | **passes** | fixed |
| canonical sweep | blocked before launch | launches, but capacity-blocked mid-sweep | partial |
| any real guidellm number | none | TTFT p50 8091.8ms, TPOT median 31.6ms (c=1) | first data point |

## Artefacts

- `bench-output/2026-07-06-dsv4-mtp-tp4-c1/benchmarks.json`
- `bench-output/2026-07-06-dsv4-mtp-tp4-c1/benchmarks.csv`
- `bench-output/2026-07-06-dsv4-mtp-tp4-c1/benchmarks.html`
- `bench-output/2026-07-06-dsv4-mtp-tp4-c1/service_stats_before.txt` /
  `service_stats_after.txt` / `service_stats_trace.jsonl` /
  `service_stats_trace_summary.md`
- (all pod-local under `/host/arle-build/bench-output/`, not copied off-pod
  this session — ephemeral per project convention)

## Notes

- What changed in code since the commissioning push: nothing this session
  (devops/bench execution task); the SSE usage fix (`e16c89968`) and the
  FlashMLA budget-reconciliation fix (separately verified,
  `2026-07-06-dsv4-flashmla-budget-reconciliation-verified.md`) were already
  landed and pulled in via `scripts/pod.sh sync`.
- Follow-ups (flagging for the calling session, not filed as issues by this
  task):
  1. **TP=8 retry** once GPU1 clears — the only way to get the canonical
     sweep's concurrency range on this checkpoint+MTP combination.
  2. **Investigate DSv4 CUDA multiproc SSE chunking granularity**
     (`coordinator.rs`'s relay-to-SSE path) — content is batched into ~2
     large chunks instead of streaming per-token, making TTFT ≈ full latency;
     likely a missing per-token flush somewhere between the engine's output
     stream and the coordinator's SSE writer.
  3. **`requirements-bench.txt`'s guidellm pin needs enforcing at pod-setup
     time**, not just declared — a bare system-wide `pip install guidellm`
     (0.7.0) silently overrides it if anyone runs that instead of installing
     from the requirements file.
- Cleanup: server (PGID 1374657) killed via exact PGID after the c=1 run
  completed; guidellm client processes exited on their own. `nvidia-smi`
  confirmed GPUs 4-7 at 0 MiB after. GPU1's foreign tenant (PID 1198140,
  `Qwen3.6-27B-FP8` terminal-bench serve) and GPU0 (idle, untouched) were
  never touched.
