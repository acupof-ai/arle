# Long-context bench (prompt=2048) all-rejected — default infer max_input is ~1997

## Context

Date: 2026-05-10 12:47-12:48 KST
Bench: W4A16 conc=1 prompt=2048 single-var A/B vs prior conc=1 prompt=512
baseline (per `8d32576` Arm C). Goal: first long-context perf data point
for "world-first 长序列推理引擎" claim per user's standing directive.

Server config: default flags, no `--max-seq-len` override.

## Symptoms

Bench produced 0 successful requests, 4926 rejections in 60s window:

```text
2026-05-10T12:48:28 WARN admission.rs:105
  Rejecting prompt with 2049 tokens: scheduler max_input=1997 max_request=2002
```

(`prompt_tokens=2048` produces tokenized length ~2049 per Qwen3 tokenizer
with BOS).

`benchmarks.json` exists (70 MB) but all entries are
`status=rejected_request_size_too_large`. **Substantive NULL result.**

## Root Cause

Source: `infer/src/main.rs:23` `const DEFAULT_SEQ_LEN: usize = 4096`
+ `:853` `args.max_seq_len.unwrap_or(DEFAULT_SEQ_LEN)`.

Default `--max-seq-len 4096` translates to scheduler `max_input=1997`
(observed from rejection log). The ~2× headroom (4096 → 1997) is
allocated to:
- Output tokens budget (max 128 in this bench)
- KV cache padding / page alignment
- Other admission-time reserves

So the long-context ceiling at default config is **prompt ≤ 1997 tokens**.
For 2048+ prompts, must pass `--max-seq-len` explicitly higher (e.g.
`--max-seq-len 8192` or 16384).

## Fix

**No code fix this entry — config discovery + future bench guidance.**

To benchmark long-context paths properly:

```bash
# For prompt=2048 + output=128 → max_seq_len ≥ 4400 (2048 + 128 + ~10% overhead)
RUST_MIN_STACK=33554432 \
  setsid target/release/infer \
    --model-path infer/models/Qwen3-4B-GPTQ-W4A16-marlin-zpfix \
    --max-seq-len 8192 \                    # ← EXPLICIT overide
    --port 8000 \
    > /tmp/longctx-server.log 2>&1 &
```

For 4k context: `--max-seq-len 16384` (rule of thumb: 2× headroom).
For 8k context: `--max-seq-len 32768`.

## Rule

When benching long-context paths, **explicitly set `--max-seq-len` ≥ 2×
desired prompt_tokens**. Default 4096 caps prompt at ~2k tokens — not
sufficient for "world-first 长序列推理引擎" target (which implies ≥8k
ideally ≥32k contexts).

This generalizes a precondition for any future long-ctx work:
- Per `M_rope-yarn-scaling` (Task #39 LANDED): YARN scaling enables
  Qwen3-4B at 64k+ ctx
- BUT — bench harness `pf85_bench_v11_user.sh` and `bench_guidellm.sh`
  do NOT pass `--max-seq-len` by default
- Future long-ctx benches must add explicit `--max-seq-len` argument,
  OR the bench scripts should default to 16384/32768 for long-ctx
  workloads

**Procedural rule for future ticks**: when a bench reports 100%
rejected requests AND the rejection log mentions `max_input=N`, the
fix is `--max-seq-len 2*prompt_tokens`, not a code change.

## Cross-references

- `8d32576` W4A16 conc=1/2/4 baseline (prompt=512, was within budget)
- Task #39 M_rope-yarn-scaling Phase 3a (`4efd30b`) — long-ctx
  substrate landed but only smoke-tested at 50 tokens
- `infer/src/main.rs:23` DEFAULT_SEQ_LEN=4096
- `infer/src/scheduler/cuda/runtime/admission.rs:105` rejection logic
- SKILL `kernel-optimization` v1.12.0 #34b (server log first — caught
  this in 1 tick because I checked the log immediately when bench
  output dir was empty)
- `bench-output/2026-05-10-w4a16-longctx-prompt2048/benchmarks.json`
  (70 MB of all-rejected entries, kept as evidence)
- `/tmp/w4a16-longctx-2048.log` (server log with rejection cascade)
