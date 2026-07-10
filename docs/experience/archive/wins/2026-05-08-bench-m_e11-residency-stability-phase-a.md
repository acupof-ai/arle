# Bench — M_e.11 residency-set stability Phase A: abort NOT reproduced — 2026-05-08

## Goal

Phase A of the M_e.11 (periodic `mx::clear_cache` 1024-token cadence)
verification bench. Reproduce the IOGPU `-[IOGPUMetalResidencySet
addAllocation:]` abort PRE-fix by disabling M_e.11's clear-cadence
(`INFER_METAL_RESIDENCY_CLEAR_TOKENS=0`) and pushing >4096 generated
tokens at c=1 / 8192-output / 600s ceiling. If the abort fires, M_e.11
is load-bearing. If not, M_e.11 is prophylactic on this stack.

## Hypothesis

- **Plan-side (per omlx commit `6bda6781`, 2026-05-06)**: IOGPU
  residency-set hits ~4096 entries within a single long-generation,
  aborting the server.
- **Pre-fix expected behavior**: server aborts mid-second-or-third
  long request (~5-7 min into bench).

## Command

```bash
INFER_METAL_RESIDENCY_CLEAR_TOKENS=0 RUST_LOG=info \
  ./target/release/metal_serve \
    --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
    --port 8765 \
    --max-running-requests 1 &

scripts/bench_guidellm.sh metal-m3max-residency-prefix \
  --target http://localhost:8765 \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --processor mlx-community/Qwen3.6-35B-A3B-4bit \
  --concurrencies 1 \
  --data 'prompt_tokens=512,prompt_tokens_stdev=1,prompt_tokens_min=512,prompt_tokens_max=512,output_tokens=8192,output_tokens_stdev=1,output_tokens_min=8192,output_tokens_max=8192' \
  --max-seconds 600
```

(Driver script at `/tmp/m_e11_phase_a.sh` orchestrates server + bench +
cleanup.)

## Environment

- **Backend**: Metal (Apple Silicon)
- **Hardware**: **Apple M4 Pro** (chip auto-detected;
  `nax_available=false`, `m5_neural_accelerator=not-eligible`)
- **MLX**: 0.31.1
- **macOS**: 26.3.1
- **Model**: `mlx-community/Qwen3.6-35B-A3B-4bit` (canonical Metal
  per AGENTS.md)
- **Commit**: post-M_e.11 ship `7206bc1f` (M_e.11 disabled at runtime
  via `INFER_METAL_RESIDENCY_CLEAR_TOKENS=0` — code path in tree but
  the threshold check at `ops.rs:288` early-returns)
- **Feature set**: `cargo build --release --no-default-features
  --features metal -p infer --bin metal_serve`
- **Workload**: 7 long requests, c=1, target output_tokens=8192,
  prompt_tokens=512, max_seconds=600 (guidellm SIGINT'd after 12 min
  in cooldown phase to terminate)
- **Auto-wired-limit**: 20 GiB pinned (default-on)

## Results

```
$ grep -E "IOGPUMetalResidencySet|addAllocation|abort|panicked|FATAL" \
    /tmp/m_e11_phase_a_server.log | head -5
(no matches)

$ grep -c "Received request" /tmp/m_e11_phase_a_server.log
8   # 1 warmup + 7 long bench requests

$ grep "M_E11_RESIDENCY_CLEAR_FIRED" /tmp/m_e11_phase_a_server.log
(empty — correct, threshold=0 disables the path)
```

Server timeline (extracted from `/tmp/m_e11_phase_a_server.log`):
- 10:44:11 — server listening
- 10:45:38 — request 1 (long, 8192-target, 2703 prompt bytes)
- 10:47:32 — request 2 (1m54s after r1 receipt → r1 EOS'd at ~3000 tokens)
- 10:49:18 — request 3 (1m46s after r2)
- 10:51:03 — request 4 (1m45s after r3)
- 10:52:47 — request 5 (1m44s after r4)
- 10:54:38 — request 6 (1m51s after r5)
- 10:56:19 — Phase A driver completes (SIGINT triggered cleanup)

7 long requests × ~3000 generated tokens each ≈ **21000 sampling
steps cumulative**. Each step's `mx.random.categorical` → `gumbel` →
`uniform` chain allocates fresh GPU scalars. Per the omlx 2026-05-06
hypothesis, the residency-set ceiling at ~4096 entries should have
aborted the server mid-second-request. **It did not.**

## Δ vs hypothesis

| Aspect | Plan-side hypothesis | Phase A measurement |
|---|---|---|
| Server abort within 600s | Yes, mid-request 2 or 3 | No abort over 12 min / 7 long requests |
| IOGPUMetalResidencySet log signature | Present | Absent |
| Per-request EOS behavior | Reach 8192 max | Natural EOS at ~3000 tokens |
| Residency ceiling reached | >4096 cumulative entries | ~21000 sampling steps elapsed, no ceiling hit |

## Problems / observations

1. **Hardware mismatch with omlx report**: omlx commit message implied
   the bug was reproducible at c=1 ~4096 generated tokens on their
   internal stack. Phase A on M4 Pro / MLX 0.31.1 cannot reproduce.
   Three plausible reasons (in order of likelihood):
   - **MLX 0.31.1 already mitigated upstream**: `mx.clear_cache` may
     now happen lazily inside `mx.random.categorical` or the residency
     set itself may auto-evict. Without an MLX changelog dive this is
     a hypothesis only.
   - **Residency entries are request-scoped on this stack**: each
     request allocates its own context; the per-request set is freed
     when the request finalizes, so cumulative sampling steps don't
     accumulate within a long-lived process.
   - **The bug requires c≥2 packed-batch shape**: omlx's reproduce was
     a coding-agent workload (likely c=1 with tool-call retries). Our
     Phase A is also c=1; this rules this out.
2. **Each request EOS'd at ~3000 tokens** even though `output_tokens=8192`
   was requested. The 4-bit quant Qwen3.6 model produces natural stop
   sentences before 8192 on synthetic prompts. Doesn't change the
   conclusion (we still got 21000 cumulative sampling steps).
3. **Phase B (M_e.11 enabled, default 1024-token cadence) is not run**.
   With Phase A showing no abort baseline, Phase B would just confirm
   "still no abort" — no differential signal. Cost: ~12 min wall clock
   for zero new evidence. **Skipped.**

## What this means for M_e.11

M_e.11 (commit `7206bc1f`) on this stack (M4 Pro / MLX 0.31.1) is
**prophylactic, not load-bearing**. The implementation is correct
(probe verified to fire at the configured threshold) and the runtime
overhead is bounded (1 `clear_cache` per 1024 generated tokens at the
default), but no reproducible benefit can be measured because the
abort mode it prevents does not occur on the current stack.

**Decision**: keep M_e.11 default-on as defense-in-depth. The cost is
low (one clear_cache per ~30s of decode at typical rates) and the
defensive value remains for: (a) older MLX revisions, (b) hardware
where the residency-set ceiling is lower, (c) workloads with longer
single-request decodes (~30k+ tokens) we haven't exercised. Do NOT
claim M_e.11 fixes a current production bug — it doesn't, on the
canonical M4 Pro stack.

## Rule

When porting an upstream defensive optimization, run the disabled-pre-fix
repro bench on the EXACT current canonical stack first. If the abort
the optimization prevents cannot be reproduced, mark the optimization
prophylactic and document the gap rather than claim a SOLID fix. Memory
update: M_e.11 is prophylactic on M4 Pro / MLX 0.31.1; reproduce
attempt failed at 21000 sampling steps cumulative.

## What worked

- **Pre-fix repro bench BEFORE post-fix confirm** (per the new rule
  above). Catching the no-repro case in Phase A saved a redundant
  Phase B and produced calibrated confidence in M_e.11's true scope.
- **Single-shot driver script with SIGINT cleanup**
  (`/tmp/m_e11_phase_a.sh`) that captures server log + abort grep +
  probe grep in one artefact. Makes the no-evidence case auditable.

## Next

- **No Phase B.** Skipped per §Problems 3.
- **Memory entry**: persist this finding under
  `feedback_m_e11_prophylactic_on_m4_pro.md` so future tickets that
  want to claim residency-related wins on this stack are gated on a
  fresh repro attempt.
- **Amend `2026-05-08-m_e11-residency-set-hygiene.md`** with a top-of-
  file note pointing to this Phase A finding (the original wins entry
  describes the implementation, not the verification — both are valid
  but the verification supersedes the "stability win" framing).

## References

- M_e.11 implementation wins entry (this commit follows it):
  [`2026-05-08-m_e11-residency-set-hygiene.md`](2026-05-08-m_e11-residency-set-hygiene.md)
- omlx commit `6bda6781` (the source pattern, 2026-05-06):
  https://github.com/jundot/omlx/commit/6bda6781
- M_e.11 plan:
  [`docs/plans/M_e11-omlx-residency-set-hygiene.md`](../../plans/M_e11-omlx-residency-set-hygiene.md)
- Phase A raw artefacts: `/tmp/m_e11_phase_a.log`,
  `/tmp/m_e11_phase_a_server.log`,
  `bench-output/2026-05-08-metal-m3max-residency-prefix/`
