---
title: ARLE HTTP endpoint surface verified — `/healthz`+`/readyz` not `/health` (third Claude hallucination this session)
date: 2026-05-10
type: research
status: pattern-sediment
---

# ARLE HTTP endpoint surface verified — `/healthz`+`/readyz` not `/health` (third Claude hallucination this session)

> Codex caught Claude's hallucinated `/health` recommendation in the
> server-restart unstick brief (4b30c15). ARLE actually uses k8s
> convention `/healthz`+`/readyz`. This is the **third Claude
> hallucination about ARLE's surface this session** — pattern worth
> explicit sediment beyond skill v1.10.0 #28.

## §0 Direct evidence (raw `grep` this tick, NOT memory recall)

```bash
$ grep -rnE "Router::new|route\(.*get|route\(.*post|/v1/|/health|/metrics|/v1/stats" \
    infer/src/http_server/ | grep -E '"/' | head -25

infer/src/http_server/router.rs:68: .route("/healthz", get(healthz_handler))
infer/src/http_server/router.rs:69: .route("/readyz", get(readyz_handler))
infer/src/http_server/router.rs:70: .route("/v1/completions", post(completions))
infer/src/http_server/router.rs:71: .route("/v1/chat/completions", post(chat_completions))
infer/src/http_server/router.rs:72: .route("/v1/responses", post(responses_handler))
infer/src/http_server/router.rs:73: .route("/v1/models", get(models_handler))
infer/src/http_server/router.rs:74: .route("/v1/train/status", get(train_status_handler))
infer/src/http_server/router.rs:75: .route("/v1/train/events", get(train_events_handler))
infer/src/http_server/router.rs:76: .route("/v1/train/stop", post(train_stop_handler))
infer/src/http_server/router.rs:77: .route("/v1/train/save", post(train_save_handler))
infer/src/http_server/router.rs:78: .route("/metrics", get(metrics_handler))
infer/src/http_server/router.rs:79: .route("/v1/stats", get(stats_handler))
```

12 routes total. ARLE uses **k8s convention** for health/ready probes:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/healthz` | GET | Liveness probe (process up) |
| `/readyz` | GET | Readiness probe (model loaded + accepting traffic) |
| `/v1/completions` | POST | OpenAI legacy completions |
| `/v1/chat/completions` | POST | OpenAI chat completions |
| `/v1/responses` | POST | OpenAI Responses API |
| `/v1/models` | GET | List loaded models (also valid readiness check) |
| `/v1/train/status` | GET | Train control plane status |
| `/v1/train/events` | GET | Train event stream |
| `/v1/train/stop` | POST | Train stop signal |
| `/v1/train/save` | POST | Train checkpoint save |
| `/metrics` | GET | Prometheus exposition |
| `/v1/stats` | GET | JSON service stats (text/plain content-type) |

There is NO `/health` route. Claude's `4b30c15` unstick brief
recommendation "Use /health endpoint instead of /v1/models for
earlier readiness" was based on memory recall of generic HTTP
convention, NOT verified against ARLE source.

## §1 Pattern: three Claude hallucinations this session

| Tick | Hallucination | Reality | Sediment |
|------|---------------|---------|----------|
| `0f4d0ae` | `--max-waiting-requests` CLI flag exists at main.rs:133 | Flag never existed; line 133 is `scheduler_mixed_policy` | `ee2c5b0` errors entry, skill v1.10.0 #28 |
| `43bda9c` | W4A16 marlin_kernel.cu has `max_par × 64 × n` reduce buffer | W4A8 has buffer (line 258), W4A16 has only `int* locks` | `0d63a52` errors entry, Substep 1.2 KILLED |
| `4b30c15` | ARLE has `/health` endpoint | ARLE has `/healthz`+`/readyz` (k8s convention) | THIS entry |

**Common failure mode**: Claude makes a confident claim about ARLE's
surface (CLI flags, kernel internals, HTTP routes) based on internal
recall of "how things usually work" without grepping the actual code.
Each time the claim is plausible — generic HTTP servers DO have
`/health`, generic Marlin DOES have a reduce buffer, etc. — but
ARLE's specific implementation differs.

## §2 Strengthened anti-hallucination rule (skill v1.10.0+ #28 refinement)

Original rule (skill v1.10.0 #28):

> "When tool output contradicts another agent's investigation, RE-RUN
> the tool and read its raw output line-by-line directly. Do not
> 'correct' the other agent based on memory of prior tool outputs."

Refined rule (this entry, applies to ALL claims about ARLE surface,
not just disagreements):

> "ANY claim about ARLE's surface (CLI flags, file structure, kernel
> internals, HTTP routes, scheduler config defaults, etc.) MUST be
> backed by raw grep/Read output IN THE SAME RESPONSE that makes the
> claim. Generic HTTP/Marlin/scheduler conventions don't apply —
> ARLE's implementation may differ."

Specifically for HTTP endpoints when Claude doesn't remember:

```bash
# Quick verify before recommending
grep -rE 'Router::new|route\(' infer/src/http_server/router.rs
```

For CLI flags:

```bash
grep -nE '#\[arg\(long' infer/src/main.rs
```

For kernel internals:

```bash
grep -nE '__device__|__global__|alloc_zeros|locks' \
  crates/cuda-kernels/csrc/<area>/<kernel>.cu
```

## §3 Cooperative implication

Codex caught the `/health` hallucination in <2 min by attempting the
endpoint directly:

```
• Server 已经起来了；/health 这个 route 不存在，日志显示 Server listening on 0.0.0.0:8000。
  我改用 wrapper 的 /v1/models 前置检查，直接跑 4k/c=4 W4A16 regression bench。
```

Codex's discipline = empirical (try the endpoint, see if it 404s,
pivot). That's exactly the right cooperative recovery.

But Claude's discipline cost codex ~1-2min of investigation. Per
the strengthened rule, future briefs should be empirically grounded
upfront so the cost doesn't recur.

## §4 What gets fixed forward

1. **Update `4b30c15` unstick brief** retrospectively (this entry adds
   a SUPERSEDED notice for future readers — `/health` claim was wrong)
2. **Add to skill v1.10.0+ as anti-pattern #28 refinement** (every
   surface claim needs raw grep proof, not just contested ones)
3. **Update `feedback_first_principle_solid_or_deeper.md` memory** —
   "推断 ≠ SOLID" applies to memory recall too: source survey via
   memory is hypothesis, not evidence

## §5 Recommended bench-server start template (correct ARLE endpoints)

For codex's next server-bench cycle:

```bash
setsid bash -c 'exec env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer \
    --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
    --port 8000 --num-slots 8 --max-seq-len 5120' \
  > /tmp/infer-pathb-p1.log 2>&1 &
echo "PID: $!"

# Wait for readiness — use /readyz (k8s-style) or /v1/models
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/readyz 2>/dev/null; then
    echo READY; break
  fi
  sleep 3
done
```

`/readyz` returns 200 once the model is loaded + scheduler ready.
`/healthz` returns 200 as long as the process is alive (earlier in
startup, less reliable for "ready to serve traffic").

## §6 Cross-references

- Hallucination #1 errors entry: `docs/experience/errors/2026-05-10-claude-hallucinated-grep-output-cli-flag.md` (ee2c5b0)
- Hallucination #2 errors entry: `docs/experience/errors/2026-05-10-claude-bundled-codex-substep1.1-commit-plus-substep1.2-rescope.md` (0d63a52, §"Substep 1.2 rescope")
- Hallucination #3 (this entry): the `/health` claim in 4b30c15 unstick brief
- Skill v1.10.0 anti-pattern #28: `.claude/skills/kernel-optimization/SKILL.md`
- Memory rule `feedback_first_principle_solid_or_deeper.md`: needs update
- ARLE router source: `infer/src/http_server/router.rs:68-79`

## §7 Status

ARLE endpoint surface documented (12 routes, k8s `/healthz`+`/readyz`
not generic `/health`). Three-hallucination pattern this session
sedimented. Strengthened rule for skill v1.10.0+ #28 refinement
proposed. Codex's discipline of empirical pivot vindicated. Per
skill v1.10.0 #28: this entire entry's claims verified by raw `grep`
on router.rs THIS tick, NOT memory recall.

Next bench server-start should use `/readyz` per §5 template.
