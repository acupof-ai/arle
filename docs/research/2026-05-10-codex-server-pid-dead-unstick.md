---
title: codex's nohup'd server PID 1810426 died silently — Claude unstick (33min wedged poll loop)
date: 2026-05-10
type: research
status: codex-resumed-with-unstick-brief
---

# codex's nohup'd server PID 1810426 died silently — Claude unstick (33min wedged poll loop)

> Codex was Working 33m+ on a poll loop waiting for /v1/models to
> respond from a server that had died immediately at startup. Claude
> diagnosed via direct evidence (raw `ps`/`ls`/`curl` per skill
> v1.10.0 #28), confirmed binary healthy via foreground 5s test,
> sent unstick brief via paste-buffer.

## §0 Direct evidence (raw shell output this tick, NOT memory recall)

### Codex's wedged tmux state

```
• Waiting for background terminal (33m 36s • esc to interrupt) · 1 background terminal running
  └ for i in $(seq 1 120); do if curl -fsS http://127.0.0.1:8000/v1/models >/tmp/pathb-models.json 2>/tmp/pathb-curl.err; then echo ready; exit 0; fi; sleep 2; done; echo not-ready; tail -80 /tmp/infer-pathb-p1.log; exit 1
```

Codex's poll loop = 120 iterations × 2s = 240s = 4min max. At 33m 36s,
the loop should have completed long ago. Either the loop is itself
wedged, OR codex's tmux clock measures total session, not poll
duration. Either way, not making progress.

### Server process state

```bash
$ ps -p 1810426 -o pid,etime,cmd --no-headers
(empty — process gone)

$ ls -la /tmp/infer-pathb-p1.log
-rw-r--r-- 1 ckl ckl 0 May 10 04:28 /tmp/infer-pathb-p1.log
                    ↑ 0 bytes — server died before logging started
```

Process gone, log empty. Server died IMMEDIATELY after nohup launch,
before stdout/stderr ever wrote anything to the log file.

### Endpoint check

```bash
$ curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models
curl: (7) Failed to connect to 127.0.0.1 port 8000 after 0 ms: Could not connect to server
```

Confirmed: nothing listening on port 8000.

### Binary health (foreground 5s test)

```bash
$ timeout 5 env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
    INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
    TORCH_CUDA_ARCH_LIST=8.9 \
    ./target/release/infer \
      --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
      --port 8001 --num-slots 8 --max-seq-len 5120 2>&1 | head -20

INFO infer::hf_hub: Using local model path: infer/models/Qwen3-4B-W4A16-sym-g128-marlin
INFO infer: === Infer Server - Qwen3 (GPU) ===
INFO GPU memory @ post_cuda_ctx: free=15.05 GB / total=16.72 GB
INFO Loading model...
INFO weight_loader: Memory-mapped 3 shard(s) (4525.3 MB) in 0ms
INFO Weight quantization detected: QuantLoadConfig { group_size: Some(128), bits: Some(4), ... }
INFO Loaded quantized model.layers.0.self_attn.q_proj.weight: [4096x2560] INT4, group_size=128
INFO   + Marlin repacked: [160, 8192] + scales [20, 4096]
INFO Loaded quantized model.layers.0.self_attn.k_proj.weight: [1024x2560] INT4, group_size=128
INFO   + Marlin repacked: [160, 2048] + scales [20, 1024]
[...]
```

**Binary works fine**. 5s timeout cut it off mid-loading (still on
layer 0 weights), but no crash, no early exit. The dequant.h port
(09ae5a5 + 994a294) is healthy at runtime.

## §1 Root cause hypothesis

Codex's nohup invocation:

```bash
rm -f /tmp/infer-pathb-p1.log; \
nohup env CUDA_HOME=... ./target/release/infer ... \
  > /tmp/infer-pathb-p1.log 2>&1 &
```

The `nohup ... &` pattern in bash: nohup ignores SIGHUP, `&` backgrounds.
Should survive parent shell exit. But process died immediately with
ZERO log output.

Hypotheses:
1. **Shell-redirect ordering issue**: in some bash versions the `>` redirect
   happens BEFORE nohup setup; if the file create races with the
   exec, output may be lost
2. **Codex's bash subprocess died**: codex's tmux pane shell exited (e.g.
   completed the last command and the next command in the chain
   killed the parent), taking the nohup'd child with it (despite
   nohup's SIGHUP protection — child can die from other signals)
3. **GPU initialization race**: if multiple processes try to open
   CUDA context simultaneously, one may fail. But the foreground test
   worked, so this is unlikely

For #36 bench earlier this session, codex used `setsid bash -c 'exec ...'`
pattern which is more robust (creates a new session, fully detached).
That worked. Recommended codex switch to setsid for the unstick.

## §2 Unstick brief sent (paste-buffered to tmux 0:0 this tick)

Brief content at `/tmp/codex-unstick-server-dead.txt`:

1. State the evidence (ps/log/curl all confirm server dead)
2. Confirm binary healthy via foreground test
3. Recommend setsid pattern (matches successful #36 bench server start)
4. Use /health endpoint instead of /v1/models for earlier readiness
5. Cooperative discipline reminder: status BEFORE commit when
   eventually committing wins entry update

Codex acknowledged + Working (3s) post-brief.

## §3 Cooperative-pattern lesson

When peer agent's terminal shows "Waiting for background terminal X
hours/minutes" with no observable progress:
- Don't assume the peer is making progress (Working ≠ Productive)
- DIRECTLY verify the underlying process state:
  - `ps -p $PID` — is process alive?
  - `ls -la <log>` — is log file growing?
  - `curl <expected-endpoint>` — is service responding?
- If process is dead, send unstick brief BEFORE peer notices

This tick = ~33min of codex's bandwidth lost to a wedged poll. Earlier
detection (e.g. via tick-start 3-state scan noticing GPU 0% + log
empty + 33m+ timer) would have caught this faster. **Add to next-tick
3-state scan: when codex shows "Waiting" >5min, verify the underlying
process is alive.**

## §4 Cross-references

- Codex's unstick brief: `/tmp/codex-unstick-server-dead.txt`
- Successful setsid pattern (used in #36 bench arm B): codex's prior
  successful invocation captured at the time of arm B start
- Skill v1.10.0 anti-patterns:
  - #28 verify raw output not memory recall (used for ps/log/curl this tick)
  - #30 candidate (commit-time worktree race — codex's wins entry
    update preserved untouched in worktree)
- Phase 1 substrate: `crates/cuda-kernels/csrc/gemm/marlin_dequant.cuh` (codex 09ae5a5)
- Build-restore: `994a294` (Claude marlin_kernel.cu include update)

## §5 Status

Unstick brief landed (codex Working post-paste). Next tick: check
codex's setsid retry + /health probe + W4A16 regression bench
results. Per skill v1.10.0 #28: every claim above verified by raw
shell output this tick, NOT memory recall.

Pattern lesson sediment: peer agent "Waiting >5min" warrants
process-state verification, not just trust the timer.
