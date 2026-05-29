# L4 day-1 bench — reproducibility handover from V100 session

## Why

V100 perf-gap work (
[`projects/2026-05-29-v100-perf-gap-closure.md`](../projects/2026-05-29-v100-perf-gap-closure.md))
queue pivoted to sm_80+ once the cheap V100 levers were exhausted.
L4 sm_89 hardware is "马上来" per ckl's heads-up
(2026-05-29). This doc captures the L4 day-1 plan so the first
session on L4 is `bash scripts/...` and read numbers — not "fight
substrate for an hour".

## Day-1 deliverables

In priority order, each one a separable session-end:

1. **L4 build sanity** — `infer --features cuda` builds with no
   substrate workarounds (V100 needed 3; L4 should need 0). Confirm
   the build flags below + the binary loads Qwen3.5-4B.
2. **L4 Step 1 bench** — same guidellm `--profile concurrent --rate
   "1,4,8" --data prompt_tokens=128,output_tokens=128 --max-seconds
   30` per precision (bf16 / int8 / fp8 / int4). Ship a wins entry
   mirroring
   [`wins/2026-05-29-guidellm-ttft-throughput-v100-qwen35.md`](../experience/wins/2026-05-29-guidellm-ttft-throughput-v100-qwen35.md)
   with the L4 numbers.
3. **L4 industry comparison** — install vLLM/SGLang in a fresh venv
   on the same L4 box. Same shape. Ship wins/ entry vs the published
   numbers cited in
   [`wins/2026-05-29-vs-industry-v100-qwen35.md`](../experience/wins/2026-05-29-vs-industry-v100-qwen35.md).
   This is where the **live "we beat X" claim** can land — V100 was
   substrate-bound; L4 sm_89 is where TileLang AOT already beats
   FA-v2 per prior April-May 2026 wins entries.

## L4 vs V100 substrate diff (what to enable / disable on day 1)

| flag / knob | V100 (sm_70) | L4 (sm_89) |
|---|---|---|
| `ARLE_CUDA_DISABLE_FLASHMLA=1` | required (sm_90+ FP8 `__nv_fp8_e8m0`) | **probably required** — sm_89 still lacks `__nv_fp8_e8m0`; verify with a clean build first. See `docs/projects/2026-05-29-flashmla-sm89-sm90-build-gate.md` (Task I) for the canonical answer once that work lands. |
| `ARLE_TILELANG_SRC` + `ARLE_TILELANG_CUTLASS_INCLUDE` | required (tilelang Python broken in venv) | **probably not required** — if the L4 box has a working `pip install tilelang` env, the canonical `INFER_TILELANG_PYTHON=<venv>/bin/python` is enough. |
| `ARLE_TILELANG_AOT_FALLBACK` | required (build.rs hash-drift workaround) | not required — only matters when local-only build.rs patches change the OUT_DIR hash. L4 should land the upstream build.rs unmodified. |
| `--quant-format marlin_w4a8` | available but no Qwen3.5-4B checkpoint | **same model gap** — Marlin code path itself is sm_80+; L4 sm_89 hits it natively. Still needs a Marlin/AWQ Qwen3.5-4B checkpoint, but on L4 the H1 unblock is "pip install autoawq + quant" not "fight httpx" because the box should have working network. |
| `TORCH_CUDA_ARCH_LIST` | `7.0` | `8.9` (or `8.0;8.9` if A100 also bench-target) |
| `INFER_DETERMINISTIC=1` | **not** needed for guidellm bench (only for parity audit) | same |
| `--num-slots` | 8 (matched V100 budget) | bump to fit L4's ~24 GB if running larger c |

## Day-1 commands (reproducibility-ready)

Substitute `<L4_HOST>` and `<QWEN35_4B_DIR>` for your box's values.

```bash
# 1. Sanity build on L4 (no V100-style env overrides)
ssh <L4_HOST>
cd ~/arle  # or wherever
git pull
export CUDA_HOME=/usr/local/cuda-12.4   # or whichever 12.x toolchain
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=8.9
export INFER_TILELANG_PYTHON=$(which python3)  # any python with tilelang installed
export ARLE_CUDA_DISABLE_FLASHMLA=1  # remove if sm_89 FP8 FlashMLA file builds

cargo build --release -p infer --bin infer --features cuda

# 2. Step 1 sweep — one server per precision
for prec in bf16 int8 fp8 int4; do
  pkill -9 -f target/release/infer 2>/dev/null
  sleep 3
  mkdir -p /tmp/bench_runs/$prec
  nohup ./target/release/infer \
    --model-path <QWEN35_4B_DIR> \
    --port 8000 \
    --kv-cache-dtype "$prec" \
    --num-slots 16 \
    > /tmp/bench_runs/$prec/server.log 2>&1 &
  SERVER_PID=$!
  # wait for /v1/models 200
  for i in $(seq 1 180); do
    curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && break
    sleep 1
  done
  # canonical guidellm (forkserver MP, env -i to dodge httpx proxy URL bug if the box has it)
  env -i HOME=$HOME PATH=/usr/bin GUIDELLM__MP_CONTEXT_TYPE=forkserver \
    $HOME/<venv>/bin/guidellm benchmark \
      --target http://localhost:8000 \
      --model Qwen3.5-4B --processor <QWEN35_4B_DIR> \
      --profile concurrent --rate "1,4,8" \
      --data "prompt_tokens=128,output_tokens=128" \
      --max-seconds 30 \
      --backend-kwargs '{"validate_backend":"/v1/models","request_format":"/v1/completions"}' \
      --disable-console-interactive \
      --output-path /tmp/bench_runs/$prec/guidellm.json \
      > /tmp/bench_runs/$prec/guidellm.log 2>&1
  kill -INT $SERVER_PID 2>/dev/null
  wait $SERVER_PID 2>/dev/null
done

# 3. Extract numbers per precision
for prec in bf16 int8 fp8 int4; do
  echo "=== $prec ==="
  grep -A8 "Request Latency.*TTFT" /tmp/bench_runs/$prec/guidellm.log | head -10
done
```

Output drops into `docs/experience/wins/2026-05-XX-guidellm-ttft-throughput-l4-qwen35.md`
(mirror the V100 wins entry skeleton).

## Industry comparison setup (Day-2 once sanity is green)

```bash
# vLLM via the existing in-tree wrapper
bash scripts/vllm_serve_control.sh           # foreground, port 8000
# in another shell: same guidellm command as above, change target if needed

# SGLang via the existing in-tree wrapper
bash scripts/bench_sglang_longctx.sh sg-l4-128-128 --smoke
```

Both scripts already handle their own venv + model fetch. The
guidellm client-side command is identical to the infer one above —
that's the whole "apples-to-apples" point.

## Substrate watch-list (likely issues to hit on L4)

Borrowed from the V100 session's hard-won list, ranked by V100
likelihood-of-recurrence:

1. **`std::strcasecmp` build error in `quantized_gemv.cu`** — already
   fixed at commit `7b7e1066`, but if someone's branch reintroduces
   it the L4 build dies the same way V100 did. Grep for it before
   the first `cargo build`.
2. **NCCL feature-gate compile error in `main.rs`** — fixed at
   `c971dabc`. If the worker bin build fails on a `new_nccl` /
   `ep_nccl` method, that gate regressed. Grep similarly.
3. **FlashMLA SM90 `__nv_fp8_e8m0` undefined** — fixed by env knob
   `ARLE_CUDA_DISABLE_FLASHMLA=1`. Until Task I from
   `2026-05-29-flashmla-sm89-sm90-build-gate.md` lands, set this on
   any non-sm_90 build (including L4 sm_89).
4. **guidellm 0.6.0 httpx `Invalid port: ':'` bug** — sometimes
   triggered by corp proxy env vars on the box. `env -i HOME=$HOME
   PATH=/usr/bin GUIDELLM__MP_CONTEXT_TYPE=forkserver` clears it.
5. **guidellm `/health` 404** — must pass
   `--backend-kwargs '{"validate_backend":"/v1/models","request_format":"/v1/completions"}'`.
   infer ships `/healthz` + `/v1/models`, not `/health`.

## Out of scope for day 1

- INT4 KV CLI was added at commit `591d1bf6`; reuse as-is on L4.
- Marlin / W4A8 weights still need a checkpoint — defer to day 2 +
  the H1 unblock once a Qwen3.5-4B AWQ variant is available.
- Long-context (4K / 32K) shapes — defer to day 3; the 128/128
  comparison is the credibility-anchor.

## Rule

The pre-flight checklist above is the deliverable. Don't repeat the
substrate-fight from the V100 session; if any of #1-#5 surfaces on
day 1, fix it the same way (commit ref + flag from the list) and
keep moving. The V100 session burned ~2 hours on substrate before
the first bench ran — L4 day 1 should be at first bench within
30 minutes if the watch-list is read first.
