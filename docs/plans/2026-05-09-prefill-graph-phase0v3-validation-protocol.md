---
title: Prefill graph Phase 0v3 (#24) functional validation protocol — pre-built
date: 2026-05-09
type: plan
status: pre-built
audience: codex (when #24 implementation lands), Claude (auditor)
---

# Phase 0v3 (#24) functional validation protocol — concrete commands ready when codex commits

> Pre-built validation infra so codex commit → Claude immediately verifies
> Phase 0v3 license without delay。NOT throughput gate(#37 Phase 2 工作),only
> functional + nsys-evidence + Phase-0-anti-pattern check。

## 0. Pre-flight — codex commit 检查

Codex commit landing 后,Claude 立即:

```bash
git fetch origin main && git log --oneline -3 origin/main
git diff --stat $PRIOR_HEAD..origin/main -- infer/src/ crates/cuda-kernels/csrc/
```

**LOC budget audit**:
- Per task #24 description:**单 PR 不超过 380 LOC**(insertions only)
- 检查 `git diff --shortstat`:`<N> insertions`
- ≤ 380 LOC ✓ proceed validation
- 380-500 LOC ⚠ message codex justify scope blow
- > 500 LOC ❌ require split commit before audit

## 1. Correctness gate(必过)

```bash
# Build with cuda feature(release only,debug too slow per CLAUDE.md)
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  cargo check --release -p infer --features cuda 2>&1 | tail -10

CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  cargo clippy --release -p infer --features cuda -- -D warnings 2>&1 | tail -5

# Test the prefill graph path explicitly opt-in
INFER_PREFILL_GRAPH=1 \
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
  cargo test --release -p infer --features cuda --test e2e 2>&1 | tail -20

INFER_PREFILL_GRAPH=1 \
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
  cargo test --release -p infer --features cuda --test greedy_consistency 2>&1 | tail -20
```

**Expected** — all 4 pass。**Failure** = bug,KILL Phase 0v3,errors entry。

## 2. Functional smoke — server boot + 200 OK

Match codex baseline command exactly,加 `INFER_PREFILL_GRAPH=1`:

```bash
INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1 \
  CUDA_HOME=/opt/cuda \
  NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  RUST_LOG=info \
  ./target/release/infer \
    --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix \
    --port 8000 --num-slots 8 --max-seq-len 8192 \
    --admission-policy prefix-aware \
    2>&1 | tee /tmp/infer-phase0v3.log &

sleep 35    # 给 model load + tilelang AOT 编译 时间
curl -fsS -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-4B-W4-hybrid-zpfix","prompt":"hello","max_tokens":4,"stream":false}' \
  | head -50
```

**Expected**:
- ✓ 200 OK + 4 token completion
- ✓ no panic in `/tmp/infer-phase0v3.log`
- ✓ "prefill graph capture" log line(if codex 加了 capture log,per Phase 0 precedent)

**Failure modes**:
- ❌ panic on model load → graphsafe_batched_weight extension bug
- ❌ panic on inference → MarlinPrefillScratch lifetime bug
- ❌ 500 internal error → graph capture invalidation loop

## 3. nsys evidence — W4 weights 实际 enter capture path

```bash
# nsys capture 60s of c=4 4k/256 prefill
INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1 \
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/profile_nsys_guidellm.sh phase0v3-w4-graph-evidence \
  --concurrencies 4 --max-seconds 60 \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'

# Analyze nsys report for graph capture / replay events
nsys stats --report cuda_api_sum --report cuda_gpu_kern_sum \
  bench-output/2026-05-09-phase0v3-w4-graph-evidence-*/profile.nsys-rep \
  | grep -iE 'cudaGraphLaunch|cudaGraphInstantiate|cudaStreamBeginCapture' | head -20
```

**Expected**(per `2026-05-08-m_pgc-phase0-killed` precedent log format):
- `prefill graph capture key: ≥ 30`(类似 Phase 0 数据)
- **W4 layer 在 capture path**:`grep "marlin\|w4" /tmp/infer-phase0v3.log | grep "graph"` 应有 hit
- `cudaGraphLaunch` 计数 ≥ N(N = num requests × num prefill chunks)

**KILL signals**(re-replay Phase 0 anti-patterns):
- ❌ `prefill graph fallback reason=token-count: > capture key count` → tail-token issue 没 fix
- ❌ `prefix cache pressure fallback: > 5%` → KV pressure 太大(应 use auto KV mode 不是 BF16-forced)
- ❌ `Plan labels: prefill > N`,N >> codex baseline 179 → envelope clamp serialization 没 remove

## 4. Matched-control bench(non-strategic,只验 not regression)

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh phase0v3-functional-noregression \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
```

**Expected** Phase 0v3 alone:
- TTFT 4k/c=4:**1500-1700 ms**(codex baseline 1639 ± noise,Phase 0v3 单独可能 small win 0-10%,不 expect throughput gate close)
- ITL:**unchanged ± 5%** 
- σ:**< 5%** across run

**KILL signals**(any 触发 → Phase 0v3 broken):
- ❌ TTFT > 1900 ms(类 Phase 0 KILL 1961 ms regression 重演)
- ❌ ITL > 30 ms(Phase 0 KILL 25.6 ms regression 重演)
- ❌ out tok/s < 175(Phase 0 KILL 122.95 regression 重演)

## 5. License decision matrix

| 维度 | PASS 条件 | FAIL 条件 |
|------|----------|----------|
| Correctness | 4/4 cargo test pass | any fail |
| Server smoke | 200 OK + no panic | any panic / 500 |
| nsys evidence | W4 layer 进入 capture path + capture key ≥ 30 | W4 not in capture OR capture key 0 |
| No regression | TTFT/ITL/tok-s 在 baseline ± 10% noise | Phase 0 KILL 数值重演(±) |
| LOC budget | ≤ 380 insertions | > 500 require split |

**Pass all 5** → Phase 0v3 ✓ → 立即 brief codex on **#37 Phase 2 multi-key cache + tail handling**(my next pickup brief)。
**Fail any** → KILL with errors entry,document failure mode,re-scope #24 sub-fix。

## 6. Phase 0 anti-pattern check(per skill v1.7.0 #6)

每 validation 必检:
- ✓ "Capture exists" 不等同 "capture reused" — 看 `cudaGraphLaunch` count vs `cudaGraphInstantiate` count
  - 健康:launch >> instantiate
  - 病态:launch ≈ instantiate(Phase 0 病)
- ✓ Matched-control:Phase 0v3 是否用 same KV format vs codex baseline?
  - 健康:auto FP8 OR auto W4-hybrid 同 baseline
  - 病态:BF16-forced(Phase 0 KILL contamination)

## 7. Cross-references

- Architecture brief:`docs/research/2026-05-09-prefill-graph-w4-prereq-architecture.md`
- Phase 0 KILL evidence:`docs/experience/errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md`
- Codex baseline:`docs/experience/wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`
- Bench wrapper:`scripts/bench_guidellm.sh`
- nsys wrapper:`scripts/profile_nsys_guidellm.sh`
- Skill anti-pattern #6:License on capture reuse,not capture exists

## 8. 状态

#24 Phase 0v3 validation protocol pre-built。Codex commit 落地后 Claude 立即可
run 5 个 gate 验证(correctness / smoke / nsys / no-regression / LOC budget)+
Phase 0 anti-pattern check。Pass → brief #37 Phase 2;Fail → KILL with errors entry。
