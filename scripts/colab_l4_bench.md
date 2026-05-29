# L4 Colab cell-by-cell bench harness

Paste each numbered block into a Colab cell, run top-to-bottom. The
final cell prints all four precision results. Paste that output back
to whoever's coordinating the bench.

Mirrors `docs/plans/2026-05-29-l4-day1-bench-handover.md` runbook but
runs in-Colab rather than over SSH.

## Cell 1 — sanity checks (5 s)

```python
!nvidia-smi -L
!nvcc --version | tail -1
!free -g | head -3
!df -h / | tail -1
```

Expected: `Tesla L4` or `L4` line; nvcc 12.x; ≥ 50 GB free.

## Cell 2 — install build deps (~2 min)

```python
%%bash
apt-get install -qy build-essential cmake ninja-build pkg-config libssl-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.95.0 --profile minimal
source $HOME/.cargo/env
rustc --version
pip install -q tilelang
python -c "import tilelang; print('tilelang ok')"
```

## Cell 3 — clone arle + fetch Qwen3.5-4B (~3 min, ~10 GB)

```python
%%bash
cd /content
git clone --depth 1 https://github.com/cklxx/arle.git
cd arle
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', local_dir='/content/Qwen3.5-4B')"
ls -lh /content/Qwen3.5-4B/*.safetensors 2>/dev/null | head
```

## Cell 4 — build infer with cuda features (~6-8 min, L4 sm_89)

```python
%%bash
source $HOME/.cargo/env
cd /content/arle
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=8.9
export INFER_TILELANG_PYTHON=$(which python3)
export ARLE_CUDA_DISABLE_FLASHMLA=1   # sm_89 still lacks __nv_fp8_e8m0
cargo build --release -p infer --bin infer --features cuda 2>&1 | tail -8
ls -la target/release/infer
```

## Cell 5 — install guidellm bench client (~30 s)

```python
%%bash
pip install -q guidellm==0.6.0
guidellm --version 2>&1 | head -1
```

## Cell 6 — 4-precision sweep, c=1/4/8 (~12 min total)

```python
%%bash
cd /content/arle
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
MODEL=/content/Qwen3.5-4B
mkdir -p /content/bench

for prec in bf16 int8 fp8 int4; do
  echo "============================================================"
  echo "=== $prec ==="
  echo "============================================================"
  pkill -9 -f target/release/infer 2>/dev/null
  sleep 3
  mkdir -p /content/bench/$prec
  nohup ./target/release/infer \
    --model-path "$MODEL" \
    --port 8000 \
    --kv-cache-dtype "$prec" \
    --num-slots 16 \
    > /content/bench/$prec/server.log 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" > /content/bench/$prec/server.pid
  for i in $(seq 1 240); do
    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
      echo "server up after ${i}s"; break; fi
    sleep 1
  done
  if ! curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "SERVER FAILED — tail of server.log:"; tail -30 /content/bench/$prec/server.log
    kill -9 $SERVER_PID 2>/dev/null; continue
  fi
  env -i HOME=$HOME PATH=/usr/bin GUIDELLM__MP_CONTEXT_TYPE=forkserver \
    guidellm benchmark \
      --target http://localhost:8000 \
      --model Qwen3.5-4B --processor "$MODEL" \
      --profile concurrent --rate "1,4,8" \
      --data "prompt_tokens=128,output_tokens=128" \
      --max-seconds 30 \
      --backend-kwargs '{"validate_backend":"/v1/models","request_format":"/v1/completions"}' \
      --disable-console-interactive \
      --output-path /content/bench/$prec/guidellm.json \
      > /content/bench/$prec/guidellm.log 2>&1 || true
  kill -INT $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
done
echo "ALL-DONE"
```

## Cell 7 — extract numbers (paste this output back)

```python
%%bash
for prec in bf16 int8 fp8 int4; do
  echo "=== $prec ==="
  grep -A8 "Request Latency.*TTFT" /content/bench/$prec/guidellm.log 2>/dev/null | head -10
  echo
  grep "TokenKVPool.*format=\|scales=.*MB" /content/bench/$prec/server.log 2>/dev/null | tail -3
  echo
done
```

## Notes

- Total wall-clock ≈ **25-30 min** end-to-end (~5 min deps + ~3 min
  fetch + ~6-8 min build + ~12 min sweep + ~30 s extract).
- If Colab times out mid-sweep: precisions run sequentially, so
  whatever finished is in `/content/bench/<prec>/`. Run Cell 7 alone
  to extract partial results.
- For long-context (4K / 32K) or higher concurrency, edit Cell 6's
  `--data` and `--rate` strings; everything else stays.
