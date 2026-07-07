#!/bin/bash
# Full Terminal-Bench (80 core==0.1.1 tasks) on the 27B + format LoRA, pass@1,
# --dataset-path over the local cache (no registry fetch). Heavy tasks
# (qemu/kernel/swe-bench) fail on pod infra not capability — read per-task.
set +e
ROOT=/host/arle-build; ARLE=$ROOT/target/release/arle; MODEL=/host/Qwen3.6-27B-FP8
PY=/host/guidellm-venv/bin/python
W=/host/tb_full; rm -rf $W; mkdir -p $W
PORT=${PORT:-18300}; GPU=${GPU:-5}   # 5-7 free; 1-4 held by another pod job
LORA=/host/tb_lora/adapters_replay/adapter_model.safetensors
POOL=/root/.cache/terminal-bench/terminal-bench-core/0.1.1
export PATH="$HOME/.local/bin:$PATH"; export DOCKER_HOST=unix:///run/podman/podman.sock

CUDA_VISIBLE_DEVICES=$GPU nohup $ARLE serve --model-path $MODEL --bind 0.0.0.0 --port $PORT \
  --max-running-requests 4 --lora-adapters $LORA --lora-alpha 32 > $W/serve.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 10; done

OPENAI_API_BASE=http://127.0.0.1:$PORT/v1 OPENAI_API_KEY=dummy NO_PROXY=127.0.0.1,localhost,::1 \
  tb run --dataset-path $POOL -a terminus -m openai/Qwen3.6-27B-FP8 \
  --n-attempts 1 --n-concurrent 4 --global-agent-timeout-sec 600 --global-test-timeout-sec 300 \
  --output-path $W/run > $W/eval.log 2>&1

R=$(ls -td $W/run/*/ 2>/dev/null | head -1)
$PY - "$R/results.json" > $W/score.txt 2>&1 <<'PYEOF'
import json,sys,collections
rows=json.load(open(sys.argv[1])).get("results",[])
res=sum(1 for x in rows if x.get("is_resolved")); n=len(rows)
fm=collections.Counter(x.get("failure_mode","") for x in rows if not x.get("is_resolved"))
print(f"FULL TB pass@1: {res}/{n} = {res/n:.1%}" if n else "no results")
print("failure_modes:", dict(fm))
PYEOF
pkill -f "arle serve.*--port $PORT" 2>/dev/null
echo "FULLTB_DONE $(date -u)"; cat $W/score.txt
