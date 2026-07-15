#!/bin/bash
# Gen a hard pool -> pass@3 label on a format-fixed model (base's JSON deficit
# else reads every task as fail) -> keep 0<pass<attempts -> $WORK/calib feeds
# tbench_opd_loop.sh. The eval is the only signal for "hard *for the model*".
set +e

ROOT=/host/arle-build; ARLE=$ROOT/target/release/arle; MODEL=/host/Qwen3.6-27B-FP8
PY=${PY:-python3}
WORK=${WORK:-/host/tb_calib}; PORT=${PORT:-18200}; GPU=${GPU:-0}
LORA=${LORA:-/host/tb_lora/adapters_replay/adapter_model.safetensors}  # +5pp format fix
N=${N:-36}; MINC=${MINC:-1}; MAXC=${MAXC:-3}   # full spread by default; band sits at c3
export PATH="$HOME/.local/bin:$PATH"; export DOCKER_HOST=unix:///run/podman/podman.sock

rm -rf $WORK; mkdir -p $WORK; POOL=$WORK/pool

# 1. gen a difficulty-spread pool
$PY $ROOT/scripts/gen_terminal_tasks.py --out $POOL --n $N --min-complexity $MINC --max-complexity $MAXC \
  --seed 7 --self-check > $WORK/gen.log 2>&1
echo "pool: $(ls $POOL | wc -l) tasks"

# 2. serve base + format LoRA (4-wide: 8-wide starved each constrained-decode
#    request and stalled the eval into all-timeout)
CUDA_VISIBLE_DEVICES=$GPU nohup $ARLE serve --model-path $MODEL --bind 0.0.0.0 --port $PORT \
  --max-running-requests 4 --lora-adapters $LORA --lora-alpha 32 > $WORK/serve.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 10; done

# 3. pass@3 difficulty label (600s agent cap bounds runaway-reasoning waste)
OPENAI_API_BASE=http://127.0.0.1:$PORT/v1 OPENAI_API_KEY=dummy NO_PROXY=127.0.0.1,localhost,::1 \
  tb run --dataset-path $POOL -a terminus -m openai/Qwen3.6-27B-FP8 \
  --n-attempts 3 --n-concurrent 4 --global-agent-timeout-sec 600 --global-test-timeout-sec 60 \
  --output-path $WORK/run > $WORK/eval.log 2>&1

# 4. keep the in-band tasks + print the per-family band histogram
R=$(ls -td $WORK/run/*/ 2>/dev/null | head -1)
$PY $ROOT/scripts/filter_inband.py "$R/results.json" $POOL $WORK/calib > $WORK/band.txt 2>&1
pkill -f "arle serve.*--port $PORT" 2>/dev/null
echo "CALIB_DONE $(date -u)"; cat $WORK/band.txt
