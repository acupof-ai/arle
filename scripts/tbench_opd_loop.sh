#!/bin/bash
# STaR-style agentic-OPD loop over a difficulty-calibrated task pool (pod-side).
# Each round: serve (base|+LoRA) -> pass@N eval -> distil the sweet-spot
# (0<passes<attempts) passing trajectories -> new LoRA. Logs a pass@1 curve.
# DATASET_PATH — an in-band pool (e.g. from tbench_calibrate.sh) — is required;
# the filter is what keeps every round producing gradient (Tmax).
set +e

ROOT=/host/arle-build; ARLE=$ROOT/target/release/arle; MODEL=/host/Qwen3.6-27B-FP8
PY=${PY:-python3}
WORK=${WORK:-/host/tbench_opd}
DATASET_PATH=${DATASET_PATH:?set DATASET_PATH to a calibrated task pool}
ROUNDS=${ROUNDS:-3}; N_ATTEMPTS=${N_ATTEMPTS:-3}; EPOCHS=${EPOCHS:-2}
GPU=${GPU:-0}; PORT=${PORT:-18200}
export PATH="$HOME/.local/bin:$PATH"; export DOCKER_HOST=unix:///run/podman/podman.sock

mkdir -p $WORK
CURVE=$WORK/curve.tsv; echo -e "round\tpass1\ttrials\tnew_records\tcum_records" > $CURVE
CUM=$WORK/records_cum.jsonl; : > $CUM
LORA=${INIT_LORA:-}   # seed from a format-fixed LoRA to isolate the capability gradient

kill_serve(){ pkill -f "arle serve.*--port $PORT" 2>/dev/null; sleep 5; }  # port-scoped
wait_serve(){ for i in $(seq 1 60); do curl -s --max-time 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && return 0; sleep 10; done; return 1; }

for r in $(seq 0 $((ROUNDS-1))); do
  echo "===== ROUND $r $(date -u) (lora=${LORA:-base}) ====="
  kill_serve
  LORA_FLAGS=""; [ -n "$LORA" ] && LORA_FLAGS="--lora-adapters $LORA --lora-alpha 32"
  CUDA_VISIBLE_DEVICES=$GPU nohup $ARLE serve --model-path $MODEL --bind 0.0.0.0 --port $PORT \
    --max-running-requests 4 $LORA_FLAGS > $WORK/serve_r$r.log 2>&1 &
  wait_serve || { echo "serve failed round $r"; break; }

  RUN_ROOT=$WORK/round$r
  OPENAI_API_BASE=http://127.0.0.1:$PORT/v1 OPENAI_API_KEY=dummy NO_PROXY=127.0.0.1,localhost,::1 \
    tb run --dataset-path $DATASET_PATH -a terminus -m openai/Qwen3.6-27B-FP8 \
    --n-attempts $N_ATTEMPTS --n-concurrent 4 --global-agent-timeout-sec 600 --global-test-timeout-sec 60 \
    --output-path $RUN_ROOT > $WORK/eval_r$r.log 2>&1
  RUN=$(ls -td $RUN_ROOT/*/ 2>/dev/null | head -1)

  # pass@1 + sweet-spot passing trials (0<passes<attempts carry gradient;
  # always-pass = zero-gradient, dropped — this is what unflattens the STaR curve)
  read PASS1 TRIALS PASSING < <($PY - "$RUN/results.json" <<'PYEOF'
import json,sys,collections,re
rows=json.load(open(sys.argv[1])).get("results",[])
by=collections.defaultdict(dict)
for x in rows:
    tn=x.get("trial_name",""); m=re.search(r"\.(\d+)-of-\d+", tn)
    by[x.get("task_id")][int(m.group(1)) if m else 1]=(1 if x.get("is_resolved") else 0, tn)
p1=sum(1 for t in by if by[t].get(1,(0,))[0]==1)
sweet=[t for t in by if 0 < sum(o for o,_ in by[t].values()) < len(by[t])]
passing=[v[1] for t in sweet for _,v in by[t].items() if v[0]==1]
print(p1, len(rows), " ".join(passing))
PYEOF
)
  echo "round $r: pass@1=$PASS1/${TRIALS:-?}  sweet-spot passing=$(echo $PASSING | wc -w)"

  # collect passing trajectories -> records, append to cumulative corpus.
  # Pass the raw trial names — terminus_to_records globs `{name}*`, so no brittle
  # regex strip (the [a-z0-9-] class silently dropped underscored gen families).
  NEWREC=0
  if [ -n "$PASSING" ]; then
    $PY $ROOT/scripts/terminus_to_records.py "$RUN" $MODEL $WORK/records_r$r.jsonl $PASSING > $WORK/conv_r$r.log 2>&1
    NEWREC=$(wc -l < $WORK/records_r$r.jsonl 2>/dev/null)
    cat $WORK/records_r$r.jsonl >> $CUM
  fi
  CUMREC=$(wc -l < $CUM 2>/dev/null)
  echo -e "$r\t$PASS1\t$TRIALS\t$NEWREC\t$CUMREC" >> $CURVE

  # distil on the cumulative corpus -> LoRA for the next round (skip if empty/last)
  if [ "$CUMREC" -gt 0 ] && [ "$r" -lt "$((ROUNDS-1))" ]; then
    kill_serve
    CUDA_VISIBLE_DEVICES=$GPU $ARLE train agent-opd --student-model $MODEL \
      --replay-records $CUM --replay-epochs $EPOCHS --lora-rank 16 --lora-alpha 32 \
      --lora-target-set attention-qv --writeback-window 1024 --writeback-cap 60 \
      --save-lora-adapters $WORK/lora_r$r > $WORK/distill_r$r.log 2>&1
    [ -f "$WORK/lora_r$r/adapters_replay/adapter_model.safetensors" ] && LORA=$WORK/lora_r$r/adapters_replay/adapter_model.safetensors
  fi
done
kill_serve
echo "===== LOOP DONE $(date -u) ====="; cat $CURVE
echo "RUN_EXIT=done $(date -u)"
