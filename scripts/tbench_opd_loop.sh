#!/bin/bash
# Automated STaR-style agentic-OPD loop on Terminal-Bench (pod-side).
# Each round: serve (base or +LoRA) -> pass@3 eval -> collect execution-passing
# trajectories -> append to the cumulative corpus -> distill -> new LoRA. Logs a
# per-round pass@1 curve. Env-overridable: ROUNDS, N_ATTEMPTS, EPOCHS, TASKS.
set +e

ROOT=/host/arle-build
ARLE=$ROOT/target/release/arle
MODEL=/host/Qwen3.6-27B-FP8
PY=/host/guidellm-venv/bin/python
WORK=/host/tbench_opd
ROUNDS=${ROUNDS:-3}
N_ATTEMPTS=${N_ATTEMPTS:-3}
EPOCHS=${EPOCHS:-2}
GPU=${GPU:-1}
DS=terminal-bench-core==0.1.1
TASKS=${TASKS:-"hello-world chess-best-move fibonacci-server fix-git fix-permissions csv-to-parquet openssl-selfsigned-cert configure-git-webserver git-workflow-hack nginx-request-logging password-recovery heterogeneous-dates grid-pattern-transform"}
TASK_FLAGS=""; for t in $TASKS; do TASK_FLAGS="$TASK_FLAGS -t $t"; done

mkdir -p $WORK
CURVE=$WORK/curve.tsv; echo -e "round\tpass1\ttrials\tnew_records\tcum_records" > $CURVE
CUM=$WORK/records_cum.jsonl; : > $CUM
LORA=""
export PATH="$HOME/.local/bin:$PATH"
export DOCKER_HOST=unix:///run/podman/podman.sock

kill_serve(){ pkill -f "arle serve" 2>/dev/null; sleep 5; }
wait_serve(){ for i in $(seq 1 60); do curl -s --max-time 3 http://127.0.0.1:18200/v1/models >/dev/null 2>&1 && return 0; sleep 10; done; return 1; }

for r in $(seq 0 $((ROUNDS-1))); do
  echo "===== ROUND $r $(date -u) (lora=${LORA:-base}) ====="
  # 1. serve
  kill_serve
  LORA_FLAGS=""; [ -n "$LORA" ] && LORA_FLAGS="--lora-adapters $LORA --lora-alpha 32"
  CUDA_VISIBLE_DEVICES=$GPU nohup $ARLE serve --model-path $MODEL --bind 0.0.0.0 --port 18200 \
    --max-running-requests 4 $LORA_FLAGS > $WORK/serve_r$r.log 2>&1 &
  wait_serve || { echo "serve failed round $r"; break; }

  # 2. eval pass@N
  RUNDIR=$WORK/round$r
  OPENAI_API_BASE=http://127.0.0.1:18200/v1 OPENAI_API_KEY=dummy NO_PROXY=127.0.0.1,localhost,::1 \
    tb run -d $DS -a terminus -m openai/Qwen3.6-27B-FP8 $TASK_FLAGS \
    --n-attempts $N_ATTEMPTS --n-concurrent 3 --global-agent-timeout-sec 900 --global-test-timeout-sec 300 \
    --output-path $RUNDIR > $WORK/eval_r$r.log 2>&1
  RUN=$(ls -td $RUNDIR/*/ 2>/dev/null | head -1)

  # 3. pass@1 + passing trials
  read PASS1 TRIALS PASSING < <($PY - "$RUN/results.json" <<'PYEOF'
import json,sys,collections
d=json.load(open(sys.argv[1])); rows=d.get("results",[])
by=collections.defaultdict(dict)
for x in rows:
    tn=x.get("trial_name",""); tid=x.get("task_id"); ok=1 if x.get("is_resolved") else 0
    # attempt idx from "task.N-of-K"
    import re; m=re.search(r"\.(\d+)-of-\d+", tn); a=int(m.group(1)) if m else 1
    by[tid][a]=(ok,tn)
p1=sum(1 for t in by if by[t].get(1,(0,))[0]==1)  # attempt-1 = pass@1
passing=[v[1] for t in by for a,v in by[t].items() if v[0]==1]
print(p1, len(rows), " ".join(passing))
PYEOF
)
  echo "round $r: pass@1=$PASS1/${TRIALS:-?}  passing_trials=$(echo $PASSING | wc -w)"

  # 4. collect passing trajectories -> records, append to cumulative
  NEWREC=0
  if [ -n "$PASSING" ]; then
    # strip timestamps: keep task.N-of-K prefix for the converter glob
    PATS=$(for p in $PASSING; do echo "$p" | grep -oE '^[a-z0-9-]+\.[0-9]+-of-[0-9]+'; done | sort -u)
    $PY $ROOT/scripts/terminus_to_records.py "$RUN" $MODEL $WORK/records_r$r.jsonl $PATS > $WORK/conv_r$r.log 2>&1
    NEWREC=$(wc -l < $WORK/records_r$r.jsonl 2>/dev/null)
    cat $WORK/records_r$r.jsonl >> $CUM
  fi
  CUMREC=$(wc -l < $CUM 2>/dev/null)
  echo -e "$r\t$PASS1\t$TRIALS\t$NEWREC\t$CUMREC" >> $CURVE

  # 5. distill on cumulative corpus -> LoRA for next round (skip if empty)
  if [ "$CUMREC" -gt 0 ] && [ "$r" -lt "$((ROUNDS-1))" ]; then
    kill_serve
    CUDA_VISIBLE_DEVICES=$GPU $ARLE train agent-opd --student-model $MODEL \
      --replay-records $CUM --replay-epochs $EPOCHS \
      --lora-rank 16 --lora-alpha 32 --lora-target-set attention-qv --writeback-window 1024 \
      --writeback-cap 60 --save-lora-adapters $WORK/lora_r$r > $WORK/distill_r$r.log 2>&1
    [ -f "$WORK/lora_r$r/adapters_replay/adapter_model.safetensors" ] && LORA=$WORK/lora_r$r/adapters_replay/adapter_model.safetensors
  fi
done
kill_serve
echo "===== LOOP DONE $(date -u) ====="; cat $CURVE
echo "RUN_EXIT=done $(date -u)"
