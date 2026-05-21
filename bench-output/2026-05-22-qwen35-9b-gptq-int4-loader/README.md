# Qwen3.5-9B GPTQModel W4 Loader Probe

Model:

```text
/home/ckl/.cache/modelscope/hub/DavidWen2025/Qwen3___5-9B-GPTQ-4bit
```

Result:

- Experimental GPTQModel W4 physical-layout loader reached HTTP readiness.
- Peak observed GPU memory while loaded: `15230 MiB / 16376 MiB`.
- Multi-token generation quality failed: all 3 prompts produced repeated `!`.
- Default path is therefore fail-closed; `fail-closed.log` captures the
  actionable error that requires `INFER_EXPERIMENTAL_GPTQMODEL_W4=1`.

Key artifacts:

- `serve.log`: experimental loader run with HTTP readiness.
- `completion_*.json`: failed generation outputs.
- `fail-closed.log`: default safety gate after the failed quality run.
- `nvidia-smi-*.txt`: memory snapshots.
