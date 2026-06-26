# Terminal-Bench eval setup — Qwen3.6-27B-FP8 via `arle serve`

Status: harness wired + serve up; smoke in progress. Two new scripts + this doc.
Owner: ckl. Pivot target: attack Terminal-Bench with our model.

## What Terminal-Bench is

[Terminal-Bench](https://www.tbench.ai/) (laude-institute/terminal-bench) is a
benchmark for AI agents in real terminal environments. Each task is:
an English instruction, a **Docker container** the agent operates in by issuing
shell commands, a **test script** that verifies completion, and a reference
("oracle") solution. The harness scores pass/fail per task.

- **Docker is mandatory.** `tb run` builds and runs a per-task container; the
  agent's shell commands execute inside it and the test script runs there.
- Core dataset: `terminal-bench-core` (v0.1.1, ~80 tasks incl. `hello-world`,
  `fix-permissions`, `swe-bench-*`, …; the registry lists the exact
  `task_id_subset`). Full set is slow; run a subset for smokes.
- CLI: installed as `tb` (and `terminal-bench`). `tb run` is the entry.

## The integration: how OUR model is wired in

Terminal-Bench has its own agent loop (`terminus`), so we do NOT use the
in-repo `crates/agent` harness. Instead we expose Qwen3.6-27B-FP8 as an
OpenAI-v1 endpoint and let terminus drive it through **litellm**.

Mechanism (grounded in TB source @ `main`, 2026-06-26):

```
tb run --agent terminus --model openai/Qwen3.6-27B-FP8 -k api_base=<url>
  └ runs.py _process_agent_kwargs: merges model_name + every `-k key=value`
    └ Harness._agent_kwargs ──► terminus_1.Terminus(model_name, api_base, …)
        └ LiteLLM(model_name, api_base) ──► litellm.completion(model=…, api_base=…)
            └ `openai/` prefix ⇒ POST <api_base>/chat/completions (OpenAI-compatible)
```

- `terminus.__init__(self, model_name, temperature, api_base=None, **kwargs)`
  passes `api_base` straight to `LiteLLM`, which passes it to
  `litellm.completion(api_base=…)`. So `-k api_base=http://localhost:8000/v1`
  is the exact knob (`terminal_bench/llms/lite_llm.py:159`).
- litellm's `openai/` provider **requires** a non-empty `OPENAI_API_KEY` even for
  a local server (any dummy string; our serve does not check it).
- Our serve (`arle serve`) exposes `/v1/chat/completions`, `/v1/completions`,
  `/v1/models` (`crates/infer-server/src/http.rs`). The served model id is the
  model-dir basename — `Qwen3.6-27B-FP8` (`model_id_from_path`). Default port
  8000, default bind `127.0.0.1` (`crates/cli/src/args.rs`).

## Topology (why split across two hosts)

| | host | role |
|---|---|---|
| TB harness | **Mac** | has Docker (29.4.0); runs per-task containers |
| `arle serve` | **8×H20 pod** | has the GPUs + the model; NO docker (it's itself a k8s container) |

Evidence: `pod 'docker --version'` → command not found (the pod is a static-pod
container). Docker is local-only. The pod container runs with **host
networking** (verified: a port bound inside the container is visible on the
node's localhost), so a serve bound to `0.0.0.0:PORT` is the node's
`localhost:PORT`. The Mac reaches it with `ssh -L PORT:127.0.0.1:PORT` over the
existing `127.0.0.1:12222` tn tunnel — `terminal_bench_eval.sh` sets this up
automatically.

```
Mac:localhost:8000  ──ssh -L──►  node:localhost:8000  ──host-net──►  container arle serve :8000
   (litellm api_base)              (tn tunnel 12222)                  (Qwen3.6-27B-FP8, TP)
```

### GPU selection (important)

`Qwen3.6-27B` (`model_type=qwen3_5`, arch `Qwen3_5ForConditionalGeneration`)
classifies as `Qwen35` and takes the **multiproc TP** serve path; the TP world
size is the count of `INFER_CUDA_DEVICES`. **Set it to FREE GPUs.** On the
shared box GPUs 0-3 are often held by another DSv4 serve (port 8095); use
`INFER_CUDA_DEVICES=4,5,6,7` for a TP=4 serve on the free half. Without a free
GPU the load OOMs at `upload tensor lm_head.weight`.

## The two scripts

### `scripts/terminal_bench_serve.sh` — run ON THE POD

Starts `arle serve` for Qwen3.6-27B-FP8 bound to `0.0.0.0`.

```bash
# on the pod (e.g. via a pod tmux/setsid session), on the free GPU half:
INFER_CUDA_DEVICES=4,5,6,7 PORT=8100 scripts/terminal_bench_serve.sh
```

Env: `ARLE_BIN` (default `/host/arle-build/target/release/arle`), `MODEL_PATH`
(`/host/Qwen3.6-27B-FP8`), `PORT` (8000), `BIND` (0.0.0.0),
`INFER_CUDA_DEVICES` (TP device list), `EXTRA_FLAGS`.

### `scripts/terminal_bench_eval.sh` — run ON THE MAC

Ensures the serve is reachable (opens `ssh -L` over the tn tunnel, probes
`/v1/models`), then runs `tb run` with terminus on a task subset.

```bash
# smoke (1 task; defaults to a 1-task run if no subset given):
scripts/terminal_bench_eval.sh --task-id hello-world

# a few tasks:
scripts/terminal_bench_eval.sh -n 5
scripts/terminal_bench_eval.sh --task-id hello-world --task-id fix-permissions

# full core set (~80 tasks, slow):
scripts/terminal_bench_eval.sh --all --n-concurrent 4
```

It forwards any unrecognized flag straight to `tb run` (e.g. `--n-concurrent`,
`--livestream`, `--n-attempts`). Key env: `PORT` (8000 — match the forwarded
serve port), `MODEL_ID` (`Qwen3.6-27B-FP8`), `API_BASE`, `OPENAI_API_KEY`
(dummy `arle-local`), `DATASET_NAME`/`DATASET_VER`, `N_CONCURRENT`,
`OUTPUT_DIR` (`runs/terminal-bench`), `SKIP_TUNNEL=1` (forward yourself),
`SKIP_PROBE=1`. Results land in `OUTPUT_DIR` (TB writes per-trial logs + a
results json scored pass/fail).

> If the serve port on the pod is not 8000 (e.g. 8100 to dodge another
> workload), run the eval with `PORT=8100` so both the `ssh -L` forward and the
> litellm `api_base` line up.

## Env requirements

- **Mac**: Docker running; `uv tool install terminal-bench` (or `pip install
  terminal-bench`) so `tb` is on PATH; the tn keepalive tunnel up
  (`lsof :12222`).
- **Pod**: a built `arle` binary, the model at `/host/Qwen3.6-27B-FP8`, free
  GPUs for the TP serve.

## Smoke result

See the dated wins entry under `docs/experience/wins/` (or the commit body if
`pending-remote`): records whether terminus reached our serve, issued shell
commands, and got a pass/fail verdict on `hello-world`.

## What's not yet covered

- Full-suite scoring + a leaderboard-comparable number (this is harness setup,
  not the eval campaign).
- A `pending-remote` path if the Mac cannot keep Docker + the tunnel up for a
  long full run; consider a Docker-capable Linux host co-located with the pod
  network to drop the `ssh -L` hop.
