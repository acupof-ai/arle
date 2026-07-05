# Terminal-Bench on the H20 pod: 27B frontier baseline = 42.86% pass@1

> Status: Active — 2026-07-05 · Driver: ckl ("换 terminal bench 2.1" →
> frontier-calibrated OPD substrate). Supersedes the 7-instance SWE-Pro
> ansible corpus as the OPD dataset.

## Goal

Establish a **frontier-calibrated** dataset for the agentic-OPD capability
curve, replacing the 7 hand-scraped SWE-Pro ansible instances (not a dataset —
3 too-easy, 4 too-hard, the frontier band never measured). Run the standardized
Terminal-Bench harness against the Qwen3.6-27B-FP8 student served on one H20,
and measure the base pass@1 to locate the OPD frontier band.

## Why this pivot (the essence)

The prior self-distill experiment on SWE-Pro returned Δ≈0 held-out, and the
root cause was **not** infra — it was that OPD needs a *capability gradient*:
the distilled trajectories must contain something the student can't already do.
That run distilled the student's own trajectories on instances it already
solved 3/3 (zero information); the instances where lift was possible produced no
passing trajectories (bootstrap wall). The fix is a dataset that (a) the student
fails on-policy — headroom, (b) a teacher can solve — a trajectory source, (c)
has enough tasks and cheap-enough scoring to measure a Δpp above noise.
Terminal-Bench (89 standardized tasks, difficulty tiers, execution-scored)
supplies (a)+(c); (b) is the next phase.

## Params

- **Harness**: `terminal-bench` (pip, "TB 2.x"); dataset
  `terminal-bench-core==0.1.1` (89-task launch set; `head` download is broken —
  `FileNotFoundError .../tasks`).
- **Agent**: `terminus` (TB-native, lighter than claude-code — avoids CC's
  ~17K-token system prompt re-prefill), `--n-concurrent 3`.
- **Model**: `openai/Qwen3.6-27B-FP8` via `OPENAI_API_BASE=http://127.0.0.1:18200/v1`
  (ARLE serve, `--max-running-requests 4`, GPU 1). Thinking ON by default
  (Qwen3.6 template) — the 43% is already a think-on number.
- **Tasks (curated 14, light/medium; heavy qemu/kernel builds excluded)**:
  hello-world, chess-best-move, count-dataset-tokens, fibonacci-server, fix-git,
  fix-permissions, csv-to-parquet, openssl-selfsigned-cert,
  configure-git-webserver, git-workflow-hack, nginx-request-logging,
  password-recovery, heterogeneous-dates, grid-pattern-transform.

## Env

- Pod: sglang-test container on the 8×H20 box, amd64, uid=0. NO docker →
  **podman 4.9.3** (apt) + crun 1.14.1 + docker-compose-v2 2.40.3 +
  podman-docker shim. storage `driver=vfs` on `/host` (overlay-on-overlay fails
  nested). `podman system service` manual (no systemd) →
  `DOCKER_HOST=unix:///run/podman/podman.sock`. Native x86_64 containers.
- Egress: Docker Hub blocked from the pod → `tn proxy` reverse SOCKS5 on
  `127.0.0.1:1080` (Mac's network), `ALL_PROXY/HTTP(S)_PROXY=socks5h://…` +
  `NO_PROXY=127.0.0.1,localhost` (local model calls must bypass). Full recipe:
  memory `reference_terminal_bench_on_pod_podman_proxy`.
- ARLE HEAD at the CC-distill-channel commits + the `student_lora` PEFT-name fix.

## Results

**Base 27B pass@1 = 42.86% (6/14 resolved).** Non-saturated → a valid frontier.

| Verdict | Tasks |
|---|---|
| PASS (6) | hello-world, heterogeneous-dates, git-workflow-hack, fix-permissions, csv-to-parquet, openssl-selfsigned-cert |
| FAIL (8) | count-dataset-tokens, grid-pattern-transform, chess-best-move, configure-git-webserver, fix-git, nginx-request-logging, fibonacci-server, password-recovery |

- End-to-end validated: `hello-world` resolves 100% in ~63s (image build via
  proxy → terminus → 27B thinking+action → container exec → test PASS).
- 3 FAILs were 360s agent-timeouts (csv-to-parquet and openssl still resolved
  on another trial; password-recovery timed out to FAIL) — the per-turn
  re-prefill makes long episodes slow; a higher `--global-agent-timeout-sec`
  may recover a few.
- The 8 FAIL tasks are the OPD target band.

## Problems (hard-won, all in the memory recipe)

1. Pod has no container runtime — installed podman-in-container (root + userns +
   /dev/fuse; vfs storage).
2. Docker Hub unreachable from the pod — solved by `tn proxy` (ckl); DaoCloud
   `docker.m.daocloud.io` is a fallback mirror.
3. `docker compose` missing — podman-docker + docker-compose-v2 cli-plugin.
4. tb SOCKS: needs `--with socksio --with requests[socks] --with httpx[socks]`.
5. **terminus max_tokens bug**: `llms/lite_llm.py::call` omits max_tokens → serve
   truncates → `OutputLengthExceededError`. Patched
   `kwargs.setdefault("max_tokens", 16000)` (Qwen3.6 thinking needs headroom).
6. `--n-tasks 1` picks by duration → grabbed `qemu-alpine-ssh` (>40min VM
   build). Always pick task-ids explicitly.

## Learnings / next

- Terminal-Bench is the right OPD substrate: 43% base pass@1 is squarely in the
  frontier band, unlike the saturated toy corpus or the too-hard SWE-Pro tail.
- terminus already runs the 27B think-on, so "same-27B think-on teacher" is not
  a separate lever here. The two real levers for the 8 FAIL tasks:
  **(1) pass@k self-mining** — N attempts/task, distill the lucky passes (pure
  on-policy, no external teacher); **(2) a stronger external teacher** driving
  terminus.
- **Next run**: pass@k mining on the 8 FAIL tasks (`--n-attempts 5`) → identify
  which are solvable-on-retry → extract terminus trajectories from the run's
  agent-logs → distill → re-eval pass@1 → Δpp.

Raw: pod `/host/tb_runs/2026-07-04__18-53-35/results.json`,
`/host/tb_baseline_run.log`.
