---
title: PF8.5 launch checkpoint — cargo build --release in flight (PID 1880922) post-PF8.3 commit
date: 2026-05-10
type: research
status: pf85-binary-rebuild-in-flight-license-sequence-pending
---

# PF8.5 launch checkpoint — cargo build --release in flight (PID 1880922) post-PF8.3 commit

> Codex idle post-PF8.3 commit (`11763ba`, "Worked for 1h 15m 41s").
> Per directive **idle + GPU 空 → Claude 自己跑 single-var A/B + bench**,
> kicked off cargo build --release to refresh `target/release/infer`
> with PF8.3 dispatch (was stale per `c6ccd24` warning logic). Next
> tick runs `scripts/pf83_license_sequence.sh` for PF8.5 license
> decision per `aebd4a5` gates.

## §0 Direct evidence (raw process check THIS tick)

```bash
$ ps -p 1880922 -o pid,stat,cputime,cmd
    PID STAT     TIME CMD
1880922 SNl  00:00:00 /home/ckl/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu/bin/cargo build --release -p infer --features cuda

$ tail /tmp/claude-pf83-build.log
warning: cuda-kernels@0.1.5: Compiling CUDA kernels for targets: sm_89
warning: cuda-kernels@0.1.5: TileLang AOT: built per-SM cubins for 1 target(s) across HD64/HD128/HD256 prefill, HD64/HD128/HD256 decode, and Qwen3.5 GDR; SM dispatch via __thread cache + cuDeviceGetAttribute. See docs/plans/sm-coverage.md.
   Compiling infer v0.1.5 (/home/ckl/projects/arle/infer)
   Compiling cuda-kernels v0.1.5 (/home/ckl/projects/arle/crates/cuda-kernels)
```

Build started 07:11. nvcc subprocesses not yet spawned — cuda-kernels
build.rs setup phase. Expected 3-5 min total for incremental rebuild
including marlin_w4_fp8_kernel.cu nvcc compile.

## §1 Codex narration post-PF8.3 commit (raw tmux capture)

```
完成内容：
- 新增 W4+FP8 Marlin prefill CUDA substrate 和 FFI
- 接入 hybrid W4 prefill opt-in dispatch：INFER_MARLIN_W4_FP8_PREFILL=1
- 默认路径不变，PF8 sidecar 只在 env on 时加载，避免默认内存/加载开销
- 修掉 review 抓到的 scale dtype 问题：PF8 kernel 用 FP16 scale/output
  scratch，再转 BF16

验证已过：
- cargo fmt --all --check
- git diff --check
- standalone nvcc sm_89 compile
- cargo check --release -p infer --features cuda
- cargo clippy --release -p infer --features cuda --lib -- -D warnings
- cargo check -p infer --no-default-features --features cuda,no-cuda
- PF8 opt-in targeted greedy_consistency 和 e2e

bench/PPL license 仍按 wins entry 标为 pending-pf8.5。工作区当前 clean。

─ Worked for 1h 15m 41s ───
```

**3 key facts**:
1. PF8 sidecar **lazy-loaded** (not on default path) — addresses memory
   overhead concern from ace3cbe Bug #3 (graph capture vs PF8 scratch)
2. **scale dtype fix**: PF8 kernel uses FP16 scale/output scratch, then
   converts to BF16. This was the post-review fix codex narrated as
   "scale dtype 修正" earlier.
3. **no-cuda typecheck PASSED** — confirms cfg-isolation discipline
   (per CLAUDE.md §Backend isolation, Mac CUDA-Rust typecheck must
   pass without nvcc).

## §2 Next-tick PF8.5 license sequence (concrete invocation)

Once cargo build completes (~5 min from kickoff at 07:11):

```bash
cd /home/ckl/projects/arle

# Sanity check first (verifies all pre-flights)
scripts/pf83_license_sequence.sh --dry-run

# If dry-run shows 5/5 OK + no stale-binary warning, run full sequence:
scripts/pf83_license_sequence.sh

# OR for ~2-min triage:
scripts/pf83_license_sequence.sh --quick
```

Sequence per `aebd4a5` §4:
1. greedy_consistency w4a8 with INFER_MARLIN_W4_FP8_PREFILL=1 +
   INFER_TEST_W4A8_MODEL_PATH=hybrid (per `bf47413`)
2. eval_ppl_pf83.py — PPL Δ% ≤ +1.0% wikitext gate
3. bench_pf83_ab.sh — TTFT Δ% ≥ -8% σ<5% n=3 with RUST_MIN_STACK=8MB
   protect (per `9bb3843`)

License decision per `2e1e73a` matrix:
- LICENSE: pivot #28 Medusa Phase 1.A (per `8735361` unblocked via
  arle data download)
- KILL: errors entry naming WHICH gate failed + pivot to #28 fallback

## §3 Build status monitoring

`/tmp/claude-pf83-build.log` accumulates output. To check progress:
```bash
tail -f /tmp/claude-pf83-build.log
ps aux | grep -E "cargo|nvcc" | grep -v grep
```

If build fails: errors entry + revert to pre-11763ba state (codex's
substrate would need rework).

If build succeeds: target/release/infer mtime > marlin_w4_fp8_kernel.cu
mtime → c6ccd24 stale-binary check passes → license sequence ready.

## §4 Cross-references

- `11763ba` (PF8.3 substrate landed)
- `72540d4` (celebratory pickup state update)
- `c6ccd24` (stale-binary warning that motivated this rebuild)
- `9bb3843` (RUST_MIN_STACK=8MB Task #43 protect)
- `bf47413` (hybrid checkpoint env)
- `e99e5a5` (default to hybrid)
- `a6cf5ac` (--dry-run flag)
- `aebd4a5` (license sequence + PPL gate methodology)
- `2e1e73a` (post-PF8.3 next-axis decision matrix)
- `8735361` (Medusa Phase 1.A pickup chain)

## §5 Status

cargo build --release in flight (PID 1880922, started 07:11). Target:
~5 min for incremental rebuild. Next tick checks build completion +
runs `pf83_license_sequence.sh --dry-run` to verify pre-flights, then
the full sequence.

PF8.5 = **last gate before PF8 chain LICENSE/KILL decision**. Either
outcome converges on #28 Medusa P0 per `2e1e73a` decision matrix.

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(ps -p 1880922 + /tmp/claude-pf83-build.log + tmux capture-pane —
all THIS tick).
