# TileLang Phase 0 — H100 verification runbook

Companion to [`tilelang-integration.md`](tilelang-integration.md). Step-by-
step procedure the operator runs on a remote H100 to take Phase 0 from
"committed locally" to a reviewable bench delta. Every step has a
copy-paste command and an explicit pass/fail signal so the run is
reproducible without referring back to other docs.

Target commits (everything from this range must be present on the H100
checkout):

```text
022e8dd feat(cuda): add TileLang prefill HD128 path behind tilelang-attn feature
76e044b refactor(cuda): TileLang prefill HD128 specializes per Qwen3 head config
9896d25 fix(cuda): make tilelang-attn actually reach the prefill path
```

Whatever follows on `main` after these is also fine. The only
hard requirement is that `git log -1 -- crates/cuda-kernels/build.rs |
grep -q tilelang_target` returns 0 (i.e., the TileLang AOT track is in
the tree).

---

## 1 · Pre-flight (don't skip)

Each line below should pass before touching the build.

```bash
# 1.1 GPU is what we expect (sm_90, ≥40 GB free)
nvidia-smi --query-gpu=name,compute_cap,memory.free --format=csv,noheader
# Expect: H100*, 9.0, free ≥ 40000 MiB

# 1.2 CUDA toolkit is reachable
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version | grep release
# Expect: CUDA_HOME=/usr/local/cuda (or equivalent), release 12.x

# 1.3 Repo is at a Phase-0-bearing commit
git log -1 --oneline
git log --oneline | grep -E '022e8dd|76e044b|9896d25' | wc -l
# Expect: 3

# 1.4 TileLang Python is installed and importable
pip install -e ".[tilelang]"          # idempotent
python3 -c "import tilelang; print(tilelang.__version__)"
# Expect: a version string. Record the exact version in the bench entry §6.
```

If any line fails, stop. Don't paper over a missing prerequisite — fix it
or stop the run.

---

## 2 · Build verification

Two binaries: an `off` (FlashInfer) and an `on` (TileLang). They MUST be
matched A/B per `feedback_matched_ab_for_small_bench_effects.md`: same
commit, same machine, same nvcc, same TileLang version — built
back-to-back, not on different days.

### 2.1 `off` build (default FlashInfer)

```bash
CUDA_HOME=/usr/local/cuda \
  cargo build --release --features cuda 2>&1 | tee /tmp/build-off.log
```

Pass: `Finished \`release\` profile`, no errors.
Fail: any `error[E…]` or linker error. Stop and triage.

### 2.2 `on` build (TileLang)

```bash
CUDA_HOME=/usr/local/cuda \
  cargo build --release --features cuda,tilelang-attn 2>&1 | tee /tmp/build-on.log
```

What to grep for in the build log (each is a positive signal):

```bash
grep "TileLang AOT enabled"           /tmp/build-on.log   # should see ≥1 line
grep "tilelang_aot/batch_prefill_paged_hd128_q[0-9]*_kv8" /tmp/build-on.log
ls target/release/build/cuda-kernels-*/out/tilelang_aot/
# Expect 4 dirs:
#   batch_prefill_paged_hd128_q16_kv8
#   batch_prefill_paged_hd128_q32_kv8
#   batch_prefill_paged_hd128_q40_kv8
#   batch_prefill_paged_hd128_q64_kv8
# Each contains tilelang_batch_prefill_paged_hd128_q*_kv8.{cubin,c}
```

Risk gate per `tilelang-integration.md` §5:

- **#1 AOT export fails on sm_90** → the build panics with
  `TileLang AOT generator failed for tilelang_batch_prefill_paged_hd128_q*_kv8_run`.
  Action: write `docs/experience/errors/2026-04-…-tilelang-aot-sm90-blocker.md`,
  `git revert 9896d25 76e044b 022e8dd` in that order, push the revert.
  Phase 0 closed with a recorded blocker.
- **#2 paged-KV primitive rejected by tilelang.compile()** → same panic,
  but the underlying error message names the offending kernel construct
  (likely `KV_indices[...]` indirect indexing). Same revert path.

If only one of the four configs fails (e.g. `(64,8)` only), narrow
`TILELANG_PREFILL_HD128_HEAD_CONFIGS` in `crates/cuda-kernels/build.rs`
to the working subset, rebuild, and proceed — record the narrowing in
the bench entry §Problems.

### 2.3 Confirm the binary is actually self-contained

```bash
# 1) Binary contains the cubin bytes (cuModuleLoadData wrapper, not path)
strings target/release/infer | grep -c "tilelang_batch_prefill_paged_hd128_q.._kv8_run" || true
# Expect ≥4 (one symbol per supported head config).

# 2) Cubin embedded, not stored as a string path
strings target/release/infer | grep -E "tilelang_aot/batch_prefill_paged_hd128_q.._kv8/.+\.cubin" && {
  echo "FAIL: binary references cubin by absolute path; cubin embedding broke." >&2
  exit 1
} || echo "PASS: no embedded cubin paths in binary"
```

The second check is the codex P2 #1 regression guard. If it ever finds
a path string, the AOT generator regressed back to `cuModuleLoad` —
fix `gen_tilelang_aot.py::write_c_wrapper` before continuing.

---

## 3 · Numerical parity (correctness gate)

Both binaries must produce numerically equivalent output on the
deterministic e2e baseline. Different output here = TileLang kernel is
wrong; A/B perf numbers are meaningless.

```bash
# 3.1 off
cargo test --release --features cuda --test e2e -- --nocapture 2>&1 | tee /tmp/e2e-off.log
# Expect: test result: ok. <N> passed; 0 failed

# 3.2 on
cargo test --release --features cuda,tilelang-attn --test e2e -- --nocapture 2>&1 | tee /tmp/e2e-on.log
# Expect: same N passed, 0 failed.

# 3.3 confirm both ran the same baseline (Qwen3-4B by default)
diff <(grep '^test' /tmp/e2e-off.log) <(grep '^test' /tmp/e2e-on.log)
# Expect: empty diff (same test names, same outcomes).
```

If 3.2 fails: STOP. Open `docs/experience/errors/2026-04-…-tilelang-prefill-hd128-numerical.md`
with the exact failing case, prompt, expected substring, observed
output. Revert and close Phase 0 — there is no point benchmarking a
wrong kernel.

If 3.3 produces a diff: investigate which test flips outcome between
runs. Numerical parity must be exact substring-match per
`infer/test_data/Qwen3-4B.json`.

---

## 4 · Bench A/B sweep

Both runs use the canonical `scripts/bench_guidellm.sh` with no
parameter changes. Run them back-to-back on a quiet machine
(`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` reads
0 % between runs).

```bash
# 4.1 off — FlashInfer baseline
INFER_FEATURES="cuda" scripts/start_infer.sh models/Qwen3-4B 8000 &
SERVER_PID=$!
# wait for the model to be loaded; rough proxy:
until curl -s http://localhost:8000/v1/models > /dev/null; do sleep 5; done
scripts/bench_guidellm.sh tilelang-prefill-off
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

# 4.2 on — TileLang
INFER_FEATURES="cuda,tilelang-attn" scripts/start_infer.sh models/Qwen3-4B 8000 &
SERVER_PID=$!
until curl -s http://localhost:8000/v1/models > /dev/null; do sleep 5; done
scripts/bench_guidellm.sh tilelang-prefill-on
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
```

Expected artefacts per run:

```text
bench-output/<date>-tilelang-prefill-{off,on}/
├── benchmarks.json
├── benchmarks.csv
├── benchmarks.html
├── service_stats_before.txt
├── service_stats_trace.jsonl
├── service_stats_after.txt
└── service_stats_trace_summary.md
```

If the `on` run shows the same TTFT / out-tok-s as `off` to ≤1 %, double-
check that the binary really has TileLang on:

```bash
file target/release/infer
strings target/release/infer | grep -c tilelang_batch_prefill_paged_hd128_q
# Expect ≥4. If 0, INFER_FEATURES did not propagate — re-build manually
# with `cargo build --release --features cuda,tilelang-attn` and rerun §4.2.
```

---

## 5 · Decision matrix

Per [`tilelang-integration.md`](tilelang-integration.md) §5, with the
exact thresholds spelt out:

| `on` Δ vs `off`                                  | Action |
|---|---|
| TTFT p50 ≥ −10 % at synchronous **AND** out-tok-s ≥ +10 % at saturation | **Phase 1 starts.** Open `docs/plans/tilelang-decode.md` from this template; migrate decode HD128/HD256. |
| Both metrics within ±5 %                          | **Ship-and-hold.** Update bench entry to "flat", land it; do not start Phase 1. Feature stays in tree, default off. |
| TTFT regresses ≥5 % at synchronous **OR** out-tok-s regresses ≥5 % at saturation | **Revert.** `git revert 9896d25 76e044b 022e8dd`, write errors/ entry, push. |
| Anywhere in §1–§4 a step failed | **Revert per the action listed at that step.** |

The 5–10 % no-go band is intentional. A 7 % win does not justify
carrying two attention paths long-term; a 7 % loss is too small to
matter relative to bench noise.

---

## 6 · Bench entry handoff

Replace
[`docs/experience/wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md`](../experience/wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md)
with two new dated entries — one for `off`, one for `on` — using
[`TEMPLATE-bench-guidellm.md`](../experience/wins/TEMPLATE-bench-guidellm.md).
The `on` entry's `## Δ vs baseline` cites the `off` entry as the
matched A/B pair.

Both entries record:

- exact `git rev-parse HEAD`
- `tilelang.__version__` from §1.4
- nvcc release from §1.2
- the four AOT cubins observed in §2.2

Then delete the pending-remote stub.

---

## 7 · If you have to roll back

```bash
# Latest first; otherwise dependent commits leave the tree in a
# half-applied state.
git revert --no-edit 9896d25 76e044b 022e8dd
git push origin main
```

The revert sequence is intentional: 9896d25 wires forwarding for
022e8dd's feature, and 76e044b changes the kernel module 022e8dd
introduced. Reverting in newest-first order keeps every intermediate
state buildable.

After the revert, write the errors/ entry per §5. Phase 0 is then
formally closed; future TileLang work starts from a fresh plan.
