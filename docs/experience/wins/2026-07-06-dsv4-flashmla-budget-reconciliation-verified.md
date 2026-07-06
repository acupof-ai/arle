# DSv4 FlashMLA/`affordable`-gate reconciliation (`ba36fbd39`) — pod-verified: clean reject, no crash

## Context

Round 3 of a verification chain on real CUDA hardware (8×H20):

1. [round 1](2026-07-06-dsv4-max-total-tokens-pod-verify.md) — no-flags DSv4
   serve auto-resolved `max_seq_len` to the checkpoint's native
   `max_position_embeddings` (1,048,576) and hard-failed at engine build
   (`kv_layout.rs`'s FlashMLA pool `ensure!`, pages=3344 need>=4098) on all 4
   worker ranks.
2. [round 2](2026-07-06-dsv4-auto-context-ceiling-still-crashes-tp4.md) —
   `29fdda704` capped the auto-resolve ceiling at 32768, but the same failure
   still reproduced at the smaller value (pages=74 need>=130) — proving the
   real defect is a reconciliation gap between two independent budget checks
   (`dsv4.rs`'s `affordable` gate vs. `kv_layout.rs`'s FlashMLA pool
   constructor), not "wrong default value."
3. **This round** — pod-verifies `ba36fbd39`, which adds a
   `dsv4_flashmla_slot_pages` pre-check directly inside `dsv4_kv_budget_plan`
   (`crates/infer-cuda/src/dsv4.rs`), so the reconciliation gap is caught
   *before* reaching `kv_layout.rs`'s pool constructor, with the same
   `ensure!`-style clean-reject error the existing `affordable > 0` gate
   already uses.

## Pod state

- GPU 1 held by a foreign tenant throughout (51,373 MiB, 0-96% util,
  untouched). GPU 0 also carried another agent's terminal-bench session
  (`arle serve --model-path /host/Qwen3.6-27B-FP8`, PID 1198140, untouched) and
  a live `opd_base3.sh` shell (PID 1243469, untouched). GPUs 2, 3, 4, 5, 6, 7
  free (0 MiB) at session start. Used **GPUs 4,5,6,7 (TP=4,
  `INFER_CUDA_DEVICES=4,5,6,7 INFER_TP_SIZE=4`)** — same topology as rounds 1-2,
  for direct comparability (GPU1's occupancy still precludes TP=8).
- `scripts/pod.sh sync` → pod tree confirmed `pod tree @ ba36fbd3 fix(cuda):
  reconcile DSv4's affordable gate with the FlashMLA pool's own page floor`.
- Build: `cargo build --release --features cuda,nccl,deepep --bin arle` →
  `BUILD_EXIT=0 (compiled 6 crates)` in 54.00s (warm `target/` cache; the
  default `scripts/pod.sh build` with no args only enables plain `cuda` — had
  to pass `cuda,nccl,deepep` explicitly to get the multiproc/NCCL TP=4 path
  this scenario needs).
- Checkpoint: `/host/DeepSeek-V4-Flash-FP8`, `max_position_embeddings =
  1048576` (unchanged from prior rounds).

## Key finding — the terminology correction

Reading `crates/infer-cuda/src/attention/kv_layout.rs:1017-1025`, the
pre-existing FlashMLA pool check that used to fire (and still exists as a
defense-in-depth backstop) is `anyhow::ensure!(...)` — an `anyhow::Result::Err`
return, **not a Rust `panic!`/unwind**. Neither the pre-fix nor post-fix
failure path is a literal panic with a backtrace; both rounds 1-2's quoted
logs and this round's log show `[arle-worker rank=N] failed: worker rank N
engine build: <message>` with no `thread '...' panicked at`/`RUST_BACKTRACE`
text, in both the old and new cases. **The actual improvement `ba36fbd39`
delivers is not "eliminating a panic"** (there wasn't a literal panic to
eliminate) — it is:
1. **Firing earlier**: the new `ensure!` in `dsv4_kv_budget_plan`
   (`dsv4.rs`) runs before `kv_layout.rs`'s `TokenKVPool::with_format` even
   allocates the shared pool, vs. the old path which only caught the mismatch
   deep inside that constructor, after real work had already happened.
2. **A materially clearer, actionable message**: new — `"DSv4 KV budget
   rejected startup: the shared FlashMLA pool's per-layer remainder (2MB)
   cannot hold even one slot's band at max_seq_len 32768 (130 pages, 4MB).
   Lower --max-total-tokens or free VRAM."` vs. old — `"DSv4 FlashMLA pool page
   mismatch: page_size=64 pages=74 need page_size=64 pages>=130"` (opaque
   internals, no remediation hint).
3. **Style unification**: the new check reuses the exact reject-style already
   used by the pre-existing `affordable > 0` gate, so there is now one
   consistent "rejected startup" failure mode for both per-slot and
   per-layer-pool infeasibility, instead of two different error shapes from
   two different call sites.

Both before and after, the top-level behavior (worker exits `Some(1)`,
coordinator aborts cleanly, `RUN_EXIT=1`, no zombie GPU memory) was already
correct — `ba36fbd39` improves *where* and *how clearly* the rejection fires,
not whether the process exits cleanly.

## No-flags boot (the exact round-2 crash scenario)

```
INFER_CUDA_DEVICES=4,5,6,7 INFER_TP_SIZE=4 ./target/release/arle serve \
  --model-path /host/DeepSeek-V4-Flash-FP8 --backend cuda --port 18191 \
  --max-running-requests 2
```

Log (verbatim, key lines):
```
INFO cli::serve: serve.rs:347 DSv4 max context: auto-resolved to 32768 from
  /host/DeepSeek-V4-Flash-FP8/config.json (max_position_embeddings=1048576)
...
[rank*] INFO infer_cuda::executor: executor.rs:2255 [vram-probe] after model
  load (weights+experts): used 74745MB free 22763MB
[rank*] INFO infer_cuda::dsv4: dsv4.rs:1818 DSv4 KV budget: free 24171MB,
  per_slot 317MB (slot-state 295MB + DSA key-cache 21MB + DSA batched 0MB;
  FP8 arena in shared pool), shared DSA 56MB, shared MoE decode 0MB, shared
  expert scratch 2MB, shared MLA decode 2MB, pool_per_layer 2MB, affordable 68
[arle-worker rank=2] failed: worker rank 2 engine build: DSv4 KV budget
  rejected startup: the shared FlashMLA pool's per-layer remainder (2MB)
  cannot hold even one slot's band at max_seq_len 32768 (130 pages, 4MB).
  Lower --max-total-tokens or free VRAM.
WARN infer_server::multiproc_relay: [relay-coordinator] worker rank 2
  completion reader failed: relay read header: Connection reset by peer
  (os error 104)
[arle-worker rank=0] failed: ... (same message)
[arle-worker rank=3] failed: ... (same message)
[arle-worker rank=1] failed: ... (same message)
WARN cli::serve_multiproc: worker rank 0 exited Some(1)
WARN cli::serve_multiproc: worker rank 1 exited Some(1)
WARN cli::serve_multiproc: worker rank 2 exited Some(1)
WARN cli::serve_multiproc: worker rank 3 exited Some(1)
[ARLE serve] multiproc coordinator setup failed: worker rank 1 exited (code
  Some(1)) during engine build (0/4 ready); aborting coordinator
RUN_EXIT=1
```
`grep -i "panic\|backtrace\|SIGABRT" run-noflags4.log` → no match.

**Verdict: clean reject — the fix worked as intended.** All 4 ranks produce
the new "DSv4 KV budget rejected startup..." message (not the old opaque
"FlashMLA pool page mismatch" from `kv_layout.rs`), the coordinator aborts
gracefully with a normal `anyhow::Error`-driven exit, `RUN_EXIT=1`, GPUs 4-7
returned to 0 MiB with no zombie processes. This does **not** make
`max_seq_len=32768` boot successfully at TP=4 (it never can, per the fixed
free-VRAM budget at this GPU count) — it only makes the failure legible and
consistent instead of surfacing from a different, less-informative call site.

## Explicit-override regression check (`--max-total-tokens 16384`)

```
INFER_CUDA_DEVICES=4,5,6,7 INFER_TP_SIZE=4 ./target/release/arle serve \
  --model-path /host/DeepSeek-V4-Flash-FP8 --backend cuda --port 18192 \
  --max-running-requests 2 --max-total-tokens 16384 --max-prompt-tokens 16000
```
Boots clean: `CUDA engine: executor clamped slots 256 -> 121; scheduler
follows`, all 4 ranks `engine-ready ack sent`, `all 4 worker engines ready;
opening HTTP`, `serving OpenAI v1 on http://127.0.0.1:18192`.

Decode-check (`curl /v1/chat/completions`, "What is the capital of France?
Answer in one word.", `max_tokens=200`, `temperature=0`):
```json
{"content":"Paris","reasoning_content":"The user asks for the capital of
France and specifies to answer in one word. The answer is straightforward:
Paris.","finish_reason":"stop"}
```
Correct — unaffected by the fix, as expected (16384 never trips the new
per-layer-remainder ensure).

## Boundary color (not required, cheap to get)

Bisected between the known-good 16384 and known-bad 32768 at TP=4 for this
checkpoint:

| `--max-total-tokens` | affordable | pool_per_layer | Result |
|---|---|---|---|
| 16384 | (untested exact value, prior rounds confirm boot) | — | boots, decodes correctly |
| 24576 | 87 | 3MB | boots clean (`clamping num_slots to 87`) |
| 28672 | 76 | 5MB | boots clean (`clamping num_slots to 76`) |
| 32768 | 68 | 2MB | **rejected cleanly** (new message) |

Boundary sits strictly between 28672 (works) and 32768 (rejects) at TP=4 on
this checkpoint/GPU-count. Not bisected further — sufficient color, not the
gate.

## Ruled out / not confounded

- Not a build miss: `BUILD_EXIT=0` on the exact `ba36fbd39` pod HEAD
  (`scripts/pod.sh sync` confirmed before build), `cuda,nccl,deepep` features
  (the default `pod.sh build` with no args only builds plain `cuda` — caught
  and corrected before the key test).
- Not a GPU-pollution artifact: GPU1 (foreign tenant) never in
  `INFER_CUDA_DEVICES`; GPUs 4-7 showed 0 MiB before every boot in this
  session.
- Not a leftover process/zombie-GPU-memory issue: every boot (crash and clean)
  self-terminated, GPUs 4-7 returned to 0 MiB each time, confirmed via
  `nvidia-smi` after each kill.

## Rule

- **A `Result`-returning `ensure!` deep in a constructor and a `Result`-
  returning `ensure!` at the budget-gate call site produce the same top-level
  process behavior (clean `Err` → `RUN_EXIT=1`, no panic) — the value of
  moving the check earlier is legibility and actionable messaging, not
  crash-vs-no-crash.** Don't let a commit message's "instead of a hard crash"
  framing stand unverified — grep the actual macro (`ensure!` vs `panic!`/
  `assert!`) before writing up "this used to be a hard panic."
- **`scripts/pod.sh build` with no args silently drops `nccl,deepep`** — for
  any DSv4 multiproc/TP>1 scenario, pass the full feature set explicitly
  (`build <label> --release --features cuda,nccl,deepep --bin arle`); the
  bare default only proves the `cuda`-only lib compiles, not that the
  multiproc path you're about to exercise does.
- **`scripts/pod.sh run <label> <gpu>` only pins ONE GPU
  (`CUDA_VISIBLE_DEVICES=<gpu>`, `world_size=1`)** — it cannot express TP>1.
  For TP=4/8 scenarios, hand-roll the detached launch with
  `INFER_CUDA_DEVICES=a,b,c,d INFER_TP_SIZE=4` directly (still `setsid` +
  redirect to a log + `RUN_EXIT=` marker, same discipline), then kill by the
  exact recorded PID/PGID — `pod.sh kill` only knows the single-GPU-run PID
  file convention.

## Cleanup

All three boots (no-flags reject, explicit-16384, and the two boundary
bisection boots at 24576/28672) self-terminated or were killed by exact PID
(`kill -- -<pgid>`); confirmed via `nvidia-smi` that GPUs 4-7 returned to 0 MiB
after each. GPU1 (foreign tenant, 51,373 MiB throughout) and GPU0's
`Qwen3.6-27B-FP8` terminal-bench session (PID 1198140) plus `opd_base3.sh`
(PID 1243469) were never touched.
