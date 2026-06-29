# Agent-OPD writeback backward OOMs on the `causal_mask` `[1, seq, seq]` N×N buffer — NOT the per-layer activation grads, so `--lora-layer-start` cannot fix it (verdict c)

## Context

Goal: capture the FIRST real agent-OPD value signal — `trained_pairs>0` → AdamW
step → held-out eval Δ — on the 8×H20 box. Two prior walls were already cleared
and understood:
- **Load-death** = the non-persistent `~/bin/pod` (crictl exec) foreground launch
  being reaped when the exec session ended; fixed by launching in a **persistent
  tmux** that survives the teardown
  ([persistent-tmux entry](../wins/2026-06-29-agent-opd-persistent-tmux-confirms-launch-reap-writeback-backward-oom-is-next-wall.md)).
- **Scoring** (`ccedd788`+`34a955df`, test_patch path-reset gated on the `---`
  header) — a real accept now scores `passed=true`.

This run was the brief's prescribed **CHEAP FIT**: raise `--lora-layer-start`
from 32 → **48** (train only the top 16 of 64 layers; detach the autograd
backward before layer 48) to bound the backward activation-grad VRAM, expecting
forward ~50 GB + grads ~22 GB ≈ 72 GB to fit under the 97.8 GB H20 ceiling.

Launch (all the brief's hygiene held): `tmux new-session -d -s aopd2` →
`exec -a arleCKL /host/arle-ckl-aopd/target/release/arle train agent-opd …` on
the free GPU 1, marker `arleCKL`, log `/host/run_final.log`. Config:
`--samples-per-prompt 4 --writeback-cap 1 --rounds 1 --eval-every 1
--eval-temperature 0.0 --rollout-temperature 1.0 --max-turns 16 --max-tokens 768
--lora-layer-start 48 --rollout-num-slots 1`, student `/host/Qwen3.6-27B-FP8`
(64 layers, hidden 5120), 1 train task (`ansible__ansible-f327e65`) + 3 held-out
eval tasks. Binary built Jun 28 15:50 (carries the scoring fix + pre-CUDA
sandbox-spawner). Persistent-tmux launch confirmed working: `arleCKL` ran ~63 min
and survived the exec teardown, all 8 GPUs free at start.

## Root Cause

The writeback ran **further than ever** — past the forward, past the fused-CE,
into the **backward** — but OOMed there on a buffer `--lora-layer-start` does not
touch. The self-caught error (ground truth, ARLE's own error path; corroborated
independently by a file-tail read AND a log-grep monitor):

```
[masked-writeback] seq_len=18168 total_targets=1475 chunk_rows=2048
[masked-writeback] phase=forward_hidden_states seconds=2935.440   # ~49 min forward, 64 layers
[masked-writeback] phase=fused_ce seconds=0.454                   # CE on the logits tile is ~free
[ARLE train] error: masked CE writeback (round 0): cuda htod copy failed:
  shape=[1, 18168, 18168] len=330076224 bytes=1320304896
  err=DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

The OOM tensor is **identified exactly by arithmetic**: `len=330076224 = 18168²`
and `bytes=1320304896 = 18168²×4` (f32). This is the **full `[1, seq_len,
seq_len]` causal attention mask** built host-side and htod-copied in
`crates/autograd/src/ops/attention.rs:520` `causal_mask()`:

```rust
fn causal_mask(seq_len: usize, store: &mut TensorStore) -> Result<TensorId> {
    let mut data = vec![0.0; seq_len * seq_len];          // host 18168² f32 = 1.32 GB
    for row in 0..seq_len { for col in (row+1)..seq_len { data[row*seq_len+col] = -inf; } }
    Ok(store.alloc(Tensor::new(data, vec![1, seq_len, seq_len], false)?))  // → htod [1,18168,18168]
}
```

`causal_mask` is called in BOTH the forward (`attention.rs:80`) and the
**backward recompute** (`attention.rs:342`, `causal_sdpa_recompute_backward_device`).
The backward had already pushed GPU to **88.4 GB** (the 16-layer grads — measured
live), and this unconditional 1.32 GB N×N mask htod (plus the SDPA score buffers
it feeds) crossed 97.8 GB → OOM, GPU released cleanly to 0 MiB, `arleCKL` exited
(zombie reaped), tmux gone. **No `eval_round_1.jsonl`, no adapter dir** — the
AdamW step never ran.

**Why `--lora-layer-start` is the wrong lever (the brief's CHEAP-FIT premise was
incomplete):** the head-chunked SDPA (`qwen35.rs:287 head_chunked_causal_sdpa`)
correctly bounds the `[chunk_heads, seq, seq]` *scores*, but the `causal_mask` it
calls is a **single full `[1, seq, seq]` buffer per SDPA call — NOT chunked, and
independent of the trainable-layer count**. Raising `--lora-layer-start` cut the
per-layer activation grads exactly as intended (forward + fused_ce completed at
88 GB), but the N×N mask is a fixed O(seq²) tax that no layer-count reduction
removes. It is on the **critical path because the value-producing accept is the
longest rollout**: this run's accepted trajectory was **18168 tokens** (vs 10780
the prior run), so the mask grew to 1.32 GB and the 16-layer headroom (~9 GB at
the 88 GB peak) wasn't enough.

## Fix

**Verdict (c): still OOM at the brief's sane `--lora-layer-start 48`** — but with
a sharper, code-located root cause than the brief's hypothesis (it is NOT the
whole-sequence activation grads `--lora-layer-start`/`--writeback-window` target;
it is the unchunked N×N `causal_mask`). The durable fix is **q-chunked masking in
the SDPA forward+backward recompute so the full `[1, seq, seq]` mask is never
materialized** — the primitive already exists: `causal_mask_window(q_len, kv_len,
q_start)` at `attention.rs:530` builds a `[1, q_len, kv_len]` tile. Route
`head_chunked_causal_sdpa` / `causal_sdpa_recompute` through a q-chunked path
(O(q_chunk·seq) mask instead of O(seq²)), matching the score-chunking that is
already there. This is the **out-of-scope-for-this-capture** O(window) peak fix
the brief flagged as the fallback.

Cheaper interim probes (untested, for the next run): (a) generate the causal
mask **on-device** (a `tril`/iota kernel) instead of host-build+htod — removes
the 1.32 GB host buffer and the htod copy, though the device `[1,seq,seq]` tensor
still costs 1.32 GB; (b) cap the accepted trajectory length fed to writeback
(reject >~12k-tok accepts for the first signal) — but that discards the
value-producing case, so it is a measurement crutch, not a fix.

## Case-as-fact (decoded, both sides)

**Train accept (the CE-writeback target) — a genuinely correct patch.** Task
`ansible__ansible-f327e65` (ansible/ansible): "Collection Name Validation Accepts
Python Keywords" — FQCN validation must reject `def.collection`, `return.module`,
`assert.test`. `fail_to_pass` = `test_fqcn_validation[assert.this-False]`,
`[ns4.return-False]`, `[import.*-False]`. **All 4 rollout samples passed
(`passed=true :: [exit 0]`)**; the accepted `git diff` of
`lib/ansible/utils/collection_loader/_collection_finder.py`:

```python
+        import keyword
         collection_name = to_text(collection_name)
-        return bool(re.match(AnsibleCollectionRef.VALID_COLLECTION_NAME_RE, collection_name))
+        if not re.match(AnsibleCollectionRef.VALID_COLLECTION_NAME_RE, collection_name):
+            return False
+        namespace, collection = collection_name.split('.')
+        return not keyword.iskeyword(namespace) and not keyword.iskeyword(collection)
```

Clean, idiomatic, correct — the hidden `test_patch` applies and the tests pass.
So the scoring fix is verified working AGAIN (`trained_pairs` would be 1) and the
distillation target is high quality. This is NOT the failure; the failure is
purely the writeback VRAM.

**Held-out baseline (the would-be Δ reference), `eval_round_base.jsonl`:**
- `ansible__ansible-0ea40e0` passed=true (edited)
- `ansible__ansible-12734fa` passed=false (edited — wrong fix)
- `ansible__ansible-5e36960` passed=false (edited — wrong fix)
- **pass_rate = 0.3333 (1/3 tasks, 3 edited)** — all 3 produce edits, 2 of 3 are
  wrong fixes. **No round-1 eval was produced** (OOM before AdamW), so the Δ is
  **NOT measurable this run.**

## Bench

Exempt: agent-OPD training path, not a serving hot path. No code change this run
(persistent-tmux launch + `--lora-layer-start 48` config only); default
serve/CLI byte-identical. Per the mandatory-bench rule this is the training-axis
capture attempt + root-cause, not a guidellm serving delta.

## Rule

- **`--lora-layer-start` / `--writeback-window` bound the per-layer activation
  grads and the logits tile — NOT the O(seq²) `causal_mask`.** The writeback
  backward holds a fixed `[1, seq, seq]` f32 causal mask (1.32 GB at seq=18168)
  per SDPA call, unchunked, independent of trainable-layer count. A long accept
  (the value-producing one is the longest) OOMs the backward on the mask even
  after the layer grads are cut. The fix is q-chunked masking
  (`causal_mask_window` already exists), not a config knob.
- **Identify an OOM'd tensor by arithmetic before blaming the phase.** `len`/`bytes`
  in the `cuda htod copy failed: shape=…` error is the smoking gun: `330076224 =
  18168²`, `1320304896 = 18168²×4` proved it was the N×N mask, not the "activation
  grads" the brief assumed — redirecting the fix from `--lora-layer-start` (wrong
  lever) to SDPA mask chunking (right lever) in one step.
- **The accepted trajectory length is a first-class writeback-VRAM variable.** The
  prior OOM was at seq=10780; this one at seq=18168 (1.7×) — same task, deeper
  rollout. Writeback VRAM scales O(seq²) via the mask AND O(seq) via grads, so
  the capture-run knob isn't just layer-start, it's also bounding/​chunking the
  sequence the writeback forward+backward sees.
- **Notification-summary lines from a polling monitor are NOT log ground truth —
  the file + fd offset are.** Mid-run, monitor summaries surfaced a phantom
  `phase=backward seconds=119.834` and `ADAPTER:adapters_round_1.safetensors`
  that the authoritative `cat`/`grep` of `/host/run_final.log` (and `find` for the
  adapter) did NOT contain; `grep -c "phase=backward"` = 0 confirmed the backward
  never completed and no adapter was written. Cross-check any monitor/subagent
  summary against the file before treating it as a measured fact.
- **A self-caught `cuda htod copy failed … OUT_OF_MEMORY` (GPU released to 0 MiB,
  ARLE's error path prints it) is a clean OOM, not a reap/SIGKILL.** Distinct from
  the prior `alloc_zeros failed` OOM — same wall (writeback backward VRAM),
  different triggering allocation (the mask htod vs the grad alloc).

Claude-Session: https://claude.ai/code/session_01Vsoud3oabdLDppvb274bCr
