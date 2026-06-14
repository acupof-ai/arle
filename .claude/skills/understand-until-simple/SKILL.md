---
name: understand-until-simple
description: Use this skill BEFORE writing code for any task you're tempted to call "hard / 难 / 复杂 / 硬", any "why is X slow / 瓶颈在哪", any "should I optimize / 能不能加速 Y", any concurrency/perf/kernel/scheduler investigation, or any time you catch yourself hand-waving a root cause. It is the pre-implementation understanding GATE — decompose to the atomic (line/kernel/buffer) level, get MEASURED evidence, let measurement correct your hypotheses, until the problem is SIMPLE and quantified. Only then write code. If it still feels hard, you haven't decomposed enough.
version: 1.0.0
---

# understand-until-simple

The pre-code gate. Operationalizes §0 (SOLID) + §0.1 (抽丝剥茧到实现级) into a
checklist you clear BEFORE writing implementation code.

Born from the DSv4 concurrency case (2026-06-14): I kept calling batched decode
"very hard / 硬", proposing it as a multi-day lever on hypothesis. ckl's
correction — *"你现在的表达等于你完全不懂呢，还什么很难很硬；什么时候你搞清楚了
认为简单了咱们开工"* — was right. Reading the code + measuring collapsed it to:
**it's one `for r in 0..n` loop (`dsv4.rs:1872`); batch it → ~2×; pieces exist;
MoE won't cap it.** Simple. The "hard" was just un-decomposed.

---

## The law

**"Hard" is a confession that you haven't decomposed it yet.** Difficulty is not
a property of the problem; it is the distance between you and the atomic
mechanism. Close that distance until the problem is simple, then the code writes
itself. A plan that still sounds hard is a plan you don't understand.

**No code until you can state the fix in one simple sentence with a measured
number behind it.** ("Batch the per-row attention loop → ~2× aggregate, MoE
stays sub-linear" — that sentence is the license to start.)

---

## The gate (clear every step before coding)

### 1. Catch the hand-wave
If your description contains "hard / 难 / 硬 / 复杂 / tricky / multi-day / it's the
lever" WITHOUT a line number and a measured number, STOP. You are at hypothesis,
not understanding. The words "hard/硬" are the alarm.

### 2. Decompose to the atomic level — READ THE ACTUAL CODE
Not "attention is slow" → **"`dsv4.rs:1872` is a `for r in 0..n` loop calling
single-row `mla_attention` + 2 memcpy per row."** Not "prefill is serial" →
read `planner.rs` and find it's chunked + mixed-batched (so the real cause is
elsewhere). Name the exact file:line / kernel / buffer / loop. If you can't, keep
reading. Source survey / call-graph / "it must be X" are HYPOTHESIS, never
evidence (§0).

### 3. List your hypotheses — then refuse to guess the load-bearing one
You will have a guess for what dominates ("MoE ∝ active_experts is the real
wall"). That guess is exactly the thing you must MEASURE, not assert. The
load-bearing assumption is never the one you skip.

### 4. Measure — controlled, single-variable, the right framing
Profile / A-B the controlled variable (c=1 vs c=8; before vs after; with vs
without). step-profiler, nsys NVTX, `/v1/stats` deltas, wall-clock Δ%. Decompose
the cost into named buckets and get each bucket's scaling. Wall-clock /
per-request framing is ground truth; a narrow-window % is not (§0 framing trap).
For the perf A/B mechanics (formula → binding constraint → matched A/B →
license-or-kill) use the **kernel-optimization** skill — this skill is the gate
that precedes it.

### 5. Let the measurement CORRECT you
The point of measuring is to be wrong cheaply. (DSv4: MoE was NOT the wall —
3.70× sub-linear, not ∝c; and `--spec-type mtp` silently *disabled* the batched
lane — `executor.rs:1563`.) A wrong root cause wastes every downstream hour (§0:
root-cause 假设也要 license-or-kill). Update the picture; re-decompose if needed.

### 6. Reach simple + quantified
Write the one-sentence fix with its number and its file:line. Name the next wall
(there's always one) and why it's softer. Name the one wrinkle (DSv4: MTP↔batched
exclusivity). If you cannot compress it to a sentence + a number, you are not
done — go back to step 2.

### 7. NOW write the line-level spec, then code
The understanding produces an implementation-level spec the executor copies
verbatim (§0.1: every mutated buffer + call site + precondition). Code only after
the gate is clear.

---

## Done test

Before any implementation diff, you can fill ALL of these:

- [ ] The bottleneck as a **file:line / kernel / loop**, not a concept.
- [ ] A **measured** decomposition (numbers, not "should be"), controlled-variable.
- [ ] At least one **hypothesis the measurement killed** (if zero, you didn't test the load-bearing one).
- [ ] The fix in **one sentence + an expected number**.
- [ ] The **next wall** named, and why it's acceptable for now.
- [ ] It feels **simple**. If it still feels hard → back to step 2.

Miss any box → you're still in understanding, not implementation. Do not code.

---

## Anti-patterns (this is how the gate gets skipped)

1. **"It's hard, so it'll take a while" as a substitute for decomposing.**
   Hardness estimate ≠ understanding. The DSv4 lever went from "3-4 week new
   infra" → "wire one for-loop" purely by reading the code.
2. **Proposing the lever on a hypothesis ("X is the bottleneck, batched-Y fixes
   it") without the measured decomposition.** `plan_label`/"X must dominate" is
   reachability, not a license (cross-ref kernel-optimization anti-patterns).
3. **Skipping the measurement of the load-bearing assumption** because the code
   "looks like" it confirms the guess. Code-reading and measurement share
   blindspots only sometimes — measure the thing that decides the plan.
4. **Extrapolating from a smoke shape** (short context, c=1) to the production
   regime (long context, c=8). Re-measure at the binding shape (§ distilled
   lessons: SLO verdict from the SLO workload).
5. **Calling it understood when you can't say it simply.** Complexity in the
   explanation = gaps in the understanding.

---

## When NOT to gate

Trivial / mechanical edits (rename, config flip, a ≤3-file obvious change), or a
task already decomposed + measured this session. The gate is for the things you'd
otherwise hand-wave as "hard" — that's exactly where it pays.
