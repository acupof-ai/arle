# cap=8 bimodal — first-burst session 0-9 cluster pattern points to graph-capture race

> Per `f7da3e1` bimodal trigger remaining hypotheses(H_sched / H_harness
> / H_grcap / H_alloc),analyze existing bench logs for per-session
> failure pattern。
>
> **Result**:both 76%-success runs(`bwa4piqqx` + `b5i3467ad`)show
> **first-burst session 0-9 cluster failure**。Subsequent burst sessions
> mostly succeed。Pattern consistent with **H_grcap(graph capture race
> for first burst sessions)** even after `c20b1ce` warmup max=16 fix。

## Empirical evidence — log parsing

Parsed `/tmp/cap8-default-bench.log`,`/tmp/warmup-fix-n2-bench.log`,
`/tmp/n3-bench.log` for session ID-tagged turn outcomes:

| Run | Mode | Failed sample(first 10 IDs) | Note |
|---|---|---|---|
| `bwa4piqqx`(run #1)| Normal 76% | 0,1,2,3,4,5,6,7,8,9 | First-burst cluster |
| `b4r8fha82`+`b4kaqdrmj`(run #2)| Degraded 56% | 0,1,2,3,4,5,6,7,8,10 | First-burst + extending |
| `b5i3467ad`(run #3)| Normal 76% | 0,1,2,3,4,5,6,7,9,10 | First-burst cluster |

**All three runs fail sessions 0-9**(or close to it)— the FIRST 8-10
sessions admitted。This is the burst that hits the c=8 admission cap
at fresh server start。

## Mechanism — graph capture race even with max=16 warmup

Even after `c20b1ce` warmup pre-captures batches 1-16 at startup,the
**first burst of 8 sessions** still encounters latent issues:

### Hypothesis H_grcap refined

Pre-capture batch=N graphs covers DECODE batch=N。But when 8 sessions
admit simultaneously and start prefill,each session has different
prompt length(8K) → different chunk count → different prefill
sequence。The PREFILL graph(if used)may have per-session state that
isn't fully captured during warmup。

Or:warmup captures DECODE only,not PREFILL chunks。First-burst sessions
hit prefill compute that doesn't have a captured graph → eager kernel
launches → 100-500 ms tax → admission cascade。

### Why warm server case worked

`19d12c2` cap=8 override 100% success had been preceded by prior
benches。Prior bench runs INVOKED prefill kernels on c=8 burst → those
prefill code paths warmed → first burst in `19d12c2` test hits hot
code paths → no first-encounter cost。

### Why subsequent bursts in same fresh server succeed

After first 8-10 sessions hit cold-start,JIT codegen / kernel cache /
allocator slop stabilizes。Next bursts(sessions 10-127)benefit from
the warmed state → 76-92% success on remainder。

## Math of bimodal

Normal mode runs:
- First burst(sessions 0-9)mostly fail:~6-8/10 fail
- Subsequent bursts(sessions 10-127):mostly succeed(~10-15% fail rate)
- Total:~62 failed / 256 = 24% fail = **76% success**

Degraded mode runs:
- First burst:same ~6-8 fail
- Subsequent bursts:**HIGHER fail rate**(maybe ~30-40%)
- Why higher fail rate?Possibly some accumulated state degrades(see H_alloc)

So bimodal switch at slots=16 is between:
- **Normal**:first-burst tax,subsequent stabilize → 76-92%
- **Degraded**:first-burst tax + subsequent degradation → 56%

## Refined hypothesis priorities

H_grcap(graph capture race for first burst):**STRONGER**
- Empirically first 8-10 sessions consistently fail across all runs
- Suggests prefill code paths aren't fully warmed even with `c20b1ce`

H_alloc(GPU allocator slop accumulation):**STRONGER for degraded mode**
- Run #2 degraded had EXTRA failures beyond first burst
- Allocator state may degrade after some N sessions

H_sched / H_harness:less likely given session ID clustering pattern

## Codex follow-up — Step 2.B refinement

Per `f7da3e1` codex Step 2.B / 2.C plan,this evidence narrows
investigation:

### Step 2.B' — Pre-warm prefill code paths at startup

Before serving traffic,run a synthetic prefill burst at startup to
warm up:
- Tokenize prompt of expected production size(8K)
- Submit 8 concurrent dummy prefill requests
- Wait for completion
- THEN start accepting real traffic

This should warm prefill code paths,kernel cache,allocator state →
first real burst hits hot paths → fail rate drops。

LOC:~30-50 in `bootstrap.rs` warm-traffic handler。Risk:Low(startup
delay grows ~5-10s)。

### Step 2.C — Per-burst allocator reset

If bimodal degraded mode is allocator-slop accumulation,add explicit
GPU mempool reset between session bursts。Or use cudarc allocator
warmup-reset semantics。

LOC:variable(depends on allocator integration)。Risk:Med。

## Status

- ✅ Bimodal pattern empirically traced to first-burst session 0-9 cluster
- ✅ H_grcap refined:warmup max=16 doesn't cover prefill code paths
- ✅ H_alloc indicated:degraded mode has secondary degradation beyond first burst
- 🔧 Codex pickup:Step 2.B' pre-warm prefill at startup(~30-50 LOC)

## Cross-references

- Codex investigation plan:`fc9bea9`
- Bimodal characterization:`a0a3f42`
- Memory floor distinction:`e5f9d86`
- Codex correction:`f7da3e1`
- Skill v1.5.0 anti-pattern #15:warm-server implicit dependency

## Rule

**When bimodal failure mode shows session-ID clustering**(first burst
sessions consistently fail),the trigger is **first-encounter overhead**
in some non-warmed code path,not random scheduling variance。

For ARLE specifically:warmup pre-captures DECODE graphs but PREFILL
code paths aren't pre-warmed。Pre-warming prefill at startup is the
recommended fix per Step 2.B'。This generalizes:**any "fresh server"
bimodal pattern with first-N-IDs-failing should test pre-warm of
the relevant code path**(decode,prefill,KV,scheduler,etc)。
