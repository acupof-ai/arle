# DSv4 native-deepep MoE correctness — root-caused + fixed (double-subtract local_expert_start)

## Context

After the combine-IMA cross-stream race fix
([`2026-05-30-dsv4-native-deepep-combine-ima-crossstream-fix.md`]), native-deepep
stopped crashing but returned **numerically wrong** output: `': is? is?'` where the
allreduce backend on the *identical* config returned `'**Paris**'`. The garbage was
isolated to the native-deepep-specific dataflow (flip-only A/B), but the cause was
not yet pinpointed — total garbage pointed at a gross routing/indexing error, not a
subtle numerical one.

## What Worked

**Adversarially-verified multi-agent dataflow audit** (6 segments — routing,
dispatch-recv, expert-scatter, combine-values, output-assembly, git-diff —
each finder paired with a refuter). It produced 1 high-confidence confirmed root
cause and, crucially, **refuted three plausible-but-wrong hypotheses** (double-weight,
pre-summed-row, stale-buffer/race) by reading the actual DeepEP `intranode.cu` kernel
— preventing three wasted "fixes".

**Root cause (confirmed): the native path double-subtracts `local_expert_start`.**
DeepEP intranode dispatch ALREADY remaps `recv_topk_idx` to RANK-LOCAL expert ids on
the send side (`idx - responsible_rank*num_experts_per_rank`, in-range → `[0,
experts_per_rank)`, else `-1` — DeepEP `intranode.cu:383-387`). The native path then
fed that already-local index into `dsv4_count_local_experts_cuda` +
`dsv4_pack_local_experts_cuda` with `local_expert_start = rank*experts_per_rank`, and
those kernels subtract it AGAIN (`local = expert - local_expert_start`,
`dsv4_route.cu:512/636`). For rank `r>0`: `local = j - r*experts_per_rank < 0` → every
valid local route rejected; **only rank-0's experts survive** → ~7/8 of the expert mass
vanishes every layer → corrupted logits → degenerate decode.

The allreduce reference is correctly asymmetric: it feeds GLOBAL ids
(`dsv4_route_kernel` writes `expert ∈ [0,n_routed_experts)`), so its single subtract is
right. Same kernel + same `local_expert_start`, but global input (allreduce) vs
already-local input (native) — that asymmetry was the bug.

**Fix (a6019c7e), native-path-only**: pass `local_expert_start = 0` to count+pack (the
already-local index IS the `[0,experts_per_rank)` group key; `self.experts[j]` is
exactly the expert DeepEP labelled `j`). The allreduce path is untouched.

## Validation (8×H20, TP=8 multiproc, native-deepep, bf16 KV)

| Prompt | Before (HEAD) | After fix |
|---|---|---|
| "The capital of France is" | `': is? is? is?'` | **`'...**Paris**.'`** ✓ |
| "Compute 137 + 269 …" | garbage | **`'406'`** ✓ |
| "List the first five prime numbers" | garbage | **`'… 2, 3, 5, …'`** ✓ |

8 ranks boot (peer_handles=8), **0 IMA / combine errors**. native-deepep MoE is now
numerically correct at production TP/EP.

## Still open (separate, gated)

A second confirmed bug — `dsv4_scatter_all_route_slots_kernel` OVERWRITES instead of
accumulating (`route_out[recv_token] = …`, not `+=`) so colliding same-rank experts
clobber — affects ONLY the DeepGEMM expert backend (`mlp.rs:2994`), which we do not use
(`EXPERT_BACKEND=native` runs the accumulating `dsv4_scatter_packed_expert` kernel,
`dsv4_route.cu:699-701`). DeepGEMM-backend is also JIT-blocked on the CUDA-12.2 pod.
Tracked as a follow-up; does not affect the validated native path.

## Rule

- **Two callers sharing one kernel can disagree on a SEMANTIC precondition the kernel
  assumes.** Here both the allreduce and native paths call the identical
  count/pack/scatter kernels with `local_expert_start`, but one feeds GLOBAL expert ids
  and the other feeds DeepEP-already-localized ids. The kernel's `expert -
  local_expert_start` is correct for one and a double-offset for the other. When wiring
  a new transport (DeepEP) into an existing kernel, audit the index space each
  upstream produces, not just the kernel signature.
- **rank-0-survives bugs hide on single-rank tests.** Dropping routes on ranks 1..N-1
  is invisible at TP=1 (start=0 makes the double-subtract a no-op). Parity must run at
  the production TP/EP.
- **Adversarial verify earns its cost by killing false positives.** Three plausible
  root causes (double-weight, contract-mismatch, race) were refuted at the DeepEP
  kernel source before any code was touched — each would have been a wrong fix that
  broke a currently-correct path.
