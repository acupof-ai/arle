# Iteration ergonomics: four changes

Date: 2026-09-09 · Status: Plan, accepted by ckl · Owner: agent-infer-34

## The measurement this plan answers to

A warm no-op `cargo check` on the `arle` CPU lane is 6.05 s. rustc is not the
problem. The problem is that the project almost never runs on a warm tree:

| Quantity | Measured 2026-09-09 |
|---|---|
| `target/release/.fingerprint` entries | 1,814 |
| distinct compiled variants of `cuda-kernels` (release) | 39 |
| distinct compiled variants of `infer-api` (release) | 21 |
| distinct compiled variants of `infer-seam` (2,406 lines) | 17 |
| feature sets crossed by one `pre_push_checks.sh` run | 4 |
| `target/` total (debug + release + pre-push-quick) | 9.4 GB |

Backend selection is a compile-time feature, so every backend combination is a
separate compilation of every crate above the seam. That is the cause of the
build cost, of the 9.4 GB, and of the cache never being warm for the
combination in front of you.

Two further costs are structural, not incidental:

- No kernel can be run alone. `crates/cuda-kernels` holds 57 `.cu`, 8 `.cuh`
  and 69 TileLang `.py` behind a 3,273-line `build.rs`, and the only way to
  observe any of them is a full engine run on the pod against a real model.
- The pod tree has no version identity. `pod.sh sync` ships the working tree
  (`git ls-files -co`), so "what is built on the pod" has no name.

## W1 — Build profile: iterate on `release-fast`, measure on `release`

Owner: lane/c

`[profile.release]` is `codegen-units = 1` + `lto = "thin"`. `pod.sh build`
defaults to `--release`, so every iteration build on the pod pays full LTO.
`[profile.release-fast]` (cu=16, no LTO, incremental) already exists and only
CI uses it.

The hazard in flipping the default is that a bench number from a `release-fast`
binary is not a valid number. So the profile becomes explicit and recorded:

1. `pod.sh build` defaults to `--profile release-fast`.
2. The build receipt records the profile.
3. `pod-remote-run.sh` refuses a run labelled as a bench when the receipt's
   profile is not `release`, and says so.
4. `AGENTS.md:211` states the split: iteration builds `release-fast`, perf
   numbers and shipped artifacts `release`.
5. `pre_push_checks.sh:130-131` (the opt-in Metal block) moves to
   `release-fast`; the release Metal build stays only in the artifact lane.

**Exit gate.** A run launched from a `release-fast` build with a bench label
fails with the profile named in the error, demonstrated once; the same run
from a `release` build proceeds.

## W2 — Backend becomes a runtime choice, not a feature

Owner: lane/a

`infer_seam` is already two host-only traits. The obstacle is that
`infer_core::Engine<E, K>` is generic over them, so the backend identity
propagates as a type parameter into every crate above the seam, and from there
into `--features`.

Make the seam object-safe (the `State` associated type boxes) and select the
backend at runtime from a registry each backend crate registers into. The cost
is one virtual call per `submit`/`poll` — once per step, not per token per
layer, on a step in the 10 ms range.

Crates above the seam (`infer-core`, `infer-seam`, `infer-server`, `infer-api`,
`cli`, `arle`) must end with zero backend features. Which backend crates are in
the link stays a platform decision at the leaf, and does not fan out upward.

`no-cuda` exists so cuda-gated Rust typechecks without nvcc; keep it as a
`cuda-kernels`-local build.rs switch, not a workspace-wide feature.

**Exit gate.** In one profile, `infer-seam`, `infer-core` and `infer-server`
each have exactly one `.fingerprint` entry per test/lib target;
`cuda-kernels` has at most 4. `arle serve --backend cuda` and
`--backend metal` both dispatch from one binary on their platform. Needle
ladder x3 unchanged against the baseline envelope.

## W3 — `arle kernel`: one kernel, real tensors, a CPU reference beside it

Owner: lane/d

A kernel today is an invisible component of an end-to-end throughput number.
Give each kernel its own number:

```
arle kernel <name> --shape <dims> --ref cpu [--iters N]
```

prints the kernel's own time and the max relative error against a CPU
implementation of the same operation. The registry is explicit, so a kernel
with no entry is reported as unreferenced.

First two entries, chosen because they answer an open question:

- `fp4-gemv` — the dense NVFP4 M=1 GEMV at `quantized_gemv.cu:1193`, zero
 production callers since dispatch convergence never A/B'd
  against the Marlin tensor-core GEMM it was replaced by. This is the leading
  candidate explanation for the 84.5 -> 92.4 decode gap
  (`errors/2026-09-09-w4afp8-gemv-killed-item-dsv4-moe-only.md`).
- `marlin-fp4-gemm` — the arm actually in the serving path, so the two are
  comparable at M=1.

**Exit gate.** `arle kernel fp4-gemv` and `arle kernel marlin-fp4-gemm` each
print time and max relative error at M=1 on the pod, and the two numbers
appear in one wins entry with the matched A/B that decides the killed item.

## W4 — The pod tree gets a name, and one tree per lane

Owner: lane/b

`pod.sh sync` ships `git ls-files -co` — the working tree, including
uncommitted and untracked files. Nothing on the pod identifies which source is
built. Separately, `flock` serialises each sync and each build but not the
`sync -> build` sequence, so one lane's sync can land between another lane's
sync and build.

1. `sync` ships a commit sha; dirty overlay only with an explicit flag, and
   the receipt records that the tree is dirty.
2. `pod.sh status` prints the sha the current build came from.
3. One tree per lane (1.5 GB each; `/host` has 54 GB free), so `sync -> build`
   cannot interleave across lanes.

**Exit gate.** Two lanes sync and build concurrently, and each binary reports
its own sha; a dirty sync is labelled dirty in `status`.

## Order

W1 unblocks everyone and is the smallest. W3 answers an open measurement.
W2 is the largest and carries a correctness gate. W4 is independent.
W1, W3, W4 run in parallel; W2 starts now and lands after W1 (it will rebase
onto the profile change).
