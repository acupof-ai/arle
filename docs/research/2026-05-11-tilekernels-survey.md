# TileKernels Survey for DSv4

Date: 2026-05-11

Scope: DeepSeek official `deepseek-ai/TileKernels` repository surveyed from a
temporary clone at `/tmp/deepseek-ai-tilekernels`. No code was imported or
vendored.

Surveyed revision: `36d9e45d38e204ebb87e6f6e833821eee0482fe5`.

SOLID rule: every factual claim below cites a repository path or git commit.
Items marked Hypothesis are integration judgments, not measured performance.

## 1. Repository Structure and Kernel Inventory

- SOLID: top-level files are `.editorconfig`, `.gitignore`, `LICENSE`,
  `README.md`, `pyproject.toml`, `tests/`, and `tile_kernels/`
  (`TileKernels@36d9e45` file list).
- SOLID: package structure is `tile_kernels/{moe,quant,transpose,engram,mhc,
  modeling,torch,testing}` (`TileKernels@36d9e45:README.md`).
- SOLID: build/runtime stack is Python package + PyTorch + TileLang:
  `pyproject.toml` requires `torch>=2.10` and `tilelang>=0.1.9`; README says
  kernels are built with TileLang (`TileKernels@36d9e45:pyproject.toml`,
  `README.md`).
- SOLID: target hardware/toolchain is NVIDIA SM90 or SM100 and CUDA Toolkit
  13.1+ (`TileKernels@36d9e45:README.md`).
- SOLID: no raw `.cu`, CMake, CUTLASS, or Triton sources exist in the surveyed
  tree; kernel sources are Python TileLang modules under `tile_kernels/*`
  (`TileKernels@36d9e45` file list, `rg cutlass|triton|__global__|CMake`).

Kernel coverage:

- SOLID: MoE routing/layout kernels:
  `moe/topk_gate_kernel.py`, `top2_sum_gate_kernel.py`,
  `topk_sum_and_topk_group_idx_kernel.py`, `get_fused_mapping_kernel.py`,
  `expand_to_fused_kernel.py`, `reduce_fused_kernel.py`,
  `group_count_kernel.py`, `normalize_weight_kernel.py`,
  `mask_indices_by_tp_kernel.py`, `inplace_unique_group_indices_kernel.py`,
  `aux_fi_kernel.py`.
- SOLID: quantization kernels:
  `quant/per_token_cast_kernel.py`, `per_block_cast_kernel.py`,
  `per_channel_cast_fused_kernel.py`,
  `per_channel_cast_and_transpose_kernel.py`,
  `per_token_cast_to_e5m6_kernel.py`, `cast_back_kernel.py`,
  `cast_back_e5m6_kernel.py`, `per_block_cast_lossless_kernel.py`,
  `swiglu_forward_and_per_token_cast_kernel.py`,
  `swiglu_forward_and_per_channel_cast_and_transpose_kernel.py`,
  `swiglu_backward_and_per_token_cast_kernel.py`.
- SOLID: mHC kernels:
  `mhc/pre_big_fuse_kernel.py`, `pre_split_mixes_kernel.py`,
  `pre_apply_mix_kernel.py`, `post_kernel.py`, `sinkhorn_kernel.py`,
  `norm_fn_kernel.py`, `expand_kernel.py`, `head_compute_mix_kernel.py`,
  `multilayer_recompute_kernel.py`.
- SOLID: Engram kernels:
  `engram/engram_hash_kernel.py`, `engram_gate_kernel.py`,
  `engram_fused_weight_kernel.py`, `engram_grad_w_reduce_kernel.py`.
- SOLID: transpose kernel: `transpose/batched_transpose_kernel.py`.
- SOLID: no attention, KV-cache, MLA, CSA/HCA attention, MTP, standalone GEMM,
  or DeepGEMM-style FP8 GEMM kernel files are present (`TileKernels@36d9e45`
  file list and `rg attention|mla|csa|mtp|deepgemm`).

Dtypes and shapes:

- SOLID: MoE gating uses FP32 logits/bias/weights, INT64 selected expert
  indices, and INT32 expert/token mapping buffers
  (`moe/top2_sum_gate_kernel.py`, `moe/get_fused_mapping_kernel.py`).
- SOLID: quant supports E4M3 FP8, E2M1 FP4 packed as int8, and E5M6 packed
  format, with BF16/FP32 dequant targets (`quant/common.py`,
  `quant/per_token_cast_kernel.py`, `quant/cast_back_e5m6_kernel.py`).
- SOLID: mHC/Engram use BF16 activations plus FP32 mix/logit/reduction buffers
  (`mhc/pre_big_fuse_kernel.py`, `mhc/post_kernel.py`,
  `engram/engram_gate_kernel.py`).

## 2. License and Maintenance State

- SOLID: license is MIT (`TileKernels@36d9e45:LICENSE`,
  `TileKernels@36d9e45:pyproject.toml`).
- SOLID: current HEAD is `36d9e45d38e204ebb87e6f6e833821eee0482fe5`,
  committed 2026-04-23 18:19:20 +0800, subject `Merge pull request #1 from
  tianr22/main` (`git -C /tmp/deepseek-ai-tilekernels log -1`).
- SOLID: full fetched history has 3 commits, all on 2026-04-23: initial
  commit, an Engram comment revision, and PR #1 merge (`git rev-list --count
  HEAD`, `git log --since=2026-04-01`).
- SOLID: package classifier says `Development Status :: 3 - Alpha`
  (`TileKernels@36d9e45:pyproject.toml`).
- SOLID: README says some kernels are used in internal training and inference,
  but also says they do not represent best practices and docs/code quality are
  still being improved (`TileKernels@36d9e45:README.md`).
- Hypothesis: external production-readiness should be treated as alpha/research
  until ARLE runs correctness and benchmark gates; the repo itself provides
  pytest correctness/benchmark tests but no stable C ABI for non-PyTorch
  runtimes (`tests/*`, `tile_kernels/testing/bench.py`).

## 3. Alignment With ARLE

### Directly Usable

- SOLID: no strict direct-import candidate today. ARLE CUDA kernels are native
  CUDA C or build-time TileLang AOT under `crates/cuda-kernels/csrc/` and
  `crates/cuda-kernels/tools/tilelang/`, while TileKernels exports Python
  functions that allocate `torch.Tensor`s and launch TileLang JIT kernels
  (`ARLE:crates/cuda-kernels/AGENTS.md`,
  `ARLE:crates/cuda-kernels/tools/tilelang/README.md`,
  `TileKernels@36d9e45:moe/*.py`, `quant/*.py`).
- SOLID: SM tier is not aligned for direct import. ARLE default-builds
  SM80/86/89/90 and has SM100/120 opt-in; TileKernels requires SM90/100
  (`ARLE:docs/support-matrix.md`, `ARLE:crates/cuda-kernels/build.rs`,
  `TileKernels@36d9e45:README.md`).
- Hypothesis: a H100-only experimental import of MoE routing might be possible,
  but it is not "directly usable" under ARLE's T1 support bar because it would
  need an AOT wrapper, Rust FFI, allocation ownership, and SM policy decisions.

### Borrow Algorithms

- SOLID: MoE routing/layout is the strongest match to DSv4. TileKernels has
  `top2_sum_gate`, `topk_gate`, `get_fused_mapping`, `expand_to_fused`, and
  `reduce_fused`; ARLE DSv4 has routed/shared experts, hash vs bias routing,
  `sqrtsoftplus` scoring, `num_experts_per_tok=2`, and current CUDA MoE TODOs
  (`TileKernels@36d9e45:moe/*.py`, `tests/moe/test_top2_sum_gate.py`,
  `ARLE:crates/deepseek-spec/src/v4.rs`,
  `ARLE:infer/src/model/deepseek/mlp.rs`,
  `ARLE:infer/src/model/deepseek/reference.rs`).
- Hypothesis: MoE routing/layout can be adapted in 1-2 weeks for a CUDA DSv4
  forward slice if scoped to BF16 activations and existing Rust-owned buffers;
  it still does not provide expert GEMM.
- SOLID: mHC matches DSv4 hyper-connection math. TileKernels implements
  pre-norm/mix split, Sinkhorn normalization, pre-apply, post-combine, and a
  fused pre pipeline; ARLE reference has `gen_mhc_params`, `hc_pre`, `hc_post`,
  `hc_mult=4`, `hc_sinkhorn_iters=20`, `hc_eps=1e-6`
  (`TileKernels@36d9e45:mhc/*.py`, `tests/mhc/*.py`,
  `ARLE:infer/src/model/deepseek/reference.rs`,
  `ARLE:crates/deepseek-spec/src/v4.rs`).
- Hypothesis: mHC is the second-best adaptation target: it is DSv4-specific and
  shape-aligned, but needs ARLE AOT export and no-Torch FFI before runtime use.
- SOLID: quant/SwiGLU kernels cover FP8/FP4/E5M6 casts and fused SwiGLU +
  per-token/per-channel quant. ARLE has existing `csrc/quant/`, `csrc/gemm/`,
  W4/FP8/Marlin/TurboQuant surfaces, but DSv4 V4-only substrate is currently
  BF16 reference plus CUDA TODOs (`TileKernels@36d9e45:quant/*.py`,
  `ARLE:crates/cuda-kernels/csrc/{quant,gemm}/`,
  `ARLE:docs/support-matrix.md`).
- Hypothesis: quant kernels are useful after BF16 DSv4 forward exists; importing
  them first risks optimizing a non-wired path.
- SOLID: Engram hash/gate kernels are present, but ARLE DSv4 hash routing uses
  checkpoint tensor `gate.tid2eid` indexed by token id, not Engram n-gram hash
  embeddings (`TileKernels@36d9e45:engram/engram_hash_kernel.py`,
  `ARLE:infer/src/model/deepseek/reference.rs`,
  `ARLE:crates/deepseek-spec/src/v4.rs`).
- Hypothesis: Engram is not a DSv4 MoE routing shortcut unless a later V4 model
  variant exposes matching n-gram hash inputs.

### Not Relevant for This DSv4 Pass

- SOLID: `transpose/batched_transpose_kernel.py` is generic layout work and
  does not map to a current DSv4 blocker.
- SOLID: `modeling/*` are PyTorch autograd wrappers around TileKernels
  primitives, while ARLE hot path forbids PyTorch and owns training in Rust
  (`TileKernels@36d9e45:modeling/*`, `ARLE:AGENTS.md` project shape).
- SOLID: TileKernels has no KV-cache append/scatter/paged-attention kernels;
  ARLE already owns KV/page primitives in `crates/cuda-kernels/csrc/kv/`.

## 4. DSv4-Specific Hits

- MLA / V4 hybrid attention: SOLID no hit. TileKernels has no attention/MLA
  directory or kernel; ARLE's V4 attention remains TODO in
  `infer/src/model/deepseek/mla.rs` and legacy `mla_decode` stub remains
  unsupported in `crates/cuda-kernels/csrc/attention/mla_decode.cu`.
- CSA/HCA compressed sparse attention: SOLID no attention kernel hit.
  TileKernels has no compressor/indexer attention implementation. ARLE
  compressor/indexer reference lives in
  `infer/src/model/deepseek/reference.rs::compressor_forward` and
  `csa_selected_blocks`.
- MoE routing/layout: SOLID hit. `moe/top2_sum_gate_kernel.py` supports
  `sigmoid`, `sqrtsoftplus`, and `softmax`, bias, shared experts, fixed
  routing masks, logical-to-physical maps, EP/TP masking, and top-k weights;
  `get_fused_mapping`/`expand_to_fused`/`reduce_fused` provide expert-major
  layout plumbing. This can fill the routing/layout part of ARLE's
  "CUDA V4 MoE kernel not landed" gap, not expert GEMM.
- MTP: SOLID no hit. TileKernels has no `mtp` files or next-token prediction
  layer kernels. ARLE MTP tensor names exist in `crates/deepseek-spec/src/v4.rs`
  and remain unwired in optimized CUDA forward.
- Indexer: SOLID no DSv4 CSA indexer hit. `get_fused_mapping` is a MoE layout
  index map; `engram_hash` is n-gram embedding hashing. Neither computes V4
  CSA block scores/selects from `indexer.wq_b` and `weights_proj`.
- DeepGEMM-style FP8 GEMM: SOLID no standalone GEMM hit. Quant/SwiGLU kernels
  produce FP8/FP4/E5M6 tensors, but there is no GEMM kernel comparable to
  DeepGEMM (`TileKernels@36d9e45` file list).
- mHC: SOLID hit outside the user's named gap list. `tile_kernels/mhc/*`
  matches DSv4 hyper-connection pre/post/Sinkhorn structure and can fill a
  real DSv4 substrate gap around `hc_attn`, `hc_ffn`, and `hc_head`; it does
  not solve attention, expert GEMM, or MTP.

## 5. TileLang Relationship and ARLE AOT Fit

- SOLID: TileKernels is TileLang-first. Every low-level kernel module imports
  `tilelang` and declares `@tilelang.jit`; `pyproject.toml` depends on
  `tilelang>=0.1.9` (`TileKernels@36d9e45:tile_kernels/**/*.py`,
  `pyproject.toml`).
- SOLID: ARLE already has a TileLang 0.1.9-compatible AOT generator that
  extracts TileLang device source, compiles cubins with nvcc, emits C wrappers,
  and dispatches per SM (`ARLE:crates/cuda-kernels/tools/tilelang/gen_tilelang_aot.py`,
  `ARLE:crates/cuda-kernels/build.rs`).
- Hypothesis: TileKernels kernels are structurally portable into ARLE's AOT
  pipeline, but not drop-in. Each candidate needs a `tools/tilelang/*.py`
  entry returning a `T.prim_func`, a `WrapperSpec`, Rust FFI, allocation-free
  call ABI, per-SM build, and correctness tests against the CPU reference.
- Hypothesis: MoE routing kernels are easier to AOT-port than mHC because their
  tensor ABI is mostly flat 2D/1D buffers; mHC has more fused pipeline choices
  and forward/backward/autograd wrapper assumptions.

## Next Step Recommendations

A) Immediate import:

- SOLID: import nothing immediately. No kernel satisfies license + interface +
  dtype + ARLE SM-tier + no-PyTorch hot-path compatibility at direct-import
  quality.
- Hypothesis: the only "immediate" action worth opening is a no-vendor design
  spike that copies no source yet: define ARLE MoE route/layout FFI buffers and
  compare them against `top2_sum_gate`/`get_fused_mapping` signatures.

B) Worth 1-2 week adaptation:

- SOLID: first candidate is MoE routing/layout:
  `moe/top2_sum_gate_kernel.py`, `topk_gate_kernel.py`,
  `get_fused_mapping_kernel.py`, `expand_to_fused_kernel.py`,
  `reduce_fused_kernel.py`. This directly maps to DSv4 routed MoE TODOs and
  has correctness/benchmark tests in `tests/moe/*`.
- SOLID: second candidate is mHC:
  `mhc/pre_big_fuse_kernel.py`, `sinkhorn_kernel.py`, `post_kernel.py`,
  `norm_fn_kernel.py`, plus split/apply helpers. This maps to DSv4
  hyper-connection math in the reference path and has tests in `tests/mhc/*`.
- Hypothesis: quant/SwiGLU FP8 kernels should follow only after BF16 DSv4
  forward is wired; they are likely valuable for training/FP8 experiments but
  do not unblock first CUDA inference.

C) Punt:

- SOLID: attention/MLA/CSA/HCA, KV-cache operations, MTP, and DeepGEMM-style
  FP8 GEMM are absent from TileKernels, so ARLE must implement or source those
  elsewhere.
- SOLID: Engram and generic transpose do not close the current DSv4 optimized
  inference blockers.
