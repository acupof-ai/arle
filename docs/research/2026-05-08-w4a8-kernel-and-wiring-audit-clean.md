# W4A8 Claude-side audit — kernel + wiring + dtype CLEAN,bug surface narrowed

> Continues `4dea952`(H3c applied,output regressed,methodology iteration
> at limit per skill anti-pattern #13)。
>
> Per `4dea952` recommended action(direct kernel-internal audit),this
> entry compares ARLE kernel + wiring against PR #31 reference verbatim
> and confirms **0-diff in 3 layers**:
> 1. Kernel source(961 LOC kernel + 26-line FFI wrapper)
> 2. Linear.rs FFI call argument ordering / dtypes
> 3. Loader name conventions + storage path
>
> Bug surface narrowed to **quantize-script side only**(remaining 3
> candidate sites)。Codex iteration on script is methodologically valid;
> not a kernel-port issue。

## Layer 1 — kernel source diff

```bash
$ diff -u /tmp/marlin-w4a8/marlin/w4a8_marlin_cuda_kernel.cu \
        crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu
```

**Result**:single 26-line append at bottom — pure `extern "C"` FFI
wrapper for Rust。Kernel body 0-diff。

```cpp
// ARLE crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu:961-987 (only diff)
extern "C" int gemm_w4a8_marlin_cuda(
    const void* A, const void* B, void* C, void* D,
    void* s1, void* s2, void* s3,
    int prob_m, int prob_n, int prob_k,
    void* workspace, int groupsize, int dev,
    cudaStream_t stream, int thread_k, int thread_n,
    int sms, int max_par
) {
    return w4a8_marlin_cuda(A, B, C, D, s1, s2, s3, prob_m, prob_n, prob_k,
                            workspace, groupsize, dev, stream, thread_k,
                            thread_n, sms, max_par);
}
```

Conclusion:**kernel matches PR #31 byte-for-byte**。If PR #31 produces
correct output on A100,ARLE kernel will produce correct output given
correct inputs。

## Layer 2 — wiring(linear.rs FFI call)

`infer/src/ops/linear.rs:777-859 run_marlin_w4a8_linear`:

```rust
ffi::gemm_w4a8_marlin_cuda(
    xq_ptr   as *const i8,        // A — INT8 quantized activation [M×K]
    mp_ptr   as *const u8,        // B — packed W4 weight (qweight)
    reduce_ptr as *mut i32,       // C — INT32 reduce buffer
    yf_ptr   as *mut ffi::Half,   // D — FP16 output
    s1_ptr   as *const f32,       // s1 — activation per-token scale (M floats)
    s2_ptr   as *const f32,       // s2 — s_channel (per-channel, F32)
    s3_ptr   as *const ffi::Half, // s3 — s_group (per-group, FP16)
    m as i32, n as i32, k as i32,
    ws_ptr as *mut i32, weight.group_size as i32,
    ctx.ordinal() as i32, ctx.stream.cu_stream(),
    -1, -1, sms, max_par as i32,
)
```

Cross-check against PR #31 kernel signature(`/tmp/marlin-w4a8/marlin/w4a8_marlin_cuda_kernel.cu:858-877`):

```cpp
int w4a8_marlin_cuda(
  const void* A, const void* B, void* C, void* D,
  void* s1, void* s2, void* s3,
  int prob_m, int prob_n, int prob_k,
  void* workspace, int groupsize = -1, int dev = 0,
  cudaStream_t stream = 0, int thread_k = -1, int thread_n = -1,
  int sms = -1, int max_par = 16
)
```

ARLE arg ordering ✓ matches。

PR #31 internal dtype expectations:

| Arg | PR #31 cast | Internal type | ARLE provides |
|-----|-------------|----------------|----------------|
| A   | `(const int4*)` | INT8 array (M×K, row-major) | `xq_ptr` from w4a8_activation_quant.cu i8 ✓ |
| B   | `(const int4*)` | packed uint32 (W4 → uint32) | `mp_ptr` from quant_pack uint32 ✓ |
| s1  | `(const float*)` | float32 per-token | `s_activation` f32 from quantizer ✓ |
| s2  | `(const int4*)` → `Vec<float, 2>` (FragS_CHANNEL) | float32 per-channel | `s_channel` f32 from quant script ✓ |
| s3  | `(const int4*)` → half2 internal | FP16 per-group | `s_group` Half from quant script ✓ |

(See `/tmp/marlin-w4a8/marlin/w4a8_marlin_cuda_kernel.cu:55` for FragS_CHANNEL =
Vec<float, 2>;line 369-403 for s2 stride/layout;line 715-718 for write
function consuming frag_s1 + frag_s2 + frag_c。)

Conclusion:**dtypes match across all 7 args**。No silent f16/f32 misuse。

## Layer 3 — loader / safetensors naming

Quantize script(`/tmp/quantize_qwen3_w4a8.py:159-161`):
```python
out_tensors[f"{prefix}.marlin_w4a8_qweight"] = qweight
out_tensors[f"{prefix}.marlin_w4a8_s_channel"] = s_channel
out_tensors[f"{prefix}.marlin_w4a8_s_group"] = s_group
```

Loader(`infer/src/weight_loader.rs:669-671`):
```rust
let packed_name = name.replace(".weight", ".marlin_w4a8_qweight");
let channel_scales_name = name.replace(".weight", ".marlin_w4a8_s_channel");
let group_scales_name = name.replace(".weight", ".marlin_w4a8_s_group");
```

Conclusion:**naming matches**。Loader reads the exact tensor names quantize
script writes。

Then linear.rs:786-787 reads the loaded matrix:
```rust
let s_channel = weight.marlin_channel_scales.as_ref().unwrap();
let s_group   = weight.marlin_scales.as_ref().unwrap();
```

`marlin_channel_scales` is the F32 tensor from `.marlin_w4a8_s_channel`,
`marlin_scales` is the FP16 tensor from `.marlin_w4a8_s_group` ✓。

## Layer 4 — activation quantizer

`crates/cuda-kernels/csrc/gemm/w4a8_activation_quant.cu:33`:
```cpp
s_act = max/127.0  // symmetric per-token max-abs INT8
```

Per PR #31 paper(QQQ §3.3),activation is symmetric INT8 per-token quant。
ARLE `w4a8_activation_quant.cu` produces `s_act = max/127.0` which **does
match** the symmetric max-abs convention。

`linear.rs:792-812` calls `quantize_bf16_rows_to_int8_cuda` which writes
`xq` (i8 array) + `s_activation` (f32 array)。✓ matches A_ptr + s1_ptr
expected by kernel。

## Bug surface remaining

Eliminated:
- ✅ Kernel source(0-diff)
- ✅ FFI argument ordering(matches signature)
- ✅ Argument dtypes(f32 / f32 / Half match kernel cast expectations)
- ✅ Tensor naming convention(loader reads what script writes)
- ✅ Activation quantizer math(symmetric max-abs per-token,matches QQQ §3.3)

Remaining candidate sites(in `/tmp/quantize_qwen3_w4a8.py` only):

A. **Tile permute lines 112-115**:
   ```python
   tile = 16
   w = w.reshape((k // tile, tile, n // tile, tile))
   w = w.permute((0, 2, 1, 3)).reshape((k // tile, n * tile))
   res = w.reshape((-1, perm.numel()))[:, perm].reshape(w.shape)
   ```
   PR #31 has identical 4-line block(line 303-307)。Should be 0-diff,
   but worth byte-confirm。

B. **Bit-packing stride 8**(line 117-119):
   ```python
   q = np.zeros((res_np.shape[0], res_np.shape[1] // 8), dtype=np.uint32)
   for i in range(8):
       q |= res_np[:, i::8] << (4 * i)
   ```
   PR #31 line 308-312 identical for `groupsize != k` branch。Should be
   0-diff。

C. **H3c was correct,but s_channel storage post-permute encoding wrong**:
   Maybe PR #31 stores s_channel as `(rows=1, n)` while ARLE writes
   `(rows=n//8, 8)` after `.reshape((-1, n))` — kernel expects strided
   layout that doesn't match a flat `(1, n)` reshape。

   Worth double-check:after H3c permute,does `.reshape((-1, n)).contiguous()`
   produce same byte layout as PR #31 line 299?

D. **groupsize != k check assumption**:
   ARLE script calls `pack_w4a8` only on Linear with valid `marlin_w4a8_aligned`
   shape。Is `groupsize == 128` for Qwen3-4B Linear,and `k` always
   `> 128`(so `groupsize != k`)? If somehow per-tensor quant lands
   in the `groupsize == k` branch,kernel would receive s3=nullptr but
   ARLE always passes s3 → kernel would crash or return 0xff data。
   
   `weight.group_size` propagates from quant script GROUP_SIZE = 128。
   Probably OK but worth confirming with `python -c "import torch; ..."`
   on quantized output。

## Recommended next investigation

**Option 1(highest ROI):single-linear-layer unit test**
- Take 1 Linear layer from Qwen3-4B (e.g. `model.layers[0].mlp.gate_proj`)
- Quantize via `pack_w4a8(weight)` → save tensors
- Load via ARLE weight_loader → run `run_marlin_w4a8_linear` on a known
  input vector(e.g. `[1, 1, ..., 1]`)
- Compare output to BF16 reference(`weight @ input`)
- This **isolates** quant + kernel from the full model + activation quantizer
  + loader runtime

If this unit test passes → bug is in some interaction across layers /
all-layer quant /loader full pipeline → narrow to integration。

If this unit test fails → bug 100% in `pack_w4a8`(remaining 3 candidate
sites) or kernel-with-this-quant-input。

**Option 2(continued iteration)**:
- Try reverting H3c (revert to H3+H3b state which was "closest to English")
- Inspect the H3+H3b output more carefully:**which output channels are right
  vs which are wrong**? May reveal which permute layer is broken。
- If "first 8 channels right,others wrong" → tile permute issue。
- If "every 4th channel right" → bit-packing stride issue。

**Option 3(bypass)**:
- Use AutoGPTQ or vLLM W4A8 fork to produce known-good Qwen3-4B W4A8
  checkpoint
- Load that into ARLE — if **passes**,confirms ARLE wiring + kernel are
  fine and our pack script is the bug。
- If **fails**,kernel-internal interaction with input format we missed。

Recommended:**Option 1 first(2-4 hours codex implementation),then Option 3
if Option 1 still fails**。

## Probability estimate

Given:
- Kernel verbatim from PR #31 ✓
- Wiring matches signature ✓
- All script-side perm math iterated 3× without convergence

P(bug in `pack_w4a8` remaining 3 sites)= 60%
P(bug in PR #31 quantize-side code we copied,not kernel)= 25%
P(bug in PR #31 tile permute documentation we mis-read)= 10%
P(bug in some interaction we haven't enumerated)= 5%

Methodology:**stop iterating script blind**;run unit test to isolate。

## Cross-references

- H3c applied still wrong: [`4dea952`](2026-05-08-w4a8-h3c-applied-still-broken.md)
- H3c confirmed: [`d0f030b`](2026-05-08-w4a8-bug-h3c-confirmed-permute-before-divide.md)
- H3b confirmed: [`3479a87`](2026-05-08-w4a8-bug-h3b-confirmed-scale-perm-single-deleted.md)
- H3 row stride: [`25391f3`](2026-05-08-w4a8-bug-h3-confirmed-perms-row-stride.md)
- Kernel source: `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`(961+26 LOC)
- PR #31 reference: `/tmp/marlin-w4a8/marlin/w4a8_marlin_cuda_kernel.cu`(961 LOC)
- Linear FFI call: `infer/src/ops/linear.rs:777-859`
- Loader naming: `infer/src/weight_loader.rs:669-671`
- Quant script: `/tmp/quantize_qwen3_w4a8.py:77-121`
- W4A8 garbage gate: `81b6481`

## Rule

When iterating on a quant script vs upstream reference,the **third
elimination iteration** is the right time to escalate to:
1. Direct byte-diff of kernel(rule out kernel divergence)
2. Single-layer unit test with known input(isolate from full pipeline)
3. Known-good reference checkpoint(rule out script entirely)

Iterating on script alone past 3× without convergence indicates **wrong
methodology**,not insufficient effort。Per skill anti-pattern #13:NULL
elimination is real progress when paired with non-iterative escalation
on the 4th attempt。
