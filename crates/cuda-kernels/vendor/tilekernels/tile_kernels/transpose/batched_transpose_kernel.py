import os
import torch
import tilelang
from tilelang import language as T


def create_loop_layout_fn(block_x: int, num_threads: int = 256):
    def loop_layout_fn(i, j):
        elems = i * block_x + j
        forward_thread = (elems // 4) % num_threads
        forward_local = elems % 4 + elems // (num_threads * 4) * 4
        return forward_thread, forward_local

    return loop_layout_fn


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_batched_transpose_kernel(shape_x_mod_128: int, shape_y_mod_128: int, dtype: T.dtype):
    assert shape_x_mod_128 in (0, 64) and shape_y_mod_128 in (0, 64)
    # Runtime symbols
    num_batches = T.dynamic('num_batches')
    shape_x = T.dynamic('shape_x')
    shape_y = T.dynamic('shape_y')
    stride_x = T.dynamic('stride_x')

    num_threads = 256
    block_x = 128 if shape_x_mod_128 == 0 else 64
    block_y = 128 if shape_y_mod_128 == 0 else 64
    block_k = 4
    num_threads_per_row = block_y // block_k

    loop_layout = T.Fragment((block_y, block_x), forward_fn=create_loop_layout_fn(block_x, num_threads))

    @T.prim_func
    def batched_transpose_kernel(
        x: T.StridedTensor[(num_batches, shape_x, shape_y), (shape_x * stride_x, stride_x, 1), dtype],
        out: T.Tensor[(num_batches, shape_y, shape_x), dtype],
    ):
        with T.Kernel(shape_y // block_y, shape_x // block_x, num_batches, threads=num_threads) as (pid_y, pid_x, pid_batch):
            # Shared padding to reduce bank conflict
            out_shared = T.alloc_shared((block_y, block_x + block_k), dtype)
            tid = T.get_thread_binding()
            row, col = tid // num_threads_per_row, tid % num_threads_per_row

            T.assume(shape_x % block_x == 0)
            T.assume(shape_y % block_y == 0)
            T.assume(stride_x % block_k == 0)

            # Read and transpose
            tmp = T.alloc_local((block_k, block_k), dtype)
            tmp_row = T.alloc_local((block_k,), dtype)
            for i_ in T.unroll(block_x // block_k // (num_threads // num_threads_per_row)):
                i = i_ * (num_threads // num_threads_per_row) + row
                # Read into registers
                for j in T.unroll(block_k):
                    for k in T.vectorized(block_k):
                        tmp_row[k] = x[pid_batch, pid_x * block_x + i * block_k + j, pid_y * block_y + col * block_k + k]
                    for k in T.unroll(block_k):
                        tmp[k, j] = tmp_row[k]

                # Copy into shared memory
                for j in T.unroll(block_k):
                    swizzle_j = (j + tid // (8 // dtype.bytes)) % block_k
                    for k in T.vectorized(block_k):
                        out_shared[col * block_k + swizzle_j, i * block_k + k] = tmp[swizzle_j, k]

            T.sync_threads()
            # Write into output
            for i, j in T.Parallel(block_y, block_x, loop_layout=loop_layout):
                out[pid_batch, pid_y * block_y + i, pid_x * block_x + j] = out_shared[i, j]

    return batched_transpose_kernel


def transpose(x: torch.Tensor) -> torch.Tensor:
    """Transpose a 2D tensor using a tiled GPU kernel.

    Args:
        x: Input 2D tensor of shape ``(M, N)`` with dimensions divisible by 64.

    Returns:
        Transposed tensor of shape ``(N, M)``.
    """
    x = x.unsqueeze(0)
    out = batched_transpose(x)
    out = out.squeeze(0)
    return out


def batched_transpose(x: torch.Tensor) -> torch.Tensor:
    """Transpose a batched 3D tensor using a tiled GPU kernel.

    Args:
        x: Input 3D tensor of shape ``(B, M, N)`` with ``M`` and ``N``
            divisible by 64.

    Returns:
        Transposed tensor of shape ``(B, N, M)``.
    """
    assert x.dim() == 3
    num_batches, shape_x, shape_y = x.shape

    assert shape_x % 64 == 0 and shape_y % 64 == 0 and x.stride(-2) % 4 == 0 and x.stride(-1) == 1

    # Get kernel implement
    kernel = get_batched_transpose_kernel(shape_x % 128, shape_y % 128, T.dtype(x.dtype))

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    out = torch.empty((num_batches, shape_y, shape_x), dtype=x.dtype, device='cuda')
    if num_batches > 0 and shape_x > 0 and shape_y > 0:
        kernel(x, out)

    return out
