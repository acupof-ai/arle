#!/usr/bin/env python3
"""
Host-side proof of the inverse of Marlin's GPTQ repack permutation.

FORWARD is transcribed literally from
  crates/cuda-kernels/csrc/gemm/marlin/gptq_marlin_repack.cuh  (has_perm = false arm)
  crates/cuda-kernels/csrc/gemm/marlin/marlin.cuh              (tile constants)
plus the host-side pre-transposes in crates/cuda-kernels/src/tensor.rs:
  FP4 (num_bits=4): gptq[j*N + n] = LE u32 of packed[n*(K/2) + 4j .. 4j+4]   (tensor.rs:3271-3277)
  FP8 (num_bits=8): gptq_bytes[((k/4)*N + n)*4 + (k%4)] = W[n,k]             (tensor.rs:3026)

INVERSE is the closed form proposed for the CUDA de-permute kernel. Every element
of every shape is checked both ways.

No GPU, no CUDA, pure integer arithmetic.
"""

import random

TILE_SIZE = 16          # marlin.cuh:21
TILE_K = TILE_SIZE      # marlin.cuh:29
TILE_N = TILE_K * 4     # marlin.cuh:30  == 64
TC_OFFSETS = (0, 1, 8, 9)


# ---------------------------------------------------------------- FORWARD ----

def forward_repack(gptq, K, N, num_bits):
    """Literal transcription of gptq_marlin_repack_kernel<*, num_bits, false>.

    gptq: flat list of uint32, shape [K/pack_factor, N] row-major.
    returns: flat list of uint32, the marlin buffer.
    """
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    k_tiles = K // TILE_K
    n_tiles = N // TILE_N
    tile_ints = TILE_K // pack_factor
    stage_n_threads = TILE_N // 4
    sh_stride = 64
    out_tile_words = TILE_K * TILE_N // pack_factor
    out = [0] * (k_tiles * n_tiles * out_tile_words)

    for k_tile_id in range(k_tiles):
        for n_tile_id in range(n_tiles):
            # --- fetch_to_shared (no-perm arm), int4 loads flattened to u32 ---
            first_n = n_tile_id * TILE_N
            first_k = k_tile_id * TILE_K
            first_k_packed = first_k // pack_factor
            sh = [0] * (tile_ints * sh_stride)
            for tid in range(tile_ints * stage_n_threads):      # stage_size
                k_id = tid // stage_n_threads
                n_id = tid % stage_n_threads
                for j in range(4):                              # the int4 = 4 u32
                    sh[k_id * sh_stride + n_id * 4 + j] = \
                        gptq[(first_k_packed + k_id) * N + first_n + n_id * 4 + j]

            # --- repack_tile ---
            for warp_id in range(4):
                for th_id in range(32):
                    tc_col = th_id // 4
                    tc_row = (th_id % 4) * 2
                    cur_n = warp_id * 16 + tc_col

                    b1 = [sh[cur_n + sh_stride * i] for i in range(tile_ints)]
                    b2 = [sh[cur_n + 8 + sh_stride * i] for i in range(tile_ints)]

                    vals = [0] * 8
                    for i in range(4):
                        cur_elem = tc_row + TC_OFFSETS[i]
                        cur_int = cur_elem // pack_factor
                        cur_pos = cur_elem % pack_factor
                        vals[i] = (b1[cur_int] >> (cur_pos * num_bits)) & mask
                        vals[4 + i] = (b2[cur_int] >> (cur_pos * num_bits)) & mask

                    out_offset = (k_tile_id * n_tiles + n_tile_id) * out_tile_words

                    if num_bits == 4:
                        pack_idx = (0, 2, 4, 6, 1, 3, 5, 7)
                        res = 0
                        for i in range(8):
                            res |= vals[pack_idx[i]] << (i * 4)
                        out[out_offset + th_id * 4 + warp_id] = res
                    else:
                        pack_idx = (0, 2, 1, 3)
                        res1 = res2 = 0
                        for i in range(4):
                            res1 |= vals[pack_idx[i]] << (i * 8)
                            res2 |= vals[4 + pack_idx[i]] << (i * 8)
                        out[out_offset + th_id * 8 + warp_id * 2 + 0] = res1
                        out[out_offset + th_id * 8 + warp_id * 2 + 1] = res2
    return out


# ---------------------------------------------------------------- INVERSE ----

def inv_fp4(n, k, K, N):
    """(n,k) -> (marlin word index, nibble slot). num_bits == 4."""
    n_tiles = N // TILE_N
    kt, nt = k // TILE_K, n // TILE_N
    kk, nn = k % TILE_K, n % TILE_N
    half = (nn % 16) // 8
    warp_id = nn // 16
    th_id = (nn % 8) * 4 + ((kk % 8) // 2)
    slot = (kk & 1) * 4 + half * 2 + (kk >> 3)
    word = (kt * n_tiles + nt) * 128 + th_id * 4 + warp_id
    return word, slot


def inv_fp4_gather(word, slot, K, N):
    """(marlin word index, nibble slot) -> (n,k). The gather-side form."""
    n_tiles = N // TILE_N
    tile, rem = divmod(word, 128)
    kt, nt = divmod(tile, n_tiles)
    th_id, warp_id = rem // 4, rem % 4
    tc_col, tc_row = th_id // 4, (th_id % 4) * 2
    cur_n = warp_id * 16 + tc_col
    dk = (slot & 1) * 8 + ((slot >> 2) & 1)      # {0,8,1,9} indexed by low/high bits
    dn = ((slot >> 1) & 1) * 8
    return nt * TILE_N + cur_n + dn, kt * TILE_K + tc_row + dk


def inv_fp8(n, k, K, N):
    """(n,k) -> byte offset into the marlin buffer. num_bits == 8."""
    n_tiles = N // TILE_N
    kt, nt = k // TILE_K, n // TILE_N
    kk, nn = k % TILE_K, n % TILE_N
    half = (nn % 16) // 8
    warp_id = nn // 16
    th_id = (nn % 8) * 4 + ((kk % 8) // 2)
    word = (kt * n_tiles + nt) * 256 + th_id * 8 + warp_id * 2 + half
    byte = ((kk & 1) << 1) | (kk >> 3)
    return word * 4 + byte


# ------------------------------------------------------------------ TESTS ----

def test_fp4(K, N, seed):
    rng = random.Random(seed)
    W = [[rng.randrange(16) for _ in range(K)] for _ in range(N)]     # E2M1 nibbles

    # checkpoint layout [N, K/2] u8, even k in the low nibble
    packed = bytearray(N * (K // 2))
    for n in range(N):
        for k in range(K):
            packed[n * (K // 2) + k // 2] |= W[n][k] << ((k % 2) * 4)

    # tensor.rs:3271-3277 host transpose to gptq [K/8, N]
    gptq = [0] * ((K // 8) * N)
    for n in range(N):
        for j in range(K // 8):
            base = n * (K // 2) + j * 4
            gptq[j * N + n] = int.from_bytes(packed[base:base + 4], "little")

    marlin = forward_repack(gptq, K, N, 4)
    assert len(marlin) == K * N // 8, (len(marlin), K * N // 8)

    # (a) scatter form: inv_fp4(n,k) must name the slot holding W[n][k]
    bad = 0
    for n in range(N):
        for k in range(K):
            word, slot = inv_fp4(n, k, K, N)
            if (marlin[word] >> (4 * slot)) & 0xF != W[n][k]:
                bad += 1
    # (b) gather form: walk every slot, rebuild the [N,K/2] packed buffer
    rebuilt = bytearray(N * (K // 2))
    touched = bytearray(N * K)
    for word in range(len(marlin)):
        for slot in range(8):
            n, k = inv_fp4_gather(word, slot, K, N)
            v = (marlin[word] >> (4 * slot)) & 0xF
            rebuilt[n * (K // 2) + k // 2] |= v << ((k % 2) * 4)
            touched[n * K + k] += 1
    bij = all(t == 1 for t in touched)
    ok = (bad == 0) and bij and (rebuilt == packed)
    print(f"  FP4 K={K:4d} N={N:4d}  words={len(marlin):7d}  "
          f"scatter_mismatch={bad}  bijective={bij}  gather_bytes_equal={rebuilt == packed}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def test_fp8(K, N, seed):
    rng = random.Random(seed)
    W = [[rng.randrange(256) for _ in range(K)] for _ in range(N)]    # raw E4M3 bytes

    # tensor.rs:3026 host transpose to gptq [K/4, N] (byte granular, no bias)
    gb = bytearray((K // 4) * N * 4)
    for n in range(N):
        for k in range(K):
            gb[((k // 4) * N + n) * 4 + (k % 4)] = W[n][k]
    gptq = [int.from_bytes(gb[i * 4:i * 4 + 4], "little") for i in range((K // 4) * N)]

    marlin = forward_repack(gptq, K, N, 8)
    assert len(marlin) == K * N // 4, (len(marlin), K * N // 4)
    mbytes = b"".join(w.to_bytes(4, "little") for w in marlin)

    bad = 0
    touched = bytearray(N * K)
    rebuilt = bytearray(N * K)
    for n in range(N):
        for k in range(K):
            off = inv_fp8(n, k, K, N)
            if mbytes[off] != W[n][k]:
                bad += 1
            touched[off] += 1
            rebuilt[n * K + k] = mbytes[off]
    bij = all(t == 1 for t in touched)
    ref = bytes(v for row in W for v in row)
    ok = (bad == 0) and bij and (rebuilt == ref)
    print(f"  FP8 K={K:4d} N={N:4d}  bytes={len(mbytes):8d}  "
          f"mismatch={bad}  bijective={bij}  gather_equal={rebuilt == ref}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------- FP4 group-scale tail inverse ---
# tensor.rs:3327-3345. Verified separately from the weight nibbles.

def scale_forward(src, K, N):
    """src: [N, K/16] bytes (E4M3 codes). Returns the permuted tail, ints."""
    num_groups = K // 16
    sflat = [0.0] * (num_groups * N)
    for n in range(N):
        for g in range(num_groups):
            sflat[g * N + n] = src[n * num_groups + g]        # value opaque here
    sperm = [0.0] * len(sflat)
    for blk in range(len(sflat) // 64):
        base = blk * 64
        for out in range(64):
            sperm[base + out] = sflat[base + (out % 8) * 8 + (out // 8)]
    for q in range(0, len(sperm), 4):
        sperm[q + 1], sperm[q + 2] = sperm[q + 2], sperm[q + 1]
    return sperm


def scale_inverse(t, K, N):
    """tail index t -> source index into [N, K/16]."""
    R = (0, 2, 1, 3)
    blk, f = divmod(t, 64)
    q, r = divmod(f, 4)
    x = q * 4 + R[r]
    src_local = (x % 8) * 8 + (x // 8)
    src_flat = blk * 64 + src_local
    g, n = divmod(src_flat, N)
    return n * (K // 16) + g


def test_scale(K, N, seed):
    rng = random.Random(seed)
    src = [rng.randrange(256) for _ in range(N * (K // 16))]
    tail = scale_forward(src, K, N)
    touched = [0] * len(src)
    bad = 0
    for t in range(len(tail)):
        i = scale_inverse(t, K, N)
        touched[i] += 1
        if src[i] != tail[t]:
            bad += 1
    bij = all(c == 1 for c in touched)
    ok = bad == 0 and bij
    print(f"  SCALE K={K:4d} N={N:4d}  tail={len(tail):7d}  mismatch={bad}  "
          f"bijective={bij}  -> {'PASS' if ok else 'FAIL'}")
    return ok



# ------------------------------------------------ NEGATIVE CONTROL (mutants) --
# Each mutant perturbs exactly one term of the inverse. All must FAIL, otherwise
# the round-trip assertions above are not actually constraining that term.

def _run_guarded(fn, tester, K, N):
    """Returns True if the tester passes with `fn` installed. Exceptions = fail."""
    import contextlib, io as _io
    g = globals()
    name = "inv_fp4" if tester is test_fp4 else "inv_fp8"
    orig = g[name]
    g[name] = fn
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            return tester(K, N, seed=7)
    except Exception:
        return False
    finally:
        g[name] = orig


def mutation_controls(K=128, N=128):
    b4, b8 = inv_fp4, inv_fp8
    nt, kt_ = N // TILE_N, K // TILE_K
    m4 = {
        "th_id: (kk%8)//2 -> kk//2":
            lambda n, k, K, N: ((k // 16 * nt + n // 64) * 128
                                + ((n % 64 % 8) * 4 + (k % 16) // 2) * 4 + (n % 64) // 16,
                                b4(n, k, K, N)[1]),
        "slot: drop the half*2 term":
            lambda n, k, K, N: (b4(n, k, K, N)[0], (k % 16 & 1) * 4 + (k % 16 >> 3)),
        "slot: swap the low/high k roles":
            lambda n, k, K, N: (b4(n, k, K, N)[0],
                                (k % 16 >> 3) * 4 + (n % 64 % 16 // 8) * 2 + (k % 16 & 1)),
        "word: th_id*4+warp -> warp*32+th_id":
            lambda n, k, K, N: (b4(n, k, K, N)[0] // 128 * 128
                                + (n % 64 // 16) * 32 + (n % 64 % 8) * 4 + (k % 16 % 8) // 2,
                                b4(n, k, K, N)[1]),
        "tile order: n-major instead of k-major":
            lambda n, k, K, N: ((n // 64 * kt_ + k // 16) * 128 + b4(n, k, K, N)[0] % 128,
                                b4(n, k, K, N)[1]),
    }
    m8 = {
        "byte: ((kk&1)<<1)|(kk>>3) -> kk%4":
            lambda n, k, K, N: b8(n, k, K, N) // 4 * 4 + k % 16 % 4,
        "byte: un-swap to ((kk>>3)<<1)|(kk&1)":
            lambda n, k, K, N: b8(n, k, K, N) // 4 * 4 + ((k % 16 >> 3) << 1 | (k % 16 & 1)),
        "word: warp_id*2+half -> half*2+warp_id":
            lambda n, k, K, N: ((b8(n, k, K, N) // 4)
                                - (n % 64 // 16) * 2 - (n % 64 % 16 // 8)
                                + (n % 64 % 16 // 8) * 2 + (n % 64 // 16)) * 4 + b8(n, k, K, N) % 4,
    }
    allbad = True
    for name, f in m4.items():
        ok = _run_guarded(f, test_fp4, K, N)
        allbad &= not ok
        print(f"  FP4 mutant [{name}]: passes={ok}  -> {'ok (rejected)' if not ok else 'BAD'}")
    for name, f in m8.items():
        ok = _run_guarded(f, test_fp8, K, N)
        allbad &= not ok
        print(f"  FP8 mutant [{name}]: passes={ok}  -> {'ok (rejected)' if not ok else 'BAD'}")
    return allbad


def sanity_nontrivial(K=128, N=128):
    """The forward really permutes: marlin bytes differ from the source bytes,
    while the multiset of values is preserved."""
    rng = random.Random(1)
    W = [[rng.randrange(16) for _ in range(K)] for _ in range(N)]
    packed = bytearray(N * (K // 2))
    for n in range(N):
        for k in range(K):
            packed[n * (K // 2) + k // 2] |= W[n][k] << ((k % 2) * 4)
    gptq = [0] * ((K // 8) * N)
    for n in range(N):
        for j in range(K // 8):
            b = n * (K // 2) + j * 4
            gptq[j * N + n] = int.from_bytes(packed[b:b + 4], "little")
    m = forward_repack(gptq, K, N, 4)
    mb = b"".join(w.to_bytes(4, "little") for w in m)
    diff = mb != bytes(packed)
    same_multiset = (sorted((w >> (4 * s)) & 0xF for w in m for s in range(8))
                     == sorted(v for r in W for v in r))
    print(f"  forward is a real permutation (bytes differ): {diff}")
    print(f"  forward preserves the value multiset:         {same_multiset}")
    return diff and same_multiset


if __name__ == "__main__":
    # (K, N): min tile counts, multi-tile both axes, non-square, and one
    # matching a real Qwen3.8-27B ratio (down_proj-shaped 256x640).
    shapes = [(64, 64), (128, 128), (256, 192), (128, 320), (256, 640), (64, 448)]
    allok = True
    print("num_bits == 4  (NVFP4 weight nibbles)")
    for K, N in shapes:
        allok &= test_fp4(K, N, seed=K * 7919 + N)
    print("num_bits == 8  (per-channel FP8 bytes)")
    for K, N in shapes:
        allok &= test_fp8(K, N, seed=K * 104729 + N)
    print("FP4 group-scale tail (tensor.rs:3327-3345)")
    for K, N in shapes:
        allok &= test_scale(K, N, seed=K + N)
    print()
    print("Sanity: forward is non-degenerate")
    allok &= sanity_nontrivial()
    print("Negative control: single-term mutants of the inverse")
    allok &= mutation_controls()
    print()
    print("ALL ROUND-TRIPS EXACT, ALL MUTANTS REJECTED" if allok else "FAILURE")
    raise SystemExit(0 if allok else 1)
