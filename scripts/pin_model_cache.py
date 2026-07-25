#!/usr/bin/env python3
"""Pin model weight files in RAM so a cold serve boot never re-reads them (#181).

    pin_model_cache.py <model-dir> [--glob '*.safetensors'] [--max-fraction 0.5]

Why: the H20 pod's /host reads at ~0.2 GB/s and does not scale with concurrency,
so a cold DeepSeek-V4-Flash boot spends ~25 min in `loader prefetch` before
engine-ready; warm it is 90 s. The page cache already holds the weights after one
read — this just makes that residency GUARANTEED (mlock) instead of merely likely.

Runs in the foreground and holds the lock until killed: mlock is a property of
this process's mappings, so exiting releases it (the pages stay in the page cache
afterwards, just evictable again). Start it detached and leave it:

    nohup python3 scripts/pin_model_cache.py /host/DeepSeek-V4-Flash-FP8 \
        > /host/pin-model-cache.log 2>&1 &

Needs CAP_IPC_LOCK to raise RLIMIT_MEMLOCK (the pod runs as root; the default
`ulimit -l` of 64 KB is nowhere near enough and is raised here automatically).

Locked pages are NOT reclaimable. The guard refuses to pin more than
--max-fraction of MemTotal so this can never be the reason something else on the
box gets OOM-killed.
"""

import argparse
import ctypes
import os
import resource
import signal
import sys
from pathlib import Path

PROT_READ = 0x1
MAP_SHARED = 0x01
MAP_POPULATE = 0x8000  # Linux: prefault the pages, i.e. do the slow read here
MAP_FAILED = ctypes.c_void_p(-1).value

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def mem_total_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemTotal missing from /proc/meminfo")


def vm_locked_bytes() -> int:
    for line in Path(f"/proc/{os.getpid()}/status").read_text().splitlines():
        if line.startswith("VmLck:"):
            return int(line.split()[1]) * 1024
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--glob", default="*.safetensors")
    ap.add_argument("--max-fraction", type=float, default=0.5)
    args = ap.parse_args()

    files = sorted(args.model_dir.glob(args.glob))
    if not files:
        print(f"no {args.glob} under {args.model_dir}", file=sys.stderr)
        return 1
    total = sum(f.stat().st_size for f in files)
    ram = mem_total_bytes()
    share = total / ram
    print(
        f"{len(files)} files, {total / 1e9:.1f} GB, {share:.1%} of {ram / 1e9:.0f} GB RAM"
    )
    if share > args.max_fraction:
        print(
            f"refusing: {share:.1%} exceeds --max-fraction {args.max_fraction:.0%}; "
            "mlocked pages are unreclaimable",
            file=sys.stderr,
        )
        return 1

    # +1 GiB slack for the interpreter's own locked pages, if any.
    want = total + (1 << 30)
    try:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (want, want))
    except (ValueError, OSError) as e:
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        print(
            f"cannot raise RLIMIT_MEMLOCK to {want / 1e9:.1f} GB ({e}); "
            f"current soft limit {soft / 1e6:.1f} MB — need CAP_IPC_LOCK",
            file=sys.stderr,
        )
        return 1

    held = []  # keep the mappings alive; munmap would unlock them
    done = 0
    for f in files:
        size = f.stat().st_size
        fd = os.open(f, os.O_RDONLY)
        try:
            addr = libc.mmap(
                None, size, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0
            )
            if addr == MAP_FAILED:
                err = ctypes.get_errno()
                print(f"mmap {f.name}: {os.strerror(err)}", file=sys.stderr)
                return 1
            if libc.mlock(ctypes.c_void_p(addr), size) != 0:
                err = ctypes.get_errno()
                print(
                    f"mlock {f.name} ({size / 1e9:.1f} GB) failed after "
                    f"{done / 1e9:.1f} GB: {os.strerror(err)}",
                    file=sys.stderr,
                )
                return 1
            held.append((addr, size))
            done += size
            print(f"  {f.name}: +{size / 1e9:.1f} GB ({done / 1e9:.1f} GB locked)", flush=True)
        finally:
            os.close(fd)  # the mapping keeps its own reference

    print(
        f"resident: {done / 1e9:.1f} GB, VmLck {vm_locked_bytes() / 1e9:.1f} GB\n"
        "holding — kill this PID to release (pages stay cached, just evictable)",
        flush=True,
    )
    signal.pause()
    return 0


if __name__ == "__main__":
    sys.exit(main())
