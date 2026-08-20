"""Sum GGUF tensor bytes by residency class.

Answers "how many bytes does a decode step actually stream from memory" — the
denominator of the roofline check. `token_embd` never reaches the device (the
Vulkan loader gathers one row per token on host) and trailing MTP blocks are
skipped, so both are reported separately rather than folded into the total.
"""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "gguf_probe", Path(__file__).with_name("gguf_probe.py")
)
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)

# GGML type -> (block element count, bytes per block). Only the types this
# checkpoint mixes are listed; anything else is a loud KeyError rather than a
# silently wrong byte count.
_TYPES = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    8: (32, 34),    # Q8_0
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
}


def main(path, num_layers):
    with open(path, "rb") as fh:
        r = _probe.Reader(fh)
        if r.raw(4) != b"GGUF":
            raise SystemExit("not a GGUF file")
        r.scalar("<I", 4)
        n_tensors = r.scalar("<Q", 8)
        n_kv = r.scalar("<Q", 8)
        for _ in range(n_kv):
            r.string()
            r.value(r.scalar("<I", 4))

        buckets = {"device": 0, "host_embd": 0, "mtp_skipped": 0}
        # Qwen3.8 interleaves linear-attention and full-attention blocks, so
        # per-layer bytes are bimodal — report both rather than one mean.
        per_layer = {}
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.scalar("<Q", 8) for _ in range(r.scalar("<I", 4))]
            ttype = r.scalar("<I", 4)
            r.scalar("<Q", 8)  # offset
            elems = 1
            for d in dims:
                elems *= d
            per_block, block_bytes = _TYPES[ttype]
            size = elems // per_block * block_bytes

            layer = None
            if name.startswith("blk."):
                layer = int(name.split(".")[1])
            if layer is not None and layer >= num_layers:
                buckets["mtp_skipped"] += size
            elif name.startswith("token_embd"):
                buckets["host_embd"] += size
            else:
                buckets["device"] += size
                if layer is not None:
                    per_layer[layer] = per_layer.get(layer, 0) + size

    for key, total in buckets.items():
        print(f"{key:14} {total / 1e9:8.3f} GB")

    sizes = sorted(per_layer.values())
    print(f"\nlayers         {len(sizes)}")
    print(f"min layer      {sizes[0] / 1e6:8.1f} MB")
    print(f"max layer      {sizes[-1] / 1e6:8.1f} MB")
    print(f"mean layer     {sum(sizes) / len(sizes) / 1e6:8.1f} MB")
    print(f"3 x max layer  {3 * sizes[-1] / 1e6:8.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
