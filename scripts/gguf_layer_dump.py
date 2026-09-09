"""Print every tensor of a chosen `blk.<N>` from a GGUF, header-only.

Companion to `gguf_probe.py`, which collapses by suffix — this one answers
"what exactly does that one odd layer carry?".
"""

import importlib.util
import sys
from pathlib import Path

_probe = Path(__file__).with_name("gguf_probe.py")
_spec = importlib.util.spec_from_file_location("gguf_probe", _probe)
gguf_probe = importlib.util.module_from_spec(_spec)
sys.modules["gguf_probe"] = gguf_probe
_spec.loader.exec_module(gguf_probe)


def tensors(path):
    with open(path, "rb") as fh:
        r = gguf_probe.Reader(fh)
        if r.raw(4) != b"GGUF":
            raise SystemExit("not a GGUF file")
        r.scalar("<I", 4)
        n_tensors = r.scalar("<Q", 8)
        n_kv = r.scalar("<Q", 8)
        for _ in range(n_kv):
            r.string()
            r.value(r.scalar("<I", 4))
        out = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.scalar("<Q", 8) for _ in range(r.scalar("<I", 4))]
            ggml_type = r.scalar("<I", 4)
            r.scalar("<Q", 8)  # offset
            out.append((name, dims, ggml_type))
        return out


def main(path, *layers):
    all_tensors = tensors(path)
    for layer in layers:
        prefix = f"blk.{layer}."
        print(f"=== {prefix} ===")
        for name, dims, ggml_type in all_tensors:
            if name.startswith(prefix):
                print(f"  {name:45s} {str(dims):22s} type={ggml_type}")


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
