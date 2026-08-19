"""Dump GGUF metadata + tensor names without loading weights.

Ad-hoc probe for checking whether a new checkpoint fits an existing loader's
tensor-name schema before spending a full model load on the answer.
"""

import struct
import sys

# GGUF value type tags.
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)

_FIXED = {
    U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError(f"short read: wanted {n}, got {len(b)}")
        return b

    def scalar(self, fmt, n):
        return struct.unpack(fmt, self.raw(n))[0]

    def string(self):
        return self.raw(self.scalar("<Q", 8)).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype in _FIXED:
            return self.scalar(*_FIXED[vtype])
        if vtype == STR:
            return self.string()
        if vtype == ARR:
            elem = self.scalar("<I", 4)
            count = self.scalar("<Q", 8)
            # Vocab arrays run to 150k+ entries; keep a head sample, skip the rest.
            if elem == STR:
                head = [self.string() for _ in range(min(count, 5))]
                for _ in range(count - len(head)):
                    self.raw(self.scalar("<Q", 8))
                return f"[{count} strings] {head[:3]}..."
            fmt, size = _FIXED[elem]
            vals = [self.scalar(fmt, size) for _ in range(min(count, 64))]
            self.raw((count - len(vals)) * size)
            return vals if count <= 64 else f"[{count} vals] {vals[:16]}..."
        raise ValueError(f"unknown gguf value type {vtype}")


def main(path):
    with open(path, "rb") as fh:
        r = Reader(fh)
        magic = r.raw(4)
        if magic != b"GGUF":
            raise SystemExit(f"not a GGUF file: {magic!r}")
        version = r.scalar("<I", 4)
        n_tensors = r.scalar("<Q", 8)
        n_kv = r.scalar("<Q", 8)
        print(f"gguf v{version}  tensors={n_tensors}  kv={n_kv}\n")

        print("=== metadata ===")
        for _ in range(n_kv):
            key = r.string()
            val = r.value(r.scalar("<I", 4))
            if key.startswith("tokenizer.ggml.") and isinstance(val, str):
                val = val[:80]
            print(f"  {key} = {val}")

        print("\n=== tensors ===")
        names = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.scalar("<Q", 8) for _ in range(r.scalar("<I", 4))]
            ggml_type = r.scalar("<I", 4)
            r.scalar("<Q", 8)  # offset
            names.append((name, dims, ggml_type))

    suffixes = {}
    for name, dims, ty in names:
        parts = name.split(".")
        suffix = ".".join(parts[2:]) if name.startswith("blk.") else name
        suffixes.setdefault(suffix, (dims, ty, name))

    print(f"total tensors: {len(names)}")
    print("\ndistinct suffixes (dims / ggml_type / example):")
    for suffix, (dims, ty, example) in sorted(suffixes.items()):
        print(f"  {suffix:34s} {str(dims):22s} type={ty:<3} e.g. {example}")

    layers = {int(n.split(".")[1]) for n, _, _ in names if n.startswith("blk.")}
    if layers:
        print(f"\nblk layer indices: {min(layers)}..{max(layers)} (count {len(layers)})")


if __name__ == "__main__":
    main(sys.argv[1])
