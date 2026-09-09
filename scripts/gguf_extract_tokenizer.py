"""Write a checkpoint's `tokenizer.json` + `chat_template.jinja` out of a GGUF.

A GGUF converted by llama.cpp carries its whole tokenizer in metadata, but the
serving path (`OpenAiTokenizer::from_model_dir`) loads a sibling
`tokenizer.json`. A GGUF downloaded on its own therefore fails at startup with
`load tokenizer <dir>/tokenizer.json failed: (os error 2)` even though every
byte it needs is inside the file. This script closes that gap without a network
round-trip, which matters when the checkpoint came from a mirror that does not
publish the HF sidecars.

The reconstruction MIRRORS `crates/infer-gguf/src/tokenizer.rs`
(`GgufTokenizer::from_gguf`) on purpose — same BPE vocab/merge split, same
ByteLevel pre-tokenizer/decoder/post-processor flags, same rule for which token
types become special. If you change one, change the other, or a model will
tokenize differently depending on which path loaded it.

Supports `tokenizer.ggml.model == "gpt2"` (byte-level BPE) only, which is what
the Qwen3.5 / 3.6 / 3.8 family ships.

Usage:
    python scripts/gguf_extract_tokenizer.py <model.gguf> [--out-dir DIR]

Writes `tokenizer.json` and, when the GGUF declares one, `chat_template.jinja`
into `--out-dir` (default: the GGUF's own directory, which is where the serving
path looks). Refuses to clobber existing files unless `--force`.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

# GGUF value type tags.
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)

_FIXED = {
    U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}

# `tokenizer.ggml.token_type` values. 3 = CONTROL, 4 = USER_DEFINED: these are
# the `<|im_start|>` / `<think>` style tokens that must encode atomically.
TYPE_CONTROL, TYPE_USER_DEFINED = 3, 4


class Reader:
    """Full-fidelity GGUF metadata reader.

    Deliberately NOT shared with `gguf_probe.py`: that one truncates every
    array to a head sample so it can dump a 248k-entry vocab without holding it,
    which is exactly what this script must not do.
    """

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
            if elem == STR:
                return [self.string() for _ in range(count)]
            fmt, size = _FIXED[elem]
            return [self.scalar(fmt, size) for _ in range(count)]
        raise ValueError(f"unknown gguf value type {vtype}")


def read_metadata(path: Path) -> dict:
    with open(path, "rb") as fh:
        r = Reader(fh)
        if r.raw(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        version = r.scalar("<I", 4)
        if version not in (2, 3):
            raise SystemExit(f"{path}: unsupported GGUF version {version}")
        r.scalar("<Q", 8)  # tensor count — unused, we never touch tensor data
        kv_count = r.scalar("<Q", 8)
        kv = {}
        for _ in range(kv_count):
            key = r.string()
            kv[key] = r.value(r.scalar("<I", 4))
        return kv


def build_tokenizer_json(kv: dict) -> dict:
    model = kv.get("tokenizer.ggml.model")
    if model != "gpt2":
        raise SystemExit(
            f"tokenizer.ggml.model = {model!r}; only 'gpt2' byte-level BPE is supported "
            "(same limit as GgufTokenizer::from_gguf)"
        )
    tokens = kv.get("tokenizer.ggml.tokens")
    merges_raw = kv.get("tokenizer.ggml.merges")
    token_types = kv.get("tokenizer.ggml.token_type")
    if not tokens or not merges_raw:
        raise SystemExit("GGUF is missing tokenizer.ggml.tokens or .merges")
    if token_types is not None and len(token_types) != len(tokens):
        raise SystemExit(
            f"token_type len {len(token_types)} != tokens len {len(tokens)}"
        )

    # GGUF stores each merge as "left right". Token strings are already
    # byte-level encoded (space -> 'G-with-dot'), so they never contain a raw
    # space and splitting on the FIRST one is unambiguous.
    merges = []
    for m in merges_raw:
        left, sep, right = m.partition(" ")
        if not sep:
            raise SystemExit(f"malformed merge rule {m!r} (no space)")
        merges.append([left, right])

    # Array index IS the id.
    vocab = {tok: i for i, tok in enumerate(tokens)}
    if len(vocab) != len(tokens):
        # A duplicate token string would silently shift ids for everything after
        # it; refuse rather than emit a subtly wrong vocab.
        raise SystemExit(
            f"vocab has duplicate token strings ({len(tokens)} entries, "
            f"{len(vocab)} unique) — cannot build an id-stable map"
        )

    added = []
    if token_types:
        for i, t in enumerate(token_types):
            if t in (TYPE_CONTROL, TYPE_USER_DEFINED):
                added.append({
                    "id": i,
                    "content": tokens[i],
                    "single_word": False,
                    # Match `AddedToken::from(.., true).special(true)`: the Rust
                    # side takes tokenizers' defaults for stripping, which are
                    # false/false, and normalized=false for a special token.
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                })

    # ByteLevel flags mirror the Rust path exactly:
    #   ByteLevelPre::new(add_prefix_space=false, trim_offsets=true, use_regex=true)
    #   ByteLevelPost::new(add_prefix_space=false, trim_offsets=true, use_regex=true)
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            # The byte-level vocab is exhaustive, so there is no <unk>: every
            # input byte maps to a known base token.
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": merges,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: the GGUF's own directory (where the serving path looks)",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    out_dir = args.out_dir or args.gguf.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    kv = read_metadata(args.gguf)
    tok = build_tokenizer_json(kv)

    written = []
    tok_path = out_dir / "tokenizer.json"
    if tok_path.exists() and not args.force:
        print(f"refusing to overwrite {tok_path} (pass --force)", file=sys.stderr)
        return 1
    tok_path.write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")
    written.append(
        f"{tok_path}  ({len(tok['model']['vocab'])} tokens, "
        f"{len(tok['model']['merges'])} merges, {len(tok['added_tokens'])} special)"
    )

    template = kv.get("tokenizer.chat_template")
    if template:
        tpl_path = out_dir / "chat_template.jinja"
        if tpl_path.exists() and not args.force:
            print(f"refusing to overwrite {tpl_path} (pass --force)", file=sys.stderr)
            return 1
        tpl_path.write_text(template, encoding="utf-8")
        written.append(f"{tpl_path}  ({len(template)} chars)")
    else:
        print(
            "note: GGUF declares no tokenizer.chat_template; the server will fall "
            "back to a builtin and warn",
            file=sys.stderr,
        )

    # `tokenizer_config.json` is where the serving path looks FIRST for the
    # template and for the bos/eos token strings it feeds the Jinja context.
    # Emit the token strings so a chat template that references `bos_token` /
    # `eos_token` renders instead of silently interpolating nothing.
    cfg = {}
    for name, key in (
        ("bos_token", "tokenizer.ggml.bos_token_id"),
        ("eos_token", "tokenizer.ggml.eos_token_id"),
        ("pad_token", "tokenizer.ggml.padding_token_id"),
    ):
        tid = kv.get(key)
        if tid is not None and 0 <= tid < len(kv["tokenizer.ggml.tokens"]):
            cfg[name] = kv["tokenizer.ggml.tokens"][tid]
    if cfg:
        cfg_path = out_dir / "tokenizer_config.json"
        if cfg_path.exists() and not args.force:
            print(f"refusing to overwrite {cfg_path} (pass --force)", file=sys.stderr)
            return 1
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(f"{cfg_path}  ({', '.join(sorted(cfg))})")

    for line in written:
        print("wrote", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
