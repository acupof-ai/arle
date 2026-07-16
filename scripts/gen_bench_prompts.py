#!/usr/bin/env python3
"""Generate a varied-prompt bench dataset (bench_throughput.py --prompts-jsonl).

Usage: gen_bench_prompts.py [out.jsonl] [count] [chars_per_doc] [output_tokens]

Each doc gets a unique header + per-sentence doc/note indices, so no two
prompts share a long prefix (prefix-cache reuse can't mask prefill cost) and
the text is non-degenerate (unlike the repeated-filler bench-prompts.jsonl).
count must be >= the highest benched concurrency.
"""
import json
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "bench-prompts-64.jsonl"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 64
chars = int(sys.argv[3]) if len(sys.argv) > 3 else 13400
output_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 256

topics = [
    "the cooling system in rack {i} requires quarterly inspection of pump seals and coolant purity",
    "note {i} of the logistics ledger records a shipment of titanium fasteners routed through the northern depot",
    "in observation {i} the telescope array captured a transient radio burst lasting fourteen milliseconds",
    "entry {i} describes how the fermentation batch reached target acidity two days ahead of schedule",
    "audit item {i} confirms the bridge expansion joints were resealed before the winter freeze",
    "measurement {i} shows the river gauge rising three centimeters after the upstream reservoir release",
    "paragraph {i} of the training manual explains recovering a stalled compressor without venting refrigerant",
    "archive box {i} contains correspondence from the 1970s expansion of the coastal rail line",
]

with open(out, "w") as f:
    for d in range(count):
        text = f"Document {d}: field journal, volume {d + 1}, compiled for archive shelf {d * 7 % 100}."
        i = 0
        while len(text) < chars:
            i += 1
            text += (
                " "
                + topics[i % len(topics)].format(i=i)
                + f" This is note {i} in document {d}."
            )
        f.write(json.dumps({"text": text, "output_tokens": output_tokens}) + "\n")

print(f"wrote {count} docs x ~{chars} chars -> {out}")
