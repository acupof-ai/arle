#!/usr/bin/env python3
"""Per-token logit-lens settlement report over an ARLE probe JSONL (stdlib-only).

Input: the JSONL written by `arle serve --probe-out <path> --probe-lens-layers N`
(record schema: crates/infer-cuda/src/probe.rs). Multi-request sessions are
sliced into request groups; per group the decode stream is analysed for the
layer at which each token's top-1 prediction settles (agrees with the finally
emitted token for ALL deeper layers).

usage: probe_report.py <probe.jsonl> [--tokenizer-dir DIR] [--turns turns.json]
                       [--early-settle 37] [--slack-settle 40]

Token decode uses `tokenizers` or `transformers` from DIR if importable;
otherwise falls back to raw token ids (stated in the output). No hard deps.
"""

import argparse
import json
import math
import statistics
import sys

TIE_EPS = 1e-4  # bf16 argmax tie: final-layer agree=false but top1_logprob == -nll


def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def slice_requests(records):
    """Group records into requests.

    A request starts at a `prefill` record with pos==0, OR at a `prefill`
    record that follows a real decode RUN (>=2 consecutive non-lens decode
    records) — this survives prefix-cache reuse, where a later request's
    prefill positions do not restart at 0. Single interleaved decode records
    (the discarded chunk-boundary samples of chunked prefill) do not split.
    """
    groups, cur, decode_run = [], None, 0
    for rec in records:
        phase = rec.get("phase")
        if phase == "meta":
            continue
        if phase == "prefill":
            if cur is None or rec["pos"] == 0 or decode_run >= 2:
                cur = {"prefill": [], "decode": [], "lens": {}}
                groups.append(cur)
            decode_run = 0
            cur["prefill"].append(rec)
        elif phase == "decode":
            if cur is None:
                continue
            decode_run += 1
            cur["decode"].append(rec)
        elif phase == "lens":
            if cur is None:
                continue
            cur["lens"].setdefault(rec["pos"], {})[rec["layer"]] = rec
    for g in groups:
        g["prompt_len"] = 1 + max(r["pos"] for r in g["prefill"])
        # Boundary decode samples of non-final prefill chunks sit at pos < prompt_len.
        g["decode"] = sorted(
            (r for r in g["decode"] if r["pos"] >= g["prompt_len"]),
            key=lambda r: r["pos"],
        )
        g["lens"] = {p: v for p, v in g["lens"].items() if p >= g["prompt_len"]}
    return groups


def settle_layer(lens_by_layer, layers):
    """Smallest layer L such that agree holds for all layers >= L.

    Returns (settle, tie_adjusted) — settle is None when even the FINAL layer
    genuinely disagrees (not a bf16 argmax tie).
    """
    final = layers[-1]
    agrees, tie_adjusted = {}, False
    for layer in layers:
        rec = lens_by_layer[layer]
        agree = rec["agree"]
        if layer == final and not agree and abs(rec["top1_logprob"] + rec["nll"]) <= TIE_EPS:
            agree, tie_adjusted = True, True
        agrees[layer] = agree
    if not agrees[final]:
        return None, tie_adjusted
    settle = final
    for layer in reversed(layers[:-1]):
        if not agrees[layer]:
            break
        settle = layer
    return settle, tie_adjusted


def build_tokens(groups, layers):
    """Per decode token with a full lens set: settle layer + final stats."""
    tokens = []
    for gi, g in enumerate(groups):
        decode_by_pos = {r["pos"]: r for r in g["decode"]}
        for pos in sorted(g["lens"]):
            lens = g["lens"][pos]
            if sorted(lens) != layers:
                continue
            dec = decode_by_pos.get(pos)
            if dec is None:
                continue
            settle, tie = settle_layer(lens, layers)
            tokens.append(
                {
                    "group": gi,
                    "pos": pos,
                    "token": dec.get("token"),
                    "entropy": dec["entropy"],
                    "nll": dec.get("nll"),
                    "settle": settle,  # None = genuine final-layer disagreement
                    "tie_adjusted": tie,
                    "lens": lens,
                }
            )
    return tokens


def make_decoder(tokenizer_dir):
    if not tokenizer_dir:
        return None, "no --tokenizer-dir given: printing raw token ids only"
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(f"{tokenizer_dir}/tokenizer.json")
        return (lambda tid: tok.decode([tid], skip_special_tokens=False)), "tokenizers"
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        return (lambda tid: tok.decode([tid])), "transformers"
    except Exception:
        return None, "neither `tokenizers` nor `transformers` importable: raw token ids only"


def fmt_stats(values):
    if not values:
        return "n=0"
    return (
        f"n={len(values)} mean={statistics.fmean(values):.4f} "
        f"median={statistics.median(values):.4f}"
    )


def layer_ppl_table(tokens, layers, title):
    print(f"\n## Layer-PPL table — {title} (n={len(tokens)} decode tokens with full lens)")
    print("| layer | mean_nll | PPL | agree% | median_nll | p90_nll | worst10%_nll_share |")
    print("|---|---|---|---|---|---|---|")
    for layer in layers:
        nlls = [t["lens"][layer]["nll"] for t in tokens]
        agree = sum(1 for t in tokens if t["lens"][layer]["agree"])
        nlls_sorted = sorted(nlls)
        p90 = nlls_sorted[min(len(nlls) - 1, int(0.9 * len(nlls)))]
        worst10 = sum(nlls_sorted[int(0.9 * len(nlls)) :])
        total = sum(nlls)
        share = worst10 / total if total > 0 else float("nan")
        mean = statistics.fmean(nlls)
        print(
            f"| {layer} | {mean:.3f} | {math.exp(mean):.2f} | {100 * agree / len(tokens):.1f}% "
            f"| {statistics.median(nlls):.4f} | {p90:.3f} | {100 * share:.1f}% |"
        )


def settle_histogram(tokens, layers, title):
    settled = [t for t in tokens if t["settle"] is not None]
    print(f"\n## Settle-layer histogram — {title} (n={len(settled)})")
    print("| settle layer | count | % | cum% |")
    print("|---|---|---|---|")
    cum = 0
    for layer in layers:
        cnt = sum(1 for t in settled if t["settle"] == layer)
        cum += cnt
        print(
            f"| {layer} | {cnt} | {100 * cnt / len(settled):.1f}% | {100 * cum / len(settled):.1f}% |"
        )
    med = statistics.median(t["settle"] for t in settled)
    print(f"median settle layer: {med}")
    return settled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--tokenizer-dir")
    ap.add_argument("--turns", help="client turns.json for usage cross-check")
    ap.add_argument("--early-settle", type=int, default=37)
    ap.add_argument("--slack-settle", type=int, default=40)
    args = ap.parse_args()

    records = load_records(args.jsonl)
    meta = next((r for r in records if r.get("phase") == "meta"), {})
    print(f"# Probe settlement report — {args.jsonl}")
    print(f"meta: {json.dumps(meta)}  records: {len(records)}")

    layers = sorted({r["layer"] for r in records if r.get("phase") == "lens"})
    if not layers:
        sys.exit("no lens records in file")
    final = layers[-1]
    groups = slice_requests(records)

    turns = None
    if args.turns:
        with open(args.turns) as f:
            turns = json.load(f)

    print(f"\n## Requests (lens layers {layers[0]}..{final})")
    print("| req | prompt_len | prefill_recs | decode_toks | lens_positions | usage (prompt/completion) |")
    print("|---|---|---|---|---|---|")
    for gi, g in enumerate(groups):
        usage = ""
        if turns and gi < len(turns):
            u = turns[gi].get("usage", {})
            usage = f"{u.get('prompt_tokens')}/{u.get('completion_tokens')}"
        print(
            f"| {gi} | {g['prompt_len']} | {len(g['prefill'])} | {len(g['decode'])} "
            f"| {len(g['lens'])} | {usage} |"
        )

    tokens = build_tokens(groups, layers)
    genuine_disagree = [t for t in tokens if t["settle"] is None]
    ties = sum(1 for t in tokens if t["tie_adjusted"])
    print(
        f"\ndecode tokens with full lens: {len(tokens)}  "
        f"(final-layer bf16 argmax ties treated as agree: {ties})"
    )
    if genuine_disagree:
        print(
            f"\n!!! ANOMALY: {len(genuine_disagree)} tokens where the FINAL layer "
            f"genuinely disagrees with the emitted token under greedy decode:"
        )
        for t in genuine_disagree[:20]:
            lrec = t["lens"][final]
            print(
                f"  group={t['group']} pos={t['pos']} token={t['token']} "
                f"lens_top1={lrec['top1']} top1_lp={lrec['top1_logprob']:.4f} nll={lrec['nll']:.4f}"
            )
    else:
        print("genuine final-layer disagreements: 0 (as expected under greedy)")

    layer_ppl_table(tokens, layers, "ALL turns")
    settled = settle_histogram(tokens, layers, "ALL turns")
    for gi in sorted({t["group"] for t in tokens}):
        gtoks = [t for t in tokens if t["group"] == gi]
        layer_ppl_table(gtoks, layers, f"request {gi}")
        settle_histogram(gtoks, layers, f"request {gi}")

    # Headline: early-decided fraction + hard-tail attribution of the PPL cliff.
    slack = [t for t in settled if t["settle"] <= args.slack_settle]
    print(
        f"\n## Headline\nsettle <= {args.slack_settle} (>= {final - args.slack_settle} layers of slack): "
        f"{len(slack)}/{len(settled)} = {100 * len(slack) / len(settled):.1f}%"
    )
    pre_final = layers[-2]
    early_at_pre = [t for t in settled if t["settle"] <= pre_final]
    late = [t for t in settled if t["settle"] == final]
    for name, grp in [
        (f"settled by layer {pre_final} (n={len(early_at_pre)})", early_at_pre),
        (f"settle == {final} (n={len(late)})", late),
    ]:
        if not grp:
            continue
        nlls = [t["lens"][pre_final]["nll"] for t in grp]
        mean = statistics.fmean(nlls)
        print(
            f"layer-{pre_final} NLL for {name}: mean={mean:.3f} (PPL {math.exp(mean):.2f}) "
            f"median={statistics.median(nlls):.4f}"
        )
    all_pre = [t["lens"][pre_final]["nll"] for t in settled]
    late_share = (
        sum(t["lens"][pre_final]["nll"] for t in late) / sum(all_pre) if sum(all_pre) > 0 else 0
    )
    print(
        f"share of total layer-{pre_final} NLL carried by settle=={final} tokens: "
        f"{100 * late_share:.1f}% (their population share: {100 * len(late) / len(settled):.1f}%)"
    )

    # Entropy / NLL correlation with settle depth.
    early = [t for t in settled if t["settle"] <= args.early_settle]
    print(f"\n## Final-layer entropy/NLL vs settle depth")
    print(f"early (settle <= {args.early_settle}): entropy {fmt_stats([t['entropy'] for t in early])}")
    print(f"late  (settle == {final}):        entropy {fmt_stats([t['entropy'] for t in late])}")
    print(f"early (settle <= {args.early_settle}): nll     {fmt_stats([t['nll'] for t in early if t['nll'] is not None])}")
    print(f"late  (settle == {final}):        nll     {fmt_stats([t['nll'] for t in late if t['nll'] is not None])}")

    decode_fn, how = make_decoder(args.tokenizer_dir)
    print(f"\n## Examples (token decode: {how})")

    def show(t):
        text = ""
        if decode_fn and t["token"] is not None:
            text = json.dumps(decode_fn(t["token"]))
        print(
            f"  req={t['group']} pos={t['pos']} settle={t['settle']} token={t['token']} "
            f"H={t['entropy']:.4f} nll={t['nll'] if t['nll'] is None else round(t['nll'], 4)} {text}"
        )

    print("15 earliest-settling tokens:")
    for t in sorted(settled, key=lambda t: (t["settle"], t["entropy"]))[:15]:
        show(t)
    print(f"15 settle=={final} tokens (highest final entropy first):")
    for t in sorted(late, key=lambda t: -t["entropy"])[:15]:
        show(t)


if __name__ == "__main__":
    main()
