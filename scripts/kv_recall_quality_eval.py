#!/usr/bin/env python3
"""KV-as-memory recall-quality probe / mini-benchmark for Qwen3.6 (qwen3_5_moe).

Verifies the doc thesis (https://redacted.larkoffice.com/docx/QHWydhfofob92Yx8NSpcP50Znnc):
does InfLLM-style semantic block-recall, applied ONLY to the 10/40 full-attention
layers (the rest are GatedDeltaNet linear attn, no KV cache), restore long-context
retrieval that StreamingLLM loses — on the 4-bit MoE canonical Metal model?

Three conditions on the full-attn layers' decode attention (prefill always full):
  full    attend all cached KV                          kv_attended = S
  stream  StreamingLLM: sink + local only               kv_attended = n_init + n_local
  recall  mean-key top-k block recall                   kv_attended = n_init + top_k*l_bs + n_local

`kv_attended` is the per-full-attn-layer token count attended at decode (the doc's
"working memory"). NOTE: full KV stays RESIDENT here — recall only restricts
attention SCOPE to measure quality; it does not evict (memory saving = Phase-1).

Distractor haystack:
  --distractor uniform  one repeated sentence (easy: needle key trivially distinct)
  --distractor diverse  varied sentence pool incl. number-bearing lines (BINDING test)

Tasks:
  single needle  --needles 1  accuracy across --depths
  multi needle   --needles N  N keys at even depths, one list-all Q, fraction retrieved

Thinking is disabled via the chat template (enable_thinking=False) so the answer is
direct — otherwise a 12-token budget is eaten by <think> (case-as-fact: a 0/N from
think-truncation is a harness artifact, not a recall result).

  python3 scripts/kv_recall_quality_eval.py --ctx 16000 --distractor diverse --needles 4

# ponytail: recall is decode-only + absolute KV positions (no InfLLM re-positioning).
"""
import argparse, re, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models import qwen3_next as qn

CFG = {"mode": "full", "n_init": 32, "n_local": 256, "l_bs": 32, "top_k": 8}
KV_ATTENDED = {}  # mode -> tokens attended per full-attn layer at the last decode step
KEYS = ["739154", "281607", "930472", "615838", "472916", "108253", "664019", "357284"]
UNIFORM = "The grass is green and the sky is blue. The sun is bright today. "
DISTRACTORS = [
    "The capital of France is Paris.", "Photosynthesis converts sunlight into energy.",
    "The river flowed past the old stone bridge.", "In 1969 humans first landed on the moon.",
    "The recipe calls for two cups of flour.", "Mountains form over millions of years.",
    "The library closes at nine on weekdays.", "A group of crows is called a murder.",
    "The serial number on the device is 48213.", "Water boils at one hundred degrees.",
    "The train departs from platform seven.", "Honey never spoils if stored properly.",
    "The meeting was moved to next Tuesday.", "The manuscript was written in faded ink.",
    "Order confirmation number 90571 was processed.", "The garden was full of tulips.",
    "Sound travels faster in water than air.", "The committee approved the new budget.",
    "Room 312 is at the end of the hall.", "The novel spans three generations.",
    "Invoice 55820 is due at the end of the month.", "The cat slept on the warm windowsill.",
    "The bridge was painted a deep shade of red.", "Account balance is 71640 as of today.",
]


def _patched_attn_call(self, x, mask=None, cache=None):
    """Drop-in for Qwen3NextAttention.__call__ (mlx_lm 0.31.2, qwen3_next.py:121).
    Identical to upstream except decode-step attention is filtered per CFG."""
    B, L, D = x.shape
    q_proj_output = self.q_proj(x)
    queries, gate = mx.split(
        q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
    )
    gate = gate.reshape(B, L, -1)
    keys, values = self.k_proj(x), self.v_proj(x)
    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

    if cache is not None:
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = self.rope(queries)
        keys = self.rope(keys)

    mode = CFG["mode"]
    S = keys.shape[2]
    n_init, n_local, l_bs, top_k = CFG["n_init"], CFG["n_local"], CFG["l_bs"], CFG["top_k"]
    if mode != "full" and L == 1 and S > n_init + n_local + l_bs:
        mid_lo, mid_hi = n_init, S - n_local
        sel = list(range(0, n_init))
        if mode == "recall":
            nb = (mid_hi - mid_lo) // l_bs
            if nb > 0:
                kh = keys[:, :, mid_lo:mid_lo + nb * l_bs, :]
                nkv, hd = kh.shape[1], kh.shape[3]
                reps = kh.reshape(B, nkv, nb, l_bs, hd).mean(axis=3)
                g = self.num_attention_heads // self.num_key_value_heads
                qk = queries.reshape(B, nkv, g, hd).mean(axis=2)
                score = (reps * qk[:, :, None, :]).sum(axis=(1, 3))
                k = min(top_k, nb)
                idx = mx.argpartition(-score[0], k - 1)[:k]
                idx = sorted(int(i) for i in idx.tolist())
                for bi in idx:
                    sel.extend(range(mid_lo + bi * l_bs, mid_lo + (bi + 1) * l_bs))
        sel.extend(range(S - n_local, S))
        KV_ATTENDED[mode] = len(sel)
        sel_idx = mx.array(sel, dtype=mx.int32)
        keys = mx.take(keys, sel_idx, axis=2)
        values = mx.take(values, sel_idx, axis=2)
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=None
        )
    else:
        if L == 1:
            KV_ATTENDED[mode] = int(keys.shape[2])
        from mlx_lm.models.base import scaled_dot_product_attention
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output * mx.sigmoid(gate))


qn.Qwen3NextAttention.__call__ = _patched_attn_call


def build_prompt(tok, ctx_tokens, placements, multi, diverse):
    """placements: list of (depth, key). Bury each key at its depth among distractors.
    Returns a chat-formatted, thinking-disabled prompt string."""
    if diverse:
        unit = max(1, ctx_tokens // (sum(len(tok.encode(s + " ")) for s in DISTRACTORS) // len(DISTRACTORS)))
        sents = [DISTRACTORS[i % len(DISTRACTORS)] + " " for i in range(unit)]
    else:
        unit = max(1, ctx_tokens // len(tok.encode(UNIFORM)))
        sents = [UNIFORM] * unit
    for pos, k in sorted((min(unit - 1, int(unit * d)), k) for d, k in placements):
        sents[pos] = sents[pos] + f"\n\nThe pass key is {k}. Remember it: {k}.\n\n"
    body = "".join(sents)
    if multi:
        body += "\n\nList ALL the pass keys (the 6-digit numbers) mentioned above, separated by commas."
    else:
        body += "\n\nWhat is the pass key? Answer with the 6 digits only."
    return tok.apply_chat_template(
        [{"role": "user", "content": body}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )


def run(model, tok, prompt, max_tokens):
    return generate(model, tok, prompt=prompt, max_tokens=max_tokens,
                    sampler=make_sampler(temp=0.0), verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.6-35B-A3B-4bit")
    ap.add_argument("--ctx", type=int, default=6000)
    ap.add_argument("--depths", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--modes", default="full,stream,recall")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--lbs", type=int, default=32)
    ap.add_argument("--ninit", type=int, default=32)
    ap.add_argument("--nlocal", type=int, default=256)
    ap.add_argument("--needles", type=int, default=1)
    ap.add_argument("--distractor", choices=["uniform", "diverse"], default="uniform")
    args = ap.parse_args()
    CFG.update(n_init=args.ninit, n_local=args.nlocal, l_bs=args.lbs, top_k=args.topk)
    diverse = args.distractor == "diverse"

    print(f"loading {args.model} ... (distractor={args.distractor})", flush=True)
    t0 = time.time()
    model, tok = load(args.model)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    modes = args.modes.split(",")

    if args.needles > 1:
        n = min(args.needles, len(KEYS))
        keys = KEYS[:n]
        depths = [(i + 1) / (n + 1) for i in range(n)]
        prompt = build_prompt(tok, args.ctx, list(zip(depths, keys)), multi=True, diverse=diverse)
        ntok = len(tok.encode(prompt))
        print(f"multi-needle: {n} keys at depths {[round(d,2) for d in depths]} | ctx~{ntok} | {args.distractor}", flush=True)
        for mode in modes:
            CFG["mode"] = mode
            out = run(model, tok, prompt, max_tokens=16 + 12 * n)
            dig = "".join(re.findall(r"\d", out))
            hits = sum(1 for k in keys if k in dig)
            print(f"  {mode:<7} {hits}/{n} keys  kv_attended={KV_ATTENDED.get(mode,'?')}/layer  "
                  f"out={out.strip()[:100]!r}", flush=True)
        return

    depths = [float(d) for d in args.depths.split(",")]
    results = {m: {} for m in modes}
    for depth in depths:
        prompt = build_prompt(tok, args.ctx, [(depth, KEYS[0])], multi=False, diverse=diverse)
        ntok = len(tok.encode(prompt))
        for mode in modes:
            CFG["mode"] = mode
            out = run(model, tok, prompt, max_tokens=12)
            ok = KEYS[0] in "".join(re.findall(r"\d", out))
            results[mode][depth] = ok
            print(f"ctx~{ntok:>6} depth={depth:<4} {mode:<7} {'OK ' if ok else 'MISS'} "
                  f"kv={KV_ATTENDED.get(mode,'?'):>6}  out={out.strip()[:50]!r}", flush=True)

    print(f"\n=== passkey grid (ctx~{ntok}, {args.distractor}) | kv/layer: " +
          ", ".join(f"{m}={KV_ATTENDED.get(m,'?')}" for m in modes) + " ===")
    print("depth ".ljust(8) + "".join(f"{m:>9}" for m in modes))
    for depth in depths:
        print(f"{depth:<8}" + "".join(f"{'OK' if results[m][depth] else '.':>9}" for m in modes))
    for m in modes:
        print(f"  {m:<7} acc={sum(results[m].values())/len(depths):.2f}")


if __name__ == "__main__":
    main()
