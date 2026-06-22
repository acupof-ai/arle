#!/usr/bin/env python3
"""KV-as-memory recall-quality probe for Qwen3.6 (qwen3_5_moe, hybrid linear+full attn).

Verifies the doc thesis (https://bytedance.larkoffice.com/docx/QHWydhfofob92Yx8NSpcP50Znnc):
does InfLLM-style semantic block-recall, applied ONLY to the 10/40 full-attention
layers (the rest are GatedDeltaNet linear attn with no KV cache), restore passkey
retrieval at long context on the 4-bit MoE canonical Metal model?

Three conditions on the full-attn layers' decode attention (prefill always full):
  full    baseline upper bound (attend all cached KV)
  stream  StreamingLLM lower bound (attend only sink + local window; drop middle)
  recall  mean-key top-k block recall (attend sink + top_k middle blocks + local)

Metric: passkey exact-match accuracy across needle depths {0,.25,.5,.75,1.0}.
Pass = recall ~ full AND recall >> stream. If stream already retrieves, the
linear layers carry it and full-attn recall is not the deciding factor (also a
finding). See docs/experience/wins on landing.

  python3 scripts/kv_recall_quality_eval.py --ctx 6000 --topk 8 --depths 0,0.5,1.0

# ponytail: recall is decode-only + keeps original absolute KV positions (no InfLLM
# re-positioning). Simplest faithful-to-cache variant; re-position is a §5 follow-up.
"""
import argparse, re, sys, time, types
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models import qwen3_next as qn

# ---- global recall config consulted by the patched attention -----------------
CFG = {"mode": "full", "n_init": 32, "n_local": 256, "l_bs": 32, "top_k": 8}


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
    # Only filter at decode (L==1) when there's a real middle to drop/recall.
    if mode != "full" and L == 1 and S > n_init + n_local + l_bs:
        mid_lo, mid_hi = n_init, S - n_local
        sel = list(range(0, n_init))
        if mode == "recall":
            nb = (mid_hi - mid_lo) // l_bs            # whole blocks only
            if nb > 0:
                kh = keys[:, :, mid_lo:mid_lo + nb * l_bs, :]       # (B,nkv,nb*l_bs,hd)
                nkv, hd = kh.shape[1], kh.shape[3]
                reps = kh.reshape(B, nkv, nb, l_bs, hd).mean(axis=3)  # (B,nkv,nb,hd)
                # GQA: average query heads within each kv group -> (B,nkv,hd)
                g = self.num_attention_heads // self.num_key_value_heads
                qk = queries.reshape(B, nkv, g, hd).mean(axis=2)
                score = (reps * qk[:, :, None, :]).sum(axis=(1, 3))   # (B,nb) sum over kv+hd
                k = min(top_k, nb)
                idx = mx.argpartition(-score[0], k - 1)[:k]
                idx = sorted(int(i) for i in idx.tolist())           # keep temporal order
                for bi in idx:
                    sel.extend(range(mid_lo + bi * l_bs, mid_lo + (bi + 1) * l_bs))
        sel.extend(range(S - n_local, S))
        sel_idx = mx.array(sel, dtype=mx.int32)
        keys = mx.take(keys, sel_idx, axis=2)
        values = mx.take(values, sel_idx, axis=2)
        mask = None  # single query attends to a chosen past set, no causal masking
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
    else:
        from mlx_lm.models.base import scaled_dot_product_attention
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output * mx.sigmoid(gate))


qn.Qwen3NextAttention.__call__ = _patched_attn_call


def build_prompt(tok, ctx_tokens, depth, key):
    """Filler of ~ctx_tokens with the passkey buried at `depth`, then the question."""
    filler = "The grass is green and the sky is blue. The sun is bright today. "
    need = max(1, ctx_tokens // len(tok.encode(filler)))
    secret = f"\n\nThe pass key is {key}. Remember it: {key}.\n\n"
    q = "\n\nQuestion: What is the pass key? Answer with the digits only.\nAnswer: The pass key is"
    n_before = int(need * depth)
    body = filler * n_before + secret + filler * (need - n_before)
    return body + q


def run(model, tok, prompt, max_tokens=12):
    sampler = make_sampler(temp=0.0)
    return generate(model, tok, prompt=prompt, max_tokens=max_tokens,
                    sampler=sampler, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.6-35B-A3B-4bit")
    ap.add_argument("--ctx", type=int, default=6000, help="approx context tokens")
    ap.add_argument("--depths", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--modes", default="full,stream,recall")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--lbs", type=int, default=32)
    ap.add_argument("--ninit", type=int, default=32)
    ap.add_argument("--nlocal", type=int, default=256)
    args = ap.parse_args()
    CFG.update(n_init=args.ninit, n_local=args.nlocal, l_bs=args.lbs, top_k=args.topk)

    print(f"loading {args.model} ...", flush=True)
    t0 = time.time()
    model, tok = load(args.model)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    depths = [float(d) for d in args.depths.split(",")]
    modes = args.modes.split(",")
    key = "739154"  # fixed 6-digit passkey
    results = {m: {} for m in modes}

    for depth in depths:
        prompt = build_prompt(tok, args.ctx, depth, key)
        ntok = len(tok.encode(prompt))
        for mode in modes:
            CFG["mode"] = mode
            out = run(model, tok, prompt)
            digits = re.findall(r"\d{4,}", out)
            ok = key in "".join(re.findall(r"\d", out))
            results[mode][depth] = ok
            print(f"ctx~{ntok:>6} depth={depth:<4} {mode:<7} "
                  f"{'OK ' if ok else 'MISS'}  out={out.strip()[:60]!r}", flush=True)

    print("\n=== passkey accuracy grid (ctx~%d tok, S>%d triggers recall) ===" %
          (ntok, CFG['n_init'] + CFG['n_local'] + CFG['l_bs']))
    hdr = "depth ".ljust(8) + "".join(f"{m:>9}" for m in modes)
    print(hdr)
    for depth in depths:
        row = f"{depth:<8}" + "".join(f"{'OK' if results[m][depth] else '.':>9}" for m in modes)
        print(row)
    for m in modes:
        acc = sum(results[m].values()) / len(depths)
        print(f"  {m:<7} acc={acc:.2f}")


if __name__ == "__main__":
    main()
