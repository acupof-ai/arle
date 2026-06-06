# DSv4 official DSA indexer default-on

## Context

The official DeepGEMM/DSA indexer path replaces ARLE's legacy scalar
`dsv4_csa_select` selector. The first single-prompt 4096 needle was not a valid
gate: legacy failed it too, so it measured the raw-token harness/prompt rather
than the official selector.

## What Worked

Correctness was gated by relative inference against the legacy selector's own
same-config non-determinism floor. The gate ran six prompt lengths sequentially
in one engine session, with legacy repeated three times and official DSA once:

| prompt tokens | legacy floor first diff | official first diff vs legacy1 | within floor | legacy ms/token | official ms/token |
|---:|---:|---:|:---:|---:|---:|
| 64 | none | none | yes | 30.72 | 28.05 |
| 256 | 0 | 0 | yes | 27.60 | 25.98 |
| 512 | 0 | 1 | yes | 25.68 | 25.94 |
| 1024 | 0 | 0 | yes | 25.74 | 26.42 |
| 2048 | 0 | 3 | yes | 48.03 | 26.05 |
| 4096 | 1 | 1 | yes | 124.58 | 26.09 |

Run artifact: `/tmp/dsv4-dsa-floor-gate3/summary.json` on the H20 pod.

A short-form needle sanity check also passed token-exact: legacy and official
produced identical `clean_tokens` on `/tmp/short_needle_forms.list`.

The selector is now default-on. The legacy scalar selector remains as an
explicit fallback via `ARLE_DSV4_DSA_INDEXER=0`.

## Rule

For DSv4 long-context correctness, use correct-inference relative to the
reference path's same-config non-determinism floor. Do not block a path on a
needle prompt that the reference path itself cannot retrieve. The 4096 raw-token
needle failure remains a separate follow-up: verify whether the harness is
missing the DSv4 chat template or whether ARLE has a real long-context QA issue.
