# Requential Coding — takeaways for the OPD/distill lane

> Reference — arXiv:2607.11883 (ICML). Deep-read (Chinese):
> https://bytedance.larkoffice.com/docx/YV2tdclxMoQM0HxIvQhcAxWinMd

Compression-theory paper (student self-generates candidates, teacher approves
one via REC; code length = cumulative teacher-student KL). Mechanism is
isomorphic to on-policy distillation. Not relevant to the RL lane.

Adopt when the P8 privileged-self-distill lane lands
([plan §P8](../plans/2026-07-16-agent-rl-unified-infra.md)):

1. **Teacher EMA smoothing** — measured evidence that raw-teacher SGD noise is
   unlearnable cost the student pays for nothing; EMA it away. Our GKD/SOPD
   path already has `ema_alpha` — this licenses default-on.
2. **Iso-loss projection** — periodically reset teacher := student, retrain
   teacher on real data back to its loss: the min-KL teacher at equal
   performance. Cheap A/B as a teacher-refresh schedule.
3. **Framing** — cumulative teacher-student KL = information transferred;
   "KL area" from metrics.jsonl is a principled distill progress measure.

Caveats worth keeping (doc §9/§11): teacher-student KL smallness is
architecture-coupled (same-arch same-recipe), and "larger models more
compressible" carries a compute-coupled confound.
