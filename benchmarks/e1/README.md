# E1 — training regime, Qwen3-0.6B-Base + pointer head, decision-v7 train (2026-09-22)

Scored with Kev's own harness on the frozen suites: `clean` = decision-v7 dev (in-distribution),
`transfer` = transfer-v4 dev (held-out sources). Single seed (2) per arm; H100.

| arm | lora | lr | wall | v7 dev | transfer-v4 | ECE | cov@5% | Brier |
|---|---|---|---|---|---|---|---|---|
| (a) released config, reproduced | 16 | 1e-4 | 16.6 m | 0.796 | 0.614 | 0.161 | 0.108 | 0.538 |
| (b) LoRA, half LR | 16 | 5e-5 | 14.9 m | 0.793 | **0.625** | 0.136 | 0.076 | 0.505 |
| (c) full fine-tune | 0 | 5e-5 | 11.0 m | 0.650 | 0.483 | 0.097 | 0.006 | 0.604 |
| (d) full fine-tune | 0 | 2e-5 | 11.6 m | 0.775 | 0.581 | 0.108 | 0.093 | 0.520 |
| published Kev-0.6B (Qwen3 gen) | 16 | 1e-4 | — | — | 0.620 | — | — | — |

Finding: at 0.6B the adaptation regime is not a lever. LoRA lands at 0.61–0.63 at either LR;
full fine-tuning forgets the base (mmlu 0.34, emotion 0.40 at 2e-5) even while it learns rule
composition better (and_or 0.59 vs 0.44–0.47, authorization 0.75 vs 0.50–0.65). The published
anchor reproduces within run noise (0.614 vs 0.620). GPU: 0.90 h.

Files: `<arm>.json` = Kev `result.json`; `<arm>.prov.json` = provenance (exact config, hashes).
