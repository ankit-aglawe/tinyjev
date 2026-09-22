# E3 — encoder (ModernBERT-base, 149M) on Kev's data and suites (2026-09-22)

`tools/encoder_train.py`: Laya-style `[MASK]`-marker shared scorer on ModernBERT-base, trained on decision-v7's
training partition with Kev's augmentation, batching, loss reduction and OneCycle schedule (3,144 steps, the same
as the decoder trials), 2048-token window with no truncation (census: longest sequence 899 tokens), one temperature
fitted on the calibration partition, scored by `kev.benchmark.evaluate_records`. H100, 4.4 min per run.

| run | lr | decision-v7 dev (raw) | transfer-v4 dev (raw) | ECE raw / calibrated | cov@5% |
|---|---|---|---|---|---|
| mb-base-lr2e-5-s2 | 2e-5 | 0.714 | **0.532** | 0.090 / 0.087 | 0.003 |
| mb-base-lr5e-5-s2 | 5e-5 | 0.745 | **0.531** | 0.121 / 0.107 | 0.000 |
| tinyjev-0.6b (decoder, same data) | 5e-5 | 0.793 | 0.625 | 0.136 / 0.074 | 0.076 |

Per source (transfer-v4, lr 5e-5): mmlu 0.26, sciq 0.78, qnli 0.58, paws 0.45, emotion 0.48, tweet_offensive 0.58,
rules 0.28–0.90. Knowledge is at chance, and the classification block does not beat the decoder either.

Gate (RECIPE.md): publish as tinyjev-0.15b only if ≥ 0.55 on transfer-v4 dev. **Not met; the encoder does not ship.**
The number stands as the first sub-0.5B result on Kev's frozen suites. The research frame expected 3–8 pp below
the 0.6B; the measured gap is 9 pp, so Ettin's matched-size MNLI advantage for encoders does not carry to held-out
typed decisions with instructions at this data scale.
