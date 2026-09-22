# Locked-test reads (2026-09-22) — one read per candidate, never used for selection

Kev's frozen suites keep a test partition that is read once per promoted candidate. Neither of our trials
passes Kev's promotion gates (his own 0.6B reads are all `-ungated` too), so these are recorded as
exploratory reads under the same convention. Scored by Kev's harness on H100.

| checkpoint | config | decision-v7 test (n=1200) | transfer-v4 test (n=656) | ECE | cov@5% |
|---|---|---|---|---|---|
| **tinyjev-0.6b** (`tinyjev-e1-lora/01-trial-1`) | Qwen3-0.6B-Base, LoRA r16, lr 5e-5, seed 2 | 0.8075 | **0.6631** | 0.082 | 0.069 |
| baseline (`tinyjev-e1-lora/00-trial-0`) | Kev's released config (lr 1e-4), seed 2, reproduced | 0.8133 | 0.6570 | 0.108 | 0.078 |
| Kev-0.6B, published (Qwen3 generation) | same as baseline | — | 0.642 | — | — |

Reading: tinyjev-0.6b is +0.6 pp over its matched baseline and +2.1 pp over the published number, with
lower ECE. One point is ~7 questions of 656. Development-set numbers for the same checkpoints are in
benchmarks/e1 (0.625 / 0.614). This is anchor parity with better calibration, not a jump.
