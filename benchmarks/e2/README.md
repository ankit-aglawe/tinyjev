# E2 — distillation from Kev-4B via Kev's anchor loss (2026-09-22)

Student: Qwen3-0.6B-Base + pointer head, LoRA r16, lr 5e-5, decision-v7 train (the E1(b) regime).
Teacher targets: `targets-v7-kev4b.json` = `jaredpalmer/kev-4b` pointer-head distributions over every
training question (15,576 questions, 100% coverage, 0 key mismatches). Loss = CE + 1.0·KL(teacher‖student),
aligned by option key; `anchor_temp` softens teacher and student logits (Hinton KD). Scored with Kev's
harness; `transfer` = transfer-v4 dev (held-out sources), 656 clean questions.

| trial | seed | KD | T | v7 dev | transfer-v4 | ECE | cov@5% | matched CE | Δ |
|---|---|---|---|---|---|---|---|---|---|
| E1(b) | 2 | no | – | 0.793 | 0.6250 | 0.136 | 0.076 | — | — |
| trial-0 | 3 | no | – | 0.796 | 0.6235 | 0.128 | 0.046 | — | — |
| trial-1 | 2 | yes | 1 | 0.804 | 0.6098 | 0.157 | 0.091 | 0.6250 | −1.5 pp |
| trial-2 | 3 | yes | 1 | 0.787 | 0.6113 | 0.147 | 0.052 | 0.6235 | −1.2 pp |
| trial-4 | 3 | yes | 3 | 0.800 | 0.6128 | 0.164 | 0.085 | 0.6235 | −1.1 pp |
| trial-3 | 2 | yes | 3 | pending | | | | 0.6250 | |

Finding: distillation from the 4B teacher hurts held-out accuracy at 0.6B on every matched pair, at
either temperature. The targets are near-one-hot on the teacher's own training set (p(gold) 0.95–1.00
on classification and rule sources; only the ordinal sources sst5/yelp carry real soft mass), so the KL
mostly pulls the student toward the teacher's memorized answers; the rule-composition blocks drop from
~0.52 to 0.44–0.50. CE seeds 2 and 3 agree to 0.15 pp (0.6250 / 0.6235): the CE recipe is stable.

Precommitted gate (docs/research/RECIPE.md): mean paired KD − CE ≤ 0 → stop tuning the 0.6B.
Outcome: **stopped.** tinyjev-0.6b = the E1(b) checkpoint, at anchor parity.
