# Accuracy plan, v2

Written 2026-09-25 after the first same-inputs benchmark (`benchmarks/opendecision/`).
Goal set by the owner: raise accuracy by studying what the other open Jev-like models did
and experimenting, not by copying Kev; ship one smaller and one bigger model beside 0.6B.

## What the benchmark says

On OpenDecision's 500 never-seen cases, same inputs for every row:

| model | correct | note |
|---|---:|---|
| Claude Opus 5.5 | 496 | the ceiling |
| kev-0.8b (Qwen3.5-0.8B base) | 463 | Kev's current small model, via its own server |
| kev-0.6b (raw logits) | 441 | the checkpoint tinyjev reproduces |
| lostargon/Tiny-Jev (the name collision) | 445 | its own head, inside tinyjev's CI |
| **tinyjev-0.6b** | **440** | |
| OpenDecision's own engine | 428 | inside tinyjev's CI |
| agent-jev-0.6b | 415 | same backbone, permutation-equivariant set head |
| von 1.2 (395M encoder) | 414 | |
| laya, English (421M encoder) | 367 | |
| raw Qwen3-0.6B, letter logits | 354 | same weights, no head |

Two facts drive the plan:

1. **The head is worth 86 cases on identical weights** (354 → 440). Head design is a real
   lever. The three heads on the same 0.6B backbone (tinyjev/Kev pointer 440, agent-jev
   set head 415, NanoJev marker head 114 on text) do not converge; the pointer head is
   ahead, and nothing says it is the ceiling.
2. **Calibration was the wrong number, not the model.** The shipped temperature (1.464,
   fitted in-distribution) flattens confidence out of distribution: undone, ECE goes
   0.071 → 0.019 and gate-0.85 coverage 59% → 76%, matching Kev-0.6B exactly.

Measured today as well: Kev-0.8B on the Qwen3.5-0.8B base scores 463, +23 over the Qwen3-0.6B
pair, at 173 ms on Apple Silicon through Kev's torch server; the base-model generation is worth
more than the head tweaks, and the Mac penalty for DeltaNet at 0.8B is 2×, not the 4× the 4B pays.
So E11's base choice is open: Qwen3-4B-Base for Mac speed, or Qwen3.5-4B-Base for Kev's +2.7 on
transfer-v4 at a 4× latency cost. Default Qwen3-4B; run both if the ledger allows.

And one fact from Kev's own card: capacity is the bottleneck at this size. Kev transfer-v4
dev: 0.6B 0.625, 0.8B 0.648, 4B 0.817, 9B 0.822, Jev 0.857. Nothing at 0.6B closes a
19-point gap; a 4B does most of it.

## What nobody in the 65-model field has tested at this size

Every 0.6B decision model in the catalog trains on Kev's 12.5k decision-v7 examples or
something smaller. Laya trains an encoder with RL against proper scoring rules. von adds
option-order invariance and a Brier term. CLM scales a bi-encoder with 60M pairs on an
8B (not transferable, see the research memory). No one has run **data scale** on a
small cross-encoder: the recipe was tuned at 12.5k examples and every "capacity is the
bottleneck" claim was measured there.

## The ladder

Pre-registered like E1–E6. Every gate is scored on the existing harness: OpenDecision-500
(`benchmarks/opendecision`, baselines logged) and Kev's transfer-v4 dev; the locked test
is read once per shipped checkpoint, never during selection. A result gets posted only
next to its same-inputs baseline row.

### E7 — Calibration partition. $0, today.

Fit the served temperature on an out-of-distribution partition that is not the reporting
suite: OpenDecision's `typesafe_public` 80 cases. Report ECE, gate coverage and
coverage-at-2%-error on OD-500 and transfer-v4 dev at the old and new T.
**Gate:** ECE on OD-500 ≤ 0.03 with OD-500 untouched during fitting; transfer-v4 ECE not
worse than 0.10. Ships as 0.1.4 (a one-number change in `tinyjev.json`).

**Run 2026-09-25 (`benchmarks/opendecision/fit_temperature.py`): gate missed.** Fitting on
the 51 typesafe_public choice cases gives T = 0.80 (NLL 0.533 vs 0.632 served). On the
untouched OD-500 that is ECE 0.039 (from 0.071), gate-0.85 coverage 81.8% at 95.8% (from
59.2% at 98.0%), coverage at ≤2% error 69.0% (from 63.4%). Better on every axis than the
served value, above the 0.03 gate. T = 1.0 would pass (ECE 0.019) but picking it because
OD-500 likes it is fitting to the test set, so it is not picked. Next: a larger OOD
calibration partition (a few hundred cases from domains outside both suites), then refit.
Nothing ships from this run.

### E11 — The bigger model. ~$5, one evening.

`tinyjev-4b`: Qwen3-4B-Base (attention-only, so MLX stays fast on a Mac; Kev's Qwen3.5
checkpoints hit the DeltaNet trap), E1(b) recipe (LoRA r16, lr 5e-5, decision-v7 train),
one H100 via Modal, ≈ 1–1.5 GPU-hours. Convert to tinyjev-v2, MLX fp16 and INT8.
**Gate:** OD-500 ≥ 0.94 and transfer-v4 dev ≥ 0.79 (Kev-4B on the Qwen3 base: 0.790);
INT8 within 0.5 pt of fp16; ≤ 400 ms per short question on a base M1.
This is the fastest visible accuracy win and the second post.

### E8 — Data scale at 0.6B. ~$10–30 API + ~$8–16 GPU. The real bet.

Build a 100k+ typed-decision corpus and retrain the 0.6B on 10× and 30× the data with
the same recipe. Two sources, kept apart and labelled:
- **Held-out portions of the ten decision-v7 source datasets** (banking77, ag_news, …)
  that decision-v7 did not use, cast into choice/noul/score with the same templates.
- **Synthetic domains** written by a frontier model to a domain list that excludes every
  OD-500 domain family and every transfer-v4 source, with the generator prompt and
  seed logged, so the contamination statement stays true.
**Gate:** transfer-v4 dev ≥ 0.655 (+3 over 0.625) and OD-500 ≥ 0.90, at 10× or 30×;
if 30× ≤ 10×, the curve is flat and the bet is closed with the number published.
**Innovation claim if it passes:** "the 0.6B ceiling was a data ceiling", with the
scaling curve as the artifact.

### E9 — Option-order invariance. ~$4, one H100-hour.

von reports 0.0% answer flips under option permutation after v1.2 (49.5% before).
Measure tinyjev's flip rate on OD-500 (permute options, re-run, count argmax changes),
then train one E1(b) arm with permutation augmentation (each example seen under k
random option orders).
**Gate:** flip rate ≤ 2% with OD-500 accuracy within 0.5 pt of the un-augmented arm.

### E12 — The smaller model. After E8.

Qwen2.5-0.5B or SmolLM2-360M, trained on E8's best data mix (not on 12.5k; the E3 149M
encoder at 0.532 was the 12.5k recipe, not a size verdict).
**Gate:** transfer-v4 dev ≥ 0.55; OD-500 ≥ 0.80; ≤ 40 ms per short question on M1.

### E10 — Trained logit scale. ~$6–8. Lowest priority.

CLM trains a global scale with the CE gradient; tinyjev uses 1/√d plus a post-hoc T. One
matched seed pair. Expect a null (same degree of freedom as T). Only worth running if E7
leaves ECE above the gate.

## Order and money

E7 → E11 → E8 → E9 → E12 → E10. Modal ledger, September: $19.07 spent of the $30 free
tier, self-ceiling $25. E7 is free; E11 lands at ≈ $24. E8 and after are October. No run
starts without the ledger total stated first.

## What does not go on the list

Games and structured numeric state (measured: not state-sensitive). Fixed-taxonomy
sentiment (measured: FinSense 0.953 vs 0.717). Relational comparison as a target (9/20 on
citation_relation and entity_matching is a head-design problem for a later ladder, not a
data problem). Any refit on OD-500 itself.
