# Research Question Brief — tinyjev

Date: 2026-09-22. Mode: full. Prepared by the orchestrating session (research_question_agent + research_architect_agent roles collapsed; the user delegated scoping decisions).

## Main research question

**Which combination of backbone, decision head, training objective, and data lets a model of at most 0.6B parameters (target 0.1–0.3B) answer typed decisions — `choice` over K options, `noul` yes/no probability, `score` ordinal — with accuracy and calibration approaching the best open Jev-style models (Kev-4B: 0.797 dev / 0.837 test on held-out sources; Jev hosted: 0.857 dev), while running on Apple Silicon via MLX, on CPU/CUDA via PyTorch, and plausibly on a phone?**

## Sub-questions

1. **Backbone.** Encoder (ModernBERT-base 149M, mmBERT, EuroBERT, DeBERTa-v3) versus small decoder (Qwen3-0.6B, Qwen2.5-0.5B, SmolLM2-135M/360M, Gemma-3-270M): which gives the best accuracy per parameter and per millisecond on typed decisions, and what does the multiple-choice-QA / cross-encoder literature say about small encoders on option-scoring tasks?
2. **Head.** Pointer head (Kev), set-attention over candidates (NanoJev), per-candidate scalar with `[MASK]` markers (Laya), option-logit reading on a stock LM: which generalizes to unseen label sets, survives 50+ options, and is permutation-invariant?
3. **Objective and calibration.** Cross-entropy vs proper scoring rules vs distillation from a strong teacher with soft targets: what closes the small-model gap most? How is the product metric — fraction of decisions automatable at a 5% error budget — raised: temperature scaling, conformal prediction, learned abstention?
4. **Data.** What mixture transfers to new sources (Kev decision-v7: 10 public datasets + generated policy examples; Laya's mix; frontier-model synthesis)? How much data does a 0.1–0.6B model need? What contamination discipline keeps the frozen suites honest?
5. **Footprint.** Effect of 4/8-bit quantization on decision accuracy for sub-1B models on MLX; pruning/distillation (MiniLM, TinyBERT); Matryoshka-style multi-size training; what actually runs on an iPhone (MLX Swift, CoreML) at what latency for 100–600M models.
6. **Landscape.** Has anyone published a sub-0.5B open model that beats Kev-0.6B (0.620/0.642 Qwen3 gen; 0.652/0.684 Qwen3.5 gen) on Kev's frozen suites? If so, how?

## FINER

| | |
|---|---|
| Feasible | Yes. Frozen public suites (jaredpalmer/kev-suites), open baselines with published numbers, training recipes public, Modal H100 available. |
| Interesting | The category's own scaling data says accuracy collapses below 4B; a tiny model that does not collapse is the open problem. |
| Novel | No published sub-0.5B model beats Kev-0.6B on Kev's suites as of the last landscape check; distillation and encoder backbones are largely untested in this category. |
| Ethical | Public data, public models, attributed. No human subjects. |
| Relevant | Determines the one recipe tinyjev will build; every downstream decision hangs on it. |

## Scope

**In:** English typed decisions; accuracy, Brier, ECE, coverage-at-5%-error on Kev's frozen `decision-v7` (in-distribution) and `transfer-v4` / `transfer-v9` (held-out sources) suites; latency on a base M1 and on-device feasibility; models ≤ 0.6B.
**Out:** multilingual (Laya's strength, not our target); training the Qwen3.5 hybrid family; anything requiring TypeSafe API access beyond published numbers; browser/agent harness integration.

## Evidence policy

- A measured number on a frozen public suite outranks a README claim. README numbers without a controlled comparison are labelled as such.
- Every claim carries a citation with a resolvable link. A reference that cannot be shown to exist is excluded, not hedged.
- Local research logs (Kev `PLAN.md`, `experiments/*.json`, Laya `research/`, NanoJev `TRAINING_RECIPE.md`) are primary sources for "what was tried and what happened" and are cited by file path.
- Contamination: any dataset in Kev's `transfer-v4` holdout list (mmlu, emotion, tweet_offensive, qnli, …) or `transfer-v9` cannot appear in a proposed training mix.

## Methodology blueprint

Design: evidence synthesis in three streams — (i) local artifacts (code, research logs, experiment JSON) read directly; (ii) literature (peer-reviewed and arXiv) on small-model option scoring, distillation, calibration, selective prediction, quantization, on-device inference; (iii) landscape (awesome-jev, Hugging Face, jevbench) for competing open models and their measured numbers. Analytical frame: for each candidate recipe, expected accuracy on transfer-v4, expected coverage-at-5%-error, parameters, latency on M1, on-device feasibility, and evidence grade for each expectation. Output: one ranked recipe, an E1–E5 experiment ladder scored on Kev's suites with H100-hour estimates, and a risk register.

## Devil's Advocate — Checkpoint 1

**Verdict: PASS with two recorded cautions.**

1. *Bias toward a wanted answer.* The user's framing ("same accuracy, tiny") presumes the gap is closable. The investigation must report the case that it is not — Kev's own scaling (0.6B → 0.65, 4B → 0.84) is the null hypothesis, and the report must state what accuracy a 0.1–0.3B model realistically reaches if distillation only partially closes the gap.
2. *Category is nine days old.* Most "evidence" is self-reported READMEs from repos built in a week. The synthesis must weight peer-reviewed results on analogous tasks (MC-QA, cross-encoding, distillation of small classifiers) above in-category READMEs, and say so where it does.

Scope is answerable: the frozen suites make every recommendation testable, and the ladder converts the question into runs.
