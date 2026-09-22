# tinyjev — training recipe decision

Date: 2026-09-22. Derived from docs/research/phase2_investigation (4 verified evidence streams) and phase3_synthesis (synthesis + devil's-advocate checkpoint 2). This file is the decision; the research files are the receipts.

## The call

Build **tinyjev-0.6b** first: Qwen3-0.6B-Base + pointer head, full fine-tune at low LR, soft-target distillation from Kev-4B/9B, on Kev's public decision-v7 data, scored on Kev's frozen suites. Then **tinyjev-0.15b**: ModernBERT-base with the same data and the same teacher targets, as the honest mobile size.

Target: beat Kev-0.6B (0.620 transfer-v4 dev, the public anchor at the same size) by 2–6 pp. Not 4B parity. The corpus does not support 4B accuracy below 0.6B, and says so 17 different ways.

## Why each choice

| choice | pick | reason (evidence) |
|---|---|---|
| backbone, 0.6b | Qwen3-0.6B-Base | Only sub-1B arm the corpus ranks above the anchor. Attention-only (MLX-native, no DeltaNet). Read consistently, every 150–360M encoder arm lands 3–8 pp below Kev-0.6B (DA2). |
| backbone, 0.15b | ModernBERT-base 149M | Ettin (ICLR 2026, matched): 150M encoder beats 150M decoder by 3.6 MNLI; 400M encoder beats 1B decoder. Wins the classification block, loses the knowledge block. INT8 quantizes free. Honest expectation: below the 0.6b. |
| head | Kev pointer head (`<decide>` · each `</opt>`) | Handles 255 options, no letter-binding (small decoders can't bind option letters — chance even at 6.7B). Skip `option_isolation`: costs 3–4 pp (verified). Permutation handled by option-order shuffling at train time. |
| objective | plain CE + KL to teacher soft targets | Kev's 5-arm screen: no loss beats CE on coverage; label smoothing collapses coverage@5% 0.576→0.006 (forbidden). Distillation-trained models have the best selective-prediction metrics of any regime (Galil, 523 models). No source has run decoder→student soft-target KD on held-out sources: this is the untested lever. |
| training regime | full FT, LR sweep 2e-5 / 5e-5; LoRA r16 as control | LR was Kev's largest recipe effect (+4.7 pp at 4B, replicated: less drift from base). The 0.6B anchor was only ever run LoRA at 2e-4. Cheapest possible upside. |
| teacher | Kev-4B (1-epoch parent) and Kev-9B | Teacher *calibration* predicts student accuracy (R² 0.92) better than teacher accuracy (0.68); bigger teachers are not better teachers. The 1-epoch 4B parent is Kev's best-calibrated checkpoint. |
| data | Kev decision-v7 train split, hash-pinned | Same data as the anchor → clean comparison. More public data lowered transfer at 4B (10.9k: 0.704 vs 3.4k: 0.747); don't add. KD needs no labels, so extra *unlabeled* states are the one cheap expansion if E2 wants more. |
| calibration | one temperature fitted on decision-v7 dev; softmax-response for abstention | Learned abstention heads don't beat max-softmax (Feng, Jaeger, Varshney). TS transfers OOD on the Qwen3.5 generation (ECE 0.106→0.042); verify on ours. |
| eval | `kev.benchmark --remote` against our `/v1/systemone` server | Comparable to every published number. decision-v7 dev = in-distribution; transfer-v4 dev = held-out; locked test read once, at the end. |
| footprint | fp16 + INT8 both sizes; 4-bit only if E5 shows < 1 pp | 4-bit costs sub-1B decoders 3–10 MMLU points (per-group GPTQ 47.1→44.0, RTN →37.3). INT8 is free on both encoders and decoders. |

## Contamination rule

Never train on anything in transfer-v4 / v9 holdouts: `mmlu`, `emotion`, `tweet_offensive`, `qnli`, `paws`, `sciq`, `mmlu_pro`, plus Kev-synthetic `contrastive`, `legacy_holdout`, `composition_holdout`, `buried`, `unknowable`. Start only from raw base checkpoints — never GLUE-tuned, `gliclass-*`, or `*-zeroshot-*` variants (QNLI leaks through every GLUE-tuned checkpoint).

## Ladder

| rung | what | arms | eval | decision rule | H100-h |
|---|---|---|---|---|---|
| **E0** local, free | Zero-cost checks DA2 asked for | per-block clean counts from manifest; does Kev render SciQ with its support passage; MLM-cloze probe of ModernBERT-base on the 232 MMLU/SciQ dev items; re-score Kev-0.6B through our server (done: 0.6204) | transfer-v4 dev | fixes the block weights before any GPU | 0 |
| **E1** anchor regime | Qwen3-0.6B + pointer, decision-v7 | (a) LoRA r16 lr 2e-4 [reproduce 0.620]; (b) LoRA lr 5e-5; (c) full FT lr 2e-5; (d) full FT lr 5e-5 | v7 dev + transfer-v4 dev, per-source | best regime becomes the base for E2; if (c)/(d) beat (a) by ≥ 2 pp that alone is the headline | ~1.5 (Kev-0.6B ≈ 20 min/run per PLAN) |
| **E2** distillation | E1-best regime + KD | teacher logits from Kev-4B-1ep and Kev-9B over v7 train (~0.3 h); students with loss = CE + α·KL, α ∈ {0.5, 1.0}, per teacher → 4 runs | same | ship the best transfer-v4 dev; if no KD arm beats E1-best by ≥ 1 pp, ship E1-best and note KD null | ~3 |
| **E3** encoder arm | ModernBERT-base + [MASK]-marker scorer, full FT | lr {2e-5, 5e-5} × {CE, CE+KD with the same teacher targets} → 4 runs | same, per-source (watch the MMLU/SciQ block) | publish as tinyjev-0.15b if ≥ 0.55 transfer-v4 dev; else drop the size, report why | ~2 |
| **E4** seeds + lock | winning 0.6b recipe × 3 seeds | — | mean ± CI on dev; then ONE locked-test read | Kev's own runs spread 75.6 vs 69.5 on identical config; a single run is not a result | ~1 |
| **E5** footprint | fp16 / INT8 / 4-bit on MLX + CoreML export | — | transfer-v4 dev delta; M1 latency; iPhone latency if a device is available | ship INT8; ship 4-bit only if < 1 pp | 0 (local) |

**Total ≈ 8 H100-hours.** At Modal's H100 rate (assume ~$4/h — confirm against the ledger before launch) ≈ **$35**, call the ceiling **$50**. Nothing launches until the ledger is read and the total stated.

## Expected outcome, stated before running

| model | params | transfer-v4 dev | vs anchors |
|---|---|---|---|
| tinyjev-0.6b | 596M | **0.63–0.68** | Kev-0.6B 0.620 · Kev-4B 0.797 · Jev 0.857 |
| tinyjev-0.15b | 149M | 0.52–0.62 | first sub-0.5B model with a number on Kev's suites |

If tinyjev-0.6b lands inside that band it is the best sub-1B open model on the category's frozen suites. That is the claim. "Same accuracy as 4B" is not.

## Risks and their detectors

| risk | detector | mitigation |
|---|---|---|
| KD helps in-distribution, not held-out (regulariser signature seen once in-category: OOD +2.6, in-dist −4.1) | E2 v7-dev vs transfer-v4 gap | keep CE-only arm; report both |
| low-LR gain doesn't replicate at 0.6B | E1 (a) vs (c)/(d) | it's 1.5 H100-h to find out |
| encoder collapses on knowledge block | E3 per-source, E0 cloze probe | accept, publish as classification-strong / knowledge-weak, or drop the size |
| 4-bit kills the decoder | E5 delta | ship INT8 |
| single-run luck | E4 seeds | mean ± CI, always |
| contamination | E0 manifest audit; raw bases only | listed above |
| Kev changes suites/leaderboard under us | pin dataset revisions + suite sha256 (they're in the manifests) | cite the pins |
| teacher noise: bigger teacher isn't better | E2 compares 4B-1ep vs 9B | pick by student transfer-v4, not teacher size |
