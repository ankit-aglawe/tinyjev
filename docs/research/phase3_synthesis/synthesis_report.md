# Phase 3 — Synthesis Report: tinyjev

Date: 2026-09-22. Role: synthesis agent. Inputs (all under `docs/research/`): `phase1_scoping/research_question_brief.md`; `phase2_investigation/local_evidence.md` (LE) + `verification_local_evidence.md` (VLE); `lit_backbone_head.md` (LBH) + `verification_backbone_head.md` (VBH); `lit_objective_calibration.md` (LOC) + `verification_objective_calibration.md` (VOC); `lit_footprint_landscape.md` (LFL) + `verification_footprint_landscape.md` (VFL). This document builds the comparative frame. It does not write the final report and does not pick the single recipe.

Kev paths (`KEV/…`, `REP/…`) are as defined in LE §0; every Kev number below was reproduced at its cited line by VLE.

---

## 0. Conventions, weights, and corrections applied

**Evidence weights (from the brief's evidence policy, tightened by the verification reports).**

| Code | Meaning | Weight |
|---|---|---|
| MC | measured, controlled: same frozen items, one factor varied, seeds/CI stated | highest |
| MU | measured, uncontrolled: real run, but comparator differs in data/prompt/items, or single seed without CI | high for "what happened", low for "why" |
| PR-D | peer-reviewed, task is classification / multiple-choice / typed-decision-shaped | high; outranks any in-category README |
| PR-A | peer-reviewed, task is *analogous* (passage reranking, vision, generation) | medium; every use is labelled **inference** |
| PP | preprint / non-archival workshop | medium-low |
| SR | README / model card / leaderboard self-report, no controlled comparison | lowest; the category is nine days old and every card was created 2026-09-17..22 (VFL (c) 16) |
| IM | independently measured by a third party (JevBench, jev-benchmarks, edgejev, laya-coreml) | between MU and SR |

**Transfer labels.** Any of the following is marked "inference" wherever used: vision → text; passage reranking → typed decisions; in-distribution → held-out source; a different suite → transfer-v4; a decoder result → an encoder or vice versa; M1 Max / M5 / T4 latency → base-M1 latency.

**Corrections applied before any number was used (all 4 verification reports).** The Ettin native-decoder MNLI row (150M dec 85.6, 400M dec 88.2, 1B dec 89.9) is restored (VBH #3, (d)1). Abbes fine-tuned Llama-3.2-1B 56.8 / 3B 82.2 on SQuAD-v2 groundedness are added and the FLOPs ratio is RoBERTa-large 1.1e12 vs 8B 1.6e13 (~15x) (VBH (d)2). PriDe at 5% is +1.2/+1.3/+1.7 (VBH (d)3). Set-Encoder-330M is 0.727/0.789 (DL19) and 0.735/0.790 (DL20); monoELECTRA-330M 0.733/0.765, 0.727/0.799 (VBH (d)4). ModernBERT throughput is 133.8k vs 23.4k tokens/s long-variable, RTX 4090 (VBH (d)5). FIRST BEIR averages 54.3/53.7/50.7/50.7 (VBH (d)6). B2 and SayCan are peer-reviewed (VBH (d)7-8). The Ettin reranker blog is single-author, vendor COI (VBH (d)9). Mukhoti: LS *improves* 20NG accuracy; SST-Binary row added, on which focal is worse than CE pre-TS (VOC (d)2). Galil: Spearman(AUROC, accuracy) = 0.03 overall; the +0.76/−0.74 family coefficients are AUROC-vs-ECE (VOC (d)1). RAPS at 95%: 4.21–11.7 vs 22.5–46.3 (VOC (d)3). Wang 2023 baseline 66.5 (VOC (d)4). Di Palo is EMNLP main, Amazon COI (VOC (d)5). Kadavath is vendor self-evaluation, not load-bearing (VOC (d)13). Qwen3-0.6B 4-bit per-group g128: GPTQ 44.0 / AWQ 42.1 MMLU, alongside per-channel 40.0 / 43.1 (VFL (c)1). NanoJev MLX 4-bit figures are a dequantized-to-fp32 weight-error bound on a dev split, not an MLX-runtime measurement (VFL (c)2). laya-coreml: encoder-weight int8 and 4/6-bit palettes *failed* ANE parity; only int8 embeddings shipped (VFL (c)3). apple-silicon-llm-bench cells are mixed-session and the CoreML/ANE Qwen3-0.6B cell is INT8 cold-start (VFL (c)4). Laya latency: 39.5/158.6 ms is ModernBERT-large, 32.8/72.3 ms is mmBERT-base (VFL (c)6). Tiny-Jev public-benchmark rows were fitted on 2,000 items per dataset (VFL (c)7). Kev-0.6B ECE 0.086 dev / 0.089 locked test (VFL (c)9). Kev option-isolation cost at 4B is real in sign, uncertain in size: 0.729 vs round-best 0.767 (−3.8) or three-seed low-lr 0.759 (−3.0); "−5.8 vs 0.787" is unsupported (VLE (e)2). Rule-tree chain is `(A or B) and C` 0.69→0.91 and `or_not` 0.44→0.75 (VLE (e)1). NanoJev option-order invariance is present: flip 0, max abs(Δp) 1.19e-7 (VLE (e)6). kev 0.5B JevBench hard 30.9%; GLiNER2.5-small 33.2%; smalljev 38.2% (VLE (e)7). The Gemma-3-270M "zero-shot beats trained" TypeSafe anomaly is a majority-label artifact and there is no zero-shot held-out block at 270M (VLE (e)8). Verdict has two calibrators (T 1.4265 in the report; 2.8039 in `calibrator.json`) and its selective policy at 0.85 is 0.846 / 1.18% calibrated (VLE (e)4). The 0.6B→4B capacity ladder ran at lr 2e-4, one seed (VLE (e)12); 4B→8B is +0.4 pp [−3.9, +4.5] paired (VLE (e)13). `auto-06b-r1/04` is confounded (lr + ord_w); only `arch-06b/05` is a clean lr mutation at 0.6B (VLE (e)5).

**Anchor numbers used throughout (transfer-v4 dev, 764 records / 656 clean questions, all MC, LE #4, #5, #6, #16, #39, #50; `KEV/PLAN.md:40, 447, 503-508, 717`; `KEV/runs/jev-transfer-v4/report.json`).**

| System | transfer-v4 dev | locked test | Brier (dev) | ECE served | cov@5% (dev) |
|---|---|---|---|---|---|
| Jev 1.13.0 (hosted) | 0.857 | never run | 0.211 | 0.049 | 0.70 |
| Kev-9B (Qwen3.5) | 0.822 | 0.852 | 0.286 | 0.042 | 0.45–0.47 |
| Kev-4B (Qwen3.5) | 0.797 | 0.837 | 0.299 | 0.041 | 0.54–0.57 |
| SemIf-style untrained Qwen3.5-4B-instruct | 0.747 | — | 0.362 | — | — |
| Kev-0.8B (Qwen3.5-0.8B, LoRA r=16) | 0.652 (seeds 0.622/0.634/0.643) | 0.684 | 0.499 | 0.054 | 0.23 |
| **Kev-0.6B (Qwen3-0.6B-Base, LoRA r=16, pointer, CE)** | **0.620** (seeds 0.613/0.605/0.620) | **0.642** | 0.536 | 0.15 dev raw | not reported |
| Kev-0.5B v0.1 (Qwen2.5-0.5B, 6 sources) | 0.561 | — | 0.534 | — | — |
| Untrained Qwen3-0.6B, letter logits | 0.567 | — | 0.505 | — | — |

transfer-v4 dev composition (VLE (d)): `mmlu` 116, `sciq` 116 (knowledge 4-way, 30%); `emotion` 116, `tweet_offensive` 80, `qnli` 80, `paws` 80 (classification-shaped, 47%); `composition_holdout` 96, `legacy_holdout` 80 (policy / rule composition, 23%). Kev-0.8B per-source: QNLI 0.85, SciQ 0.91, Tweet 0.68, PAWS 0.55, MMLU 0.42, Emotion 0.54, auth 0.97, deadline 0.38, rules 0.66/0.56/0.59 (LE §4, `kev-0.8b.md`). Kev-4B: QNLI 0.91, SciQ 0.97, Tweet 0.74, PAWS 0.74, MMLU 0.70, Emotion 0.56, deadline 0.60, rules 0.91/0.88/1.00 (`kev-4b.md`). This per-block structure is the mechanism behind every accuracy estimate in §4.

---

## 1. Literature matrix

Themes: **BB** backbone · **HD** head · **OB** objective · **KD** distillation · **CS** calibration + selective · **DT** data · **FP** footprint · **LS** landscape. Stance is the source's own finding, compressed. "Level" is the evidence code from §0; "Xfer" says whether using it for a ≤0.6B typed-decision recipe is direct (D) or an inference (I) and why.

### 1a. Local stream (LE, VLE)

| ID | Source | Themes | Stance | Level | Xfer |
|---|---|---|---|---|---|
| S1 | `KEV/PLAN.md` capacity ladder (`:729-731`, `:364-365`, `:665-690`) | BB | 0.6B→4B +17.5 pp [+13.0, +22.1] (lr 2e-4, 1 seed); 4B→8B +0.4 [−3.9, +4.5]; Qwen3.5 vs Qwen3 at 4B +1.0 mean (n.s.); 0.8B vs 0.6B +4.8 [+0.2, +9.3] locked; 0.6B "saturated 0.59–0.61 regardless of knobs" | MC | D |
| S1 | `PLAN.md:307-313` lr | BB/DT | lr 2e-4→5e-5 at 4B +4.7 [+0.4, +9.6], mechanism = drift; not isolated at ≤1B | MC | D (4B) / I (≤1B) |
| S1 | `PLAN.md:289-302, 407-410` erosion | BB/DT | fine-tune loses MMLU/PAWS vs base at every size; classification-shaped tasks gain | MC | D |
| S1 | `PLAN.md:265-267, 287-288, 348` option isolation | HD | exact invariance: parity at 0.6B, −3.0 to −3.8 at 4B (comparator uncertain) | MC | D |
| S1 | `PLAN.md:341` head_dim / special embeddings | HD | no gain | MC (1 seed) | D |
| S1 | `PLAN.md:693` adapted-backbone letter probe | HD | at 4B, letter logits through the adapted backbone reproduce the pointer head (deadline 0.53 = 0.53; MMLU 0.72 vs ~0.70) → readout is not the bottleneck at 4B | MC | D (4B) / I (≤0.6B) |
| S8 | calibration-screen-review-v1 (`PLAN.md:67-110`) | OB/CS | at 4B, 1 seed, 1 epoch: CE cov@5% 0.532; LS ε=0.05 **0.006**; CE+0.5·Brier 0.526; focal γ=1 0.502 (best ECE 0.043); recalibrated parent 0.576; "no candidate advances" | MC (1 seed) | D |
| S1 | `PLAN.md:40, 51, 187` temperature | CS | Qwen3.5: single in-dist T≈2.0–2.6 transfers OOD (9B ECE 0.105→0.039, conf-err 7.5→3.2%); per-(type,K) worse OOD; Qwen3 cards: "does not transfer" (under-documented) | MC / MU | D |
| S1 | `PLAN.md:61-64, 140` coverage | CS | cov@5% is tie-aware v2; Jev 0.70; 4B 0.54–0.57; 0.8B 0.23; 59% of errors on 37% of items (TweetEval/PAWS/Emotion) | MC | D |
| S1 | `PLAN.md:41, 698-707` unknowable delta | CS/DT | uniform soft targets on evidence-free items → ≥0.9-confident share 0.00 at 4B/9B; controls unchanged | MC | D |
| S1 | `PLAN.md:418-423` anchoring | OB | KL to base zero-shot: "not a lever" | MC (1 seed) | D |
| S1 | `PLAN.md:284-286, 305-316, 339` data | DT | v4 10.9k hurt OOD vs v3 3.4k at lr 2e-4; synthetic ×3 hurts; 1 epoch better calibrated than 2 | MC | D (4B, lr 2e-4) |
| S1 | `PLAN.md:693, 730-733, 412-431` policy data | DT | compositional pairs +3.6 [−0.4, +7.8] at 0.6B; rule trees at 4B `(A or B) and C` 0.69→0.91, `or_not` 0.44→0.75; ordinal families do not move deadline | MC | D |
| S1 | `PLAN.md:42, 116-130` date_facts | DT | binding a stated day count is what works (n=40: 0.65→1.00, 14/9 flips one way) | MC | D |
| S1 | `PLAN.md:635, 146-153` | KD | soft-target distillation, self-distillation, C4–C6: **never run** | absent | — |
| S1 | `PLAN.md:634` | BB | encoder path: **proposed, never run** ("~$5") | absent | — |
| S1 | `PLAN.md:29, 593` | FP | GGUF/int8/int4: **none** in checkout | absent | — |
| S13 | `KEV/README.md` serving | FP | Kev-0.6B (Qwen3) 123 ms / 5 q on M5 bf16; 0.8B 329 ms (hybrid); bf16 vs fp32 max abs(Δp) 0.017, 0 argmax flips | MC | I (M5 → M1) |
| S14 | Laya `BENCHMARKS.md`, `common.py` | BB/HD/OB/CS/FP | ModernBERT-large 421M + [MASK]-marker scorer; RLCD reward log+0.5(0.75)·spherical+RPS; zero-shot 0.362 (random 0.318); fine-tuned-in-dist 0.766; held-out themes 0.53–0.76; Banking77 0.425 at K=77 (head budget); flip 0.15–0.23 at K=20; ships ECE 0.466 → 0.081 with per-(type,K) T; T4 39.5 ms / 1 q | MU (1 run) / MC (code) | I (own suites) |
| S19 | Verdict README + artefacts | BB/HD/OB/CS/FP | ModernBERT-base 151M + GLiClass 25-slot head, CE+1.0·Brier; K=3/5/9/17/25 → 0.97/0.96/0.91/0.78/0.72; flips 3–4.5% (low-confidence); in-dist 0.95 vs TypeSafe 337 cases 0.48; FP16 = FP32 exactly; WASM 35.6 ms K=5; abstention recall 10–24% under shift | MC (K scaling) / MU | I (narrow data) |
| S15 | NanoJev toy + pipeline v2 | HD/KD/OB | Qwen3-0.6B full FT, set-attention Choice head, exact invariance; toy OOD (n=48): gold 0.833 > Jev-dist 0.708; pipeline v2 (n=400): teacher-distilled 0.773 > gold 0.748 OOD, gold better in-dist (0.829 vs 0.788) | MC (1 seed, tiny n) | I (own splits) |
| S20 | system-one-open | BB/DT/CS | Gemma-3-270M full FT: held-out 0.585 (ECE 0.146, AURC 0.345); lite 2,500/task **0.367**; E2B LoRA 0.742–0.748; zero-shot E2B 0.661 | MC (single run, same held-out set) | I (own 23 held-out tasks) |
| S21 | decider cards | BB/HD/DT/FP | Qwen3.5-0.8B full FT letter-slot readout: in-task 0.776 / held-out 0.707; 2B 0.809 / 0.739; "what the smaller model gives up is knowledge"; 1.47M examples, half an epoch ≈ all of held-out; M1 Pro fp16 99–146 ms | MC (1 run each, no CI) | I (own suite) |
| S22 | JevBench v1.2/v1.3 | LS/CS/FP | 534 decisions, independent: Jev 74.4 (hard 74.1%); kev 0.6B 62.5 (Int 51.9, hard 40.0%); Laya 54.4 (hard 34.1%); Verdict 38.9 (hard 37.7%); kev 0.5B 33.2 (hard 30.9%); kev rows are pre-v7 Qwen3 previews; kev truncates at 384 tokens on a long-text hard tier | IM (serial over network) | I |
| S23 | jev-benchmarks pilot | CS/LS | origin of cov@5%; Jev Emotion cov 0.00, Brier 0.846 | IM (n=100/condition) | D (metric) |
| S17 | mini-jev | HD | frozen Qwen3-0.6B: letter-A prior 37/40, 12/40 correct; frozen 4B: letters ≈ JSON (Δ −0.22 pp) | MU (n=40) / MC | D |
| S18 | jev-on-a-laptop | FP/LS | stock 4-bit MLX 1.5B at majority baseline (58.3 vs 54.2); 4-bit vs fp16 accuracy never compared | MU | I |
| S16 | jevbetter | HD | rival-aware option mixer 0.916 vs 0.873, synthetic, no backbone | MU | I (synthetic) |

### 1b. Literature stream A — backbone and head (LBH, VBH)

| ID | Source | Themes | Stance | Level | Xfer |
|---|---|---|---|---|---|
| B1 | Weller et al. 2026, Ettin, ICLR 2026, https://arxiv.org/abs/2507.11412 | BB | matched data/recipe 17M–1B: enc MNLI 89.2 vs native dec 85.6 at 150M (+3.6); 400M enc 91.3 vs 1B dec 89.9; enc ≈ one decoder size step; dec→enc adaptation does not close it (85.8) | PR-D | D (classification) / I (2T-token decoders vs 36T Qwen3) |
| B2 | Gisserot-Boukhlef et al., ICLR 2026, https://arxiv.org/abs/2507.00994 | BB | same arch, CLM vs MLM (40% mask): seq-cls 82.80/84.89 (210M), 83.58/87.00 (610M), 82.68/88.23 (1B); biphasic CLM→MLM beats MLM; MLM continued-pretraining of a CLM checkpoint is the cheap route | PR-D | D |
| B3 | Warner et al., ModernBERT, ACL 2025, https://arxiv.org/abs/2412.13663 | BB/FP | base 149M GLUE 88.4 vs DeBERTa-v3-base 88.1; large 90.4 vs 91.4; 133.8k tok/s long-variable on RTX 4090 vs GTE 23.4k | PR-D (+COI) | D |
| B4 | Nielsen et al., NoDaLiDa 2025, https://aclanthology.org/2025.nodalida-1.60/ | BB | fine-tuned DeBERTa-v3-base/large mean rank 1.29/1.09 beat few-shot GPT-4 (1.44); decoders win QA, encoders win NER/acceptability | PR-D | I (few-shot decoders, not fine-tuned) |
| B5 | Abbes et al. 2025, https://arxiv.org/abs/2506.21288 | BB | pair classification: RoBERTa-large 90.2/88.5; fine-tuned Llama-3.2-1B **56.8**/84.0; 3B 82.2/86.4; 8B 91.1/92.3; ~15x FLOPs | PP | D (only fine-tuned ≤3B decoder-vs-encoder classification row) |
| B6 | Pan 2025, Tiny Reward Models, ICML-W, https://arxiv.org/abs/2507.09973 | BB/HD | ModernBERT-base 150M with both options visible: RewardBench 78.4 vs 70B RM 91.0; large 86.4 | PP (+COI) | I (pairwise preference) |
| B7 | Zhang et al., Qwen3 Embedding, https://arxiv.org/abs/2506.05176 | BB/HD | 0.6B decoder yes/no-logit reranker beats older 0.3–0.6B encoders; 0.6B→4B gap 4–9 nDCG | PP (+COI) | I (reranking) |
| B8 | Aarsen 2026, Ettin reranker blog, https://huggingface.co/blog/ettin-reranker | BB/KD/FP | distilled 150M encoder 0.5994 = Qwen3-Reranker-0.6B 0.5940; 3,237 pairs/s H100 (0.6B-class decoder 387–928) | SR (single author, COI) | I (reranking) |
| B9 | He et al., DeBERTaV3, ICLR 2023, https://arxiv.org/abs/2111.09543 | BB | base 86M backbone + 98M embeddings; MNLI 90.6/90.7; large GLUE 91.37; 512 ctx | PR-D (+cards) | D |
| B10 | Marone et al., mmBERT, https://arxiv.org/abs/2509.06888 | BB | mmBERT-small 42M non-emb GLUE 84.7; base 86.3 (ModernBERT-base 87.4 same harness) | PP | D |
| B11 | Boizard et al., EuroBERT, https://arxiv.org/abs/2503.05500 | BB | 210M/610M XNLI 81.9/84.1; English GLUE weak (81.2) | PP | I (multilingual; out of scope) |
| B12 | SmolLM2 paper + cards, https://arxiv.org/abs/2502.02737 | BB | 360M ≥ Qwen2.5-0.5B on HellaSwag/ARC/MMLU-cloze (54.5/53.0/35.8 vs 51.2/45.4/33.7); MMLU must be cloze at this scale | PP (+COI) | I (zero-shot likelihood MC) |
| B13 | Qwen3 report, https://arxiv.org/abs/2505.09388 | BB | Qwen3-0.6B-Base MMLU 52.8 vs Qwen2.5-0.5B 47.5; 36T tokens | PP (+COI) | D (backbone facts) |
| B14 | Gemma 3 270M blog + card | BB/FP | 100M non-embedding + 170M embedding; ARC-c 29.0 (≈ SmolLM2-135M); INT4 on Pixel 9 Pro | SR (vendor) | D (facts) |
| B15 | sbert MS MARCO cross-encoders | FP | MiniLM-L6 → L12 doubles cost for +0.01 nDCG | SR | I (reranking) |
| B16 | mxbai-rerank-v2 blog | BB | 0.5B decoder reranker +1.6 nDCG over 568M encoder (different training) | SR (vendor) | I |
| B17 | Li et al., LS-LLaMA, https://arxiv.org/abs/2310.01208 | HD | drop causal mask at fine-tune time helps token tasks (7B only) | PP | I (scale) |
| H1 | Devlin et al., BERT, NAACL 2019 | HD | shared-vector MC head; BERT-large SWAG 86.3 vs GPT 78.0 (test) | PR-D | D |
| H2 | Schick & Schütze, PET, NAACL 2021 | HD/BB | [MASK]-verbalizer 223M encoder: SuperGLUE 74.0 vs GPT-3 350M/760M 56.2/56.8 and 175B 71.8 | PR-D | D |
| H3 | Yang et al., UniMC, EMNLP 2022 | HD | one [O-MASK] per option, options **cannot attend to each other**; 235M beats 11B–540B zero-shot on ANLI | PR-D | D |
| H4 | GLiNER, NAACL 2024 | HD | marker token per label, all labels in one pass | PR-D | I (NER) |
| H5 | GLiClass 2025, https://arxiv.org/abs/2508.07662 | HD/FP/BB | one-pass label tokens: 1→128 labels −7% throughput vs 52x slowdown for per-label cross-encoder; DeBERTa > ModernBERT backbones | PP (+COI) | D |
| H6 | Yin et al. 2019 + Laurer card | HD | NLI-as-classification: 0.676 (-c) / 0.673 mean F1; cost linear in K | PR-D / SR | D |
| H7 | Set Transformer, ICML 2019 | HD | ISAB/PMA: permutation-invariant set attention, linear in set size | PR-A | I (no NLP) |
| H8 | Schlatt et al., Set-Encoder, ECIR 2025, https://arxiv.org/abs/2404.06912 | HD | per-candidate position reset + [INT] tokens: 330M 0.727/0.789 vs monoELECTRA 0.733/0.765 (n.s.); invariant by construction; set attention helps only under a comparative loss | PR-A | I (reranking) |
| H9 | Rank-DistiLLM, ECIR 2025 | KD | 110M distilled from 7B/GPT-4 rankers: 0.720/0.711 vs MS-MARCO-only 0.687/0.698 (+3.3/+1.3); ≥ teacher | PR-A | I (reranking) |
| H10 | RankT5, SIGIR 2023 | HD/OB | enc-only score head ≈ enc-dec; listwise softmax > pointwise; needs list ≥20–30 | PR-A | I |
| H11 | monoT5, Findings EMNLP 2020 | HD | true/false-token head precedent; RankT5 +1.2 to +1.9 with a direct head | PR-A | I |
| H12 | RankGPT, EMNLP 2023 | KD | DeBERTa-v3-large distilled from GPT-4 permutations beats monoT5-3B on BEIR (53.03 vs 51.36) | PR-A | I |
| H13 | FIRST, EMNLP 2024 | HD | identifier-logit head + RankNet; ~50% latency cut at 7B | PR-A | I |
| H14 | PriDe, ICLR 2024 | HD/CS | letter-ID **token** bias, not position; PriDe 5% +1.2/+1.3/+1.7 | PR-D | D (≥1.5B) |
| H15 | Robinson et al., MCSB, ICLR 2023 | HD | GPT-2-scale PPA ≈ chance: sub-B decoders cannot bind letters zero-shot | PR-D | D |
| H16 | Pezeshkpour & Hruschka, Findings NAACL 2024 | HD | reordering moves accuracy 13–75/85% | PR-D | D |
| H17 | Wiegreffe et al., ICLR 2025 | HD | symbol binding is a late, sparse-head capability; Qwen2.5-0.5B studied mechanistically | PR-D | D |
| H18/H19 | Holtzman 2021; Zhao 2021 | HD/CS | PMI / contextual calibration for likelihood MC | PR-D | D |
| H20–H23 | DRRN, Wolpertinger, CALM, SayCan | HD | state×action interaction head; retrieve-then-score for 1M actions | PR-A / PP | I (RL) |

### 1c. Literature stream B — objective, distillation, calibration (LOC, VOC)

| ID | Source | Themes | Stance | Level | Xfer |
|---|---|---|---|---|---|
| 3a.1 | Mukhoti 2020, NeurIPS | OB/CS | post-TS ECE gap between CE and Brier/LS/focal shrinks to 1–2 pts; 20NG: LS improves accuracy, focal costs 1.3; SST-Binary: focal worse pre-TS | PR (mixed modality) | I (CNN/TreeLSTM) |
| 3a.2 | Müller 2019, NeurIPS | OB/KD | LS ≈ TS for ECE; LS-trained teacher makes a worse student | PR (vision/MT) | I |
| 3a.3 | Xia 2025, ICLR | OB/CS | LS collapses cov@1% risk 15.66→0.05%; logit-norm recovers | PR (vision) | I — but Kev S8 reproduces the sign on text |
| 3a.4 | Hui & Belkin 2021, ICLR | OB | square loss ≥ CE on 12/14 NLP accuracy tasks incl. BERT (MRPC 83.8 vs 82.1) | PR-D | D (accuracy only) |
| 3a.5 | Shao 2024, ICML | OB | Brier/spherical trainable at 7–13B (BLEU) | PR-A | I |
| 3a.6–3a.9 | CORN/CORAL; Beckham; SORD; Haas | OB | rank-consistent ordinal heads beat CE on MAE (vision/tabular) | PR / PP | I (no text) |
| 3b.1–3b.5 | Hinton; DistilBERT; TinyBERT; MiniLM/v2; MobileBERT | KD | 66M keeps 97% of 110M; 14.5M keeps 96.8%; 30M from 355M keeps ~93%; 81M student ≈ base; task-specific distillation is the largest ablation term (−7.1 without it) | PP / PR-D | D (in-distribution GLUE) / I (OOD) |
| 3b.6 | Turc 2019 | KD/DT | PD > PF > TF; 11M recovers teacher with 8M in-domain transfer set; transfer-set domain and size govern the gain | PP | D |
| 3b.7 | Furlanello 2018, ICML | KD | self-distillation ~1 pt; gain persists with non-argmax info removed | PR (vision) | I |
| 3b.8 | Stanton 2021, NeurIPS | KD | student agreement 80–90%; optimization is the bottleneck | PR (vision) | I |
| 3b.9 | Cho & Hariharan 2019, ICCV | KD | bigger teacher ≠ better student; early-stopped teacher better | PR (vision) | I |
| 3b.10 | Beyer 2022, CVPR | KD | function matching + consistent views + very long schedules close a 4x gap | PR (vision) | I |
| 3b.11 | Wang 2023, ACL | KD | attention transfer > logit-only KD by 2–6 GLUE | PR-D | D |
| 3b.12 | Tang 2019 | KD | soft logits + augmentation: +4–5 pts over hard labels at 1M scale | PP | D |
| 3b.13 | Hsieh 2023, Findings ACL | KD | rationale supervision lifts T5-220M on ANLI 43.6→49.6 | PR-D | I (seq2seq student) |
| 3b.14–3b.17 | Wang 2021; Pangakis 2024; Di Palo 2024; ZeroGen | KD/DT | LLM hard labels ≈ −0.04 F1 vs human; synthetic data far below supervised on NLI/QA | PR-D | D |
| 3b.20–3b.21 | Mishra 2023; Kim 2025 | KD/CS | augmentation, not soft targets, drives student ECE; teacher calibration R² 0.92 with student accuracy | PP / PR (vision) | I |
| 3c.1 | Guo 2017, ICML | CS | TS: SST ECE 6.63→1.84; argmax unchanged | PR (mixed) | D |
| 3c.2 | Kull 2019, NeurIPS | CS | Dirichlet-L2 beats TS on classwise ECE; n.s. on deep nets | PR | I |
| 3c.3 | Desai & Durrett 2020, EMNLP | CS | BERT/RoBERTa in-domain ECE 1–3%; TS does little under shift (12.62→12.83) | PR-D | D |
| 3c.4 | Chen 2023, ACL | CS | scale lowers ECE only where accuracy rises; TS best unlearnable method; learned calibrators cut confident errors | PR-D | D |
| 3c.5 | Kadavath 2022 | CS | ECE falls with scale 800M–52B (figure-only, vendor) | PP (COI) | not load-bearing |
| 3c.7 | APS/RAPS; Angelopoulos & Bates | CS | conformal sets with 1−α coverage; RAPS sizes 2–4 on ImageNet | PR (vision) | I |
| 3c.8 | Kumar 2023 | CS | LAC conformal on LLaMA-13B 4-option MCQ: coverage 91–94%, set size 2.4–3.7 | PP | I (13B) |
| 3c.9–3c.13 | Geifman 2017/2019; Deep Gamblers; SAT; Feng 2023 | CS | learned abstention heads' gains come from a better classifier; SR on the same net wins | PR (vision) | I |
| 3c.14 | Jaeger 2023; Traub 2024 | CS | nothing beats MSR across shifts; AURC → AUGRC changes rankings 5/6 | PR (vision) | I (metric definitions direct) |
| 3c.15 | Galil 2023, ICLR | KD/CS | KD regimes give the largest median AUROC/ECE gain; TS improves AUROC on 523 multiclass models; Spearman(AUROC, acc)=0.03 | PR (vision, observational) | I |
| 3c.16 | Xin 2021, ACL | CS | SR beats MC-dropout for BERT; larger encoders rank better | PR-D | D |
| 3c.17 | Varshney 2022, Findings ACL | CS | nothing consistently beats MaxProb for BERT-base across IID/OOD/ADV | PR-D | D |
| 3c.18 | Kamath 2020, ACL | CS | calibrator with known-OOD: cov@80% acc 48.2→56.1 | PR-D | I (extractive QA) |

### 1d. Literature stream C — footprint and landscape (LFL, VFL)

| ID | Source | Themes | Stance | Level | Xfer |
|---|---|---|---|---|---|
| A1–A4 | Q8BERT; I-BERT; Bondarenko; BinaryBERT | FP | encoder INT8 QAT lossless (RoBERTa 86.0→86.3); per-tensor W8A8 PTQ breaks BERT (83.06→71.03), per-group fixes (82.45); W4A8 QAT −0.4; W1A8 −0.4 MNLI | PR-D | D (BERT-era encoders) / I (ModernBERT) |
| A5 | LLM.int8(), NeurIPS 2022 | FP | naive absmax int8 breaks OPT-125M (ppl 25.65→87.76); vector-wise fixes | PR | D (mechanism) |
| A6 | Zheng 2025, Qwen3 quantization | FP | Qwen3-0.6B MMLU 47.1; 8-bit 47.0; 4-bit per-group g128 GPTQ 44.0 / AWQ 42.1; per-channel AWQ 43.1; 3-bit collapses | PP | I (MMLU generative/MC, no head) |
| A7 | Lee 2025, IJCAI | FP | Llama-3.2-1B: FP8/W8A8 lossless; AWQ-4 −3.6; GPTQ-4 −7.2; small models suffer more | PR | I (≥1B) |
| A8 | Srivastava 2026, Findings ACL | FP | Qwen2.5-0.5B-Instruct 4-bit −40% relative on arithmetic; 8-bit lossless | PR | I (arithmetic) |
| A9 | Husom 2025 | FP | Pi 4: Q4 accuracy inside noise; energy halves at Q8 | PP | I |
| A10 | Melton 2026, Zenodo | FP | MLX default 4-bit: 220 vs 125 discordant pairs favour bf16; loss shrinks +0.52 pt/B | PP (12 days old) | I (code gen) |
| A11 | NanoJev-mlx-4bit card | FP | 4-bit g64 weights, head fp16: 99.17% argmax agreement, KL 2.9e-4 — **dequantized-fp32 evaluation on a dev split** | SR (4 days old) | I |
| A12 | edgejev README | FP | ONNX INT8 CPU: Laya 322M AG News 92.8→91.2 (stable), Emotion noise; kev 0.5B 90→84 / 44→21 (n=100) | IM (2 days old) | D-ish |
| B1–B6 | Turc; TinyBERT; MobileBERT; MiniLM; CoFi; ShortGPT | FP | 25M encoder −0.6 GLUE at 62 ms on Pixel 4; CoFi 95% sparsity ≤20 GPU-h; ShortGPT ≥2.8B only | PR / PP | I |
| C1–C6 | MRL; MatFormer; Gemma 3n; LayerDrop; PABEE; DeeBERT | FP | one run → nested sizes; LayerDrop 12→6 layers MNLI 82.9 vs DistilBERT 81.6; early exit 1.6x with ±1 pt | PR / SR | I (no calibration) |
| D1 | Apple ANE transformers 2022 | FP | DistilBERT 66M seq 128 on iPhone 13 ANE: 3.47 ms | SR (vendor, code) | I |
| D2 | MobileLLM, ICML 2024 | FP | 125M decoder iPhone 13: 15.6 ms/token | PR | I (decode) |
| D4 | apple-silicon-llm-bench | FP | Qwen3-0.6B iPhone 17 Pro MLX 4-bit 158.8–178.8 tok/s; CoreML/ANE INT8 cold 37.7; Qwen2.5-0.5B M4 Max MLX 531 tok/s, 390 MB | SR (mixed sessions) | I (decode) |
| D6 | Kev README / cards | FP | Kev-0.6B 0.12 s per 5-question request on M5 bf16 | SR/MC | I (M5 → M1) |
| D7 | laya-coreml card | FP | mmBERT-base 322M CoreML FP16: L512 27.5 ms CPU+ANE, 9.0 ms all units (M5 Pro); **encoder-weight int8 and 4/6-bit palettes fail ANE parity** | SR (hours old) | I |
| D8 | dev-0.4b card | FP | ModernBERT-large 399M M1 Max MPS ~27.6 ms/forward | SR | I |
| D9 | Verdict | FP | 151M WASM single-thread K=5 35.58 ms; FP16 = FP32 | SR | D |
| §4 | Landscape table (all cards) | LS | no sub-0.5B open model measured on Kev suites; best sub-0.5B on JevBench = Laya/jeff 54.4 < kev 0.6B 62.5; strongest self-reported in-domain numbers collapse off-suite (Verdict 0.95 → 0.48) | SR / IM | D |
| §6 | Hume 2026 | LS | Jev ≈ prefill-only causal model, ~10B active (inference); RLCD assumed | SR (blog) | not citable as fact |

---

## 2. Per-sub-question synthesis

### SQ1 — Backbone

**Converges (3+ sources).**
1. *At matched size and data, a bidirectional encoder beats a causal decoder on fine-tuned classification by roughly one decoder size step.* Ettin 150M enc 89.2 vs dec 85.6, 400M enc 91.3 vs 1B dec 89.9 (B1, PR-D); CLM vs MLM at the same architecture +2.1/+3.4/+5.6 at 210M/610M/1B (B2, PR-D); DeBERTa-v3-base beats GPT-4 few-shot on ScandEval English (B4, PR-D); RoBERTa-large 90.2 vs fine-tuned Llama-3.2-1B 56.8 and 3B 82.2 on pair classification (B5, PP); PET 223M vs GPT-3 350M/760M +18 (H2, PR-D). Five sources, four peer-reviewed, all direct.
2. *Capacity is the dominant lever inside the decoder family on Kev's OOD suite.* 0.6B→4B +17.5 pp [+13.0, +22.1] (S1, MC, 1 seed, lr 2e-4); 0.6B saturated at 0.59–0.62 across eight single-factor mutations (S1, MC); 270M vs E2B −16 pp on system-one-open's held-out set (S20, MC single run); decider 0.8B vs 2B −3.2 held-out, gap concentrated on knowledge tasks (S21, MU); Qwen3-Reranker 0.6B→4B −4 to −9 nDCG (B7, PP, inference). What is lost at small size is *knowledge and date arithmetic* (MMLU 0.42 at 0.8B vs 0.70 at 4B; deadline 0.38 vs 0.60), not the decision format (decider card; Kev per-source rows).
3. *Fine-tuning erodes base capability at every decoder size* (MMLU, PAWS drop vs base; S1 MC), and *learning-rate/drift is the mechanism at 4B* (lr 2e-4→5e-5 +4.7; S1 MC). Not isolated at ≤1B (VLE (c)10).
4. *Newer base generation is worth ~0–2 pp, concentrated on `deadline`* (Qwen3.5 vs Qwen3, S1 MC; base probes +0.8/+2.1/−0.3 n.s.).

**Diverges, and why.**
- *Decoder rerankers beat encoder rerankers* (B7 Qwen3-Reranker-0.6B > bge/gte encoders; B16 mxbai 0.5B > bge-m3) vs *distilled 150M encoder = Qwen3-Reranker-0.6B* (B8). Mechanism: uncontrolled training data and a distillation teacher; B7/B16 compare a 36T-token decoder to 2021-era encoders, B8 compares a freshly distilled encoder to the same decoder. Both are reranking → inference for typed decisions, and both are vendor-reported. Net reading: on English pair scoring the encoder edge survives only when training is matched; the 36T-token pretraining gap is real and is not captured by Ettin's 2T-token matched pairs.
- *Local encoders under-perform local decoders on JevBench* (Laya 421M hard 34.1%, Verdict 151M hard 37.7% vs kev 0.6B 40.0%; S22 IM) vs the literature's encoder advantage. Mechanism: Verdict was trained on Banking77+CLINC150 only and caps K at 25; Laya's head budget is 192 tokens across options; JevBench's hard tier is long-text and both encoders (512 ctx / Laya 1024) and Kev (384-token state) truncate; JevBench's kev rows are pre-v7 previews. This is a data-breadth and context comparison, not an architecture comparison (VLE (c)5, (c)30 below). It does not contradict Ettin.
- *"Small model gives up knowledge, not format"* (decider) vs *fine-tuned 1B decoder collapses to 56.8 on groundedness* (B5). Mechanism unknown — B5 does not say which head or how many epochs; treat as a warning that small decoders fine-tuned on pair classification can fail badly with a naive recipe, and as the reason a pointer/marker head (not a generative or CLS head) is the default.

**Strongest evidence.** B1+B2 (two ICLR 2026 papers, matched training, sizes bracketing the target) for encoder-vs-decoder on classification; S1's capacity ladder (MC on the target suite) for the size effect. The two are not in conflict: the encoder advantage is per-parameter *at matched pretraining*; Qwen3-0.6B's 36T tokens buy knowledge that no ≤0.6B encoder in the corpus has been pretrained to hold. The per-block structure of transfer-v4 (30% knowledge MCQ, 47% classification-shaped, 23% rules) is where the two findings meet — §4 uses it.

**Silent (<2 sources).** Any encoder on transfer-v4 or on any Kev suite (0; PLAN.md:634 unrun). Any encoder on 4-way knowledge MCQ (MMLU/SciQ) at 150–400M (0). Any trained decoder + head at ≤360M on a held-out-source suite other than system-one-open's 270M (1). MLM continued-pretraining of Qwen3-0.6B (B2's cheapest route) applied to decisions (0). Long-state degradation curve for any small model (0; PLAN_27b A3 unrun).

### SQ2 — Head

**Converges.**
1. *Zero-shot letter-logit readout is not viable at ≤0.6B.* GPT-2-scale PPA ≈ chance (H15, PR-D); symbol binding is a late, sparse capability (H17, PR-D); frozen Qwen3-0.6B letter-A prior 37/40 (S17, MU); untrained Qwen3-0.6B 0.567 on transfer-v4 (S1, MC) vs trained pointer 0.620. Four sources.
2. *A shared scorer over per-option marker states, with no inter-option attention, is sufficient and well-precedented.* BERT-MC (H1), PET (H2), UniMC blocks inter-option attention and beats 540B zero-shot (H3), GLiNER/GLiClass one-pass label tokens (H4/H5), Laya's [MASK]-per-option (S14), Kev's `</opt>` pointer (S11). Set attention adds nothing measurable on relevance: Set-Encoder ≈ monoELECTRA (H8, PR-A, inference); Kev `head_dim`/special-embedding variants no gain (S1 MC); NanoJev's set head was never ablated against its scalar head (S15). jevbetter's rival-aware mixer wins only on synthetic menus without a backbone (S16 MU).
3. *After training, the readout is not the accuracy bottleneck at 4B*: letter logits through the adapted backbone reproduce the pointer head's deadline (0.53 = 0.53) and MMLU (0.72 vs ~0.70) (S1 MC). decider's letter-slot readout at 0.8B full-FT reaches 0.707 held-out on its own suite (S21). Whether this holds at ≤0.3B is untested (see C23).
4. *Permutation sensitivity scales inversely with size and is cheap to remove at small size.* Flip rate Kev-0.5B 0.21 → 0.6B 0.07 → 9B 0.03 → Jev 0.00 (S1 MC); Laya 0.15–0.23 at K=20; Verdict 3–4.5%, concentrated at low confidence. Exact invariance via `option_isolation` is free at 0.6B and costs 3–4 pp at 4B (S1 MC; comparator uncertain, VLE (e)2); Set-Encoder's per-candidate position reset is invariant at no measured cost (H8, inference); NanoJev's per-candidate paths are invariant (flip 0) at K× compute (S15).
5. *High-cardinality option sets are a token-budget problem, not a head problem.* GLiClass one pass 1→128 labels −7% throughput vs 52x slowdown for per-label NLI (H5); Laya Banking77 0.425 because 77 options share 192 tokens (S14); Verdict 0.97→0.72 from K=3 to 25 with a fixed 25-slot head (S19 MC); Kev-0.5B Banking77 0.860 in-distribution with no per-option budget (S13); decider CLINC 151-way 0.88 (S21).

**Diverges.**
- *Listwise softmax needs ≥20–30 candidates to beat pointwise* (H10, PR-A) vs Kev's CE over K=2–5 (S11). Mechanism: RankT5's lists are retrieval negatives with one positive; typed-decision options are short, mutually exclusive, and the softmax *is* the likelihood. Not comparable; the H10 finding matters only for Score/large-K types trained with sampled negatives.
- *PriDe/debiasing gains* (H14: +1.2–1.7 at 5% cost) apply to letter-ID readouts on ≥1.5B models; pointer/marker heads sidestep the ID token entirely, so the local replicas never needed it. Divergence is a scope difference.

**Strongest evidence.** H3 (UniMC) + H8 (Set-Encoder) + Kev's isolation/head ablations: independent-option scoring is the safe default; set attention buys invariance, not accuracy. H15/H17 + S17 settle the letter-logit question at ≤0.6B.

**Silent.** Set attention vs shared scorer *on typed decisions with a trained backbone* (0 controlled comparisons; NanoJev has both code paths, no ablation read). Per-candidate position reset on a RoPE encoder (ModernBERT) without re-pretraining (0; Set-Encoder used ELECTRA). Ordinal (cumulative-threshold) head for Score (0 built; Kev's RPS term "did not help", 1 source). Head behaviour at K=50+ on held-out sources for any system (0; all K>25 numbers are in-distribution).

### SQ3 — Objective and calibration

**Converges.**
1. *Once a temperature is fitted, the training loss barely moves ECE; CE is not the problem.* Kev's matched screen: CE, CE+Brier, focal within 0.01 ECE and 0.03 coverage of each other, "no candidate advances" (S8, MC, 1 seed); Mukhoti post-TS gaps 1–2 pts on vision and ~0 on text (3a.1, PR, inference); Hui & Belkin square ≈ CE on BERT accuracy (3a.4, PR-D); Guo TS alone takes SST ECE 6.6→1.8 (3c.1). Laya's RLCD objective has "the same optimum as log loss, RL adds variance" (Kev's reading, `PLAN.md:607`) and Laya still shipped at ECE 0.466 (S14). Five sources.
2. *Label smoothing destroys selective prediction.* Kev cov@5% 0.532→**0.006** at ε=0.05 (S8, MC, text, 1 seed); Xia cov@1% 15.66→0.05% (3a.3, PR, vision, inference); Müller: LS teacher makes a worse student (3a.2, inference). The Kev result is the first text measurement of the effect and reproduces the vision sign exactly; treat as established for this project.
3. *Post-hoc: a single temperature, fitted in-distribution, is the default; per-(type,K) is worse OOD; softmax-response is the selector.* Kev Qwen3.5 T≈2.0–2.6 transfers OOD (ECE 0.105→0.039, conf-err 7.5→3.2%, coverage unchanged) and per-(type,K) is worse because OOD K values were never seen (S1 MC); Solomon: per-type T no held-out gain (`PLAN_27b.md:36`); Chen: TS is the best unlearnable method (3c.4, PR-D); nothing beats MaxProb/SR for BERT across IID/OOD/ADV (3c.16, 3c.17, PR-D); learned abstention heads' gains are a better classifier, not a better selector (3c.13, 3c.14, vision, inference); Verdict's abstention recall collapses to 10–24% under shift (S19).
4. *Coverage@5%-error is driven by accuracy and by label-noise sources, not by the loss.* 59% of Kev-9B errors sit on TweetEval/PAWS/Emotion (37% of items); ceiling 0.86 at 9B accuracy if ranking were perfect vs 0.45–0.57 achieved (S1 MC); Jev's Emotion coverage is 0.00 with Brier 0.846 (S23 IM). Learned calibrators with *known-OOD* data recover ~8 coverage points at fixed accuracy (3c.18, PR-D, extractive QA, inference).
5. *Targeted soft targets fix a targeted failure.* Uniform soft targets on evidence-free items send the ≥0.9-confident share to 0.00 at 4B/9B without touching controls (S1 MC) — the only objective-side intervention in the corpus that moved a product metric.

**Diverges.**
- *Temperature transfer flips between Qwen3 and Qwen3.5 generations* (S1; C1 below). Qwen3 side is one sentence per card plus a 0.6B/transfer-v2 observation; Qwen3.5 side has full numbers. Under-documented, not opposite-measured; the recipe should re-measure T transfer on whatever base is chosen.
- *"TS cannot reorder confidences" (Kev, retracted `PLAN.md:63`; "coverage unchanged") vs "TS consistently improves AUROC" (Galil, 523 ImageNet models).* Resolved in C21: monotone within a question and provably so at K=2 (Noul); across questions with mixed K a single T can reorder; Kev measured no coverage change at its K distribution (mostly 2–5); Galil's is 1000-class. No conflict once K is stated.
- *Per-(type,K) T: worse OOD (Kev) vs best (Laya, ECE 0.466→0.081).* Laya evaluates on its own mixed suites with a degenerate `choice:11+ = 0.1006` (S14; VLE (c)3); Kev evaluates on unseen K. Laya's number is in-distribution for the calibration partition.
- *Focal loss:* best ECE in Kev's screen (0.043) but worst coverage (0.502); Mukhoti: focal costs 1.3 pts accuracy on 20NG and is worse than CE pre-TS on SST-Binary (VOC (d)2). Direction is consistent: focal trades ranking for ECE.

**Strongest evidence.** S8 (the only matched five-arm loss screen on the target suite, albeit 1 seed at 4B) for "loss does not matter after TS, LS is catastrophic"; 3c.16/3c.17 (PR-D, BERT) for "SR is the selector"; S1's Qwen3.5 temperature rows for "single T transfers".

**Silent.** Any loss comparison at ≤1B or from scratch (0; S8 is a 4B delta). Text measurement of LS on coverage other than S8 (0). Soft-target *distillation* effect on ECE/coverage for any text student (0; VOC (c)3). Conformal prediction on typed decisions (0; Kumar is 13B MCQ). Ordinal proper scoring (RPS/CORN) on a text Score head with ECE (0). Whether the 0.6B model's raw over-confidence (Kev-0.8B conf-err 9.9% raw) is fixable beyond TS at that size (0).

### SQ4 — Data

**Converges.**
1. *Breadth of decision-shaped sources beats volume of any one source; policy/rule synthetic data transfers.* Kev decision-v7 = 1,000 rows × 10 public sources + 896 policy pairs + 1,680 rule trees (S12 MC); compositional pairs +3.6 at 0.6B, +5.1 at 4B; rule trees lift held-out `(A or B) and C` 0.69→0.91 (S1 MC); decider 93 tasks, system-one-open 92 datasets; Tiny-Jev/Laya trained on narrow data collapse off-suite (Verdict 0.95→0.48). ZeroGen: synthetic-only data far below supervised on NLI/QA (3b.17, PR-D).
2. *More of the same public data does not help and can hurt at 4B* (v4 10.9k 0.704/0.735 vs v3 3.4k 0.747/0.750 at lr 2e-4; synthetic ×3 hurts; S1 MC) — but the mechanism is drift at high lr, and the finding is not replicated at low lr or at ≤1B.
3. *Small models are data-hungry.* system-one-open 270M lite (2,500/task) 0.367 vs full (12k/task) 0.585 (S20 MC single run); Turc: transfer-set size/domain governs distillation gains, 11M recovers the teacher with 8M in-domain examples (3b.6 PP); TinyBERT: data augmentation is worth +7 (3b.3 PR-D).
4. *Teacher labels are cheaper and slightly worse than human labels* (Pangakis −0.04 F1; Wang 2021 50–96% cost saving; 3b.14/3b.15 PR-D); *knowledge MCQ sources raise MMLU but not transfer* (S1 MC, 1 seed).
5. *Contamination discipline is exact-state dedup only, everywhere* (every Kev manifest; Verdict train–test 5-grams; S12, S19).

**Diverges.**
- *Kev "more public hurts" vs 270M "more data is the whole game"* → C24: two regimes (drift-limited large model at lr 2e-4 vs under-fit small model). Prediction for 150–360M: more diverse data helps; keep lr low; avoid oversampling any one family.
- *NanoJev: teacher distributions vs gold* — toy OOD gold > teacher (n=48), pipeline v2 OOD teacher > gold (n=400). → C25: trust the better-powered result; both say gold wins in-distribution.

**Strongest evidence.** S12/S1 for the mix that produced every Kev number; S20 for the small-model data-hunger; 3b.6 for transfer-set effects.

**Silent.** Data scaling at 150–600M *on Kev's suite* (0; only the 4B v3/v4/v5 ladder). Optimal public/policy ratio at ≤1B (Kev: `public_frac` neutral at 4B, 1 seed). Laya's training-mix composition (absent from checkout). Any fuzzy or pretraining decontamination against MMLU/QNLI/PAWS/Emotion for ModernBERT/DeBERTa/Qwen3 (0).

### SQ5 — Footprint

**Converges.**
1. *8-bit is free for both families when done right.* Encoders: Q8BERT/I-BERT QAT lossless; per-group PTQ 82.45 vs 83.06 (A1–A3, PR-D); Verdict FP16 = FP32 (S19); edgejev Laya INT8 −1.6 AG News, stable (A12 IM). Decoders: Qwen3-0.6B 8-bit 47.0 vs 47.1 (A6 PP); Llama-1B FP8/W8A8 lossless (A7 PR); decider FP8 "less than eval noise" at 2B (S21). Per-tensor / absmax PTQ breaks both families (A3 −12; A5 OPT-125M ppl ×3.4; edgejev kev 0.5B −6/−23 at n=100) — the quantizer, not the bit-width.
2. *4-bit costs a sub-1B decoder 3–8 points on knowledge MC and more on reasoning; loss shrinks with size.* Qwen3-0.6B MMLU −3.1 (GPTQ g128) to −5 (AWQ g128) (A6, corrected VFL (c)1); Llama-1B AWQ −3.6, GPTQ −7.2 (A7); Qwen2.5-0.5B −40% relative on arithmetic (A8); MLX default 4-bit favours bf16 with +0.52 pt/B recovery (A10). The one decision-model datum (NanoJev 99.17% argmax agreement) is a weight-only bound on a dev split (A11, corrected). For encoders, W4A8 QAT −0.4 on BERT (A3), but *PTQ palettized 4/6-bit and even int8 encoder weights failed ANE parity gates for mmBERT-base* (D7, hours old).
3. *Latency at 100–600M is tens of milliseconds per question on Apple Silicon, and encoders are 3–5x cheaper per question than same-accuracy decoders.* Encoders: ModernBERT-large 27.6 ms/forward M1 Max MPS (D8); mmBERT-base L512 27.5 ms CPU+ANE, 9.0 ms all units on M5 Pro (D7); Verdict 151M 35.6 ms WASM single-thread; Laya-large 39.5 ms/1 q on T4; DistilBERT 3.47 ms iPhone 13 ANE (D1). Decoders: Kev-0.6B 123 ms / 5 questions M5 bf16 (S13); Kev-0.8B hybrid 329 ms (2.7x slower than Qwen3 at the same size — the hybrid DeltaNet path is a serving cost); decider 0.8B 99–146 ms M1 Pro; Qwen3-0.6B iPhone 17 Pro MLX 159–179 tok/s decode (prefill-only workloads faster). Every number is a different device (VLE (c)17); §4 normalises to base-M1 as an inference.
4. *One-run-many-sizes exists but has never been measured for calibration* (MRL, MatFormer, LayerDrop, early exit: C1–C6, PR, inference; DeeBERT's entropy exit interacts with confident-wrong).

**Diverges.**
- *Encoder INT8 lossless (T1) vs laya-coreml ANE parity failure (T3).* → C26: QAT/per-group vs PTQ palettization, and a parity gate (argmax 16/16 + max abs(Δp)) stricter than a GLUE accuracy gate. INT8 on ANE needs QAT or per-channel with an *accuracy* gate.
- *Decoder INT8 ONNX collapse (edgejev kev 0.5B) vs 8-bit lossless (A6, A7)* → C27: dynamic PTQ with merged LoRA at n=100; LLM.int8 shows why naive int8 breaks small decoders.

**Strongest evidence.** A1–A3 + A5 (PR) for the mechanism; A6 (corrected) for the 0.6B 4-bit cost; D7/D8/S13 for latency, each single-device.

**Silent.** INT8/INT4 accuracy for *any trained ≤1B decision model on a held-out suite* (0; the whole category, VLE (c)12). ModernBERT/DeBERTa/SmolLM2 on iPhone (0, VFL §7). DeBERTa-v3 quantization at all (0). MLX runtime (not dequantized) numbers for a decision model (0). Nested-size training with calibration metrics (0).

### SQ6 — Landscape

**Converges (fully supported by verified numbers, VFL (d)).**
1. *No sub-0.5B open model has been measured on Kev's suites by anyone but Kev, and none beats Kev-0.6B.* Sub-0.5B rows on transfer-v4: Qwen2.5-0.5B ablation 0.605 (v2 data), Kev-0.5B v0.1 0.561 — both below 0.620/0.642 (S1, S13 MC).
2. *On the only independent cross-system benchmark (JevBench v1.3, 534 decisions), every sub-0.5B system is below kev 0.6B (62.5): Laya 54.4, jeff 54.4, Verdict 38.9, kev 0.5B 33.2, GLiNER2.5-multi 16.6, -small 13.8* (S22 IM). Laya and Verdict beat kev 0.6B on the calibration axis (62.5 / 74.1 vs 51.1) and lose on intelligence (45.8 / 38.6 vs 51.9).
3. *Self-reported in-domain numbers do not survive a foreign suite:* Verdict 0.95 → 0.48 (TypeSafe 337); Laya 0.766 (fine-tuned on the benchmark's own train split) vs zero-shot 0.362; Tiny-Jev's public rows were fitted on 2,000 items per dataset (VFL (c)7); decider 0.8B in-task 0.776 → held-out 0.707. Four sources.
4. *The gap to Jev is 20–24 pp at 0.6–0.8B and 2–6 pp at 4–9B on transfer-v4 dev* (S1 MC), consistent with Hume's ~10B-active inference (LFL §6, SR, not citable as fact).
5. *Every landscape card is 0–5 days old and moving* (OpenThai 63.2→74.3 in one bump; Verdict 2.0 weights not downloadable; VFL (c)16). Snapshot dated 2026-09-22.

**Diverges.** JevBench composite ranks kev 0.6B above kev 4B/8B (geometric mean with cost/calibration; intelligence alone orders 8B > 4B > 0.6B) — a metric-construction artefact, not a finding (VLE (c)5).

**Silent.** Any current (v7, Qwen3.5) Kev-4B/9B run on JevBench (0). Any encoder measured on transfer-v4 (0). Any two systems on the same hardware for latency (0 outside JevBench's network runs).

---

## 3. Contradiction register

C1–C19 are LE §3's items with VLE (c)'s adjudications; C20+ are cross-stream tensions identified here. "Resolution" names the mechanism and states what the recipe frame does with it.

| # | Contradiction | Sources | Adjudication / mechanism | Consequence for §4 |
|---|---|---|---|---|
| C1 | Single in-dist T transfers OOD (Qwen3.5) vs "does not transfer" (Qwen3) | `PLAN.md:40` vs `kev-4b-qwen3.md:84`, `PLAN.md:187, 613` | Real, asymmetric: Qwen3 side is one sentence per card + a 0.6B/transfer-v2 observation on v2 data; Qwen3.5 side has full numbers on v7. Different generation, data, metric code; nothing isolates the cause (VLE (c)1). | Re-measure T transfer on the chosen base; budget a Qwen3.5 arm if the base is Qwen3 (recipe F risk). |
| C2 | Deadline: capability limit → data problem → serving question | `PLAN.md:429-431` → `:456` → `:130` | Chronological refinement, each backed by new evidence; final (binding a stated day count, n=40, 14/9 one-way flips) is best-supported but only at 4B/9B. | Date arithmetic is out of scope for a ≤0.6B model; use the `date_facts` preprocessor at serve time; do not spend training budget on it. |
| C3 | Per-(type,K) T worse OOD (Kev) vs best (Laya) vs no gain (Solomon) | `PLAN.md:40`; Laya `BENCHMARKS.md:176-177`; `PLAN_27b.md:36` | Non-comparable partitions: Laya fits and evaluates on its own mixed suites (degenerate `choice:11+` = 0.10); Kev evaluates on unseen K. | Single T is the default; per-type only if the deployment K distribution is known and matched. |
| C4 | Bigger not monotone on locked test (Qwen3 4B 0.806 > 8B 0.780; 4B coverage > 9B) | `PLAN.md:434-435`; locked summaries | Real; single reads, n=656, CI ±4–5 pp; 4B→8B paired +0.4 [−3.9, +4.5] says the step is flat for this recipe. | Kev-4B (1-epoch parent) is a better-calibrated, cheaper teacher than 9B (with 3b.9). |
| C5 | JevBench ordering/level vs transfer-v4 | S22 vs S1 | Comparability gap: pre-v7 previews, geometric-mean composite, long-text hard tier vs 384-token truncation. | Do not use JevBench to rank recipes; use it as a second, foreign read after the ladder. |
| C6 | Laya "beats Jev" | Laya `BENCHMARKS.md`; Verdict README (TF-IDF 0.661 vs Jev 0.680 on the same benchmark name) | In-distribution fine-tune; benchmark is shallow (TF-IDF+LogReg within 2 pp of Jev). | Discard as evidence for encoders; the encoder case rests on B1/B2/H2/H3. |
| C7 | 270M: zero-shot > trained on TypeSafe | system-one-open results | Majority-label artefact (zero-shot Noul 188/236 at the always-no baseline 193/236); no zero-shot held-out block exists (VLE (e)8). | Downgrade; 0.585 held-out stands as the only ≤300M decoder datum. |
| C8 | Encoder vs decoder never compared on matched data | `PLAN.md:634` | Real gap. | E1. |
| C9 | Pointer vs letter-logit never compared on same base/data | `PLAN.md:693`; decider | Real; closest evidence says readout ≠ bottleneck at 4B. | Pointer/marker head by default; a letter-slot arm only at ≥0.8B. |
| C10 | Low-lr not isolated at ≤1B | leaderboard rows | Real and stronger than stated: the only clean ≤1B lr mutation (3e-4 → 0.595) goes up; 2e-4 vs 1e-4 changed data too. | Sweep lr in E1 (1e-4 / 5e-5 / 2e-5) — cheap at ≤0.6B. |
| C11 | Objective screen is one seed, one epoch, 4B, delta | `summary.json` | Real. LS collapse is dramatic but single-seed. | Do not re-run the loss screen at small size before E2; do run LS as a *negative control* only if a smoothing-like regulariser is proposed. |
| C12 | Quantization for trained ≤1B decision models absent | all | Real. | E5 is mandatory before any on-device claim. |
| C13 | Jev never run on the locked test | `README.md:130` | Real; every "gap to Jev" is dev. | Report gaps on dev; state it. |
| C14 | Coverage@5% is two metrics | `PLAN.md:61-62` vs jev-benchmarks protocol | Real; Kev v2 tie-aware with record-group bootstrap is the stricter one. | Use Kev v2; never compare to jev-benchmarks' pilot values. |
| C15 | Pretraining contamination unknown everywhere | manifests | Real. | §6. |
| C16 | NanoJev toy: gold > teacher OOD; pipeline v2: teacher > gold OOD | S15 | Toy n=48 single seed vs 400 with 500 bootstraps; the toy teacher was Jev's rounded distribution. | Weak positive prior for soft-target KD OOD (+2.5 pp in the better-powered read); in-dist gold wins in both. |
| C17 | Latency not hardware-normalised | all | Real. | §4 normalises to base-M1 as labelled inference; E5 measures on one machine. |
| C18 | Kev-0.5B JevBench 33.2 vs in-dist 0.799 | S22, S13 | Six-source prototype; hard tier now filled (30.9%). | Consistent with transfer 0.561; no action. |
| C19 | Recorded-but-unexecuted Kev items | `PLAN.md:164-169, 626-637` | Real; encoder path, self-distillation, ordinal head, MLX, GGUF all unrun. | These *are* the ladder. |
| C20 | **Kev "capacity dominates" (0.6B→4B +17.5; 0.6B saturated) vs Ettin "400M encoder beats 1B decoder on MNLI"** | S1 MC vs B1 PR-D | Both true in their domains. Ettin: matched 2T-token pretraining, fine-tuned single-pair classification — the encoder's bidirectional objective is worth one size step *on that task class*. Kev: fixed decoder family, the saturation is over hyperparameters not architecture, and 30% of the suite is knowledge MCQ where the 36T-token decoder carries knowledge no ≤0.6B encoder has; 23% is rule composition where encoders have no measurement. B2 adds the mechanism (CLM objective costs 2–6 pts of seq-classification at 210M–1B) and B5 the warning (fine-tuned 1B decoder at 56.8). | Predict per block: encoder ≥ decoder on emotion/qnli/paws/tweet; encoder < decoder on mmlu/sciq; unknown on rules. §4 A vs D/F ranges overlap for exactly this reason; E1 resolves it with the per-source table. |
| C21 | **Kev "temperature cannot reorder" (retracted) / "coverage unchanged" vs Galil "TS improves AUROC"** | `PLAN.md:40, 63` vs 3c.15 | K-dependent. Within one question a single T is monotone; at K=2 (Noul) provably so; across questions with different K a single T *can* reorder max-probabilities (Galil's 1000-class result). Kev measured "accuracy and coverage unchanged" at its K distribution (mostly 2–5). Not in conflict. | TS is for ECE, not coverage. Coverage lifts must come from accuracy, KD (3c.15, inference), or a Kamath-style calibrator with known-OOD items (3c.18, inference). |
| C22 | **Set-Encoder "no gain from set attention" vs NanoJev set head vs jevbetter rival-aware mixer** | H8 vs S15 vs S16 | H8's authors: null under pointwise relevance labels, gain once the loss requires comparison (DA-InfoNCE). Typed decisions *are* comparative, but options are short and K is small; NanoJev never ablated set vs scalar; jevbetter has no backbone. UniMC (H3) blocks inter-option attention and wins. Kev's isolation (exact independence) is at parity at 0.6B. | Set attention is a permutation-invariance device, not an accuracy lever; adopt only if it is free (G) and test it in E3, never as the primary bet. |
| C23 | **Small-decoder letter-binding failure vs pointer heads** | H15/H17/S17 vs S11/S21 | The pointer/marker head reads hidden states at option boundaries and never touches the letter token, so MCSB is irrelevant to it. Evidence: frozen 0.6B letter-A prior 37/40 vs trained pointer 0.620; at 4B the adapted backbone's letter logits reproduce the pointer (readout ≠ bottleneck after training); decider's letter slot works at 0.8B *after full FT*. What is untested: whether a trained letter slot works at ≤0.3B. | Pointer/marker heads for every decoder recipe (D, E, F); no letter-logit arm below 0.8B. |
| C24 | Kev "more public data hurts OOD" vs system-one-open "lite 0.367 vs full 0.585 at 270M" vs Turc/TinyBERT data-hunger | S1 vs S20 vs 3b.3/3b.6 | Two regimes: drift-limited large model at lr 2e-4 (Kev's own mechanism, `PLAN.md:307`) vs under-fit small model. | At 150–360M: more diverse data, low lr, no oversampling; E6 tests the curve. |
| C25 | KD direction (NanoJev toy vs pipeline v2) | S15 | See C16. | Weak prior for E2. |
| C26 | Encoder INT8 lossless (A1–A3, edgejev) vs laya-coreml ANE parity failure | PR-D vs D7 (T3) | QAT/per-group vs PTQ palettization; parity gate (argmax 16/16 + max abs(Δp)) stricter than accuracy. | E5 uses an accuracy gate on transfer-v4 and tests QAT INT8 for ANE. |
| C27 | Decoder INT8 ONNX collapse (edgejev kev 0.5B −6/−23) vs 8-bit lossless (A6, A7, decider FP8) | A12 vs PP/PR | Dynamic per-tensor PTQ with merged LoRA at n=100 vs per-channel/vector-wise; A5 gives the mechanism (outlier features). | E5 uses per-channel/group quantizers; report n≥656. |
| C28 | "Calibration improves with scale" (Kadavath, COI) vs Chen "only where accuracy rises" vs Kev served ECE flat (0.8B 0.054, 4B 0.041, 9B 0.042) | 3c.5 vs 3c.4 vs S1 | Post-TS ECE is roughly flat across Kev sizes; raw ECE is worse at small size (0.179 vs 0.106); what scales is coverage (0.23 → 0.57). | Report ECE after TS and coverage separately; do not promise coverage from calibration. |
| C29 | PET/UniMC "small encoder beats big decoder" vs Kev "trained model loses knowledge to its own base" | H2/H3 vs S1 | Task class: NLI/MC-as-entailment vs knowledge recall. | Same as C20. |
| C30 | JevBench: Laya/Verdict < kev 0.6B on hard tier vs literature encoder edge | S22 vs B1/B2 | Uncontrolled: narrow training data (Verdict: 2 datasets), K cap 25, 512-token context on a long-text tier, head token budget. Data breadth dominates architecture at this stage. | Not evidence against A; evidence for Kev-style breadth in A's data. |
| C31 | RankT5 "listwise needs ≥20–30 candidates" vs Kev CE over K=2–5 | H10 vs S11 | Retrieval negatives vs mutually exclusive options; softmax CE is the likelihood for the latter. | No change; consider sampled negatives only for Score/large-K. |
| C32 | Ettin reranker "150M distilled encoder = 596M decoder" (B8, SR, COI) vs Qwen3 report "0.6B decoder beats 0.3–0.6B encoders" (B7, PP, COI) | B8 vs B7 | Matched-teacher distillation vs stale encoder baselines; both reranking (inference). | Supports B (KD into an encoder) as the arm most likely to close the pretraining-token gap. |

---

## 4. Candidate recipe frame

### 4.1 Recipes

| ID | Backbone | Head | Objective | Supervision | Post-hoc |
|---|---|---|---|---|---|
| **A** | ModernBERT-base 149M (raw, not gliclass-/zeroshot-tuned) | Laya-style: `[CLS] type+instr [SEP] [MASK] opt_0 [MASK] opt_1 … [SEP] state [SEP]`, gather at each `[MASK]`, shared `LayerNorm→Linear→GELU→Linear(1)`, softmax over options; Noul = 2-option; Score = expected level | hard-label CE | Kev decision-v7 (10 public × 1k + 896 policy pairs + 1,680 rule trees), train-time option shuffling, `p_none_pair` 0.25 | single T on in-dist calibration split |
| **B** | as A | as A | CE on soft targets `−Σ t·log p` (Kev `question_loss` already supports it) with teacher T ∈ {1, 2} | A's data + Kev-4B (1-epoch parent, best calibrated) soft distributions; transfer set = v7 + 3–10× generated policy states + extra rows from *trainable* sources; consistent views (same shuffle for teacher and student) | single T |
| **C** | DeBERTa-v3-base 184M (86M backbone + 98M embeddings) | as A | CE (or B's KD) | as A/B | single T |
| **D** | Gemma-3-270M (100M non-emb) or SmolLM2-360M | Kev PointerHead (`q,k` Linear d→256, `z = k(h_</opt>)·q(h_<decide>)/√256`), LoRA r=16 on attention+MLP, head at full lr | hard-label CE | decision-v7, lr sweep {1e-4, 5e-5, 2e-5} | single T |
| **E** | as D | as D | soft-target CE (KD) | as B | single T |
| **F** | Qwen3-0.6B-Base (or Qwen3.5-0.8B if C1 bites) | Kev PointerHead; LoRA r=16 (Kev) or full FT (NanoJev/decider precedent) | soft-target CE from Kev-4B/9B + hard-label replay | as B | single T |
| **G** | ModernBERT-base 149M | Set-Encoder-style: state encoded once; each option is a segment with an `[INT]` token; tokens attend within their segment + all `[INT]`s; positions reset per option segment; scorer on `[INT]` | CE | as A | single T |

Reference rows for the table: **Kev-0.6B** (measured anchor, 0.620), **Kev-0.8B** (0.652), **Kev-4B** (0.797, teacher), **untrained Qwen3-0.6B** (0.567, floor), **Jev** (0.857, ceiling).

### 4.2 Master table

Accuracy = transfer-v4 dev (656 clean), expressed as range (central). "Chain" is the evidence path; every arrow that crosses a task, size, device, or suite is an inference and is labelled `[I]`.

| Field | A: ModernBERT-base + marker scorer, CE | B: A + KD from Kev-4B/9B | C: DeBERTa-v3-base variant | D: Gemma-270M / SmolLM2-360M + pointer, LoRA, CE | E: D + KD | F: Qwen3-0.6B + pointer + KD | G: Set-Encoder-style ModernBERT-base |
|---|---|---|---|---|---|---|---|
| **Expected transfer-v4 dev acc** | **0.57–0.65 (0.61)** | **0.60–0.69 (0.645)** | **0.58–0.66 (0.62)** | Gemma **0.50–0.58 (0.54)**; SmolLM2-360M **0.54–0.61 (0.57)** | Gemma 0.53–0.62 (0.57); SmolLM2 **0.57–0.65 (0.60)** | **0.62–0.68 (0.65)** | **0.56–0.66 (0.61)** |
| **Evidence chain** | Start Kev-0.6B 0.620 (MC). Block model: classification 47% — Ettin enc +3.6 over native dec at 150M, B2 +2.1 at 210M, DeBERTa/PET/UniMC `[I: MNLI/SuperGLUE→emotion/qnli/paws/tweet]` → +0 to +5 on Kev-0.8B's 0.64 block; knowledge 30% — no encoder MMLU/SciQ measurement; Kev-0.8B block 0.665 (MMLU 0.42, SciQ 0.91); 2T-token encoder vs 36T Qwen3 `[I]` → −1 to −13 on block; rules 23% — silent, ±8. Net −1 pp central; ±4 from the confounds; JevBench Laya/Verdict < kev 0.6B is uncontrolled (C30) and not used. | A + KD lift: Rank-DistiLLM 110M +3.3/+1.3 nDCG from a 7B ranker `[I: rerank]`; RankGPT-distilled DeBERTa-large beats 3B `[I]`; Tang +4–5 at 1M scale `[I: SST-2/MNLI]`; TinyBERT task-specific distillation +7.1 (in-dist); Turc PD +0.5–1.6; NanoJev pipeline v2 OOD +2.5 (S15, 0.6B full FT, weak); Galil KD → best AUROC `[I: vision]`. Teacher ceiling 0.797 (4B) / 0.822 (9B); Stanton fidelity 80–90% `[I]`. → +2 to +6 over A. | A/C: DeBERTa-v3-base MNLI 90.6 vs Ettin-150M 89.2; GLUE 88.1 vs 88.4 (tie, B3); GLiClass "DeBERTa consistently outperforms ModernBERT" (H5, COI); ScandEval rank 1.29 (B4). → A + 0 to +2. | Kev-0.5B v0.1 0.561 (6 sources) and Qwen2.5-0.5B on v2 recipe 0.605/0.482 (MC, transfer-v2); SmolLM2-360M ≥ Qwen2.5-0.5B on zero-shot MC (B12, COI) → SmolLM2 ≈ Qwen2.5-0.5B-class ≈ 0.55–0.60 `[I: zero-shot→trained]`. Gemma-270M: 100M non-emb ≈ Ettin 68–150M decoder class; system-one-open 270M held-out 0.585 vs E2B 0.742 (−16) on its suite `[I: suite]`; ARC-c 29.0 at SmolLM2-135M level (B14) → 0.50–0.58. Capacity ladder downward from 0.6B `[I: extrapolation]` −4 to −10. | D + KD: Turc PD gain largest for the smallest students (+2 Tiny, +3 Mini, fig.); Tang +4–5; Ghita: decoder students lose general knowledge first (PP) → +2 to +5 over D. | Kev-0.6B 0.620 + KD. "Saturated 0.59–0.62" was over hyperparameters, not supervision; NanoJev pipeline v2 OOD +2.5 with teacher soft targets on the *same base* (S15, n=400, 1 seed); Kev-0.8B (hybrid) +1.4 dev / +4.8 locked (MC) is the same-size alternative base; Furlanello self-distillation ~1 `[I]`; 4B→0.6B 7x gap within Beyer's closed range `[I]`. → +0 to +6. | A ± 2: Set-Encoder-330M 0.727/0.789 vs monoELECTRA 0.733/0.765, n.s. (H8 `[I: rerank]`); Kev isolation parity at 0.6B, −3 to −4 at 4B (MC); UniMC blocks inter-option attention and wins (H3). |
| **Expected cov@5%-error (Kev v2)** | 0.15–0.30 | 0.25–0.40 | 0.15–0.30 | 0.10–0.20 | 0.15–0.30 | 0.25–0.40 | as A |
| Chain | Kev accuracy→coverage pairs: 0.652→0.23, 0.797→0.54–0.57, 0.857→0.70 (MC); 59% of errors on noisy-label sources; encoder in-domain ECE 1–3% (3c.3) does not transfer to ranking under shift. | + KD improves AUROC/ranking (3c.15 `[I: vision]`); Kamath-style calibrator +8 pts only with known-OOD `[I: QA]`. | as A | 0.8B is 0.23 at 0.652; below 0.60 the curve is near floor (Kev-0.5B/JevBench hard 30.9%). | as B | as B, from a higher accuracy base | as A |
| **Expected ECE after TS** | 0.05–0.10 | 0.04–0.07 | 0.05–0.10 | 0.05–0.10 | 0.04–0.08 | 0.04–0.07 (Qwen3 T-transfer risk, C1) | as A |
| Chain | Kev-0.8B served 0.054, Kev-4B 0.041 (MC); Desai: TS does little under shift for BERT (12.62→12.83) `[I]`; system-one-270M 0.131→0.037 at T=2 (SR); Chen: TS best unlearnable (PR-D). | + teacher calibration R² 0.92 with student quality (3b.21 `[I]`); Kev-4B 1-epoch parent is the best-calibrated teacher (S1). | as A | as A | as B | Kev-0.6B Qwen3 raw dev ECE 0.15; Qwen3 T "does not transfer" (under-documented). | as A |
| **Params** | 149M + 1–15M head | same | 184M (98M embeddings) + head | 270M (100M non-emb) / 360M + 0.5M head + ~5–9M LoRA | same | 596M + 0.5M head (+8.8M LoRA) | 149M + ~1M |
| **Expected M1 (base) latency, one 4-option question, ~300–450-token sequence, fp16** | **15–40 ms** MPS/MLX; ~10 ms CoreML ANE `[I: D8 27.6 ms for 395M on M1 Max → ×0.4 params, ×0.5 M1 vs M1 Max; D7 9.0–27.5 ms for 322M on M5 Pro ANE; S19 35.6 ms WASM]` | same | **30–80 ms** `[I: B3 "slow relative to ModernBERT"; no unpadded FA; 1.5–2.5× A]` | **30–70 ms** `[I: Kev-0.6B 123 ms / 5 q on M5 bf16 → 60–120 ms per single q on M1; ×0.45–0.6 params; decider 0.8B 99–146 ms M1 Pro]` | same | **60–150 ms** `[I: Kev-0.6B 123 ms / 5 q M5 → M1 ×2–3; prefill-dominated]` | **20–50 ms** if state is encoded once with K option segments; **K× A** if state is duplicated per candidate (Set-Encoder layout) `[I]` |
| **On-device feasibility** | fp16 ~300 MB. **INT8 free on CPU/GPU** (A1–A3 QAT lossless; edgejev −1.6 PTQ per-channel) — **ANE INT8 failed parity for mmBERT (D7)**, so ANE needs QAT + accuracy gate. 4-bit: W4A8 QAT −0.4 on BERT (A3) `[I]`; PTQ 4/6-bit palettes failed ANE parity (D7); no ModernBERT 4-bit datum. iPhone: DistilBERT 3.47 ms on ANE (D1) `[I]`. | same | fp16 ~370 MB; 98M embedding table; **no DeBERTa-v3 quantization datum**; disentangled attention export to CoreML/MLX untested (silent). 512-token cap (state 384 + options fits; no long-state headroom). | fp16 540 MB / 720 MB. **8-bit free** (A6, A7). **4-bit −3 to −5 MMLU at 0.6B (A6 g128), −40% rel. arithmetic at 0.5B (A8), Melton +0.52 pt/B** `[I]`; Gemma INT4 on Pixel (vendor); NanoJev 4-bit weight-only 99.2% argmax (dequantized). MLX-Swift 0.5B 531 tok/s on M4 Max (D4). | same | fp16 1.2 GB; **MLX 4-bit g64 336 MB**; 4-bit −3 to −5 MMLU `[I]`; 8-bit free; iPhone 17 Pro MLX 159–179 tok/s decode (D4, mixed sessions). | as A, **plus** a custom attention mask and per-segment RoPE reset with no off-the-shelf CoreML/MLX path (silent). |
| **Handles 50+ options?** | Yes with a per-option token budget: Laya's shared `head_max_len` 192 → Banking77 0.425 at K=77; fix = ≥8 tokens/option → sequence grows to ~1k (ModernBERT 8k ctx OK); GLiClass one pass 128 labels −7% (H5). | same | Yes up to the 512 cap (~40 options at 8 tokens each with a 384-token state) — **worse than A**. | Yes: Kev up to 255, Banking77 0.860 at 0.5B in-dist, CLINC 151-way 0.88 (decider). | same | Yes (Kev 255). | Yes by construction, linear in K (100 candidates 0.219 s at 330M on GPU, H8 `[I]`). |
| **Permutation invariance** | No (Laya 0.15–0.23 flips at K=20); mitigate with train-time shuffling + `perm_kl` (Kev has it, unused in release). | same | same | No (Kev-0.5B 0.21, 0.6B 0.07); `option_isolation` gives exact invariance at parity at 0.6B (MC). | same | No (0.07); isolation available on Qwen3 (attention-only), **not** on Qwen3.5 hybrid. | **Exact** (flip 0 by construction; H8, S15). |
| **Data needed** | decision-v7 (12.6k rec / 15.6k q) for a first read; 270M lite→full says small models want more — plan 3–5× via generated policy states + more rows from *trainable* sources only. | A + unlabeled transfer set 3–10× for teacher soft targets; consistent views. | as A/B | as A; 270M is data-hungry (0.367 at 2.5k/task). | as B | as B | as A |
| **Training cost (H100-h)** | **0.1–0.5** `[Kev 0.6B LoRA trial 4.5 min; Laya 421M full FT 30k q × 4 epochs = 2×T4×4.5 h ≈ 0.6–1.2 H100-h → 149M ≈ 0.3]` | **1–4** `[teacher pass: 9B 0.18 s/record → 12.6k rec ≈ 0.6 H100-h, 4B ≈ 0.4; ×3–10 transfer set → 1–4; + student 0.3]` | 0.2–0.8 | **0.1–1** `[Kev 0.6B LoRA 4.5 min; system-one-open 270M full FT 7,112 steps / 207M tokens]` | 1–4 | **1–4** (LoRA) / 2–6 (full FT, NanoJev 600 updates bs 24 A100) | 0.3–1.5 (K× tokens) |
| **Confidence in the estimate** | **LOW–MED** | **LOW–MED** | **LOW** | **MED** (closest analogues exist: Kev-0.5B, system-one-open 270M) | **LOW–MED** | **MED** | **LOW** |
| **Single biggest unknown** | Encoder accuracy on the knowledge block (MMLU/SciQ = 30% of items) and on compositional rules (23%): **zero measurements** for any 100–400M encoder. | Whether decoder→encoder soft-target KD transfers to *held-out sources* — **no source measures it** (VOC (c)3); NanoJev is split. | On-device exportability of disentangled attention; whether GLiClass's DeBERTa > ModernBERT holds for option scoring. | Whether 12.6k records are enough at 270–360M (system-one-open used ~92 datasets); the 270M vs 360M non-embedding gap (100M vs ~300M). | As B, plus whether K-way soft targets carry enough signal (Furlanello: most gain persists with non-argmax info removed → the signal is per-example confidence). | Whether KD lifts a base whose hyperparameter surface is flat ("saturated") — **no KD run exists in the Kev log**; and C1 (T transfer on Qwen3). | RoPE position reset on ModernBERT without re-pretraining (Set-Encoder used ELECTRA absolute positions); custom-mask serving path. |

### 4.3 Reading the frame

1. **The realistic band for a 150–360M model is 0.57–0.69 on transfer-v4 dev, i.e. Kev-0.6B ± 5 pp, not Kev-4B.** Every chain that ends above 0.69 passes through an unmeasured transfer (KD OOD, encoder on knowledge MCQ). The 4B row (0.797) is +17.5 pp of capacity that no source in the corpus recovers at 27x compression on held-out sources (DistilBERT/TinyBERT/MiniLM recover 93–97% *in distribution* at 2–12x; Rank-DistiLLM matches its teacher on in-domain reranking; B5's fine-tuned 1B decoder sits 33 pp below RoBERTa-large).
2. **The encoder arms (A/B/C/G) and the decoder arms (D/E/F) are separated by the knowledge block, not by architecture.** If an encoder scores ≥0.35 on MMLU-4-way and ≥0.85 on SciQ, A ≈ Kev-0.6B at 25% of the parameters and 3–5x the speed; if it scores at chance on MMLU, A ≈ 0.57 and F is the only arm that beats the anchor. E1's per-source table decides this in one run.
3. **KD is the only lever with a plausible +2 to +6 on top of any backbone**, and it is the one Kev never ran. Its direction is supported by seven peer-reviewed analogues (all inference) and one weak local read; its OOD behaviour is silent. It should be the second rung, applied to whichever backbone wins the first.
4. **Coverage@5% will track accuracy, not calibration.** No arm is expected above 0.40; Jev is 0.70. Temperature fixes ECE; it did not move coverage in Kev's measurement (C21). The unknowable-delta pattern (uniform soft targets on evidence-free items) is the one objective-side trick that moved a product metric and is cheap to include in every arm.
5. **On-device: encoders at INT8 are the only path with T1 evidence of zero loss**, but the sole ANE datum (D7) says PTQ is not enough. Decoders at 4-bit carry a measured 3–5 pt MMLU cost at 0.6B and an unmeasured cost on a decision head. Nothing in the corpus quantizes a trained decision model and scores it on a held-out suite.

---

## 5. Gap analysis → experiment ladder

Experiments no source has run and that decide between the top recipes. All scored on Kev's frozen suites (decision-v7 dev, transfer-v4 dev per-source, transfer-v9 dev for unknowable/buried), 3 seeds, record-clustered paired bootstrap vs Kev-0.6B (`v7-06b/02`), Kev v2 tie-aware cov@5%, flip rate at 4 orders, Brier, ECE (raw and post-T). Locked test read once, at the end, on the single promoted checkpoint.

| Rung | Experiment | Decides | Design | Cost (H100-h) | Decision rule |
|---|---|---|---|---|---|
| **E1 — Backbone bake-off** (fills C8, C10, C20, C23, C30) | ModernBERT-base + marker scorer (A); DeBERTa-v3-base + marker scorer (C); SmolLM2-360M + pointer LoRA (D); Gemma-3-270M + pointer LoRA (D'); Qwen3-0.6B + pointer LoRA re-run (anchor); all on decision-v7, hard CE, option shuffling, lr ∈ {1e-4, 5e-5, 2e-5} (encoders: {5e-5, 2e-5, 1e-5}), 2 epochs, 3 seeds at the best lr | A/C vs D vs anchor; the per-block hypothesis of C20 | 5 backbones × 3 lr × 1 seed + 5 × 2 extra seeds ≈ 25 runs at 0.1–0.5 h | 3–6 | Promote the top two by transfer-v4 paired Δ; record MMLU/SciQ/rules per-source rows. If the best encoder is within CI of the anchor on classification+rules and ≥ −8 pp on knowledge, the encoder line continues. |
| **E2 — Distillation arm** (fills VOC (c)3, C16/C25, F's unknown) | On E1's top two: soft-target CE from Kev-4B (1-epoch parent `485ace87…`, T ∈ {1, 2}) and Kev-9B; transfer set = v7 + 5× generated policy states + 5k extra rows from *trainable* sources; consistent shuffles; hard-label replay 20%; also a hard-label-only control on the same enlarged set (separates "more data" from "soft targets") | B/E/F vs A/D; whether KD moves OOD accuracy, coverage, ECE | teacher pass ≈ 2–3 h (9B) / 1.5 h (4B) on ~75k records; 2 backbones × 2 teachers × 2 T × 3 seeds ≈ 24 student runs | 6–12 | Promote if paired Δ ≥ +2 pp with CI excluding 0 *or* cov@5% ≥ +0.05 at equal accuracy; if the hard-label control captures the gain, KD is dropped and "more data" is kept. |
| **E3 — Head and K study** (fills C22, SQ2 silent items) | On the promoted backbone: marker scorer (A) vs Set-Encoder reset (G) vs BERT-MC per-option pass; K ∈ {2, 5, 25, 77} (Banking77 in-dist, synthetic 50-way policy menus); flip rate; latency per K on one M-series machine; per-option token budget sweep {4, 8, 16} | Whether invariance is free; the 50+ option path | 3 heads × 3 seeds + K sweep | 1–3 | Keep the head with ≤1 pp loss and the lowest flip rate; fix the per-option budget where Banking77 stops dropping. |
| **E4 — Post-hoc selective stack** | On the E2 winner: single T vs per-type T vs LAC conformal (α = 0.05) vs Kamath-style calibrator trained on 200 known-OOD items drawn from sources *outside* the holdout (e.g. `snips`, `hate_speech18` — verify against §6 first); plus the unknowable delta (255 evidence-free + 270 controls, 1 epoch, lr 2e-5) | cov@5%, conf-err, ≥0.9-confident-wrong share | CPU + one delta run | 0.1–0.3 | Adopt the stack with the best transfer-v4 cov@5% whose ECE ≤ 0.06 and whose conf-err ≤ 4%. |
| **E5 — Footprint gate** (fills C12, C26, C27, SQ5 silent) | Export the winner: MLX fp16 / 8-bit / 4-bit g64; CoreML fp16 / INT8 per-channel / INT8 QAT (if ANE parity fails) ; ONNX INT8 per-channel; score transfer-v4 dev (n=656) at each precision; argmax agreement; per-question latency on one base M1 and one iPhone (MLX-Swift or CoreML); memory | Whether the on-device claim holds; which precision ships | GPU-free except QAT (≤0.5 h) | 0–0.5 | Ship the smallest precision with ≤1 pp transfer-v4 loss and ≥98% argmax agreement; report the first quantization-vs-held-out-accuracy number in the category. |
| **E6 — Data scaling at small size** (fills C24) | At the promoted size: 3k / 12.6k / 50k records (policy-generated + trainable-source rows), fixed lr, 2 seeds | Whether Kev's "more hurts" or system-one-open's "more is everything" holds below 0.6B | 3 sizes × 2 seeds | 1–3 | Set the production data budget at the knee. |

**Three experiments that most reduce uncertainty.** E1 (the encoder-vs-decoder question has zero measurements on the target suite and separates A/C from D/F by up to 10 pp), E2 (the only lever with a plausible +2–6 and zero OOD measurements anywhere), E5 (nothing in the category has quantized a trained decision model and scored it held-out; the on-device claim is unsupported until this runs). E3 follows because its outcome changes design, not ranking.

---

## 6. Contamination discipline

**Holdout list — exact, from `evals/v4/transfer-v4/manifest.json`, `evals/v9/transfer-v9/manifest.json`, and `kev/data.py:185-280` (VLE (d)).** No proposed training mix, transfer set, calibration set, or known-OOD calibrator set may contain any row of:

| source key | HF dataset (config) | split Kev uses | dev n |
|---|---|---|---|
| `mmlu` | `cais/mmlu` (`all`) | test | 116 |
| `emotion` | `dair-ai/emotion` (`split`) | test | 116 |
| `tweet_offensive` | `cardiffnlp/tweet_eval` (`offensive`) | test | 80 |
| `qnli` | `nyu-mll/glue` (`qnli`) | test | 80 |
| `paws` | `google-research-datasets/paws` (`labeled_final`) | test | 80 |
| `sciq` | `allenai/sciq` | test | 116 |
| `contrastive` | Kev synthetic `authorization` / `deadline` families | — | 0 dev / 80 test (v9) |
| `legacy_holdout` | Kev synthetic legacy policy families | — | 80 dev / 0 test (v9) |
| `composition_holdout` | Kev rule shapes `held_and_or`, `held_or_not`, `held_conditional` (transfer); `final_combination`, `final_negation`, `final_exception` (locked; render style 2) | — | 96 |
| `mmlu_pro` (v9) | `TIGER-Lab/MMLU-Pro` rev `b189ec765aa7ed75c8acfea42df31fdae71f97be`, 10-way | — | 200 |
| `buried` (v9) | Kev-built from `paws`, `qnli`, `tweet_offensive`, `emotion` states | — | 80 |
| `unknowable` / `unknowable_control` (v9) | Kev-built | — | 110 / 110 |

Partition asymmetry to carry: v9 **dev** has `legacy_holdout`, v9 **test** has `contrastive` (VLE (e)16). Trainable sources' *test/validation splits* (`banking77`, `boolq`, `agnews`, `mnli`, `sst5`, `yelp`, `trec`, `dbpedia14`, `amazon`, `imdb`) form Kev's dev/test and must also be excluded from any enlarged transfer set. The v6 knowledge sources (ARC-Challenge, OpenBookQA, CommonsenseQA) were trainable, not holdout — usable, but Kev found them flat on transfer.

**External suites that a training mix would also contaminate:** SemIf 144 (+108 perturbations), scienthoon 900, ekzhang MMLU-Pro 1,000, TypeSafe public eval 372 pairs, JevBench 534 (111 public hard), kotoba-lang typed-decisions (already training data for `laya-typed-decisions`).

**Literature and landscape datasets that overlap the holdout (do not fine-tune on, and do not start from checkpoints tuned on):**
- **QNLI**: every GLUE-tuned number in B1, B3, B9, B10, 3b.2–3b.5, 3b.11, A1–A4, B5(CoFi), C4; Hui & Belkin BERT rows; ZeroGen DistilBERT. Consequence: **start from raw `answerdotai/ModernBERT-base`, `microsoft/deberta-v3-base`, base decoders — never from `gliclass-*`, `*-zeroshot-v2.0`, `*-mnli`, `Ettin-reranker`, or any GLUE-fine-tuned checkpoint.** Verdict started from `gliclass-modern-base-v2.0` (S19), which is a contamination risk for any transfer-v4 read of that model.
- **MMLU / MMLU-Pro**: B13 (Qwen3 report), A6 (quantization eval), H14 (PriDe), 3c.8 (Kumar), system-one-gemma's *training* data includes MMLU (LFL §4), ekzhang. Any Qwen3/Gemma/SmolLM2 base may have seen MMLU test items in pretraining; Kev's manifests certify exact-state dedup only (VLE (c)15). Report MMLU rows with that caveat.
- **Emotion (`dair-ai/emotion`)**: Laya, edgejev, jev-benchmarks pilot, and **Tiny-Jev fitted on 2,000 items** (VFL (c)7). Its Emotion 82.2 is unusable as a zero-shot comparator.
- **TweetEval offensive**: Laya's TweetEval rows.
- **PAWS**: decider's Bespoke-suite PAWS rows, OpenThai's `paws` ECE.
- **SciQ**: no literature overlap found.
- **Known-OOD calibrator data (E4)**: must be drawn from sources absent from both the holdout and the trainable list; verify each candidate against this table before use.

---

## 7. What the evidence does not support

| Hoped-for claim | Why the corpus contradicts it |
|---|---|
| "Same accuracy as Kev-4B (0.80) at 150M" | Capacity 0.6B→4B is +17.5 pp [+13.0, +22.1] (S1 MC). The encoder advantage is one decoder size step at matched pretraining (B1), not 27x. Distillation recovers 93–97% of a 2–12x larger teacher *in distribution* (3b.2–3b.5); the only 27x-class text KD result (Tang, 350x) lands 14 pp below the teacher on MNLI. A fine-tuned 1B decoder scored 56.8 vs 90.2 for a 355M encoder (B5). Nothing supports >0.69 at ≤360M. |
| "Distillation will close the gap to Jev" | Jev is ~10B active (inference), 0.857; Kev-9B is 0.822 with an open teacher ceiling of 0.822/0.852; students reach 80–90% teacher agreement (3b.8, inference); "No Jev outputs were used for training" in Kev and the Jev API is a rounded-probability teacher at best (NanoJev). |
| "A proper scoring rule or focal loss beats CE" | Kev's matched screen: no candidate advances; focal has the best ECE and the worst coverage (S8 MC); post-TS ECE gaps are 1–2 pts (3a.1); square = CE on BERT (3a.4). |
| "Label smoothing is a safe regulariser" | cov@5% 0.532 → 0.006 (S8 MC, text); cov@1% 15.66 → 0.05% (3a.3, vision); LS teachers make worse students (3a.2). |
| "Per-(type, K) temperatures are better" | Worse OOD when K is unseen (S1 MC); Solomon no held-out gain; Laya's gain is on its own calibration partition. |
| "A learned abstention head beats max-softmax" | Nothing beats MaxProb for BERT across IID/OOD/ADV (3c.17 PR-D); SR wins on the same weights in vision (3c.13); Verdict's abstention recall 10–24% under shift. |
| "Temperature scaling raises coverage@5%" | Kev: accuracy and coverage unchanged under T (S1 MC); T is monotone within a question and provably so at K=2; only cross-K reordering is possible (C21). |
| "Set attention over options improves accuracy" | Set-Encoder ≈ pointwise, n.s. (H8); `head_dim`/special-embedding no gain (S1); UniMC blocks it and wins (H3). It buys invariance only. |
| "Read option logits off a frozen small LM" | Frozen Qwen3-0.6B: letter-A prior 37/40, 12/40 correct (S17); PPA ≈ chance at GPT-2 scale (H15); untrained 0.6B 0.567 vs trained 0.620 (S1). |
| "4-bit is free for a sub-1B decoder" | Qwen3-0.6B MMLU −3.1 to −5 at 4-bit g128 (A6); Llama-1B AWQ −3.6 / GPTQ −7.2 (A7); Qwen2.5-0.5B −40% relative on arithmetic (A8); MLX default 4-bit loses to bf16 (A10); the one decision-model datum is a dequantized weight-error bound (A11). Encoder PTQ 4/6-bit palettes failed ANE parity (D7). |
| "INT8 on the Neural Engine is free" | Encoder-weight int8 failed parity for mmBERT-base; only int8 embeddings shipped (D7). Needs QAT (A1/A2) or an accuracy gate. |
| "More public training data helps" | v4 10.9k hurt OOD vs v3 3.4k at 4B (S1 MC, lr 2e-4) — with the caveat that at 270M the opposite holds on its own suite (S20); the ≤0.6B curve on Kev's suite is unmeasured (E6). |
| "Zero-shot Gemma-270M beats trained Gemma-270M" | Majority-label artefact; no zero-shot held-out block exists (VLE (e)8). |
| "Laya beats Jev" | In-distribution fine-tune on the benchmark's own train split; TF-IDF+LogReg is within 2 pp of Jev on that benchmark (S19). |
| "Someone has already beaten Kev-0.6B under 0.5B" | No sub-0.5B open model has a number on Kev's suites other than Kev's own (0.561, 0.605); all sub-0.5B JevBench rows are below 62.5 (VFL (d)). |
| "Ordinal (Score) accuracy can be fixed by an ordinal loss" | Kev's RPS term "did not help" (S1); ordinal-head evidence is vision/tabular only (3a.6–3a.9); no text Score head with ECE exists. |
| "Latency numbers from cards transfer to a base M1" | Every card is a different device (T4, M1 Pro/Max, M3 Max, M5/M5 Pro, GH200, WASM, over-network); §4's M1 column is an inference and E5 replaces it. |

---

## 8. Confidence statement

The anchors (Kev's transfer-v4 numbers, per-source rows, the loss screen, the temperature and coverage rows, the capacity ladder) are measured-controlled on the target suite and reproduce exactly at their cited lines; confidence in them as *recorded* is high, with the standing caveat of one to three seeds and ±4–5 pp CIs on 656 items. The backbone conclusion (encoder ≈ one decoder size step at matched pretraining) rests on two ICLR 2026 papers and is high-confidence *for classification-shaped items*; its application to transfer-v4 is medium-confidence because 53% of the suite is knowledge MCQ and rule composition where no encoder has been measured. The head conclusions (pointer/marker scorer; set attention buys invariance only; letter logits unusable at ≤0.6B zero-shot) are high-confidence. The objective conclusions (CE + single T; LS forbidden; SR as selector) are high-confidence on direction, medium on magnitude (one-seed screen). The distillation arm's expected gain (+2–6) is the least supported number in §4: every supporting result is a reranking, vision, or in-distribution inference, and the one local read is split. The footprint conclusions are high-confidence for 8-bit on both families and for encoder INT8 on CPU/GPU, low-confidence for anything on the ANE or at 4-bit, and absent for a trained decision model. The landscape answer (nothing under 0.5B beats Kev-0.6B; nothing under 0.5B has been measured on Kev's suites) is fully supported and dated to 2026-09-22. The recipe ranges in §4 should be read as ±4 pp bands whose centres are ordered F ≥ B ≥ C ≈ A ≈ G ≥ E(SmolLM2) ≥ D(SmolLM2) > E(Gemma) > D(Gemma), with A–G separated from F by exactly the two unknowns E1 and E2 resolve.
