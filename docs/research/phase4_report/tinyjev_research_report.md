# tinyjev: One Recipe and a Five-Rung Ladder for a Sub-0.3B Typed-Decision Model

**A deep-research synthesis on Kev's frozen suites**

Phase 4 report, initial draft. Deep-research run, full mode. Date: 2026-09-22.

Prepared for: the tinyjev build engineer.

Evidence base: Phase 1 brief (`phase1_scoping/research_question_brief.md`); Phase 2 bibliographies and verification reports for three literature streams and the local stream (`phase2_investigation/lit_*.md`, `verification_*.md`, `local_evidence.md`); Phase 3 synthesis (`phase3_synthesis/synthesis_report.md`). Corrected values from the four verification reports are used throughout; where a needed fact is absent from those materials the text says `[MATERIAL GAP]`.

Path conventions: `KEV/` is the Kev checkout described in `local_evidence.md` §0; `PLAN.md:NN` is a line in `KEV/PLAN.md`. Evidence grades (defined in §6.3): MC measured-controlled; MU measured-uncontrolled; PR-D peer-reviewed, decision-shaped task; PR-A peer-reviewed, analogous task; PP preprint; SR self-report (card, README); IM independently measured by a third party. `[I]` marks a transfer across task, size, device or suite that is an inference rather than a measurement.

---

## Abstract

We asked which backbone, decision head, training objective, data mix and calibration stack let a model of at most 0.6B parameters (target 0.1–0.3B) answer typed decisions (choice over K options, yes/no, ordinal score) with accuracy and calibration approaching the best open Jev-style models, while running under MLX and PyTorch and plausibly on a phone. Three evidence streams were synthesised: Kev's research log and frozen suites (about 110 controlled trials), 161 verified external sources on small-model option scoring, distillation, calibration, selective prediction and quantization, and a dated survey of every open sub-1B replica. The anchors are measured-controlled on Kev's transfer-v4 dev suite (656 clean questions from never-trained sources): Kev-0.6B 0.620, Kev-4B 0.797, Jev 0.857. Two ICLR 2026 papers show a bidirectional encoder is worth roughly one decoder size step on fine-tuned classification at matched pretraining; nothing in the corpus recovers the +17.5 pp capacity gap from 0.6B to 4B at 27x compression on held-out sources. We recommend building one recipe first: ModernBERT-base (149M) with a per-option `[MASK]`-marker shared scorer, trained on Kev's decision-v7 mix with soft-target distillation from the Kev-4B one-epoch parent, a single fitted temperature, and INT8 export. Expected transfer-v4 dev accuracy 0.60–0.69 (central 0.645), coverage at 5% error 0.25–0.40, 15–40 ms per question on a base M1 (inference). A five-rung ladder (10–22 H100-hours, about $40–86 at $3.95/H100-h) settles the two open questions: whether an encoder holds up on the knowledge block, and whether decoder-to-encoder distillation transfers to unseen sources.

**Keywords:** typed decisions; Jev; Kev; ModernBERT; knowledge distillation; calibration; selective prediction; coverage at error budget; quantization; MLX; on-device inference

---

## 1. Recommendation

### 1.1 Build this first: Recipe B

**ModernBERT-base + per-option marker scorer + soft-target distillation from Kev-4B, single temperature, INT8 export.**

*Backbone.* `answerdotai/ModernBERT-base`, 149M parameters, 8,192-token context, raw pretrained checkpoint (Warner et al., 2025; PR-D). Never start from a GLiClass, zero-shot-NLI, MNLI or Ettin-reranker derivative: those were tuned on QNLI-family data that sits in the transfer-v4 holdout (Appendix A).

*Head.* Laya's layout with Kev's semantics. One sequence per question: `[CLS] type + instructions [SEP] [MASK] opt_0 [MASK] opt_1 … [SEP] state [SEP]`. Gather the hidden state at each option's `[MASK]`, pass it through a shared `LayerNorm → Linear(d,d) → GELU → Linear(d,1)`, softmax over the real options. Noul is a two-option choice; Score is the expected level over an ordered option list, exactly as Kev serves it (Palmer, 2026, `kev-0.5b.md`; `REP/laya/laya/common.py:89-136`). No inter-option attention. Give every option at least 8 tokens of budget so that a 77-way menu is about 1k tokens, well inside the 8k context; Laya's 192-token shared budget is why its Banking77 stalls at 0.425 (convaiinnovations, 2026, `BENCHMARKS.md`; MU).

*Objective.* Soft-target cross-entropy, `−Σ t · log p`, which Kev's `question_loss` already implements (`KEV/kev/train.py:37-58`). Targets come from the Kev-4B one-epoch parent (Qwen3.5-4B, checkpoint `485ace87…`, the best-calibrated open teacher in the corpus: transfer-v4 dev 0.797, ECE 0.048 after its own temperature; Palmer, 2026, `runs/calibration-screen-review-v1/summary.json`; MC). Run two teacher temperatures, raw (T=1) and the teacher's fitted serving temperature (about 2.1–2.4, `PLAN.md:51`). Mix 20% hard-label replay. Shuffle option order at train time and use the same shuffle for teacher and student (consistent views; Beyer et al., 2022; PR-A `[I]`). No label smoothing, no focal term, no set attention. The hard-label-only version of this recipe (Recipe A in the frame) is trained alongside as the control that separates "soft targets" from "more data".

*Data.* Kev `decision-v7` training partition: 12,576 records / 15,576 questions, made of 1,000 rows from each of ten public sources (banking77, boolq, agnews, mnli, sst5, yelp, trec, dbpedia14, amazon, imdb), 896 legacy-policy minimal-pair records and 1,680 records from 60 random rule trees (`KEV/evals/v7/decision-v7/manifest.json`; MC). Add a transfer set for the teacher pass: about 5x generated policy states from Kev's rule-tree and minimal-pair generators plus 5k extra rows drawn only from the ten trainable sources' train splits. Keep Kev's `p_none_pair` 0.25 none-of-the-above pairs and the unknowable pattern (uniform soft targets on evidence-free items), the one objective-side change that moved a product metric in the corpus (≥0.9-confident share on evidence-free items 0.19 → 0.00 at 4B with controls unchanged; `PLAN.md:41, 698-707`; MC).

*Calibration and selection.* One scalar temperature fitted by NLL on the decision-v7 calibration split (968 records), applied to every type and K. Softmax response is the selector; coverage at 5% error is Kev's tie-aware v2 metric with record-clustered bootstrap (`PLAN.md:61-62`). Do not fit per-(type, K) temperatures: on unseen K they were worse out of distribution for Kev (`PLAN.md:40`; MC) and gave no held-out gain for Solomon (`PLAN_27b.md:36`).

*Expected numbers (transfer-v4 dev, 656 clean questions).* Accuracy 0.60–0.69, central 0.645. Coverage at 5% error 0.25–0.40. ECE after temperature 0.04–0.07. Brier `[MATERIAL GAP: no encoder Brier on transfer-v4 exists; Kev's accuracy→Brier pairs give 0.45–0.52 as the plausible band]`. Confidence LOW–MED.

*Evidence chain.* Start at Kev-0.6B 0.620 (MC). Split transfer-v4 into its three blocks. Classification-shaped items (emotion, tweet_offensive, qnli, paws; 47% of records): a 150M encoder beats a native 150M decoder by 3.6 MNLI points at matched 2T-token pretraining (Weller et al., 2026; PR-D), and the MLM objective beats CLM by 2.1 points at 210M on sequence classification (Gisserot-Boukhlef et al., 2026; PR-D); a 223M `[MASK]`-head encoder beats 350M–760M decoders by about 18 SuperGLUE points (Schick & Schütze, 2021; PR-D); expect +0 to +5 on Kev-0.8B's 0.64 block `[I: MNLI/SuperGLUE → emotion/qnli/paws/tweet]`. Knowledge MCQ (mmlu, sciq; 30%): zero encoder measurements; Qwen3's 36T-token pretraining carries knowledge that no 2T-token encoder holds; expect −1 to −13 on Kev-0.8B's 0.665 block `[I]`. Rule composition (23%): silent, ±8. Net central −1 pp for the hard-label recipe (0.61), then a distillation lift of +2 to +6 from seven analogues, all inference: a 110M encoder distilled from a 7B ranker gains +3.3/+1.3 nDCG and ends above its teacher (Schlatt et al., 2025b; PR-A); soft logits plus augmentation give +4–5 points over hard labels at 1M scale (Tang et al., 2019; PP); task-specific distillation is TinyBERT's largest ablation term, −7.1 without it (Jiao et al., 2020; PR-D); and the one in-category read, NanoJev pipeline v2, has teacher-distilled OOD 0.773 against gold 0.748 on 400 questions (C-Tianyu, 2026; MC, one seed).

*Sizes and footprint.* fp16 about 300 MB. Publish fp16 for MLX and PyTorch, INT8 per-channel for CPU/ONNX, and INT8 with quantization-aware fine-tuning for CoreML/ANE if post-training INT8 fails the parity gate. The reasons: INT8 QAT is lossless on BERT-class encoders (Zafrir et al., 2019; Kim et al., 2021; PR-D), per-group post-training INT8 costs 0.6 GLUE against 12 for per-tensor (Bondarenko et al., 2021; PR-D), edgejev's INT8 conversion of a 322M encoder cost 1.6 points on AG News (yzfly, 2026; IM), but laya-coreml reports that encoder-weight int8 and 4/6-bit palettes failed its ANE parity gate and only int8 embeddings shipped (FluidInference, 2026; SR, hours old). Do not ship 4-bit until E5 measures it: the only encoder 4-bit datum is W4A8 QAT at −0.4 GLUE on BERT-base (Bondarenko et al., 2021; PR-D `[I]`), and there is no ModernBERT 4-bit number anywhere. Ship the smallest precision with ≤1 pp transfer-v4 loss and ≥98% argmax agreement.

*Latency.* One 4-option question on a 300–450-token sequence: 15–40 ms on a base M1 via MPS or MLX, about 10 ms on CoreML with the Neural Engine. Every figure is an inference: dev-0.4b reports 27.6 ms per forward for a 399M ModernBERT-large on an M1 Max (mpnikhil, 2026; SR); laya-coreml reports 27.5 ms at L512 on CPU+ANE and 9.0 ms on all units for a 322M mmBERT-base on an M5 Pro (FluidInference, 2026; SR); Verdict's 151M model runs at 35.6 ms single-threaded in WASM (Heman10x-NGU, 2026; SR). For comparison, Kev-0.6B takes 123 ms for five questions on an M5 in bf16 (Palmer, 2026, `README.md`; MC), so the encoder should be 3–5x cheaper per question. `[MATERIAL GAP: no source reports ModernBERT-base or any 100–400M encoder on an iPhone; the nearest is DistilBERT 66M at 3.47 ms on an iPhone 13 Neural Engine (Apple Machine Learning Research, 2022; SR).]`

### 1.2 Runner-up and the exact result that switches to it

**Recipe F: Qwen3-0.6B-Base + Kev PointerHead (LoRA r=16) + soft-target CE from Kev-4B/9B with hard-label replay.** Expected 0.62–0.68 (0.65), coverage 0.25–0.40, MED confidence. It is Kev-0.6B with the one lever Kev never pulled, and its evidence is stronger per number (same base, same suite) but it sits at the size ceiling, costs 3–5 MMLU points at 4-bit (Zheng et al., 2025; PP), is 1.2 GB in fp16, and carries the Qwen3 temperature-transfer question (contradiction C1).

Switch to F if E1 returns either of these, at the best learning rate with three seeds, paired record-clustered bootstrap against the re-run Qwen3-0.6B anchor: (a) the best encoder's knowledge block (mmlu 116 + sciq 116 dev records) is more than 8 pp below the anchor's; or (b) its classification-plus-rules blocks are below the anchor with a CI excluding zero. If only (a) fires and the shortfall is under 8 pp, keep the encoder line and add the 400m tier (§1.4). If E1 passes and E2 shows the hard-label control captures the whole distillation gain, ship Recipe A with the enlarged data and drop soft targets.

### 1.3 What the evidence does not support

Say none of the following in a model card or a launch post.

1. *"Same accuracy as Kev-4B at 150M."* Capacity 0.6B→4B is +17.5 pp [+13.0, +22.1] on transfer-v3 dev (`PLAN.md:730-731`; MC, lr 2e-4, one seed). The encoder advantage is one decoder size step at matched pretraining (Weller et al., 2026), not 27x. Distillation recovers 93–97% of a 2–12x larger teacher in distribution (Sanh et al., 2019; Jiao et al., 2020; Wang et al., 2021); the one 350x text result lands 14 points below its teacher on MNLI (Tang et al., 2019). A fine-tuned 1B decoder scored 56.8 against 90.2 for a 355M encoder on pair classification (Abbes et al., 2025; PP). Nothing supports above 0.69 at ≤360M.
2. *"Distillation closes the gap to Jev."* Jev is 0.857 on transfer-v4 dev and has never been run on the locked test (`README.md:130`); the open teacher ceiling is 0.822 dev / 0.852 test (Kev-9B); students agree with teachers 80–90% of the time (Stanton et al., 2021; PR-A `[I]`).
3. *"A proper scoring rule or focal loss beats cross-entropy."* Kev's matched five-arm screen: no candidate advances; focal has the best ECE (0.043) and the worst coverage (0.502) (`runs/calibration-screen-review-v1/summary.json`; MC, one seed at 4B). Post-temperature ECE gaps are 1–2 points on vision and about zero on text (Mukhoti et al., 2020; PR-A).
4. *"Label smoothing is a safe regulariser."* Coverage at 5% error 0.532 → 0.006 at ε=0.05 (same screen; MC); coverage at 1% risk 15.66% → 0.05% on ImageNet (Xia et al., 2025; PR-A `[I]`).
5. *"Per-(type, K) temperatures are better."* Worse OOD for Kev when K is unseen (`PLAN.md:40`); Laya's gain (ECE 0.466 → 0.081) is on its own calibration partition with a degenerate `choice:11+` temperature of 0.10.
6. *"A learned abstention head beats max-softmax."* Nothing consistently beats MaxProb for BERT-base across IID/OOD/adversarial settings (Varshney et al., 2022; PR-D); abstention heads' gains come from a better classifier, not a better selector (Feng et al., 2023; PR-A); Verdict's abstention recall falls to 10–24% under shift (Heman10x-NGU, 2026; SR).
7. *"Temperature scaling raises coverage."* Kev: accuracy and coverage unchanged under T (`PLAN.md:40`; MC). T is monotone within a question and provably so at K=2.
8. *"Set attention over options improves accuracy."* Set-Encoder-330M 0.727/0.789 (DL19) against pointwise monoELECTRA-330M 0.733/0.765, not significant (Schlatt et al., 2025a; PR-A); Kev's `head_dim` and special-embedding variants gained nothing (`PLAN.md:341`; MC). It buys invariance only.
9. *"Read option logits off a frozen small LM."* Frozen Qwen3-0.6B: letter-A prior 37/40, 12/40 correct (r-ms, 2026, `PREREG.md` §0; MU); symbol binding is near chance at GPT-2 scale (Robinson et al., 2023; PR-D).
10. *"4-bit is free for a sub-1B decoder"* and *"INT8 on the Neural Engine is free."* See §1.1 and §7.5.
11. *"Someone under 0.5B already beats Kev-0.6B."* No sub-0.5B open model has a number on Kev's suites other than Kev's own (0.561; 0.605), and every sub-0.5B JevBench row is below kev 0.6B's 62.5 (fstandhartinger, 2026; IM). Snapshot 2026-09-22.

### 1.4 Naming, sizes, expected numbers

| Name | Backbone (raw checkpoint) | Params | Expected transfer-v4 dev acc | cov@5% | ECE post-T | Base-M1 latency, 1 question | Confidence |
|---|---|---|---|---|---|---|---|
| **tinyjev-150m** (build first) | ModernBERT-base | 149M + 1–15M head | 0.60–0.69 (0.645) with KD; 0.57–0.65 (0.61) hard-label | 0.25–0.40 | 0.04–0.07 | 15–40 ms `[I]` | LOW–MED |
| tinyjev-400m (accuracy tier, after E1) | ModernBERT-large | 395M + head | 0.62–0.71 `[I: Ettin 150M→400M encoder MNLI +2.1; Laya's 421M has no transfer-v4 number]` | 0.30–0.45 `[I]` | as 150m | 40–70 ms `[I: dev-0.4b 27.6 ms on M1 Max]` | LOW |
| tinyjev-70m (hold until E5 shows the phone needs it) | Ettin encoder 68M | 68M + head | 0.55–0.63 `[I: Ettin 68M→150M MNLI −2.2]` | 0.15–0.30 `[I]` | 0.05–0.10 | ≤10 ms `[I: DistilBERT 66M, 3.47 ms, iPhone 13 ANE]` | LOW |

Publish 150m and 400m. The 70m row exists so the phone question has a named answer, not because the evidence supports it yet.

---

## 2. Experiment ladder E1–E5

All rungs score on Kev's frozen suites: `decision-v7` dev (1,204 records / 1,468 questions) for in-distribution, `transfer-v4` dev (764 records / 656 clean questions) per source for held-out, `transfer-v9` dev for unknowable and buried items. Metrics on every rung: accuracy, Brier, ECE raw and after temperature, coverage at 5% error (Kev v2, tie-aware), argmax flip rate at four option orders, and the per-source table. Three seeds at the chosen setting; paired record-clustered bootstrap against the anchor. The locked test is read once, at the very end, on the single promoted checkpoint, following Kev's protocol of one read per promoted checkpoint (`KEV/runs/locked/*/summary.json`; `PLAN.md:683-690`). Base checkpoints are raw pretrained weights only; the contamination rule in Appendix A forbids any checkpoint tuned on GLUE/QNLI, MNLI, Emotion, TweetEval, PAWS, SciQ, MMLU or MMLU-Pro.

Cost assumptions. One Kev-0.6B LoRA trial takes about 4.5 min on one H100 (`PLAN.md:251`); Kev-4B trials 13.5–63 min; teacher forward passes cost 0.11 s/record at 4B and 0.18 s/record at 9B in row form (`PLAN.md:655-659`); a 149M encoder full fine-tune on decision-v7 is estimated at 0.1–0.5 H100-h from Laya's 421M run (30k questions × 4 epochs on 2×T4 in 4.5 h, about 0.6–1.2 H100-h, scaled by parameters). The dollar line uses **$3.95 per H100-hour**, which is Kev's metered Modal rate (`PLAN.md:251`; MC) and is an assumption to confirm against the current Modal price list before anything is launched.

| Rung | Hypothesis | Arms | Data | Eval | Decision rule | H100-h | Cost at $3.95 |
|---|---|---|---|---|---|---|---|
| **E1** Backbone bake-off | An encoder matches Kev-0.6B on classification and rules and loses less than 8 pp on knowledge | ModernBERT-base + marker scorer (A); DeBERTa-v3-base + marker scorer (C); SmolLM2-360M + pointer LoRA (D); Gemma-3-270M + pointer LoRA (D′); Qwen3-0.6B-Base + pointer LoRA (anchor re-run). Hard CE, option shuffling, 2 epochs; lr sweep {5e-5, 2e-5, 1e-5} encoders, {1e-4, 5e-5, 2e-5} decoders; 3 seeds at best lr | decision-v7 train | decision-v7 dev; transfer-v4 dev per source | Promote the top two by paired Δ. Encoder line continues if within CI of anchor on classification+rules and ≥ −8 pp on knowledge; else switch to F | 3–6 | $12–24 |
| **E2** Distillation | Soft targets from Kev-4B lift OOD accuracy or coverage beyond what the same extra data gives with hard labels | On E1's top two: soft-target CE from Kev-4B (`485ace87…`) and Kev-9B, teacher T ∈ {1, served}; 20% hard replay; consistent shuffles; plus hard-label control on the identical enlarged set | v7 + 5x generated policy states + 5k trainable-source rows | as E1, plus transfer-v9 unknowable | Promote if paired Δ ≥ +2 pp with CI excluding 0, or cov@5% ≥ +0.05 at equal accuracy; if the hard-label control captures the gain, drop KD and keep the data | 6–12 | $24–47 |
| **E3** Head and K | Invariance is free; per-option token budget sets the 50+ option ceiling | On the promoted backbone: marker scorer (A) vs Set-Encoder-style per-segment reset (G) vs BERT-MC per-option pass; K ∈ {2, 5, 25, 77}; budget ∈ {4, 8, 16} tokens/option | decision-v7 (Banking77 in-dist) + synthetic 50-way policy menus | transfer-v4 dev; flip rate; latency per K on one M-series machine | Keep the head with ≤1 pp loss and lowest flip rate; fix the budget where Banking77 stops falling | 1–3 | $4–12 |
| **E4** Post-hoc stack | A single T plus softmax response is the best selector; the unknowable delta cuts confident errors | Single T vs per-type T vs LAC conformal (α=0.05) vs Kamath-style calibrator on 200 known-OOD items from sources outside Appendix A; unknowable delta (255 evidence-free + 270 controls, 1 epoch, lr 2e-5) | calibration split; delta records | transfer-v4 dev cov@5%, ECE, conf-err; transfer-v9 ≥0.9-confident share on unknowable | Adopt the stack with best cov@5% whose ECE ≤ 0.06 and conf-err ≤ 4% | 0.1–0.3 | $0.40–1.20 |
| **E5** Footprint gate | INT8 is lossless on CPU/GPU; ANE needs QAT; 4-bit is not free | Export: MLX fp16 / 8-bit / 4-bit g64; CoreML fp16 / INT8 per-channel / INT8 QAT if parity fails; ONNX INT8 per-channel | none (QAT ≤0.5 h) | transfer-v4 dev (n=656) at each precision; argmax agreement; per-question latency on one base M1 and one iPhone (MLX-Swift or CoreML); memory | Ship the smallest precision with ≤1 pp loss and ≥98% argmax agreement | 0–0.5 | $0–2 |
| **Total** | | | | | | **10–22** | **$40–86** |

**E1, backbone bake-off (fills contradictions C8, C10, C20, C23, C30).** This is the run with the highest information per dollar in the whole programme, because the encoder-versus-decoder question has zero measurements on the target suite (Kev proposed it at `PLAN.md:634` "for ~$5" and never ran it) and the per-block hypothesis of C20 predicts the encoder wins the 47% classification block and loses the 30% knowledge block. Five backbones, three learning rates each, one seed, then two extra seeds for the five best cells: about 25 runs at 0.1–0.5 h. The learning-rate sweep matters because the only clean ≤1B mutation on Kev's leaderboard goes up (3e-4 → 0.595, `arch-06b/05`) and the 2e-4 → 1e-4 comparison changed the data too, so no low-lr result exists at 0.6B. Include a zero-shot `[MASK]` probe of the untrained encoder on transfer-v4 as a floor, the way Kev probes its bases (untrained Qwen3-0.6B letter logits 0.567, `PLAN.md:503-508`). Report the per-source table for every arm; the decision rule reads it, not the mean.

**E2, distillation arm (fills VOC (c)3, C16, C25).** The lever with a plausible +2 to +6 and no OOD measurement anywhere. Teacher pass: Kev-9B at 0.18 s/record over about 75k records is 3.75 h, Kev-4B about 2.3 h; then 2 backbones × 2 teachers × 2 temperatures × 3 seeds, 24 student runs at 0.1–0.5 h. The hard-label control on the same enlarged set is not optional: TinyBERT's ablation attributes more of its gain to augmentation than to soft targets (Jiao et al., 2020), Mishra et al. (2023) find augmentation rather than soft targets drives student calibration (PR-A `[I]`), and system-one-open's 270M went from 0.367 to 0.585 held-out just by lifting the per-task cap from 2,500 to 12,000 (mithalouni, 2026; MC, single run). Use the 4B parent, not the 9B, as the primary teacher: it is better calibrated, cheaper, and bigger teachers do not make better students (Cho & Hariharan, 2019; PR-A `[I]`); teacher calibration error predicts student accuracy with R² about 0.92 in vision (Kim et al., 2025; PR-A `[I]`).

**E3, head and K study (fills C22 and SQ2's silent items).** Outcome changes design, not ranking, so it runs third. The three heads differ only in whether options see each other and whether positions reset per option. Set-Encoder's per-candidate position reset gives exact invariance at no measured cost on ELECTRA (Schlatt et al., 2025a), but nobody has done it on a RoPE encoder without re-pretraining, so G is a test, not a bet. Flip rate at four orders is the primary number: Laya shows 0.15–0.23 at K=20, Kev-0.6B 0.07, Jev 0.00 (`KEV/runs/jev-transfer-v4/report.json`). The K sweep answers the 50+ option question that GLiClass answered for labels (1 → 128 labels costs 7% throughput in one pass against a 52x slowdown for per-label cross-encoding; Stepanov et al., 2025; PP).

**E4, post-hoc selective stack.** Runs on CPU except the one delta fine-tune. The single-T default is well supported (Guo et al., 2017; Chen et al., 2023; `PLAN.md:40`); the two arms that could beat it both need out-of-holdout data: LAC conformal sets at α=0.05, which reached 91–94% coverage with set sizes 2.4–3.7 on 4-option MMLU with a 13B model (Kumar et al., 2023; PP `[I]`), and a Kamath-style calibrator, which recovered 48.2 → 56.1 coverage at 80% accuracy with known-OOD items on extractive QA (Kamath et al., 2020; PR-D `[I]`). Verify every candidate known-OOD source against Appendix A before use. The unknowable delta is included in every promoted arm because it is cheap (9–15 min at 4B; less here) and it is the only thing in the corpus that moved the ≥0.9-confident-wrong share.

**E5, footprint gate (fills C12, C26, C27).** Mandatory before any on-device claim, because nothing in the category has quantized a trained decision model and scored it on a held-out suite: NanoJev's 99.17% argmax agreement is a dequantized-to-fp32 weight-error bound on a dev split (ZeroDegress, 2026; SR), Kev has no quantized artefact (`PLAN.md:29, 593`), and decider's only sub-1B datum is fp16 parity on MPS. Score transfer-v4 dev at every precision, not a 100-item slice: edgejev's kev 0.5B INT8 drop (AG News 90 → 84, Emotion 44 → 21) is at n=100 with per-tensor dynamic quantization, exactly the quantizer LLM.int8() showed breaks OPT-125M (perplexity 25.65 → 87.76; Dettmers et al., 2022; PR). Report argmax agreement per bucket plus max |Δp|, the parity gate laya-coreml uses. Measure latency on one base M1 and one iPhone with the same harness, replacing every device-mixed number in §1.1.

*Optional E6, data scaling at the promoted size (1–3 H100-h).* Three data sizes (3k / 12.6k / 50k records) at fixed lr, two seeds, to settle C24 (Kev's "more public data hurts at 4B, lr 2e-4" against system-one-open's "more is everything at 270M"). E2's hard-label control already gives one point on that curve.

---

## 3. Risk register

| # | Risk | Early detector (ladder measurement) | Mitigation |
|---|---|---|---|
| 1 | The encoder is near chance on the knowledge block (mmlu + sciq = 30% of transfer-v4), so the mean lands at 0.57 and below the anchor | E1 per-source rows: mmlu-4-way < 0.35 or sciq < 0.85 for the best encoder; knowledge block > 8 pp below the anchor | Switch to Recipe F (§1.2); or ship the encoder for classification and policy workloads and add the 400m tier; never promise knowledge MCQ |
| 2 | Soft-target distillation does not transfer to held-out sources (no source measures it; NanoJev is split across its two reads) | E2 paired Δ against the hard-label control on the identical enlarged set; cov@5% delta | Keep the enlarged data, drop the soft targets; try attention-transfer distillation, which beat logit-only KD by 2–6 GLUE points (Wang et al., 2023; PR-D) |
| 3 | Coverage at 5% error stays near the floor (0.15–0.30) whatever the calibration, because 59% of errors sit on the three noisy-label sources (37% of items) | E4 cov@5% on transfer-v4 with the per-source error share; ≥0.9-confident-wrong count | Unknowable delta; Kamath-style calibrator with known-OOD items; report coverage separately from ECE and do not promise it from calibration |
| 4 | Pretraining contamination of mmlu/qnli/paws/emotion in the base checkpoint (every manifest certifies exact-state dedup only) | E1 zero-shot `[MASK]` floor per source; a source that jumps far above its trained neighbours flags leakage; any GLUE-tuned lineage in the checkpoint name | Raw checkpoints only; report MMLU rows with the caveat; never start from `gliclass-*`, `*-zeroshot-v2.0`, `*-mnli` or Ettin-reranker weights |
| 5 | INT8 fails the ANE parity gate (laya-coreml: encoder-weight int8 and 4/6-bit palettes failed; only int8 embeddings shipped) | E5 argmax agreement 16/16 per length bucket and max |Δp|; transfer-v4 loss at each precision | INT8 QAT (Zafrir et al., 2019; Kim et al., 2021); ship fp16 on ANE and INT8 on CPU/GPU; hold 4-bit |
| 6 | Option-count ceiling and order sensitivity: Laya's shared 192-token budget capped Banking77 at 0.425 and flipped 15–23% of K=20 answers | E3 Banking77 accuracy across budgets {4, 8, 16}; flip rate at four orders | ≥8 tokens per option; train-time shuffling plus Kev's `perm_kl`; adopt the per-segment reset head only if E3 shows ≤1 pp cost |
| 7 | Small-model over-confidence and a temperature that does not transfer (Kev-0.6B raw dev ECE 0.15; Qwen3 cards say in-domain T "does not transfer", Qwen3.5 rows say it does) | E4 ECE after T on transfer-v4 against in-distribution; conf-err at ≥0.9 | Fit one T on the calibration split; re-measure transfer on the chosen base; per-type T only if the deployment K distribution is known and matched |
| 8 | Wrong data regime: more rows hurt Kev-4B at lr 2e-4 (10.9k 0.704/0.735 vs 3.4k 0.747/0.750) but helped a 270M model (0.367 → 0.585); the ≤0.6B curve on Kev's suite is unmeasured | E2 hard-label control vs E1 baseline (one point on the curve); optional E6 knee | Low learning rate; no oversampling of any single family (Kev's synthetic ×3 hurt); set the production data budget at E6's knee |

---

## 4. Introduction

Jev-style models answer typed decisions: given a state, a question, and a small set of options, return a calibrated distribution over the options in one forward pass, with no generation. Three types cover the product surface: `choice` over K options, `noul` (yes/no), and `score` (ordinal level). The hosted reference, Jev 1.13.0, scores 0.857 on Kev's transfer-v4 dev suite (Palmer, 2026, `runs/jev-transfer-v4/report.json`; MC) and is inferred from latency signatures to be a prefill-only causal model with about 10B active parameters (Hume, 2026; SR, blog inference, not citable as fact). The best open replica, Kev-4B, reaches 0.797 dev / 0.837 locked test; Kev-0.6B reaches 0.620 / 0.642. The category is nine days old at the time of the brief (2026-09-22); every model card in the survey was created between 2026-09-17 and 2026-09-22 (`verification_footprint_landscape.md` (c) 16).

tinyjev is our own family of typed-decision models at 0.1–0.3B (ceiling 0.6B), to be published for MLX and PyTorch and, if the numbers allow, run on a phone. The engineering question is not whether such a model can be built; several exist. It is which recipe to build first, and which experiments decide the questions that the existing evidence leaves open. The main research question (Phase 1 brief) asks which combination of backbone, head, objective and data gets closest to Kev-4B and Jev at that size, with accuracy and calibration measured on frozen public suites rather than self-reported in-domain splits. Six sub-questions structure the work: backbone (encoder against small decoder), head (pointer, set attention, marker scorer, letter logits), objective and calibration (cross-entropy against proper scoring rules and distillation; how to raise coverage at a 5% error budget), data (mix, volume, contamination discipline), footprint (quantization, pruning, nested sizes, phone latency), and the competing open models (has anyone already beaten Kev-0.6B under 0.5B).

The Phase 1 Devil's Advocate recorded two cautions that this report honours. First, the framing "same accuracy, tiny" presumes the gap is closable; Kev's own scaling ladder is the null hypothesis, and §1.3 states what a 0.1–0.3B model realistically reaches. Second, most in-category evidence is self-reported from repositories built in a week, so peer-reviewed results on analogous tasks are weighted above in-category READMEs, and every such transfer is labelled `[I]`.

---

## 5. Literature review

Organised by the six sub-questions. Each paragraph states what converges, what diverges and why, and where the literature is silent. Grades follow §6.3.

### 5.1 Backbone

Two ICLR 2026 papers settle the matched-pretraining question. Ettin trains paired encoders and decoders at 17M–1B on the same 2T tokens and fine-tunes both on MNLI: the 150M encoder scores 89.2 against the native 150M decoder's 85.6, the 400M encoder 91.3 against the 1B decoder's 89.9, and adapting a decoder to MLM afterwards does not close the gap (85.8) (Weller et al., 2026; PR-D). Gisserot-Boukhlef et al. (2026; PR-D) hold the architecture fixed and vary only the objective: MLM beats CLM on sequence classification by 2.1, 3.4 and 5.6 points at 210M, 610M and 1B, and continued MLM pretraining of a CLM checkpoint is the cheap route to a strong small encoder. Three further sources point the same way: fine-tuned DeBERTa-v3-base and -large out-rank few-shot GPT-4 on English ScandEval (mean rank 1.29/1.09 against 1.44; Nielsen et al., 2025; PR-D `[I]`, decoders not fine-tuned); RoBERTa-large reaches 90.2 on SQuAD-v2 groundedness where fine-tuned Llama-3.2-1B reaches 56.8 and 3B 82.2, at about 15x fewer inference FLOPs than an 8B decoder (Abbes et al., 2025; PP); and PET's 223M `[MASK]`-verbalizer encoder scores 74.0 on SuperGLUE test against 71.8 for GPT-3 175B and 56.2/56.8 dev for the 350M/760M decoders (Schick & Schütze, 2021; PR-D).

The divergence is between those papers and the reranking literature. Qwen3-Reranker-0.6B, a decoder scored by yes/no logits, beats older 0.3–0.6B encoders on MTEB-R (65.80 against 57.03 for bge-reranker-v2-m3; Zhang et al., 2025; PP, vendor `[I]`), while a freshly distilled 150M Ettin encoder equals that same decoder (0.5994 against 0.5940) at 2.3x the throughput of same-size peers (Aarsen, 2026; SR, single author, vendor `[I]`). The mechanism is uncontrolled training data: the encoder edge survives only when training is matched, and Qwen3's 36T-token pretraining (Yang et al., 2025; PP) buys knowledge that no 2T-token encoder has. Within the decoder family, Kev's capacity ladder dominates every other lever: 0.6B → 4B is +17.5 pp [+13.0, +22.1] and 0.6B is "saturated at 0.59–0.61 regardless of knobs" over eight single-factor mutations (`PLAN.md:282-283, 730-731`; MC); what small decoders lose is knowledge and date arithmetic, not the decision format (decider's 0.8B against 2B: held-out 0.707 against 0.739, gap concentrated on TruthfulQA, OpenBookQA and ARC; Mapika, 2026; MU). The literature is silent on any encoder scored on transfer-v4 or on 4-way knowledge MCQ at 150–400M.

### 5.2 Head

Four sources converge on the first point: zero-shot letter-logit readout is not viable at ≤0.6B. Symbol binding is near chance at GPT-2 scale (Robinson et al., 2023; PR-D), it is a late-emerging capability carried by one to four attention heads per layer (Wiegreffe et al., 2025; PR-D), a frozen Qwen3-0.6B places 37 of 40 answers on "A" (r-ms, 2026; MU), and the untrained base scores 0.567 on transfer-v4 against 0.620 for the trained pointer (`PLAN.md:503-508`; MC). Reordering options moves accuracy by 13–75% across models (Pezeshkpour & Hruschka, 2024; PR-D), and the bias is a token prior on the ID symbol rather than a position effect; PriDe's 5%-sample estimate removes +1.2/+1.3/+1.7 points of it on MMLU/ARC/CSQA at 1.15x cost (Zheng et al., 2024; PR-D, ≥1.5B models). Pointer and marker heads never touch the letter token, which is why Kev never needed PriDe.

The second convergence is that a shared scorer over per-option marker states, with no inter-option attention, is sufficient and well precedented: BERT's multiple-choice head (Devlin et al., 2019; PR-D), PET (Schick & Schütze, 2021), UniMC's one `[O-MASK]` per option with an attention mask that blocks options from attending to each other, which beats 11B–540B decoders zero-shot on ANLI (Yang et al., 2022; PR-D), and the one-pass label tokens of GLiNER and GLiClass (Zaratiana et al., 2024; Stepanov et al., 2025). Set attention adds nothing measurable on relevance: Set-Encoder-330M at 0.727/0.789 (DL19) and 0.735/0.790 (DL20) against monoELECTRA-330M's 0.733/0.765 and 0.727/0.799 is not significant (Schlatt et al., 2025a; PR-A `[I]`), and its authors attribute the null to relevance labels that carry no inter-passage signal, noting that set attention helps once the loss requires comparison. Kev's `head_dim` 1024 and trainable delimiter embeddings gained nothing (`PLAN.md:341`; MC, one seed). What the literature does not contain is a controlled comparison of set attention against a shared scalar scorer on typed decisions with a trained backbone; NanoJev has both code paths and never ablated them.

### 5.3 Objective and calibration

Once a temperature is fitted, the training loss barely moves ECE. Mukhoti et al. (2020; PR-A) train the same nets under CE, Brier, label smoothing and focal loss: on 20 Newsgroups the post-temperature ECEs are 2.39, 3.22, 2.54 and 2.19, label smoothing improves accuracy slightly and focal costs 1.3 points; on SST-Binary focal is worse than CE before scaling. Square loss equals or beats CE on 12 of 14 NLP accuracy tasks including BERT fine-tuning (MRPC 83.8 against 82.1; Hui & Belkin, 2021; PR-D). Temperature scaling alone takes SST-binary ECE from 6.63 to 1.84 without changing the argmax (Guo et al., 2017; PR). Label smoothing is the exception in the wrong direction: it destroys the confidence ranking, taking coverage at 1% risk from 15.66% to 0.05% on ImageNet (Xia et al., 2025; PR-A `[I]`) and making a worse distillation teacher (Müller et al., 2019; PR-A `[I]`).

For distillation, the canonical encoder results recover 97% (DistilBERT, 66M from 110M; Sanh et al., 2019; PP), 96.8% (TinyBERT4, 14.5M; Jiao et al., 2020; PR-D) and about 93% (MiniLMv2, 30M from 355M; Wang et al., 2021; PR-D) of the teacher in distribution; Turc et al. (2019; PP) show pre-training and distillation compound and that transfer-set size and domain govern the gain. The signal reaches small students from decoder teachers: a 110M cross-encoder distilled from a 7B ranker ends above its teacher (Schlatt et al., 2025b; PR-A), a DeBERTa-v3-large distilled from GPT-4 permutations beats monoT5-3B on BEIR (Sun et al., 2023; PR-A), and rationale supervision lifts a 220M T5 on ANLI from 43.6 to 49.6 (Hsieh et al., 2023; PR-D, seq2seq). The cautions are all vision: students agree with teachers only 80–90% of the time (Stanton et al., 2021), bigger teachers make worse students unless early-stopped (Cho & Hariharan, 2019), and closing a 4x gap took 9,600 epochs of consistent-view function matching (Beyer et al., 2022). LLM hard labels are about 0.04 F1 worse than human labels (Pangakis & Wolken, 2024; PR-D), and purely synthetic data sits far below supervised on NLI and QA (Ye et al., 2022; PR-D).

For selection, nothing beats max-softmax on a fixed classifier. Softmax response beats MC-dropout for BERT and larger encoders rank better (Xin et al., 2021; PR-D); nothing consistently beats MaxProb for BERT-base across IID, OOD and adversarial settings (Varshney et al., 2022; PR-D); learned abstention heads' gains vanish when their selector is replaced by softmax response on the same weights (Feng et al., 2023; PR-A), and no method beats maximum softmax across shifts in FD-Shifts (Jaeger et al., 2023; PR-A). Temperature scaling does little under shift for BERT (12.62 → 12.83 ECE; Desai & Durrett, 2020; PR-D) and is the best unlearnable method in domain (Chen et al., 2023; PR-D). Two results do raise coverage: knowledge distillation gives the largest median gain in AUROC and ECE across 523 ImageNet models (Galil et al., 2023; PR-A `[I]`), and a small calibrator trained with known-OOD data lifts coverage at 80% accuracy from 48.2 to 56.1 on extractive QA (Kamath et al., 2020; PR-D `[I]`). Conformal sets are available as a wrapper: RAPS sets at 95% coverage are 4.21–11.7 wide against 22.5–46.3 for APS on ImageNet (Angelopoulos et al., 2021; PR), and LAC on a 13B model's 4-option MCQ softmax reaches 91–94% coverage with sets of 2.4–3.7 (Kumar et al., 2023; PP). No source trains a 100–600M encoder with a proper scoring rule and reports coverage at risk; no source measures soft-target distillation from a decoder into an encoder with an ECE or coverage metric.

### 5.4 Data

Kev's own log is the primary source. The mix behind every Kev number is decision-v7: breadth of decision-shaped public sources plus generated policy data, and the policy data transfers: compositional pairs are worth +3.6 pp [−0.4, +7.8] at 0.6B and +5.1 [0.0, +9.9] at 4B, and rule trees lift held-out `(A or B) and C` from 0.69 to 0.91 and `or_not` from 0.44 to 0.75 at 4B (`PLAN.md:693, 730-733`; MC). More of the same public data hurt out of distribution at 4B and lr 2e-4 (10.9k records 0.704/0.735 against 3.4k 0.747/0.750, two seeds; `PLAN.md:284-286`), oversampling synthetic data ×3 hurt, and knowledge MCQ sources raised MMLU by 2–5 points but left transfer flat (`PLAN.md:310-311`; MC, one seed). The mechanism is drift at high learning rate; the finding is not replicated at low lr or below 1B. The opposite regime appears at 270M: system-one-open's Gemma-3-270M goes from 0.367 held-out with 2,500 rows per task to 0.585 with 12,000 (mithalouni, 2026; MC, single run), and Turc et al. (2019) show an 11M student recovers its teacher only with an 8M in-domain transfer set. Contamination discipline is exact-state deduplication everywhere; no manifest certifies fuzzy or pretraining decontamination (`KEV/evals/**/manifest.json`; Heman10x-NGU, 2026, E6).

### 5.5 Footprint

Eight-bit is free for both families when the quantizer respects outliers. On encoders, INT8 QAT is lossless (Q8BERT; I-BERT RoBERTa-base 86.0 → 86.3; Zafrir et al., 2019; Kim et al., 2021; PR-D), per-tensor W8A8 post-training breaks BERT-base (83.06 → 71.03) while per-group fixes it (82.45) and W4A8 QAT costs 0.4 (Bondarenko et al., 2021; PR-D). On decoders, naive absmax int8 breaks OPT-125M (perplexity 25.65 → 87.76) and vector-wise quantization repairs it (Dettmers et al., 2022; PR); Qwen3-0.6B loses nothing at 8-bit (47.1 → 47.0 MMLU) and 3.1–5.0 points at 4-bit with g128 grouping (GPTQ 44.0, AWQ 42.1), collapsing at 3-bit (Zheng et al., 2025; PP, corrected to the per-group table); Llama-3.2-1B loses 3.6 (AWQ) to 7.2 (GPTQ) at 4-bit and small models suffer more (Lee et al., 2025; PR); Qwen2.5-0.5B loses 40% relative on arithmetic at 4-bit (Srivastava et al., 2026; PR-A); and MLX's default 4-bit loses to bf16 on code generation with the loss shrinking at +0.52 points per billion parameters (Melton, 2026; PP, 12 days old). Latency at 100–600M is tens of milliseconds per question on Apple silicon, every figure on a different device (§1.1). One-run-many-sizes methods exist (MatFormer's nested 582M–850M submodels, Devvrit et al., 2024; LayerDrop's 12 → 6 layer RoBERTa at MNLI 82.9 against DistilBERT's 81.6, Fan et al., 2020; PR) but none has been measured for calibration. The divergence that matters is on the Neural Engine: the peer-reviewed INT8 results are GPU/CPU with QAT or per-group scaling, and the one ANE datum (FluidInference, 2026; SR) reports post-training encoder-weight int8 and 4/6-bit palettes failing a parity gate stricter than a GLUE accuracy gate.

### 5.6 Competing open models (the brief's "landscape" sub-question)

On Kev's frozen suites no external sub-0.5B model has published any number; the only sub-0.5B rows are Kev's own Qwen2.5-0.5B ablation (transfer-v2 dev 0.605, v2 data) and Kev-0.5B v0.1 (transfer-v4 dev 0.561), both below Kev-0.6B (Palmer, 2026; MC). On the one independent cross-system benchmark, JevBench v1.3 (534 decisions), every sub-0.5B system sits below kev 0.6B's 62.5: Laya 421M 54.4 (hard tier 34.1%), jeff 54.4, Verdict 151M 38.9 (hard 37.7%), kev 0.5B 33.2 (hard 30.9%), GLiNER2.5 16.6/13.8 (fstandhartinger, 2026; IM). Laya and Verdict beat kev 0.6B on the calibration axis and lose on intelligence. Self-reported in-domain numbers do not survive a foreign suite: Verdict's 0.95 on its own Banking77+CLINC150 test becomes 0.48 on 337 TypeSafe cases (Heman10x-NGU, 2026); Laya's 0.766 on typed-decisions is a checkpoint fine-tuned on that benchmark's own train split, against 0.362 zero-shot; Tiny-Jev's public-benchmark rows were fitted on 2,000 items per dataset (lostargon, 2026; SR); decider-0.8b drops from 0.776 in-task to 0.707 held-out (Mapika, 2026). The JevBench composite ranks kev 0.6B above kev 4B and 8B, a geometric-mean artefact of cost and calibration axes; its intelligence axis orders 8B > 4B > 0.6B, and its kev rows are pre-v7 Qwen3 previews truncated at 384 tokens on a long-text hard tier (`verification_local_evidence.md` (c) 5). Every card in the survey is 0–5 days old.

---

## 6. Methodology

### 6.1 Three evidence streams

*Local artefacts.* The Kev checkout (`PLAN.md`, 755 lines; `experiments/*.json`; `runs/leaderboard.md` with 130+ trial rows; 14 locked-test summaries; seven model cards; suite manifests; `kev/model.py` and `kev/train.py`) and shallow clones of ten open replicas taken 2026-09-22 (Laya, NanoJev, jevbetter, mini-jev, jev-on-a-laptop, Verdict, system-one-open, decider, jevbench, jev-benchmarks) plus the awesome-jev list (heyjunpenn, 2026). Numbers were read from machine-written files where they exist and from prose otherwise, and the two are distinguished in the grades.

*Literature.* Three annotated bibliographies compiled 2026-09-22: stream A, backbone and head (40 sources); stream B, objective, distillation, calibration and selective prediction (55 sources); stream C, footprint and competing models (66 sources including cards). Search covered arXiv, ACL Anthology, CVF, PMLR, OpenReview, Hugging Face and vendor blogs, with date preference for 2023 onward and older items admitted only as primary sources for a design still in use.

*Competing-model survey.* Hugging Face API sweeps, GitHub READMEs for every sub-1B entry in awesome-jev, JevBench's results files and site, and Kev's own external-suite runs.

### 6.2 Verification procedure

Each bibliography was verified by an independent pass that re-resolved every identifier (arXiv API, Crossref, Anthology, PMLR, CVF, OpenReview), confirmed venue against the venue's own index, and re-read every spot-checked number from PDF text or the raw card. Results: stream A 40/40 exist, 26 of 36 spot-checks confirmed, 7 misquoted, 2 grade corrections; stream B 55/55 exist, 51 of 58 confirmed, 5 misquoted with no direction reversed; stream C 66/66 exist, 57 of 69 confirmed, 8 mislabelled, 4 confirmed with an omitted condition; local stream 52 findings, 40 confirmed, 5 confirmed with a framing correction, 5 misquoted, 2 found elsewhere. Every correction was applied before synthesis and is listed in `synthesis_report.md` §0. The ones that change a number in this report: PriDe at 5% is +1.2/+1.3/+1.7; Set-Encoder's rows were shifted one model; ModernBERT throughput is 133.8k against 23.4k tokens/s on long variable-length input; Abbes's FLOPs ratio for RoBERTa-large is about 15x; Kev's option-isolation cost at 4B is −3.0 to −3.8 against recorded baselines, not −5.8 against an unrecorded one; the rule-tree chain is `(A or B) and C` 0.69 → 0.91 and `or_not` 0.44 → 0.75; Qwen3-0.6B 4-bit per-group is 44.0/42.1 MMLU; NanoJev's MLX 4-bit figures are a dequantized bound; laya-coreml's int8 encoder weights failed parity; Galil's family coefficients are AUROC-versus-ECE; Verdict has two calibrators (T 1.4265 and 2.8039); the Gemma-270M "zero-shot beats trained" TypeSafe row is a majority-label artefact with no held-out block.

### 6.3 Evidence weighting

MC (measured, controlled: same frozen items, one factor varied, seeds or CI stated) ranks highest; MU (measured but comparator differs, or single seed without CI) is high for what happened and low for why; PR-D (peer-reviewed, task is classification, multiple choice or typed decision) outranks any in-category README; PR-A (peer-reviewed, analogous task such as reranking or vision) is medium and every use is labelled `[I]`; PP (preprint or non-archival workshop) medium-low; SR (README, card, leaderboard self-report) lowest, because the category is nine days old and every card was created 2026-09-17 to 2026-09-22; IM (independently measured by a third party) sits between MU and SR. Any transfer across vision → text, reranking → typed decisions, in-distribution → held-out source, another suite → transfer-v4, decoder → encoder, or M1 Max / M5 / T4 → base M1 is marked `[I]`.

### 6.4 Contamination discipline

The holdout list (Appendix A) is exact from `evals/v4/transfer-v4/manifest.json`, `evals/v9/transfer-v9/manifest.json` and `kev/data.py:185-280`. No training mix, transfer set, calibration set or known-OOD calibrator set proposed here contains any row of those sources, nor the test or validation splits of the ten trainable sources (which form Kev's dev and test partitions). Base checkpoints must be raw: every GLUE-tuned checkpoint has seen QNLI; Verdict started from `gliclass-modern-base-v2.0` and is therefore not a clean transfer-v4 candidate. MMLU rows are reported with the caveat that Qwen3, Gemma and SmolLM2 bases may have seen MMLU test items in pretraining, since only exact-state dedup is certified.

---

## 7. Findings

### 7.1 Backbone

At matched size and pretraining, a bidirectional encoder beats a causal decoder on fine-tuned classification by about one decoder size step (Weller et al., 2026; Gisserot-Boukhlef et al., 2026; PR-D; high confidence for classification-shaped items). Inside the decoder family on Kev's suite, capacity dominates: 0.6B → 4B +17.5 pp [+13.0, +22.1] (MC, lr 2e-4, one seed); 4B → 8B +0.4 [−3.9, +4.5] paired (MC, three seeds); Qwen3.5 against Qwen3 at 4B +1.0 mean, not significant; 0.8B against 0.6B +4.8 [+0.2, +9.3] on the locked test (MC). Fine-tuning erodes base knowledge at every size (MMLU 0.6B base 0.425 → 0.46 trained; 4B 0.688 → 0.60–0.66; PAWS 0.70 → 0.56 at 0.6B; `PLAN.md:289-302`; MC, 80 items per task), and learning rate is the mechanism at 4B (2e-4 → 5e-5, +4.7 [+0.4, +9.6]; MC) but is not isolated at ≤1B. The two findings are reconciled by the suite's composition (contradiction C20): 47% of transfer-v4 is classification-shaped where encoders should win, 30% is knowledge MCQ where a 36T-token decoder carries what no 2T-token encoder holds, and 23% is rule composition where nothing has been measured for an encoder. The one ≤300M trained decoder with a held-out-source number is Gemma-3-270M at 0.585 on system-one-open's 23 held-out tasks (ECE 0.146; MC, single run, foreign suite).

### 7.2 Head

Pointer or marker scorer over per-option boundary states, no inter-option attention: high confidence. After training, readout is not the bottleneck at 4B (adapted-backbone letter logits reproduce the pointer head: deadline 0.53 = 0.53, MMLU 0.72 against about 0.70; `PLAN.md:693`; MC), untested at ≤0.3B. Permutation sensitivity scales inversely with size (flip rate Kev-0.5B 0.21, 0.6B 0.07, 9B 0.03, Jev 0.00; MC) and exact invariance via `option_isolation` is free at 0.6B and costs 3–4 pp at 4B (MC, comparator uncertain). High-cardinality option sets are a token-budget problem: GLiClass one pass 1 → 128 labels −7% throughput; Laya Banking77 0.425 at 192 shared tokens; Verdict 0.97 → 0.72 from K=3 to 25 on a fixed 25-slot head (MC); Kev-0.5B Banking77 0.860 in distribution with no per-option budget (MU).

### 7.3 Objective and calibration

Kev's matched screen (Kev-4B parent, seed 11, one epoch, 3,425 records, 656 dev questions; MC): recalibrated parent accuracy 0.797, coverage 0.576; CE continuation 0.797 / 0.532; label smoothing ε=0.05 0.802 / **0.006**; CE + 0.5·Brier 0.797 / 0.526; focal γ=1 0.799 / 0.502 with the best ECE 0.043. "No candidate advances." A single in-distribution temperature transfers OOD on Qwen3.5 (Kev-9B ECE 0.105 → 0.039, confident errors 7.5% → 3.2%, coverage unchanged; MC); per-(type, K) is worse OOD. Coverage tracks accuracy and label noise: Kev pairs 0.652 → 0.23, 0.797 → 0.54–0.57, 0.857 → 0.70 (MC); 59% of Kev-9B errors sit on TweetEval, PAWS and Emotion (37% of items); the ceiling at Kev-9B's accuracy with perfect ranking is 0.86. Uniform soft targets on evidence-free items send the ≥0.9-confident share to 0.00 at 4B and 9B with controls unchanged (MC). Anchoring to the base's zero-shot distribution is "not a lever" (MC, one seed). Served ECE after temperature is roughly flat across Kev sizes (0.8B 0.054, 4B 0.041, 9B 0.042) while raw ECE is worse at small size (0.179 against 0.106) and coverage is what scales (0.23 → 0.57).

### 7.4 Data

Breadth beats volume; policy synthetic data transfers (§5.4; MC). Small models are data-hungry on foreign suites (270M lite 0.367 against full 0.585; MC). Teacher labels are slightly worse and much cheaper than human labels (PR-D). Contamination discipline is exact-state only everywhere. Data scaling at 150–600M on Kev's suite is unmeasured.

### 7.5 Footprint

Eight-bit is lossless for both families with per-group or QAT quantizers (PR-D, PP); per-tensor and absmax quantizers break both (PR-D, PR, IM). Four-bit costs a sub-1B decoder 3–8 points on knowledge MC and more on arithmetic (PP, PR), with the loss shrinking with size; the one decision-model 4-bit datum is a weight-only dequantized bound (SR). Encoders are 3–5x cheaper per question than same-accuracy decoders on Apple silicon, every figure on a different device (SR). INT8 on the Neural Engine failed a parity gate in the only direct test (SR, hours old). Nothing in the category has quantized a trained decision model and scored it on a held-out suite.

### 7.6 Competing open models

Fully supported by verified numbers and dated 2026-09-22: no sub-0.5B open model has been measured on Kev's suites by anyone but Kev, none beats Kev-0.6B there or on JevBench, and every strong self-reported in-domain figure collapses off-suite (§5.6).

### 7.7 The recipe frame

Seven candidate recipes were scored on transfer-v4 dev accuracy, coverage, ECE, parameters, base-M1 latency, on-device feasibility, option-count handling, permutation invariance, data need, training cost and confidence. The full frame is Appendix B; the decision columns are below.

| Recipe | Backbone + head + objective | Expected transfer-v4 dev acc | cov@5% | Params | Base-M1 latency `[I]` | Confidence | Biggest unknown |
|---|---|---|---|---|---|---|---|
| A | ModernBERT-base + marker scorer, hard CE | 0.57–0.65 (0.61) | 0.15–0.30 | 149M + head | 15–40 ms | LOW–MED | Encoder on knowledge and rule blocks: zero measurements |
| **B** | A + soft-target KD from Kev-4B/9B | **0.60–0.69 (0.645)** | 0.25–0.40 | same | same | LOW–MED | Decoder → encoder KD on held-out sources: no source measures it |
| C | DeBERTa-v3-base + marker scorer | 0.58–0.66 (0.62) | 0.15–0.30 | 184M (98M embeddings) | 30–80 ms | LOW | No quantization datum; 512-token cap; CoreML/MLX export of disentangled attention untested |
| D | Gemma-3-270M / SmolLM2-360M + pointer LoRA, CE | 0.50–0.58 (0.54) / 0.54–0.61 (0.57) | 0.10–0.20 | 270M / 360M | 30–70 ms | MED | Whether 12.6k records suffice at 270–360M |
| E | D + KD | 0.53–0.62 / 0.57–0.65 (0.60) | 0.15–0.30 | same | same | LOW–MED | As B, plus whether K-way soft targets carry enough signal |
| F | Qwen3-0.6B + pointer + KD | 0.62–0.68 (0.65) | 0.25–0.40 | 596M + 0.5M head + 8.8M LoRA | 60–150 ms | MED | Whether KD lifts a base whose hyperparameter surface is flat; Qwen3 T transfer |
| G | ModernBERT-base + Set-Encoder-style reset | 0.56–0.66 (0.61) | as A | 149M + 1M | 20–50 ms; K× A if state is duplicated | LOW | RoPE position reset without re-pretraining; custom mask on CoreML/MLX |

The realistic band for a 150–360M model is 0.57–0.69, that is Kev-0.6B ± 5 pp, not Kev-4B. Every chain that ends above 0.69 passes through an unmeasured transfer. The centres order F ≥ B ≥ C ≈ A ≈ G ≥ E(SmolLM2) ≥ D(SmolLM2) > E(Gemma) > D(Gemma), with A–G separated from F by exactly the two unknowns that E1 and E2 resolve.

---

## 8. Discussion

### 8.1 Contradiction resolutions

Thirty-two contradictions were adjudicated (`synthesis_report.md` §3). The ones that shape the recommendation:

*Capacity dominates against encoders win (C20).* Both are true in their domains; the suite composition reconciles them, and E1's per-source table is the test. This is why the recipe frame's A–C and D–F ranges overlap and why the recommendation is conditional on one cheap run.

*Temperature transfer across generations (C1).* The Qwen3 side is one sentence per card plus a 0.6B observation on v2 data; the Qwen3.5 side has full numbers on v7. Under-documented rather than opposite-measured; the encoder recipe sidesteps it but E4 re-measures transfer on whatever base is chosen.

*Set attention (C22).* Set-Encoder's null result under relevance labels, its authors' caveat that comparison-requiring losses change the answer, UniMC's win with inter-option attention blocked, and Kev's isolation parity at 0.6B together say: invariance device, not accuracy lever. E3 tests it; it is never the primary bet.

*Small-decoder letter binding (C23).* Marker and pointer heads never read the letter token, so the MCSB literature does not apply to them; it does forbid a letter-logit arm below 0.8B.

*More data hurts against more data is everything (C24).* Two regimes: a drift-limited 4B model at lr 2e-4 and an under-fit 270M model. The prediction for 150–360M is that diverse data helps at low learning rate without oversampling any family; E2's hard-label control and optional E6 measure it.

*Temperature and coverage (C21).* Within a question a single T is monotone and at K=2 provably so; across questions with mixed K a single T can reorder, which is Galil's 1000-class result. Kev measured no coverage change at its K distribution (mostly 2–5). Temperature is for ECE; coverage must come from accuracy, distillation or a known-OOD calibrator.

*Encoder INT8 (C26) and decoder INT8 (C27).* QAT or per-group scaling against post-training palettization and per-tensor dynamic quantization; the quantizer, not the bit-width, explains every failure in the corpus.

### 8.2 Transfers labelled as inference

The recommendation's accuracy range rests on three inferences: MNLI and SuperGLUE encoder margins carried to emotion/qnli/paws/tweet; reranking and vision distillation gains carried to typed decisions; and Ettin's 2T-token matched pairs carried to a comparison with a 36T-token Qwen3. Its coverage range rests on Kev's accuracy-to-coverage pairs and on vision evidence that distillation improves ranking. Its latency rests on M1 Max, M5 Pro and WASM numbers scaled to a base M1 by parameter count and an assumed 2x device factor. Its quantization plan rests on BERT-era encoders and one hours-old ANE card. None of these is a measurement on the target suite or device; E1, E2, E4 and E5 replace each one.

### 8.3 Limitations

The category is nine days old. Most in-category numbers are self-reported single runs on non-comparable suites; the two independent measurements (JevBench, jev-benchmarks) run over a network with a latency assumption, and JevBench's kev rows are pre-v7 previews. Kev's anchors are the author's own record of the author's own models, mostly one to three seeds with ±4–5 pp confidence intervals on 656 items; two comparators exist only as prose. The objective screen is one seed, one epoch, at 4B, as a delta. Jev has never been run on the locked test, so every gap to Jev is a dev-partition gap. Coverage at 5% error is two metrics in the wild; only Kev's v2 tie-aware version is used here. Pretraining contamination is unknown for every system. Roughly half of the head-side literature and every row on label smoothing, distillation-and-calibration, abstention heads and conformal set size is reranking or vision. No source reports ModernBERT, DeBERTa or SmolLM2 on an iPhone. The estimate for the recommended recipe is graded LOW–MED, and the report says so wherever the number appears.

---

## 9. Conclusion

Build tinyjev-150m first: ModernBERT-base with a per-option `[MASK]`-marker shared scorer, trained on Kev's decision-v7 mix with soft targets from the Kev-4B one-epoch parent, one fitted temperature, softmax-response selection, INT8 export with a parity gate. Expect 0.60–0.69 on transfer-v4 dev (central 0.645, LOW–MED), coverage at 5% error 0.25–0.40, and 15–40 ms per question on a base M1 as an inference to be replaced by E5. Run E1 before anything else: it costs $12–24 and decides between the encoder line and the Qwen3-0.6B runner-up with a per-source table nobody has produced. Then E2 decides whether soft targets or just more data carry the gain. The whole ladder is 10–22 H100-hours, about $40–86 at Kev's metered rate, and it produces the first quantization-versus-held-out-accuracy number in the category. Do not promise Kev-4B accuracy, Jev calibration, or a free 4-bit model; the evidence supports none of them.

---

## References

Aarsen, T. (2026, May 19). Introducing the Ettin reranker family. *Hugging Face Blog*. https://huggingface.co/blog/ettin-reranker

AbdelStark. (2026). jev-benchmarks: BTZSC pilot protocol and report [Source code and results]. GitHub. https://github.com/AbdelStark/jev-benchmarks

Abbes, I., Prato, G., Fournier, Q., Rodriguez, F., Boukhary, A., Elwood, A., & Chandar, S. (2025). Small encoders can rival large decoders in detecting groundedness. *arXiv*. https://arxiv.org/abs/2506.21288

Angelopoulos, A. N., Bates, S., Malik, J., & Jordan, M. I. (2021). Uncertainty sets for image classifiers using conformal prediction. *International Conference on Learning Representations (ICLR 2021)*. https://arxiv.org/abs/2009.14193

Apple Machine Learning Research. (2022, June). Deploying transformers on the Apple Neural Engine. https://machinelearning.apple.com/research/neural-engine-transformers

Beyer, L., Zhai, X., Royer, A., Markeeva, L., Anil, R., & Kolesnikov, A. (2022). Knowledge distillation: A good teacher is patient and consistent. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022)*, 10925–10934. https://arxiv.org/abs/2106.05237

Bondarenko, Y., Nagel, M., & Blankevoort, T. (2021). Understanding and overcoming the challenges of efficient transformer quantization. *Proceedings of EMNLP 2021*. https://arxiv.org/abs/2109.12948

C-Tianyu. (2026). NanoJev [Model card, TRAINING_RECIPE.md, research reports]. Hugging Face. https://huggingface.co/C-Tianyu/NanoJev

Chen, Y., Yuan, L., Cui, G., Liu, Z., & Ji, H. (2023). A close look into the calibration of pre-trained language models. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics*, 1343–1367. https://arxiv.org/abs/2211.00151

Cho, J. H., & Hariharan, B. (2019). On the efficacy of knowledge distillation. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2019)*, 4794–4802. https://arxiv.org/abs/1910.01348

convaiinnovations. (2026). Laya [Model card; repository files BENCHMARKS.md, laya/common.py, research/results/t4_colab_benchmark.json]. Hugging Face. https://huggingface.co/convaiinnovations/laya

Desai, S., & Durrett, G. (2020). Calibration of pre-trained transformers. *Proceedings of EMNLP 2020*, 295–302. https://arxiv.org/abs/2003.07892

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. https://arxiv.org/abs/2208.07339

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186. https://aclanthology.org/N19-1423/

Devvrit, Kudugunta, S., Kusupati, A., Dettmers, T., Chen, K., Dhillon, I., Tsvetkov, Y., Hajishirzi, H., Kakade, S., Farhadi, A., & Jain, P. (2024). MatFormer: Nested transformer for elastic inference. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. https://arxiv.org/abs/2310.07707

Fan, A., Grave, E., & Joulin, A. (2020). Reducing transformer depth on demand with structured dropout. *International Conference on Learning Representations (ICLR 2020)*. https://arxiv.org/abs/1909.11556

Feng, L., Ahmed, M. O., Hajimirsadeghi, H., & Abdi, A. H. (2023). Towards better selective classification. *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2206.09034

FluidInference. (2026). laya-coreml [Model card]. Hugging Face. https://huggingface.co/FluidInference/laya-coreml

fstandhartinger. (2026). jevbench v1.2/v1.3: RESULTS-v1.2.md, results JSON and method notes [Benchmark]. GitHub and benchmarkheaven.com. https://github.com/fstandhartinger/jevbench ; https://benchmarkheaven.com/jev-models

Galil, I., Dabbah, M., & El-Yaniv, R. (2023). What can we learn from the selective prediction and uncertainty estimation performance of 523 ImageNet classifiers? *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2302.11874

Gisserot-Boukhlef, H., Boizard, N., Faysse, M., Alves, D. M., Malherbe, E., Martins, A. F. T., Hudelot, C., & Colombo, P. (2026). Should we still pretrain encoders with masked language modeling? *International Conference on Learning Representations (ICLR 2026)*. https://arxiv.org/abs/2507.00994

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning, PMLR 70*, 1321–1330. https://arxiv.org/abs/1706.04599

Heman10x-NGU. (2026). Verdict-open-jev [Source code, README, artifacts]. GitHub. https://github.com/Heman10x-NGU/Verdict-open-jev

heyjunpenn. (2026). awesome-jev [Curated list]. GitHub. https://github.com/heyjunpenn/awesome-jev

Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.-Y., & Pfister, T. (2023). Distilling step-by-step! Outperforming larger language models with less training data and smaller model sizes. *Findings of the Association for Computational Linguistics: ACL 2023*, 8003–8017. https://arxiv.org/abs/2305.02301

Hui, L., & Belkin, M. (2021). Evaluation of neural architectures trained with square loss vs cross-entropy in classification tasks. *International Conference on Learning Representations (ICLR 2021)*. https://arxiv.org/abs/2006.07322

Hume, A. (2026, September 17). Jev's architecture unmasked. *archerhume.com*. https://archerhume.com/posts/jevs-architecture-unmasked/

Jaeger, P. F., Lüth, C. T., Klein, L., & Bungert, T. J. (2023). A call to reflect on evaluation practices for failure detection in image classification. *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2211.15259

Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2020). TinyBERT: Distilling BERT for natural language understanding. *Findings of the Association for Computational Linguistics: EMNLP 2020*, 4163–4174. https://arxiv.org/abs/1909.10351

Kamath, A., Jia, R., & Liang, P. (2020). Selective question answering under domain shift. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 5684–5696. https://aclanthology.org/2020.acl-main.503/

Kim, S., Gholami, A., Yao, Z., Mahoney, M. W., & Keutzer, K. (2021). I-BERT: Integer-only BERT quantization. *Proceedings of the 38th International Conference on Machine Learning, PMLR 139*. https://arxiv.org/abs/2101.01321

Kim, S., Park, S., Lee, J., & Kwak, N. (2025). The role of teacher calibration in knowledge distillation. *IEEE Access, 13*. https://arxiv.org/abs/2508.20224

Kumar, B., Lu, C., Gupta, G., Palepu, A., Bellamy, D., Raskar, R., & Beam, A. (2023). Conformal prediction with large language models for multi-choice question answering. *arXiv*. https://arxiv.org/abs/2305.18404

Lee, J., Park, S., Kwon, J., Oh, J., & Kwon, Y. (2025). Exploring the trade-offs: Quantization methods, task difficulty, and model size in large language models from edge to giant. *Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI-25)*, 8113–8121. https://www.ijcai.org/proceedings/2025/0902.pdf

lostargon. (2026). Tiny-Jev [Model card]. Hugging Face. https://huggingface.co/lostargon/Tiny-Jev

Mapika. (2026). decider-0.8b [Model card; repository docs]. Hugging Face. https://huggingface.co/Mapika/decider-0.8b

Melton, J. (2026). What does MLX 4-bit cost? A controlled audit of quantization for code generation on Apple Silicon. *Zenodo*. https://zenodo.org/records/22698290

Mishra, I., Krishna, S. V., & Mishra, D. (2023). Distilling calibrated student from an uncalibrated teacher. *arXiv*. https://arxiv.org/abs/2302.11472

mithalouni. (2026). system-one-open [Source code, results/*.summary.json, README]. GitHub. https://github.com/mithalouni/system-one-open

mpnikhil. (2026). dev-0.4b [Model card]. Hugging Face. https://huggingface.co/mpnikhil/dev-0.4b

Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P. H. S., & Dokania, P. K. (2020). Calibrating deep neural networks using focal loss. *Advances in Neural Information Processing Systems, 33*. https://arxiv.org/abs/2002.09437

Müller, R., Kornblith, S., & Hinton, G. E. (2019). When does label smoothing help? *Advances in Neural Information Processing Systems, 32*. https://arxiv.org/abs/1906.02629

Nielsen, D. S., Enevoldsen, K., & Schneider-Kamp, P. (2025). Encoder vs decoder: Comparative analysis of encoder and decoder language models on multilingual NLU tasks. *Proceedings of NoDaLiDa/Baltic-HLT 2025*, 561–572. https://aclanthology.org/2025.nodalida-1.60/

Palmer, J. (2026). Kev: Typed-decision models; research log (PLAN.md, PLAN_27b.md), experiment configs, run ledger (runs/leaderboard.md), locked-test summaries, frozen suite manifests, model cards and README [Source code and data]. GitHub and Hugging Face. https://github.com/jaredpalmer/kev ; https://huggingface.co/jaredpalmer/kev-0.6b ; https://huggingface.co/jaredpalmer/kev-0.8b ; https://huggingface.co/jaredpalmer/kev-0.5b

Pangakis, N., & Wolken, S. (2024). Knowledge distillation in automated annotation: Supervised text classification with LLM-generated training labels. *Proceedings of the Sixth Workshop on Natural Language Processing and Computational Social Science (NLP+CSS)*. https://arxiv.org/abs/2406.17633

Pezeshkpour, P., & Hruschka, E. (2024). Large language models sensitivity to the order of options in multiple-choice questions. *Findings of NAACL 2024*, 2006–2017. https://aclanthology.org/2024.findings-naacl.130/

r-ms. (2026). mini-jev [Source code, README, PREREG.md]. GitHub. https://github.com/r-ms/mini-jev

Robinson, J., Rytting, C. M., & Wingate, D. (2023). Leveraging large language models for multiple choice question answering. *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2210.12353

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. *arXiv* (NeurIPS 2019 EMC² workshop). https://arxiv.org/abs/1910.01108

Schick, T., & Schütze, H. (2021). It's not just size that matters: Small language models are also few-shot learners. *Proceedings of NAACL-HLT 2021*. https://arxiv.org/abs/2009.07118

Schlatt, F., Fröbe, M., Scells, H., Zhuang, S., Koopman, B., Zuccon, G., Stein, B., Potthast, M., & Hagen, M. (2025a). Set-Encoder: Permutation-invariant inter-passage attention for listwise passage re-ranking with cross-encoders. *Proceedings of ECIR 2025*. https://arxiv.org/abs/2404.06912

Schlatt, F., Fröbe, M., Scells, H., Zhuang, S., Koopman, B., Zuccon, G., Stein, B., Potthast, M., & Hagen, M. (2025b). Rank-DistiLLM: Closing the effectiveness gap between cross-encoders and LLMs for passage re-ranking. *Proceedings of ECIR 2025*. https://arxiv.org/abs/2405.07920

Srivastava, G., Hussain, A., Srinivasan, S., & Wang, X. (2026). Do LLMs overthink basic math reasoning? Benchmarking the accuracy-efficiency tradeoff in language models. *Findings of the Association for Computational Linguistics: ACL 2026*. https://arxiv.org/abs/2507.04023

Stanton, S., Izmailov, P., Kirichenko, P., Alemi, A. A., & Wilson, A. G. (2021). Does knowledge distillation really work? *Advances in Neural Information Processing Systems, 34*. https://arxiv.org/abs/2106.05945

Stepanov, I., et al. (2025). GLiClass: Generalist lightweight model for sequence classification tasks. *arXiv*. https://arxiv.org/abs/2508.07662

Sun, W., Yan, L., Ma, X., Wang, S., Ren, P., Chen, Z., Yin, D., & Ren, Z. (2023). Is ChatGPT good at search? Investigating large language models as re-ranking agents. *Proceedings of EMNLP 2023*. https://arxiv.org/abs/2304.09542

Tang, R., Lu, Y., Liu, L., Mou, L., Vechtomova, O., & Lin, J. (2019). Distilling task-specific knowledge from BERT into simple neural networks. *arXiv*. https://arxiv.org/abs/1903.12136

Turc, I., Chang, M.-W., Lee, K., & Toutanova, K. (2019). Well-read students learn better: On the importance of pre-training compact models. *arXiv*. https://arxiv.org/abs/1908.08962

Varshney, N., Mishra, S., & Baral, C. (2022). Investigating selective prediction approaches across several tasks in IID, OOD, and adversarial settings. *Findings of the Association for Computational Linguistics: ACL 2022*, 1995–2002. https://arxiv.org/abs/2203.00211

Wang, W., Bao, H., Huang, S., Dong, L., & Wei, F. (2021). MiniLMv2: Multi-head self-attention relation distillation for compressing pretrained transformers. *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, 2140–2151. https://arxiv.org/abs/2012.15828

Wang, X., Weissweiler, L., Schütze, H., & Plank, B. (2023). How to distill your BERT: An empirical study on the impact of weight initialisation and distillation objectives. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Short Papers)*, 1843–1852. https://arxiv.org/abs/2305.15032

Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., & Poli, I. (2025). Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. *Proceedings of ACL 2025 (Long Papers)*, 2526–2547. https://arxiv.org/abs/2412.13663

Weller, O., Ricci, K., Marone, M., Chaffin, A., Lawrie, D., & Van Durme, B. (2026). Seq vs Seq: An open suite of paired encoders and decoders. *International Conference on Learning Representations (ICLR 2026)*. https://arxiv.org/abs/2507.11412

Wiegreffe, S., Tafjord, O., Belinkov, Y., Hajishirzi, H., & Sabharwal, A. (2025). Answer, assemble, ace: Understanding how LMs answer multiple choice questions. *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2407.15018

Xia, G., Laurent, O., Franchi, G., & Bouganis, C.-S. (2025). Towards understanding why label smoothing degrades selective classification and how to fix it. *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2403.14715

Xin, J., Tang, R., Yu, Y., & Lin, J. (2021). The art of abstention: Selective prediction and error regularization for natural language processing. *Proceedings of ACL-IJCNLP 2021*, 1040–1051. https://aclanthology.org/2021.acl-long.84/

Yang, A., Li, A., Yang, B., et al. (Qwen Team). (2025). Qwen3 technical report. *arXiv*. https://arxiv.org/abs/2505.09388

Yang, P., Wang, J., Gan, R., Zhu, X., Zhang, L., Wu, Z., Gao, X., Zhang, J., & Sakai, T. (2022). Zero-shot learners for natural language understanding via a unified multiple choice perspective. *Proceedings of EMNLP 2022*. https://arxiv.org/abs/2210.08590

Ye, J., Gao, J., Li, Q., Xu, H., Feng, J., Wu, Z., Yu, T., & Kong, L. (2022). ZeroGen: Efficient zero-shot learning via dataset generation. *Proceedings of EMNLP 2022*, 11653–11669. https://arxiv.org/abs/2202.07922

yzfly. (2026). edgejev: Local Jev-like typed-decision runtime using ONNX and quantized models [README]. GitHub. https://github.com/yzfly/edgejev

Zafrir, O., Boudoukh, G., Izsak, P., & Wasserblat, M. (2019). Q8BERT: Quantized 8bit BERT. *5th Workshop on Energy Efficient Machine Learning and Cognitive Computing (EMC²), NeurIPS 2019*. https://arxiv.org/abs/1910.06188

Zaratiana, U., Tomeh, N., Holat, P., & Charnois, T. (2024). GLiNER: Generalist model for named entity recognition using bidirectional transformer. *Proceedings of NAACL 2024*, 5364–5376. https://aclanthology.org/2024.naacl-long.300/

ZeroDegress. (2026). NanoJev-mlx-4bit [Model card]. Hugging Face. https://huggingface.co/ZeroDegress/NanoJev-mlx-4bit

Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., Huang, F., & Zhou, J. (2025). Qwen3 Embedding: Advancing text embedding and reranking through foundation models. *arXiv*. https://arxiv.org/abs/2506.05176

Zheng, C., Zhou, H., Meng, F., Zhou, J., & Huang, M. (2024). Large language models are not robust multiple choice selectors. *International Conference on Learning Representations (ICLR 2024)*. https://arxiv.org/abs/2309.03882

Zheng, X., Li, Y., Chu, H., Feng, Y., Ma, X., Luo, J., Guo, J., Qin, H., Magno, M., & Liu, X. (2025). An empirical study of Qwen3 quantization. *arXiv*. https://arxiv.org/abs/2505.02214

---

## Appendix A. Holdout and contamination list

Exact, from `KEV/evals/v4/transfer-v4/manifest.json`, `KEV/evals/v9/transfer-v9/manifest.json` and `KEV/kev/data.py:185-280` (reproduced by `verification_local_evidence.md` (d)). No training mix, transfer set, calibration set or known-OOD calibrator set may contain any row of the following.

**transfer-v4 holdout (dev 764 / test 764 records; version 3; context 384/1024/2048):**

| Source key | HF dataset (config) | Split Kev uses | Dev n | Nature |
|---|---|---|---|---|
| `mmlu` | `cais/mmlu` (`all`) | test | 116 | 4-way knowledge |
| `emotion` | `dair-ai/emotion` (`split`) | test | 116 | 6-label classification |
| `tweet_offensive` | `cardiffnlp/tweet_eval` (`offensive`) | test | 80 | binary |
| `qnli` | `nyu-mll/glue` (`qnli`) | test | 80 | binary NLI |
| `paws` | `google-research-datasets/paws` (`labeled_final`) | test | 80 | adversarial paraphrase |
| `sciq` | `allenai/sciq` | test | 116 | 4-way science MCQ |
| `contrastive` | Kev synthetic (`authorization`, `deadline` families) | — | 0 dev / 80 test (v9) | policy minimal pairs |
| `legacy_holdout` | Kev synthetic legacy policy families | — | 80 dev / 0 test (v9) | policy |
| `composition_holdout` | Kev rule shapes `held_and_or`, `held_or_not`, `held_conditional` (transfer); `final_combination`, `final_negation`, `final_exception` (locked; render style 2) | — | 96 | compositional rules |

**transfer-v9 additions (dev 1,264 / test 1,264; version 5; v4 items byte-identical):**

| Source key | Provenance | n per partition |
|---|---|---|
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` rev `b189ec765aa7ed75c8acfea42df31fdae71f97be`, 10-way | 200 |
| `buried` | Kev-built from `paws`, `qnli`, `tweet_offensive`, `emotion` states embedded among three unrelated same-source records | 80 |
| `unknowable` | Kev-built: deciding evidence removed | 110 |
| `unknowable_control` | intact controls paired with the above | 110 |

Partition asymmetry: v9 dev has `legacy_holdout`, v9 test has `contrastive`.

**Also excluded:** the test and validation splits of the ten trainable sources (`legacy-datasets/banking77`, `google/boolq`, `fancyzhx/ag_news`, `nyu-mll/multi_nli`, `SetFit/sst5`, `Yelp/yelp_review_full`, `CogComp/trec`, `fancyzhx/dbpedia_14`, `SetFit/amazon_reviews_multi_en`, `stanfordnlp/imdb`), which form Kev's dev and test partitions. The v6 knowledge sources (ARC-Challenge, OpenBookQA, CommonsenseQA) were trainable, not holdout.

**External suites a training mix would also contaminate:** SemIf authored 144 (+108 perturbations), scienthoon 900 tickets, ekzhang MMLU-Pro 1,000, TypeSafe public eval 372 pairs, JevBench 534 (111 public hard), kotoba-lang typed-decisions.

**Checkpoints that must not be used as starting points** (tuned on holdout-overlapping data): any GLUE- or QNLI-tuned checkpoint; `gliclass-*`; `*-zeroshot-v2.0`; `*-mnli`; Ettin-reranker weights; `gliclass-modern-base-v2.0` (Verdict's base). Tiny-Jev's Emotion 82.2 was fitted on 2,000 Emotion items and is unusable as a zero-shot comparator. Laya, edgejev and jev-benchmarks all report on Emotion; decider and OpenThai report on PAWS; treat those rows as in-distribution for those systems.

---

## Appendix B. Recipe frame table in full

Accuracy is transfer-v4 dev (656 clean questions), range (central). Every arrow that crosses a task, size, device or suite is an inference `[I]`. From `synthesis_report.md` §4.

**B.1 Recipes**

| ID | Backbone | Head | Objective | Supervision | Post-hoc |
|---|---|---|---|---|---|
| A | ModernBERT-base 149M (raw) | `[CLS] type+instr [SEP] [MASK] opt_0 [MASK] opt_1 … [SEP] state [SEP]`; gather at each `[MASK]`; shared `LayerNorm→Linear→GELU→Linear(1)`; softmax; Noul = 2-option; Score = expected level | hard-label CE | decision-v7 (10 public × 1k + 896 policy pairs + 1,680 rule trees), train-time option shuffling, `p_none_pair` 0.25 | single T on in-dist calibration split |
| B | as A | as A | soft-target CE `−Σ t·log p`, teacher T ∈ {1, served} | A's data + Kev-4B (1-epoch parent) soft distributions; transfer set = v7 + 3–10× generated policy states + extra rows from trainable sources; consistent views | single T |
| C | DeBERTa-v3-base 184M (86M backbone + 98M embeddings) | as A | CE (or B's KD) | as A/B | single T |
| D | Gemma-3-270M (100M non-emb) or SmolLM2-360M | Kev PointerHead (`q,k` Linear d→256, `z = k(h_</opt>)·q(h_<decide>)/√256`), LoRA r=16 on attention+MLP, head at full lr | hard-label CE | decision-v7, lr sweep {1e-4, 5e-5, 2e-5} | single T |
| E | as D | as D | soft-target CE (KD) | as B | single T |
| F | Qwen3-0.6B-Base (or Qwen3.5-0.8B if C1 bites) | Kev PointerHead; LoRA r=16 or full FT | soft-target CE from Kev-4B/9B + hard-label replay | as B | single T |
| G | ModernBERT-base 149M | Set-Encoder-style: state encoded once; each option a segment with an `[INT]` token; tokens attend within segment + all `[INT]`s; positions reset per segment; scorer on `[INT]` | CE | as A | single T |

Reference rows: Kev-0.6B 0.620 (anchor), Kev-0.8B 0.652, Kev-4B 0.797 (teacher), untrained Qwen3-0.6B 0.567 (floor), Jev 0.857 (ceiling).

**B.2 Master table**

| Field | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| Expected transfer-v4 dev acc | 0.57–0.65 (0.61) | 0.60–0.69 (0.645) | 0.58–0.66 (0.62) | Gemma 0.50–0.58 (0.54); SmolLM2-360M 0.54–0.61 (0.57) | Gemma 0.53–0.62 (0.57); SmolLM2 0.57–0.65 (0.60) | 0.62–0.68 (0.65) | 0.56–0.66 (0.61) |
| Evidence chain | Kev-0.6B 0.620 (MC). Classification 47%: Ettin enc +3.6 over native dec at 150M, B2 +2.1 at 210M, DeBERTa/PET/UniMC `[I]` → +0 to +5 on Kev-0.8B's 0.64 block. Knowledge 30%: no encoder MMLU/SciQ measurement; Kev-0.8B block 0.665 (MMLU 0.42, SciQ 0.91); 2T vs 36T tokens `[I]` → −1 to −13. Rules 23%: silent, ±8. Net −1 pp central, ±4 | A + KD: Rank-DistiLLM 110M +3.3/+1.3 nDCG `[I]`; RankGPT-distilled DeBERTa-large beats 3B `[I]`; Tang +4–5 `[I]`; TinyBERT task-specific KD +7.1 in-dist; Turc PD +0.5–1.6; NanoJev pipeline v2 OOD +2.5 (n=400, 1 seed); Galil KD → best AUROC `[I]`. Teacher ceiling 0.797/0.822; Stanton fidelity 80–90% `[I]` → +2 to +6 over A | DeBERTa-v3-base MNLI 90.6 vs Ettin-150M 89.2; GLUE 88.1 vs 88.4 (tie); GLiClass "DeBERTa > ModernBERT" (COI); ScandEval rank 1.29 → A + 0 to +2 | Kev-0.5B v0.1 0.561; Qwen2.5-0.5B v2 recipe 0.605/0.482 (MC, transfer-v2); SmolLM2-360M ≥ Qwen2.5-0.5B on zero-shot MC `[I]` → 0.55–0.60. Gemma-270M: 100M non-emb; system-one-open 270M 0.585 vs E2B 0.742 on its suite `[I]`; ARC-c 29.0 at SmolLM2-135M level → 0.50–0.58 | D + KD: Turc PD gain largest for smallest students; Tang +4–5; decoder students lose general knowledge first (PP) → +2 to +5 over D | Kev-0.6B 0.620 + KD; "saturated 0.59–0.62" was over hyperparameters, not supervision; NanoJev pipeline v2 OOD +2.5 on the same base (n=400, 1 seed); Kev-0.8B +1.4 dev / +4.8 locked as same-size alternative; Furlanello ~1 `[I]`; 4B→0.6B 7x within Beyer's closed range `[I]` → +0 to +6 | A ± 2: Set-Encoder-330M 0.727/0.789 vs monoELECTRA-330M 0.733/0.765 n.s. `[I]`; Kev isolation parity at 0.6B, −3 to −4 at 4B (MC); UniMC blocks inter-option attention and wins |
| Expected cov@5% (Kev v2) | 0.15–0.30 | 0.25–0.40 | 0.15–0.30 | 0.10–0.20 | 0.15–0.30 | 0.25–0.40 | as A |
| Coverage chain | Kev pairs 0.652→0.23, 0.797→0.54–0.57, 0.857→0.70 (MC); 59% of errors on noisy-label sources; encoder in-domain ECE 1–3% (Desai) does not transfer to ranking under shift | + KD improves AUROC `[I: vision]`; Kamath-style calibrator +8 pts only with known-OOD `[I: QA]` | as A | 0.8B is 0.23 at 0.652; below 0.60 near floor (Kev-0.5B / JevBench hard 30.9%) | as B | as B from a higher base | as A |
| Expected ECE after TS | 0.05–0.10 | 0.04–0.07 | 0.05–0.10 | 0.05–0.10 | 0.04–0.08 | 0.04–0.07 (Qwen3 T-transfer risk, C1) | as A |
| ECE chain | Kev-0.8B served 0.054, Kev-4B 0.041 (MC); Desai TS does little under shift (12.62→12.83) `[I]`; system-one-270m 0.131→0.037 at T=2 (SR); Chen TS best unlearnable (PR-D) | + teacher calibration R² ~0.92 with student quality `[I]`; Kev-4B 1-epoch parent best-calibrated teacher | as A | as A | as B | Kev-0.6B Qwen3 raw dev ECE 0.15; Qwen3 T "does not transfer" (under-documented) | as A |
| Params | 149M + 1–15M head | same | 184M (98M embeddings) + head | 270M (100M non-emb) / 360M + 0.5M head + 5–9M LoRA | same | 596M + 0.5M head (+8.8M LoRA) | 149M + ~1M |
| Base-M1 latency, one 4-option q, 300–450 tokens, fp16 `[I]` | 15–40 ms MPS/MLX; ~10 ms CoreML ANE (D8 27.6 ms / 395M M1 Max; D7 9.0–27.5 ms / 322M M5 Pro ANE; S19 35.6 ms WASM) | same | 30–80 ms (no unpadded FA; 1.5–2.5× A) | 30–70 ms (Kev-0.6B 123 ms / 5 q M5 → 60–120 ms per q on M1; ×0.45–0.6 params; decider 0.8B 99–146 ms M1 Pro) | same | 60–150 ms (Kev-0.6B 123 ms / 5 q M5 → M1 ×2–3; prefill-dominated) | 20–50 ms if state encoded once with K segments; K× A if state duplicated |
| On-device feasibility | fp16 ~300 MB. INT8 free on CPU/GPU (A1–A3 QAT lossless; edgejev −1.6 PTQ); ANE INT8 failed parity for mmBERT (D7), needs QAT + accuracy gate. 4-bit: W4A8 QAT −0.4 on BERT `[I]`; PTQ 4/6-bit palettes failed ANE; no ModernBERT 4-bit datum. iPhone: DistilBERT 3.47 ms ANE `[I]` | same | fp16 ~370 MB; 98M embedding table; no DeBERTa-v3 quantization datum; disentangled attention export untested; 512-token cap | fp16 540 / 720 MB. 8-bit free (A6, A7). 4-bit −3 to −5 MMLU at 0.6B g128, −40% rel. arithmetic at 0.5B, Melton +0.52 pt/B `[I]`; Gemma INT4 on Pixel (vendor); NanoJev 4-bit weight-only 99.2% (dequantized). MLX-Swift 0.5B 531 tok/s M4 Max | same | fp16 1.2 GB; MLX 4-bit g64 336 MB; 4-bit −3 to −5 MMLU `[I]`; 8-bit free; iPhone 17 Pro MLX 159–179 tok/s decode (mixed sessions) | as A, plus custom attention mask and per-segment RoPE reset with no off-the-shelf CoreML/MLX path |
| Handles 50+ options? | Yes with per-option token budget: Laya's shared 192 → Banking77 0.425 at K=77; fix ≥8 tokens/option → ~1k sequence (8k ctx OK); GLiClass one pass 128 labels −7% | same | Yes up to the 512 cap (~40 options at 8 tokens with 384-token state); worse than A | Yes: Kev up to 255; Banking77 0.860 at 0.5B in-dist; CLINC 151-way 0.88 (decider) | same | Yes (Kev 255) | Yes by construction, linear in K (100 candidates 0.219 s at 330M on GPU `[I]`) |
| Permutation invariance | No (Laya 0.15–0.23 flips at K=20); mitigate with shuffling + `perm_kl` | same | same | No (Kev-0.5B 0.21, 0.6B 0.07); `option_isolation` exact at parity at 0.6B (MC) | same | No (0.07); isolation on Qwen3 (attention-only), not on Qwen3.5 hybrid | Exact (flip 0 by construction) |
| Data needed | decision-v7 for a first read; plan 3–5× via generated policy states + trainable-source rows | A + unlabeled transfer set 3–10× for teacher soft targets; consistent views | as A/B | as A; 270M is data-hungry (0.367 at 2.5k/task) | as B | as B | as A |
| Training cost (H100-h) | 0.1–0.5 | 1–4 (teacher pass 9B 0.18 s/rec × 12.6k ≈ 0.6 h, 4B ≈ 0.4; ×3–10 transfer set; + student 0.3) | 0.2–0.8 | 0.1–1 | 1–4 | 1–4 (LoRA) / 2–6 (full FT) | 0.3–1.5 (K× tokens) |
| Confidence | LOW–MED | LOW–MED | LOW | MED | LOW–MED | MED | LOW |
| Single biggest unknown | Encoder on knowledge block (30%) and rules (23%): zero measurements for any 100–400M encoder | Whether decoder→encoder soft-target KD transfers to held-out sources: no source measures it; NanoJev is split | On-device exportability; whether GLiClass's DeBERTa > ModernBERT holds for option scoring | Whether 12.6k records are enough at 270–360M; 100M vs ~300M non-embedding gap | As B, plus whether K-way soft targets carry enough signal | Whether KD lifts a base whose hyperparameter surface is flat (no KD run in the Kev log); C1 | RoPE reset on ModernBERT without re-pretraining; custom-mask serving path |

---

## AI-assisted research disclosure

This report was produced by a pipeline of AI agents (Claude, Anthropic) acting as bibliography, verification, synthesis and report-compilation agents under human direction, between 2026-09-22 (scoping) and 2026-09-22 (this draft). The agents searched, fetched and read sources; re-resolved every identifier and re-read every spot-checked number from primary text; adjudicated contradictions; built the recipe frame; and drafted this text. No experiment was run by the agents; every number here is quoted from a cited source or from Kev's research log, and every estimate for an unmeasured configuration is labelled as an inference with its chain. Kadavath et al. (2022), an Anthropic vendor self-evaluation, was excluded from load-bearing use because the verification agent that graded it is itself an Anthropic model; the corresponding claim rests on Chen et al. (2023) instead. The human reader is responsible for confirming the Modal price assumption and the holdout list before launching any run, and for the final decision to build.
