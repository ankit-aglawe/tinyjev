# Literature stream B: training objective, distillation, calibration and selective prediction

Phase 2 annotated bibliography. Scope: sub-questions 3a (objective), 3b (distillation), 3c (calibration and selective prediction) for a <=0.6B (target 0.1-0.3B) single-forward-pass option scorer that must return calibrated probabilities for typed decisions (choice / yes-no / ordinal score) and be judged on coverage at a fixed error budget (e.g. 5%). This document reports what the literature measured. It does not synthesize or recommend; that is Phase 3.

Compiled 2026-09-22. All 52 references below were verified by fetching a landing page (arXiv abstract, ACL Anthology, CVF Open Access, NeurIPS/ICML proceedings, or Crossref) and, for numeric claims, by extracting the full text with `pdftotext` and reading the relevant table. Numbers marked "(fig.)" were read from a figure or figure caption rather than a table.

---

## 1. Search strategy

**Engines and indices.** Web search (general), arXiv abstract pages, ar5iv HTML renderings, ACL Anthology, CVF Open Access, NeurIPS/ICML/ICLR proceedings pages, OpenReview, Semantic Scholar landing pages, Crossref API (for one DOI whose publisher page returned 403).

**Date window.** 2015-2026. Two 2026 preprints were admitted because they bear directly on ordinal proper scoring rules (RPS) and on task-specific LLM distillation scaling.

**Query families (executed as ~40 searches).**
- 3a: `focal loss calibration Mukhoti`, `label smoothing calibration Müller Kornblith Hinton`, `square loss vs cross-entropy Hui Belkin`, `Brier score training loss calibration proper scoring rule`, `spherical score language model training`, `ordinal regression CORAL CORN rank-consistent`, `unimodal ordinal Beckham Pal`, `soft labels ordinal regression Diaz`, `label smoothing selective classification`, `ranked probability score conformal ordinal`.
- 3b: `DistilBERT`, `TinyBERT`, `MiniLM`, `MiniLMv2`, `MobileBERT`, `Well-Read Students Learn Better`, `Born Again Neural Networks`, `Does knowledge distillation really work`, `On the efficacy of knowledge distillation`, `good teacher is patient and consistent`, `How to distill your BERT`, `Distilling task-specific knowledge from BERT BiLSTM`, `Distilling step-by-step`, `GPT-3 labels reduce labeling cost`, `LLM-generated labels BERT student`, `ZeroGen dataset generation`, `Generating datasets with pretrained language models`, `LLM teacher soft labels encoder student text classification`, `teacher calibration knowledge distillation`, `distillation calibration student ECE`.
- 3c: `On Calibration of Modern Neural Networks`, `Dirichlet calibration beyond temperature scaling`, `Calibration of pre-trained transformers`, `Close look calibration pre-trained language models`, `Language models mostly know what they know`, `Revisiting calibration modern neural networks`, `Gentle introduction conformal prediction`, `APS adaptive prediction sets`, `RAPS uncertainty sets image classifiers`, `conformal prediction LLM multiple choice`, `Selective classification deep neural networks`, `SelectiveNet`, `Deep Gamblers`, `Self-adaptive training selective`, `Towards better selective classification`, `AURC bias-reduced uncertainty estimation`, `Call to reflect failure detection`, `AUGRC`, `523 ImageNet classifiers selective`, `Art of abstention selective prediction NLP`, `selective prediction IID OOD adversarial NLP`, `Selective question answering domain shift`.

**Verification procedure.** For each candidate: (1) fetch the landing page and confirm title/authors/year/venue; (2) download the PDF and run `pdftotext -layout`; (3) confirm the first line of the extracted text matches the intended title; (4) read the specific table or paragraph that supports each number reported here. Candidates whose landing page could not be fetched were excluded. One publisher page (Taylor & Francis) returned 403; the DOI was verified through the Crossref API instead.

## 2. Inclusion and exclusion criteria

**Included if** (a) peer-reviewed or a widely cited preprint; (b) reports measured results (numbers, benchmark, model size) on at least one of: training objective vs calibration/accuracy, distillation into <=1B classifiers/encoders or from LLM teachers into small students, post-hoc calibration, conformal prediction for classification, selective classification / abstention, or coverage-at-risk metrics; (c) the landing page resolves.

**Excluded if** (a) existence could not be confirmed; (b) purely theoretical with no measurement relevant to the sub-questions; (c) generation-only distillation (sequence-level KD, on-policy distillation for chat models) unless it also measures classification-type outcomes; (d) calibration of free-form LLM generations (verbalized confidence, hallucination detection) unless it includes multiple-choice / true-false formats; (e) OOD detection as a primary topic.

**Evidence grades.** PR = peer-reviewed venue (conference main track, Findings, or journal); WS = peer-reviewed workshop; PP = preprint only; FND = foundational/classic (peer-reviewed, pre-2015 or non-deep-learning).

**Tiering.** Tier 1 (T1) = core sources the Phase 3 synthesis should weight most (24 entries). Tier 2 (T2) = verified supporting sources (28 entries). Both tiers are fully annotated.

---

## 3a. Training objective: cross-entropy vs proper scoring rules, label smoothing, focal loss, ordinal losses

### 3a.1 [T1] Mukhoti et al. (2020) - focal loss vs CE, Brier, label smoothing on calibration

**APA 7.** Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P. H. S., & Dokania, P. K. (2020). Calibrating deep neural networks using focal loss. *Advances in Neural Information Processing Systems, 33*. https://arxiv.org/abs/2002.09437

**Year / venue / grade.** 2020, NeurIPS 2020, PR.

**Relevance.** The only source found that trains the *same* architectures under CE, Brier, MMCE, label smoothing and focal loss and reports ECE both before and after temperature scaling, on vision *and* a text dataset. Directly answers "does the training loss matter once you temperature-scale?".

**Key measured findings (Table 1; ECE in %, 15 bins; T = fitted temperature).**
- CIFAR-100 / ResNet-50: CE 17.52 -> 3.42 after TS (T=2.1), error 23.30; Brier 6.52 -> 3.64 (T=1.1), error 23.39; MMCE 15.32 -> 2.38 (T=1.8); LS-0.05 7.81 -> 4.01 (T=1.1), error 23.43; focal FLSD-53 4.50 -> 2.00 (T=1.1), error 23.22.
- CIFAR-10 / ResNet-50: CE 4.35 -> 1.35 (T=2.5), error 4.95; Brier 1.82 -> 1.08; LS-0.05 2.96 -> 1.67; FLSD-53 1.55 -> 0.95, error 4.98.
- 20 Newsgroups / global-pooling CNN (text): CE 17.92 -> 2.39 (T=3.4), error 26.68; Brier 13.58 -> 3.22 (T=2.3); LS-0.05 4.79 -> 2.54, error 26.03; FLSD-53 6.92 -> 2.19 (T=1.5), error 27.98.
- Pattern: Brier and focal are far better calibrated *before* TS (fitted T close to 1.0), but after TS the gap to CE narrows to roughly 1-2 ECE points; on text, CE after TS is as good as any alternative and LS/focal cost ~0.5-1.3 points of accuracy.

**Methodology.** Train from scratch, 5 seeds not reported per cell; TS fitted on validation split; ECE with 15 equal-width bins plus AdaECE and classwise ECE in appendix; a schedule for gamma (FLSD-53: gamma=5 for p<0.2, gamma=3 otherwise) chosen by a Lagrange-based argument.

**Limitations.** No transformer encoders; one text dataset; ECE binning known to be noisy; focal loss changes the implied risk and can reduce accuracy on text.

**Contribution.** Shows focal loss implicitly regularizes entropy and yields near-T=1 models; establishes the "pre-TS vs post-TS" comparison protocol reused by later work.

### 3a.2 [T1] Müller, Kornblith & Hinton (2019) - label smoothing: better calibration, worse teacher

**APA 7.** Müller, R., Kornblith, S., & Hinton, G. E. (2019). When does label smoothing help? *Advances in Neural Information Processing Systems, 32*. https://arxiv.org/abs/1906.02629

**Year / venue / grade.** 2019, NeurIPS 2019, PR.

**Relevance.** Two findings pull in opposite directions for this project: LS improves calibration roughly as much as TS, but an LS-trained teacher is a worse distillation teacher.

**Key measured findings (Table 2 and Section 5).**
- ECE, ResNet-56 / CIFAR-100: hard targets 0.150; TS (T=1.9) 0.021; LS alpha=0.05 0.024.
- ECE, Inception-v4 / ImageNet: 0.071; TS (T=1.4) 0.022; LS alpha=0.1 0.035.
- ECE, Transformer En-De: 0.056; TS (T=1.13) 0.018; LS alpha=0.1 0.019.
- Distillation, MNIST: teacher trained with hard targets 0.67% error -> distilled student 0.74%; teacher trained with LS 0.59% error (better teacher) -> distilled student 0.91% (worse student). CIFAR-10 ResNet-56 -> AlexNet shows the same ordering (fig. 6).

**Methodology.** Same architecture trained with/without LS; ECE 15 bins; penultimate-layer visualization shows LS collapses within-class representations into tight clusters, erasing inter-class similarity structure the student needs.

**Limitations.** Vision/MT only; TS numbers are the authors' own; no NLP classification.

**Contribution.** Mechanistic explanation of why LS helps calibration but hurts "dark knowledge"; widely replicated.

### 3a.3 [T1] Xia et al. (2025) - label smoothing degrades selective classification

**APA 7.** Xia, G., Laurent, O., Franchi, G., & Bouganis, C.-S. (2025). Towards understanding why label smoothing degrades selective classification and how to fix it. *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2403.14715

**Year / venue / grade.** 2024 preprint, ICLR 2025, PR.

**Relevance.** The product metric here is coverage at fixed risk; this paper measures exactly that under LS and shows a catastrophic effect at strict risk budgets.

**Key measured findings (Fig. 1 table; ResNet-50, ImageNet-1k, MSP confidence).** Coverage at 1% risk: CE 15.66%; LS alpha=0.1 0.05%; alpha=0.2 0.06%; alpha=0.3 0.02%; LS alpha=0.2 with post-hoc logit normalization 22.31%. Across CNN and ViT-S/16 and semantic segmentation, LS consistently increases AURC even though it slightly improves full-coverage accuracy (Sec. 5). Mechanism (Sec. 6): LS suppresses the max logit *more* on likely-correct samples than on likely-wrong ones, degrading the confidence rank ordering. Logit normalization (Cattelan & Silva, 2024) largely recovers the loss.

**Methodology.** Train ResNet-50 and ViT-S-16 from scratch at several alpha; risk-coverage curves and AURC; ImageNet-Sketch mixed in to test shift; gradient analysis at the logit level.

**Limitations.** Vision only; no temperature scaling of LS models reported in the main table; logit-norm hyperparameter tuned on validation AURC.

**Contribution.** First systematic demonstration that a calibration-improving regularizer can destroy selective-prediction ranking; introduces a cheap post-hoc fix.

### 3a.4 [T1] Hui & Belkin (2021) - square loss vs cross-entropy, including BERT fine-tuning

**APA 7.** Hui, L., & Belkin, M. (2021). Evaluation of neural architectures trained with square loss vs cross-entropy in classification tasks. *International Conference on Learning Representations (ICLR 2021)*. https://arxiv.org/abs/2006.07322

**Year / venue / grade.** 2021, ICLR 2021, PR.

**Relevance.** Square loss on one-hot targets is the Brier score. This is the largest accuracy comparison of Brier-as-training-loss vs CE that includes transformer fine-tuning on GLUE-style tasks.

**Key measured findings (Tables 2-3).** Fine-tuned BERT test accuracy, square vs CE: MRPC 83.8 vs 82.1; SST-2 94.0 vs 93.9; QNLI 90.6 vs 90.6; QQP 88.9 vs 88.9. F1: MRPC 88.1 vs 86.7; QQP 70.9 vs 70.7. Overall "square loss ... better or equal accuracy compared with cross-entropy in 22 out of 28 tasks" (NLP and ASR); on vision CE keeps a slight edge (CIFAR-10 WRN 96.3 vs 95.9; EfficientNet ImageNet top-1 77.0 vs 74.6). Square loss less sensitive to initialization.

**Methodology.** Rescaled square loss (class-count-dependent scaling k, M) with equal compute budgets; 5 seeds; no calibration metrics.

**Limitations.** Accuracy only; no ECE/Brier/selective metrics; text-classification numbers are close to noise level.

**Contribution.** Removes the presumption that CE is necessary for accuracy on encoder fine-tuning; enables Brier as a first-class training objective.

### 3a.5 [T1] Shao et al. (2024) - Brier and spherical scores as LM training losses

**APA 7.** Shao, C., Meng, F., Liu, Y., & Zhou, J. (2024). Language generation with strictly proper scoring rules. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*. https://arxiv.org/abs/2405.18906

**Year / venue / grade.** 2024, ICML 2024, PR.

**Relevance.** Closest published analogue to the Laya replica's use of log + spherical scoring rules; shows the token-level losses are trainable at 7-13B scale.

**Key measured findings.** WMT14 En-De BLEU: log-score Transformer 27.61; +Brier 28.01 (p<0.01); +Spherical 28.07 (p<0.01). En-Fr 41.92 / 42.50 / 42.09. Alpaca-tuned LLaMA-7B on WMT22 En-De: 25.42 -> +Brier 29.15, +Spherical 29.07; De-En 17.93 -> 21.09 / 21.05. LLaMA-13B En-De 29.35 -> 29.54 / 29.82. Uses a smoothed form S^eps (eps=0.1) to keep the bounded scores strictly proper and non-positive.

**Methodology.** Fine-tune from a log-score-pretrained model by swapping the loss; no other hyperparameters changed; generation metrics only.

**Limitations.** No calibration metric (ECE, Brier score on held-out) is reported; generation, not classification; gains may partly reflect the sharpening of the output distribution rather than better probability estimates.

**Contribution.** Demonstrates non-local proper scoring rules are practical drop-in training losses for LMs.

### 3a.6 [T1] Shi, Cao & Raschka (2023) - CORN, rank-consistent ordinal loss (with CORAL, 2020)

**APA 7.** Shi, X., Cao, W., & Raschka, S. (2023). Deep neural networks for rank-consistent ordinal regression based on conditional probabilities. *Pattern Analysis and Applications, 26*, 941-955. https://arxiv.org/abs/2111.08851

Companion: Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural networks with application to age estimation. *Pattern Recognition Letters, 140*, 325-331. https://arxiv.org/abs/1901.07884

**Year / venue / grade.** 2023 (PAA) and 2020 (PRL), PR.

**Relevance.** The score-type decisions in this project are ordinal. CORAL/CORN replace K-way softmax with K-1 conditional binary tasks whose outputs are guaranteed rank-monotone; they output probabilities P(y > r_k) that can be turned into a full ordinal distribution.

**Key measured findings (CORN Table 3, test, mean +/- sd over 5 seeds).**
- MORPH-2 / ResNet-34: CE MAE 3.73+/-0.12, RMSE 5.04; CORAL 2.99+/-0.04, 4.01; CORN 2.98+/-0.02, 3.99.
- AFAD / ResNet-34: CE 3.28+/-0.04, 4.19; CORAL 2.99+/-0.03, 3.70; CORN 2.81+/-0.02, 3.46.
- Fireman (tabular MLP): CE 0.80, CORAL 0.82, CORN 0.76.
- CORAL (2020) reports the same ordering on MORPH-2, AFAD, CACD.

**Methodology.** Shared backbone; CORAL uses weight-sharing with per-threshold biases; CORN trains on conditional subsets so consistency holds without weight sharing; MAE/RMSE on argmax-derived rank.

**Limitations.** Age-estimation image datasets and one tabular set; no calibration metrics or probabilistic scoring (RPS) reported; no transformers.

**Contribution.** Rank-consistency guarantees plus a measurable MAE gain over plain CE on ordinal targets.

### 3a.7 [T2] Beckham & Pal (2017) - unimodal (Poisson/binomial) ordinal outputs

**APA 7.** Beckham, C., & Pal, C. (2017). Unimodal probability distributions for deep ordinal classification. *Proceedings of the 34th International Conference on Machine Learning, PMLR 70*, 411-419. https://arxiv.org/abs/1705.05278

**Year / venue / grade.** 2017, ICML 2017, PR.

**Relevance.** Alternative parameterization for ordinal decisions: a single scalar plus a learned temperature tau produces a unimodal distribution over K ordered classes, giving a natural per-example uncertainty.

**Key measured findings (figs. 3-4, Adience age and Diabetic Retinopathy).** With tau=1 the unimodal heads lose accuracy to CE; with learned tau they are "on par" with CE on accuracy and QWK; binomial beats Poisson; the "expectation trick" (predict E[y]) closes most of the remaining accuracy gap. Numbers are figure-only.

**Methodology.** ResNet backbone; compares CE, squared error, Poisson-CE, binomial-CE, EMD variants.

**Limitations.** No tables; two datasets; no calibration metrics.

**Contribution.** Shows CE over ordinal classes can yield multimodal, unrankable distributions and offers a constrained alternative.

### 3a.8 [T2] Diaz & Marathe (2019) - soft ordinal labels (SORD)

**APA 7.** Diaz, R., & Marathe, A. (2019). Soft labels for ordinal regression. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019)*, 4738-4747. https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html

**Year / venue / grade.** 2019, CVPR 2019, PR.

**Relevance.** Distance-aware label smoothing for ordinal targets; the target distribution is softmax(-phi(r_t, r_i)), so the training signal is a proper-scoring-rule-friendly soft distribution rather than one-hot. Relevant to score-type decisions and to the LS-vs-selective-prediction tension (3a.3).

**Key measured findings.** Adience (Table 2): SORD 59.6+/-3.6% accuracy, MAE 0.49+/-0.05 vs CNN-POR 57.4+/-5.8, 0.55+/-0.08. Image Aesthetics (Table 1, overall): accuracy RED-SVM 64.59, CNNm 69.45, Niu et al. 68.96, CNN-POR 70.05, SORD 72.03; MAE 0.330 / 0.376 / 0.326 / 0.316 / 0.290.

**Methodology.** VGG-16 backbone; CE against SORD soft targets; no architectural change.

**Limitations.** No calibration or selective-prediction metrics; vision only.

**Contribution.** Simple, architecture-agnostic ordinal soft-label encoding with measurable MAE gains.

### 3a.9 [T2] Haas et al. (2026) - ranked probability score for conformal ordinal classification

**APA 7.** Haas, S., Killmaier, L., Javanmardi, A., & Hüllermeier, E. (2026). Reliable conformal prediction for ordinal classification using the ranked probability score. *arXiv preprint* arXiv:2606.24959. https://arxiv.org/abs/2606.24959

**Year / venue / grade.** 2026, preprint, PP.

**Relevance.** Bridges 3a and 3c for the ordinal decision type: uses RPS (the proper scoring rule Laya trains with) as a conformal non-conformity score.

**Key measured findings.** RPS-based sets are contiguous and median-centered by construction; across ordinal image and tabular datasets at alpha in {0.02, 0.05, 0.1}, LAC and APS violate contiguity while RPS sets achieve a "favorable balance between prediction set width and ordinal miscoverage" (Tables 4, 7, 9; LightGBM and CNN base models). Specific width numbers are dataset-specific; see paper Tables 7 and 9.

**Methodology.** Split conformal; base models LightGBM (tabular) and CNNs (images); compares LAC, APS, greedy interval selection, RPS.

**Limitations.** Preprint; no NLP; no comparison of RPS-*trained* vs CE-trained base models.

**Contribution.** Establishes RPS as a natural conformal score for ordinal outputs.

### 3a.10 [T2] Gneiting & Raftery (2007) - strictly proper scoring rules (foundational)

**APA 7.** Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association, 102*(477), 359-378. https://doi.org/10.1198/016214506000001437

**Year / venue / grade.** 2007, JASA, FND. (DOI verified through the Crossref API; publisher page blocks automated fetches.)

**Relevance.** Defines the family (log, Brier/quadratic, spherical, RPS/CRPS) and proves strict propriety, i.e. that each is minimized in expectation only by the true conditional distribution. This is the theoretical basis for the Laya replica's loss and for the calibration-sharpness decomposition used in 3c.

**Key findings.** Theory: characterization of proper scoring rules via convex functions; RPS as the discrete CRPS; the log score is the only *local* strictly proper rule.

**Limitations.** No neural-network experiments.

**Contribution.** Canonical reference.

---

## 3b. Distillation into small classifiers and encoders

### 3b.1 [T1] Hinton, Vinyals & Dean (2015) - knowledge distillation (foundational)

**APA 7.** Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NIPS 2014 Deep Learning Workshop*; arXiv:1503.02531. https://arxiv.org/abs/1503.02531

**Year / venue / grade.** 2015, workshop + preprint, WS/FND.

**Relevance.** Origin of soft-target distillation with temperature; the numbers give the canonical soft-vs-hard gain at small scale.

**Key measured findings (Sec. 3).** MNIST: large regularized net 67 test errors; small 2x800 ReLU net trained on hard labels 146 errors; same small net trained on the large net's soft targets at T=20 achieves 74 errors. Temperatures 2.5-4 worked best for the smaller student.

**Methodology.** Weighted sum of soft-target CE (scaled by T^2) and hard-label CE.

**Limitations.** MNIST and internal speech data; no calibration metric.

**Contribution.** Defines the method and the T^2 gradient scaling.

### 3b.2 [T1] Sanh et al. (2019) - DistilBERT

**APA 7.** Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. *arXiv preprint* arXiv:1910.01108 (NeurIPS 2019 EMC^2 workshop). https://arxiv.org/abs/1910.01108

**Year / venue / grade.** 2019, workshop + preprint, WS.

**Relevance.** Reference point for "what fraction of a ~2x larger teacher does a 66M encoder recover" under pretraining-stage soft-target distillation.

**Key measured findings (Tables 1-3).** 66M vs 110M parameters (40% smaller). GLUE dev macro: BERT-base 79.5, DistilBERT 77.0 (97% retention). Per task: MNLI 82.2 vs 86.7; QQP 89.2 vs 91.8; SST-2 91.3 vs 92.7; CoLA 51.3 vs 56.3; MRPC 87.5 vs 88.6; QNLI 88.5 vs 89.6; RTE 59.9 vs 69.3; STS-B 86.9 vs 89.0. IMDb 92.82 vs 93.46. SQuAD 1.1 F1/EM 85.8/77.7 vs 88.5/81.2. CPU inference 410 s vs 668 s (60% faster).

**Methodology.** Triple loss: MLM + soft-target KL at temperature + cosine embedding; student initialized from every other teacher layer; distilled on the pretraining corpus.

**Limitations.** No calibration metrics; single seed; RTE drop (9.4 points) shows small-data tasks suffer most.

**Contribution.** Popularized task-agnostic pretraining distillation of encoders.

### 3b.3 [T1] Jiao et al. (2020) - TinyBERT (4-layer 14.5M and 6-layer 67M)

**APA 7.** Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2020). TinyBERT: Distilling BERT for natural language understanding. *Findings of the Association for Computational Linguistics: EMNLP 2020*, 4163-4174. https://arxiv.org/abs/1909.10351

**Year / venue / grade.** 2020, Findings of EMNLP, PR.

**Relevance.** Gives the recovery fraction at 14.5M (7.5x smaller than teacher) and the ablation showing task-specific distillation + data augmentation matter more than general distillation.

**Key measured findings (Tables 2, 4).** GLUE test average: BERT-base teacher (109M) 79.5; TinyBERT4 (14.5M) 77.0 = 96.8% of teacher, 9.4x faster; TinyBERT6 (67M) 79.4 (on par). Ablation on TinyBERT4 (dev avg over MNLI-m/MRPC/CoLA): full 75.6; without general distillation 72.5; without task-specific distillation 68.5; without data augmentation 68.4.

**Methodology.** Two-stage transformer distillation (embedding, attention, hidden-state MSE, then prediction-layer soft CE); GloVe/BERT-based augmentation of task data.

**Limitations.** No calibration metrics; augmentation is a large part of the gain; comparisons at test use best-of-dev.

**Contribution.** Layer-wise distillation recipe and a clean ablation of where the gain comes from.

### 3b.4 [T1] Wang et al. (2020, 2021) - MiniLM and MiniLMv2

**APA 7.** Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. *Advances in Neural Information Processing Systems, 33*. https://arxiv.org/abs/2002.10957

Wang, W., Bao, H., Huang, S., Dong, L., & Wei, F. (2021). MiniLMv2: Multi-head self-attention relation distillation for compressing pretrained transformers. *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, 2140-2151. https://arxiv.org/abs/2012.15828

**Year / venue / grade.** 2020 NeurIPS; 2021 Findings of ACL; PR.

**Relevance.** Strongest published recovery numbers for 22-81M encoders, including a 30M student distilled from a 355M teacher (12x). These are the natural "small encoder" starting points at the 0.1-0.3B target.

**Key measured findings.**
- MiniLM (Table 2, dev): 6x768 (66M) from BERT-base: MNLI-m 84.0, SST-2 92.0, QNLI 91.0, QQP 91.0, RTE 71.5, MRPC 88.4, CoLA 49.2, SQuAD2 F1 76.4; 2.0x faster while "retaining more than 99% accuracy on SQuAD 2.0 and several GLUE tasks" relative to BERT-base.
- MiniLMv2 (Table 1, dev): 6x384 (30M) from RoBERTa-large: SQuAD2 76.4, MNLI-m 84.4, SST-2 92.0, avg 79.5 (5.3x speedup); 6x768 (81M) from RoBERTa-large: SQuAD2 81.6, MNLI-m 87.0, avg 83.8 vs RoBERTa-base (125M) avg 85.4 (MNLI 87.6, SQuAD2 83.7). Table 3: 12x768 student from RoBERTa-large avg 87.6 > RoBERTa-base 86.1.

**Methodology.** Distill only the last-layer self-attention distributions and value relations (v1) / query-key-value relations across heads (v2); teacher-assistant for very small students; no task labels during distillation.

**Limitations.** No calibration metrics; results are averaged fine-tuning runs (4 seeds) on dev; students share tokenizer with teacher.

**Contribution.** Shows a 30M encoder recovers ~93% of a 125M model's GLUE average and a 81M student can exceed a same-size base model when the teacher is larger.

### 3b.5 [T2] Sun et al. (2020) - MobileBERT (25M)

**APA 7.** Sun, Z., Yu, H., Song, X., Liu, R., Yang, Y., & Zhou, D. (2020). MobileBERT: A compact task-agnostic BERT for resource-limited devices. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 2158-2170. https://arxiv.org/abs/2004.02984

**Year / venue / grade.** 2020, ACL 2020, PR.

**Relevance.** 25M-parameter encoder within 0.6 GLUE points of BERT-base; shows depth-with-bottlenecks plus progressive layer transfer from a specially trained teacher.

**Key measured findings (Table 4 GLUE test).** BERT-base 109M: 78.3; MobileBERT 25.3M: 77.7 (MNLI 83.3/82.6, SST-2 92.8, QQP 70.2 F1, QNLI 90.6); MobileBERT-tiny 15.1M: 75.8; TinyBERT 14.5M: 75.4; DistilBERT-4L 52.2M: no avg reported. SQuAD v1.1 dev F1 90.0 (vs BERT-base 88.5). 62 ms latency on Pixel 4.

**Methodology.** Inverted-bottleneck BERT-large teacher (IB-BERT); feature-map and attention transfer layer by layer; then pre-training distillation.

**Limitations.** Requires training a bespoke teacher; no calibration metrics.

**Contribution.** Demonstrates that narrow-and-deep students recover ~99% of base GLUE at 4.3x fewer parameters.

### 3b.6 [T1] Turc et al. (2019) - Well-Read Students Learn Better (4.4M-110M students)

**APA 7.** Turc, I., Chang, M.-W., Lee, K., & Toutanova, K. (2019). Well-read students learn better: On the importance of pre-training compact models. *arXiv preprint* arXiv:1908.08962. https://arxiv.org/abs/1908.08962

**Year / venue / grade.** 2019, preprint, PP (very widely cited; released the BERT-Tiny/Mini/Small/Medium checkpoints).

**Relevance.** Only source that sweeps 24 student sizes from 4.4M to 110M and separates the contributions of pre-training vs distillation vs both, with a 340M teacher.

**Key measured findings.**
- Table 3 (6/768 student, 12/768 teacher, GLUE test meta-score): train-from-scratch+fine-tune (TF) 80.5; pre-train+fine-tune (PF) 81.6; Pre-trained Distillation (PD) 82.1; MNLI-m/mm PD 82.8/82.2 vs PF 81.8/81.1. Dev: PD 84.4 vs PF 82.8 vs DistilBERT 82.3.
- Fig. 7 (fig.; BERT-large 340M teacher): MNLI accuracy Tiny (4.4M) ~63 (PD) / ~61 (PF) / ~58 (D); Mini (11.3M) ~70 / ~67 / ~65; Base (110M) ~78 / ~76 / ~74. SST-2 Tiny ~76 / ~74 / ~72; Mini ~88 / ~86 / ~84.
- Sec. 6.2: with an 8M in-domain transfer set (Amazon Book Reviews), PD lets an 11M Transformer-Mini "recover the accuracy of the teacher" that plain distillation only reaches with a 10x larger student; PF beats plain distillation by an average of 12% on MNLI when the transfer set is moderate (1.3M) and slightly out-of-domain.

**Methodology.** Pre-train student with MLM, then distill from fine-tuned BERT-large on an unlabeled transfer set, then optionally fine-tune; 5-run dev averages.

**Limitations.** Preprint; figure-only numbers for most sizes; no calibration.

**Contribution.** Establishes that pre-training and distillation compound and that transfer-set domain/size governs distillation gains.

### 3b.7 [T1] Furlanello et al. (2018) - Born-Again Networks (identical-capacity self-distillation)

**APA 7.** Furlanello, T., Lipton, Z. C., Tschannen, M., Itti, L., & Anandkumar, A. (2018). Born again neural networks. *Proceedings of the 35th International Conference on Machine Learning, PMLR 80*, 1607-1616. https://arxiv.org/abs/1805.04770

**Year / venue / grade.** 2018, ICML 2018, PR.

**Relevance.** Evidence that soft targets improve a student *without* a larger teacher; a candidate cheap trick for the 0.1-0.3B model.

**Key measured findings (Tables 1-2, CIFAR-100 test error %).** DenseNet-80-120 teacher 16.87 -> BAN-1 16.00; DenseNet-112-33 18.25 -> 16.95 (BAN-1) -> 15.68 (BAN ensemble); DenseNet-90-60 17.69 -> 16.69; CIFAR-10 DenseNet-90-60 3.81 -> 3.5. ResNet students trained from DenseNet teachers and vice versa also improve (Table 3).

**Methodology.** Train generation k+1 with soft targets from generation k; also "confidence-weighted by teacher max" and "dark-knowledge-removed" ablations.

**Limitations.** Vision only; gains are ~1 point; no calibration metrics.

**Contribution.** Self-distillation baseline; ablations show most gain persists even when non-argmax logit information is removed.

### 3b.8 [T1] Stanton et al. (2021) - Does knowledge distillation really work? (fidelity vs generalization)

**APA 7.** Stanton, S., Izmailov, P., Kirichenko, P., Alemi, A. A., & Wilson, A. G. (2021). Does knowledge distillation really work? *Advances in Neural Information Processing Systems, 34*. https://arxiv.org/abs/2106.05945

**Year / venue / grade.** 2021, NeurIPS 2021, PR.

**Relevance.** Directly relevant to "distil teacher probabilities": measures how well students actually match teacher distributions and shows that matching is hard even with capacity.

**Key measured findings (Secs. 4-6, CIFAR-100 ResNet-56).** Self-distillation: student test agreement with teacher stays "between 80% and 90%" across augmentation strategies; train agreement only 83.3% after 5,000 epochs with SGD (vs 78.95% at standard length). Ensemble teacher (3x ResNet-56): agreement "below 80%" even as distillation data grows; but for ensemble teachers, higher fidelity does translate into higher student accuracy (Fig. 1b), whereas in self-distillation fidelity and accuracy move in opposite directions (Fig. 1a). ECE and NLL reported alongside accuracy; optimization, not data, is identified as the bottleneck.

**Methodology.** KL(teacher || student) on soft targets; vary distillation set (in-distribution, augmented, synthetic GAN images); measure top-1 agreement and KL.

**Limitations.** Vision only; CIFAR-scale models.

**Contribution.** Separates fidelity from generalization and shows students are not faithful probability copies of teachers.

### 3b.9 [T2] Cho & Hariharan (2019) - bigger teachers do not make better students

**APA 7.** Cho, J. H., & Hariharan, B. (2019). On the efficacy of knowledge distillation. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2019)*, 4794-4802. https://arxiv.org/abs/1910.01348

**Year / venue / grade.** 2019, ICCV 2019, PR.

**Relevance.** Capacity-mismatch evidence for a 10x+ teacher/student gap (the 4B -> 0.3B setting).

**Key measured findings (Table 1, ImageNet, ResNet-18 student, top-1 error %).** From scratch 30.24. KD from ResNet-18 teacher (err 30.24) 30.57; from ResNet-34 (26.70) 30.79; from ResNet-50 (23.85) 30.95. Early-stopped-teacher KD (ESKD): 29.01 (R18), 29.16 (R34), 29.35 (R50); ESKD ResNet-152 teacher 29.45. Same ordering on CIFAR-10 (Tables 3-4).

**Methodology.** T=4, alpha=0.9; full KD vs early-stopped teachers; sequential KD shown ineffective (Table 5).

**Limitations.** Vision; ResNet family only; no calibration.

**Contribution.** Teacher accuracy is a poor predictor of student accuracy; an under-trained (less confident) teacher transfers better.

### 3b.10 [T1] Beyer et al. (2022) - a good teacher is patient and consistent

**APA 7.** Beyer, L., Zhai, X., Royer, A., Markeeva, L., Anil, R., & Kolesnikov, A. (2022). Knowledge distillation: A good teacher is patient and consistent. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022)*, 10925-10934. https://arxiv.org/abs/2106.05237

**Year / venue / grade.** 2022, CVPR 2022, PR.

**Relevance.** Best-documented recipe for closing a large teacher/student gap with pure soft-target ("function matching") distillation; establishes that the student must see the *same* augmented input as the teacher and train far longer than usual.

**Key measured findings.** ResNet-50 student reaches 82.8% ImageNet top-1 (teacher BiT-M-R152x2 at 83.0%) with 9,600 distillation epochs; previous best published ResNet-50 was 78.8% (or 80.49% with heavy tuning). On Flowers102/Pets/Food101/SUN397 (Fig. 3), "fixed teacher" and "independent noise" variants plateau below "consistent teaching"; function matching with mixup is best. Student initialized from BiT-M-R50 (78.4%) reaches 81.45% (Fig. 5).

**Methodology.** KL to teacher only (no ground-truth term); aggressive crops + mixup shared between teacher and student; cosine schedule; up to 1M epochs on small sets without overfitting.

**Limitations.** Vision; compute-heavy; no calibration.

**Contribution.** Shows most "KD doesn't work" results stem from inconsistent views and short schedules.

### 3b.11 [T2] Wang et al. (2023) - How to distill your BERT (objectives and initialization)

**APA 7.** Wang, X., Weissweiler, L., Schütze, H., & Plank, B. (2023). How to distill your BERT: An empirical study on the impact of weight initialisation and distillation objectives. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Short Papers)*, 1843-1852. https://arxiv.org/abs/2305.15032

**Year / venue / grade.** 2023, ACL 2023, PR.

**Relevance.** Controlled comparison of soft-target-only ("vanilla KD") vs intermediate-layer objectives for 6-layer BERT students under identical settings.

**Key measured findings (Tables 1-2, GLUE dev avg).** Task-specific distillation (student initialized from every 4th teacher layer): vanilla KD 71.8; Hid-Seq 75.8; Att-MSE 76.5; Att-KL+Val-KL 77.5. Task-agnostic 6-layer students from BERT-base (random init): vanilla KD 79.3; Att-MSE 81.3 (best); DistilBERT 78.5; TinyBERT 79.9; MiniLM 81.0. Initializing from *lower* teacher layers improves vanilla KD on QNLI "from 68.1%" upward (Table 3).

**Methodology.** 4 runs per cell; same data and steps; grid over learning rate/batch.

**Limitations.** Short paper; no calibration; BERT-base teacher only.

**Contribution.** Attention transfer > logit-only KD by ~2-6 GLUE points; initialization layer choice matters.

### 3b.12 [T2] Tang et al. (2019) - BERT-large into a 1M-parameter BiLSTM

**APA 7.** Tang, R., Lu, Y., Liu, L., Mou, L., Vechtomova, O., & Lin, J. (2019). Distilling task-specific knowledge from BERT into simple neural networks. *arXiv preprint* arXiv:1903.12136. https://arxiv.org/abs/1903.12136

**Year / venue / grade.** 2019, preprint, PP.

**Relevance.** Extreme-ratio (350x) task-specific soft-logit distillation for classification and pair tasks; quantifies what a tiny student recovers with augmentation.

**Key measured findings (Table 1, test).** SST-2: BERT-large 94.9; BERT-base 93.5; distilled BiLSTM_SOFT 90.7; BiLSTM (hard labels) 86.7. QQP F1/acc: 72.1/89.3 (BERT-large); 68.2/88.1 (distilled); 63.7/86.2 (BiLSTM). MNLI-m/mm: 86.7/85.9; 73.0/72.6; 68.7/68.3. Parameters 335M vs 0.96M; inference 434x faster (Table 2).

**Methodology.** MSE on logits from fine-tuned BERT-large; data augmentation via masking, POS-guided word replacement, n-gram sampling.

**Limitations.** Preprint; no calibration; augmentation essential.

**Contribution.** Soft logits + augmentation recover 4-5 points over hard labels at 1M scale.

### 3b.13 [T1] Hsieh et al. (2023) - Distilling step-by-step (540B PaLM -> 220M-11B T5)

**APA 7.** Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.-Y., & Pfister, T. (2023). Distilling step-by-step! Outperforming larger language models with less training data and smaller model sizes. *Findings of the Association for Computational Linguistics: ACL 2023*, 8003-8017. https://arxiv.org/abs/2305.02301

**Year / venue / grade.** 2023, Findings of ACL, PR.

**Relevance.** Canonical "decoder LLM teacher -> small seq2seq student on classification/QA" result; shows rationale supervision beats label-only distillation for NLI and multiple-choice QA.

**Key measured findings (Table 1; 220M T5-base, 100% data, accuracy).** Standard fine-tuning: e-SNLI 88.38, ANLI 43.58, CQA 62.19, SVAMP 62.63. Distilling step-by-step with 540B PaLM rationales: 89.51, 49.58, 63.29, 65.50; with a 20B GPT-NeoX teacher: 89.12, 48.15, 63.25, 63.00. Table 2: single-task rationale+label training can be worse than fine-tuning (ANLI 43.50), multi-task is what delivers the gain. Sec. 4.2: on ANLI the method beats standard fine-tuning and standard distillation by an average of 8% and 13%; a 770M T5 exceeds 540B PaLM few-shot CoT on ANLI with 80% of the labeled data (Fig. 8); with unlabeled data only, 11B T5 beats the teacher on 3 of 4 datasets (Fig. 7).

**Methodology.** LLM produces rationales; student trained multi-task (label prediction + rationale generation) with a shared encoder-decoder; rationales not needed at inference.

**Limitations.** Seq2seq students (T5) rather than encoders; no calibration or selective metrics; ANLI gain dominates the headline.

**Contribution.** Shows LLM-generated auxiliary supervision, not just labels, transfers reasoning into sub-1B students.

### 3b.14 [T2] Wang et al. (2021) - GPT-3 as a labeler

**APA 7.** Wang, S., Liu, Y., Xu, Y., Zhu, C., & Zeng, M. (2021). Want to reduce labeling cost? GPT-3 can help. *Findings of the Association for Computational Linguistics: EMNLP 2021*, 4195-4205. https://arxiv.org/abs/2108.13487

**Year / venue / grade.** 2021, Findings of EMNLP, PR.

**Relevance.** Early hard-label LLM->encoder pipeline with an explicit cost accounting.

**Key measured findings.** To reach the same downstream performance (RoBERTa / PEGASUS students), GPT-3 labels cost 50-96% less than human labels: 96% saving on SST-2, 93.8% on Gigaword, 50-75% elsewhere (Sec. 1, Table 1: SST-2 human $0.11/label vs GPT-3 $0.0023-0.0069 depending on shots). Mixing GPT-3 pseudo-labels with a small human budget beats either alone.

**Methodology.** Few-shot GPT-3 labeling via logits over label tokens; also uses GPT-3 logits to pick uncertain samples for human relabeling.

**Limitations.** GPT-3 (175B) teacher; cost model dated; no calibration.

**Contribution.** Establishes cost-equivalence curves for LLM labels.

### 3b.15 [T2] Pangakis & Wolken (2024) - LLM-generated labels vs human labels for BERT-family students

**APA 7.** Pangakis, N., & Wolken, S. (2024). Knowledge distillation in automated annotation: Supervised text classification with LLM-generated training labels. *Proceedings of the Sixth Workshop on Natural Language Processing and Computational Social Science (NLP+CSS)*. https://arxiv.org/abs/2406.17633

**Year / venue / grade.** 2024, workshop, WS.

**Relevance.** Measures the gap between BERT/DistilBERT/RoBERTa fine-tuned on 1,000 GPT-4 labels vs 1,000 human labels across 14 tasks.

**Key measured findings (Sec. 3, median over 14 CSS tasks).** F1: DistilBERT on 1,000 human labels 0.641; BERT on human 0.624; GPT-4 few-shot 0.592; BERT on 1,000 GPT-4 labels 0.586. GPT-4-trained students have median recall 0.746 (higher than any human-label student) but precision 0.214 lower than human-label BERT. Mistral-7B labels are 0.16 F1 worse on average than GPT-4 labels.

**Methodology.** Hard labels from LLM; standard fine-tuning; per-task tables in appendix.

**Limitations.** Workshop paper; hard labels only; social-science tasks; no calibration.

**Contribution.** Quantifies the ~0.04 F1 median gap and the precision/recall shift induced by LLM labels.

### 3b.16 [T2] Di Palo, Singhi & Fadlallah (2024) - Performance-Guided KD from LLMs into BERT-base

**APA 7.** Di Palo, F., Singhi, P., & Fadlallah, B. (2024). Performance-guided LLM knowledge distillation for efficient text classification at scale. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (Industry Track)*. https://arxiv.org/abs/2411.05045

**Year / venue / grade.** 2024, EMNLP 2024, PR.

**Relevance.** Active-learning-style distillation from Claude-3 / LLaMA-3-8B teachers into BERT-base for many-class classification with few labels; reports cost/latency.

**Key measured findings (Tables 2, 4, 5).** BERT-base + PGKD (1,000 seed samples) accuracy: AG-News 0.895, Yahoo Answers 0.685, HuffPost 0.519, Amazon Reviews 0.443; removing validation-guided generation drops these to 0.893 / 0.669 / 0.501 / 0.419; removing hard-negative mining to 0.887 / 0.675 / 0.510 / 0.433. Inference: BERT-base+PGKD 0.46 s per batch on GPU vs Claude-3 zero-shot 60.64 s (~130x) and 25x cheaper on CPU.

**Methodology.** Teacher generates synthetic examples guided by the student's validation report and hard negatives; student fine-tuned iteratively.

**Limitations.** Industry track; hard labels/synthetic text, not teacher probabilities; no calibration.

**Contribution.** Practical LLM->encoder pipeline with measured cost ratios.

### 3b.17 [T2] Ye et al. (2022) - ZeroGen (data-free distillation via generated datasets)

**APA 7.** Ye, J., Gao, J., Li, Q., Xu, H., Feng, J., Wu, Z., Yu, T., & Kong, L. (2022). ZeroGen: Efficient zero-shot learning via dataset generation. *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, 11653-11669. https://arxiv.org/abs/2202.07922

**Year / venue / grade.** 2022, EMNLP 2022, PR.

**Relevance.** Synthetic-data ("data-free") distillation from a 1.5B decoder into 7M-66M students; quantifies the gap to supervised training.

**Key measured findings (Table 1).** DistilBERT (66M) trained on GPT2-XL-generated data: IMDb 84.28, SST-2 87.27, QNLI 71.19, RTE 59.93 vs supervised DistilBERT 87.24 / 89.68 / 88.05 / 58.12. LSTM (~7M): 79.80 / 78.40 / 52.26 / 58.85 vs supervised 84.60 / 76.30 / 69.00 / 54.87. ZeroGen beats prompting GPT2-XL directly on most tasks; QA (SQuAD) remains far below supervised (25.5/31.5 vs 76.3/84.7 EM/F1).

**Methodology.** Prompt-based generation of x given y, filtering, train tiny model; decoding strategy ablation (Table 2).

**Limitations.** Sentence-level tasks only; no calibration; large gap on NLI/QA.

**Contribution.** Establishes the data-free distillation baseline for small text models.

### 3b.18 [T2] Schick & Schütze (2021) - DINO (generating datasets with PLMs)

**APA 7.** Schick, T., & Schütze, H. (2021). Generating datasets with pretrained language models. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, 6943-6951. https://arxiv.org/abs/2104.07540

**Year / venue / grade.** 2021, EMNLP 2021, PR.

**Relevance.** Synthetic labeled pairs from GPT2-XL used to fine-tune a small sentence encoder; a template for generating ordinal/graded supervision (STS scores) from a decoder.

**Key measured findings (Table 1, Spearman x100, avg over 7 STS sets).** DINO (STS-x2) 75.20 vs the Sentence-RoBERTa-base row 74.21; DINO (STS-x1x2) 73.77.

**Methodology.** Instruction-style prompts for graded similarity; self-debiasing decoding; student Sentence-RoBERTa-base.

**Limitations.** Similarity regression, not classification; no calibration.

**Contribution.** Shows decoders can generate graded (ordinal-like) targets usable by small encoders.

### 3b.19 [T2] Ghita, Desai & Boier (2026) - scaling laws for task-specific LLM distillation

**APA 7.** Ghita, L., Desai, D., & Boier, I. (2026). Scaling laws for task-specific LLM distillation. *arXiv preprint* arXiv:2606.24747. https://arxiv.org/abs/2606.24747

**Year / venue / grade.** 2026, preprint, PP.

**Relevance.** Recent measurement of label-only vs soft (KL) vs chain-of-thought supervision when compressing a Qwen3-32B teacher into pruned decoder students for a finance classification task.

**Key measured findings.** Pruning + distillation compresses the teacher to 16% of its parameters while "retaining meaningful task accuracy"; in-domain accuracy degrades predictably with compression while general-knowledge benchmarks collapse earlier; blended CoT supervision preserves general knowledge better than label-only.

**Methodology.** Qwen3-32B teacher (thinking mode for CoT traces); iterative structural pruning; LoRA and logit distillation; FinHeadlineMix dataset released.

**Limitations.** Preprint; single domain; decoder students; specific accuracy tables not extracted here.

**Contribution.** Provides scaling-curve evidence on supervision format for sub-teacher students.

### 3b.20 [T2] Mishra, Krishna & Mishra (2023) - distilling a calibrated student from an uncalibrated teacher

**APA 7.** Mishra, I., Krishna, S. V., & Mishra, D. (2023). Distilling calibrated student from an uncalibrated teacher. *arXiv preprint* arXiv:2302.11472. https://arxiv.org/abs/2302.11472

**Year / venue / grade.** 2023, preprint, PP.

**Relevance.** Measures student ECE under vanilla KD vs KD with mixup/CutMix; shows KD alone barely changes calibration.

**Key measured findings (Table 1, CIFAR-100).** WRN-40-2 teacher (acc 77.09, ECE 0.103) -> ShuffleNetV1 student: scratch 70.58 / ECE 0.121; KD 71.05 / 0.108; KD+mixup 74.65 / 0.049; KD+CutMix 74.94 / 0.046. ResNet-50 (78.98, 0.104) -> MobileNetV2: scratch 63.20 / 0.169; KD 63.92 / 0.135; KD+CutMix 67.32 / 0.037.

**Methodology.** Standard KD loss with augmented inputs; ECE and overconfidence error; TinyImageNet and CIFAR-100-C also.

**Limitations.** Preprint; vision only; small teachers.

**Contribution.** Evidence that the augmentation, not the soft target, drives student calibration gains.

### 3b.21 [T2] Kim et al. (2025) - teacher calibration predicts student accuracy

**APA 7.** Kim, S., Park, S., Lee, J., & Kwak, N. (2025). The role of teacher calibration in knowledge distillation. *IEEE Access, 13*. https://arxiv.org/abs/2508.20224

**Year / venue / grade.** 2025, IEEE Access, PR.

**Relevance.** Suggests choosing / re-calibrating the 4B teacher by calibration error rather than accuracy.

**Key measured findings (Fig. 1, CIFAR-100, 17 teachers).** R^2 between teacher adaptive calibration error and student accuracy: 0.9229 (WRN-16-2 student), 0.8988 (ShuffleNetV2); between teacher accuracy and student accuracy: only 0.6751 and 0.5557. Applying temperature-style re-calibration to the teaching signal improves students across KD variants.

**Methodology.** Fixed students, varied pretrained teachers; ECE decomposed into over/under-confident parts; ACE metric.

**Limitations.** Vision; CIFAR scale; journal with lighter review.

**Contribution.** Quantitative link between teacher calibration and distillation efficacy, consistent with Cho & Hariharan and Müller et al.

---

## 3c. Calibration and selective prediction

### 3c.1 [T1] Guo et al. (2017) - temperature scaling

**APA 7.** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning, PMLR 70*, 1321-1330. https://arxiv.org/abs/1706.04599

**Year / venue / grade.** 2017, ICML 2017, PR.

**Relevance.** Defines the post-hoc baseline used by the Kev replica; documents both its effectiveness and its structural limit (does not change the argmax, so accuracy is unchanged; in binary tasks a single T is monotone in the logit margin and therefore cannot reorder confidences).

**Key measured findings (Table 1, ECE %, 15 bins).** CIFAR-100 ResNet-110: 16.53 -> TS 1.26 (matrix scaling 25.49, overfits); DenseNet-40 CIFAR-100: 10.37 -> 1.18; ImageNet ResNet-152: 5.48 -> 1.86; SVHN 0.44 -> 0.17; 20 News DAN-3: 8.02 -> 4.11; Reuters DAN-3: 0.85 -> 0.91; SST binary TreeLSTM: 6.63 -> 1.84; SST fine-grained: 6.71 -> 2.56. Sec. 4.2: "temperature scaling does not affect the model's accuracy" because it does not change the softmax maximum. Depth, width, and weight-decay reduction all worsen calibration; NLL keeps falling on train while test NLL overfits.

**Methodology.** Fit T by NLL on a validation split; compare histogram binning, isotonic, BBQ, vector/matrix scaling.

**Limitations.** No transformers; ECE binning sensitivity; only confidence (top-label) calibration.

**Contribution.** The standard post-hoc method and the calibration-vs-capacity finding.

### 3c.2 [T2] Kull et al. (2019) - Dirichlet calibration (beyond a single temperature)

**APA 7.** Kull, M., Perello-Nieto, M., Kängsepp, M., Silva Filho, T., Song, H., & Flach, P. (2019). Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration. *Advances in Neural Information Processing Systems, 32*. https://arxiv.org/abs/1910.12656

**Year / venue / grade.** 2019, NeurIPS 2019, PR.

**Relevance.** A post-hoc map that *can* reorder (full linear map on log-probabilities), with ODIR regularization to avoid the matrix-scaling overfitting seen in Guo et al.

**Key measured findings.** On 21 datasets x 11 classifiers (231 tasks), Dirichlet-L2 has the best average rank on log-loss, p-classwise-ECE and accuracy, and is in the top group on the others (Tables 1-2: average rank on p-cw-ECE DirL2 2.34, Beta 3.15, Isotonic 3.27, TempS 4.37, Uncalibrated 5.02; on log-loss DirL2 2.25, TempS 4.61). On deep nets (CIFAR/SVHN ResNets and DenseNets), Dir-ODIR has the best average rank but without statistical significance over TS; TS is nearly confidence-calibrated but not classwise-calibrated (cwECE 0.1857 TS vs 0.1795 Dirichlet in the illustrative example, Sec. 2).

**Methodology.** Friedman/Nemenyi tests across tasks; confidence-ECE, classwise-ECE, log-loss, Brier.

**Limitations.** Gains over TS on deep nets are small and not significant; tabular-heavy benchmark.

**Contribution.** Formalizes classwise calibration and provides a regularized full-matrix alternative to TS.

### 3c.3 [T1] Desai & Durrett (2020) - calibration of BERT and RoBERTa

**APA 7.** Desai, S., & Durrett, G. (2020). Calibration of pre-trained transformers. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 295-302. https://arxiv.org/abs/2003.07892

**Year / venue / grade.** 2020, EMNLP 2020, PR.

**Relevance.** Baseline ECE numbers for encoder classifiers on NLI/paraphrase/commonsense MC, in-domain and under shift, with TS and label smoothing.

**Key measured findings (Tables 2-3, ECE %).** Out-of-the-box, in-domain / out-of-domain: BERT SNLI 2.54 / MNLI 7.03; QQP 2.71 / TwitterPPDB 8.51; SWAG 2.49 / HellaSWAG 12.62. RoBERTa 1.93 / 3.62; 2.33 / 9.55; 1.76 / 11.93. Non-pretrained DA on SNLI 1.02 / 8.79, ESIM 1.33 / 12.78. With TS (MLE): BERT 1.14 / 3.61; 0.97 / 7.15; 0.85 / 12.83; RoBERTa 0.84 / 1.46; 0.88 / 7.86; 0.76 / 11.22. Label smoothing helps OOD in some cells (BERT HellaSWAG 12.62 -> 5.73 at LS) but hurts others (RoBERTa SNLI->MNLI 3.62 -> 4.50).

**Methodology.** Fine-tune, 5 seeds; ECE 10 bins; TS on in-domain dev.

**Limitations.** Small models (110-355M) only; no selective metrics.

**Contribution.** Establishes that pre-trained encoders are near-calibrated in-domain (ECE 1-3%) and that TS brings in-domain ECE below ~1.2 but does little under shift.

### 3c.4 [T1] Chen et al. (2023) - a close look at PLM calibration (scale, training dynamics, methods)

**APA 7.** Chen, Y., Yuan, L., Cui, G., Liu, Z., & Ji, H. (2023). A close look into the calibration of pre-trained language models. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics*, 1343-1367. https://arxiv.org/abs/2211.00151

**Year / venue / grade.** 2023, ACL 2023, PR.

**Relevance.** The most direct evidence on "calibration vs scale" for encoder/seq2seq classifiers at T5-small/base/large and RoBERTa-base/large, plus a head-to-head of TS, LS, ensembles and learned calibrators in-domain and OOD.

**Key measured findings (Tables 1, 5; ECE %).** T5-base MNLI in-domain: vanilla acc 86.50, conf 94.85, ECE 8.35; TS 2.75; LS 3.41; ensemble 8.28; OOD HANS: vanilla 37.30, TS 28.93; ANLI: 54.27 / 44.17. Amazon: vanilla 4.86 -> TS 1.39; SST-5 (OOD) 13.52 -> 4.94. Scale (Table 5, E-MLP calibrator): Amazon ECE T5-small 4.78 / base 4.35 / large 4.70 with accuracy 87.65 / 91.00 / 91.58; SST-5 15.23 / 15.45 / 10.24; SemEval 27.91 / 23.36 / 21.61. Conclusions: PLMs do not become calibrated during training (confidence rises monotonically even after accuracy plateaus); larger scale lowers ECE only when it also raises accuracy; TS is the best "unlearnable" method in-domain; learnable calibrators (extra model predicting correctness) sharply cut confidence on wrong predictions (CErr_neg) but raise it on correct ones.

**Methodology.** Six control factors; ID/OOD pairs (MNLI->HANS/ANLI, Amazon->SST-5/SemEval, Civil->HateSpeech/ImplicitHate); CErr_pos / CErr_neg decomposition.

**Limitations.** No selective-prediction metric; ECE only.

**Contribution.** Refines the "bigger is better calibrated" claim for sub-1B models: true only where accuracy is still improving.

### 3c.5 [T1] Kadavath et al. (2022) - calibration of LLMs on multiple-choice/true-false vs scale

**APA 7.** Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., Johnston, S., El-Showk, S., Jones, A., Elhage, N., Hume, T., Chen, A., Bai, Y., Bowman, S., Fort, S., ... Kaplan, J. (2022). Language models (mostly) know what they know. *arXiv preprint* arXiv:2207.05221. https://arxiv.org/abs/2207.05221

**Year / venue / grade.** 2022, preprint, PP.

**Relevance.** The "calibration improves with scale" reference for decoder LMs on exactly the option-scoring format used here (lettered choices, true/false); also the single-temperature fix for RLHF models.

**Key measured findings.** Fig. 4: on BIG-Bench multiple choice, ECE decreases monotonically with model size from ~800M to 52B in both 0-shot and 5-shot lettered-choice formats and in the True/False format; "calibration improves with model size, and it also improves when we pass from zero-shot to few-shot". The 52B model is well calibrated except at the tails (Fig. 8). RLHF-tuned models become miscalibrated, and "a simple temperature adjustment (with the same temperature T=2.5 for all evaluations) largely fixes calibration issues". P(IK) classifiers: smallest models have higher calibration error than the biggest (Fig. 13); in-distribution AUROC ~0.86-0.89 on TriviaQA with Brier ~0.15 (fig.), OOD (Lambada) AUROC ~0.61 (fig.).

**Methodology.** ECE with 10 equal-mass bins on top prediction; RMS calibration error; models 800M-52B (Anthropic series).

**Limitations.** Preprint; models not public; MC ECE values are figure-only; smallest model is 800M, so no data below the 0.6B target.

**Contribution.** Establishes format sensitivity (options must be presented) and the scale trend for MC calibration.

### 3c.6 [T2] Minderer et al. (2021) - calibration vs architecture and size (vision)

**APA 7.** Minderer, M., Djolonga, J., Romijnders, R., Hubis, F., Zhai, X., Houlsby, N., Tran, D., & Lucic, M. (2021). Revisiting the calibration of modern neural networks. *Advances in Neural Information Processing Systems, 34*. https://arxiv.org/abs/2106.07998

**Year / venue / grade.** 2021, NeurIPS 2021, PR.

**Relevance.** Counterpoint to Guo et al. on size: for recent (non-convolutional) architectures, size does not degrade calibration in-distribution much and improves it under shift.

**Key measured findings (Sec. 1 contributions, Figs. 2-4).** ViT and MLP-Mixer are among the best calibrated; "in-distribution calibration slightly deteriorates with increasing model size" but "under distribution shift, calibration improves with model size"; model size and pretraining amount "cannot fully explain" the differences, so architecture is a major determinant; TS remains beneficial and is applied throughout.

**Methodology.** ECE with equal-mass binning (100 bins) on ImageNet and ImageNet-C/R/A etc.; TS fitted on held-out train.

**Limitations.** Vision only.

**Contribution.** Shows the calibration-vs-size trend is architecture-dependent.

### 3c.7 [T1] Angelopoulos & Bates (2023); Romano, Sesia & Candès (2020); Angelopoulos et al. (2021) - conformal prediction for classification (gentle intro, APS, RAPS)

**APA 7.**
- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning, 16*(4), 494-591. (arXiv:2107.07511, 2021.) https://arxiv.org/abs/2107.07511
- Romano, Y., Sesia, M., & Candès, E. J. (2020). Classification with valid and adaptive coverage. *Advances in Neural Information Processing Systems, 33*. https://arxiv.org/abs/2006.02544
- Angelopoulos, A. N., Bates, S., Malik, J., & Jordan, M. I. (2021). Uncertainty sets for image classifiers using conformal prediction. *International Conference on Learning Representations (ICLR 2021)*. https://arxiv.org/abs/2009.14193

**Year / venue / grade.** 2021/2023 (FnT), 2020 (NeurIPS), 2021 (ICLR); PR.

**Relevance.** Distribution-free way to turn any scorer's softmax into prediction sets with finite-sample marginal coverage 1-alpha; a singleton set is a natural "automate" decision and set size is a confidence proxy independent of calibration.

**Key measured findings (RAPS Table 1, ImageNet val, target 90% coverage, 1,000 classes).** Naive thresholding of cumulative softmax under-covers (0.889-0.896); APS and RAPS hit 0.900. Average set size: ResNet-152 top-k 2.63, naive 9.78, APS 10.4, RAPS 2.11; ResNet-50 3.14 / 11.8 / 12.3 / 2.57; ResNet-18 5.72 / 15.5 / 16.2 / 4.43; ResNeXt-101 2.42 / 17.1 / 19.7 / 2.00. At 95% coverage RAPS sets are 4.4-11.7 vs APS 22.5-33.2 (Table, appendix). APS (Romano et al.) attains near-exact conditional coverage but large sets on ImageNet; RAPS adds a rank penalty (k_reg, lambda) that shrinks sets with a small conditional-coverage cost (size-stratified violation 0.02-0.04 vs 0.04-0.07 for APS).

**Methodology.** Split conformal on a calibration set; score = cumulative sorted softmax mass (APS) plus regularization (RAPS); LAC (threshold on 1 - p_y) as the smallest-set alternative.

**Limitations.** Marginal (not per-instance) guarantee; exchangeability assumption; sets can be large for poorly separated classes.

**Contribution.** Provides the coverage guarantee machinery; RAPS is the practical default for many-class softmax outputs.

### 3c.8 [T2] Kumar et al. (2023) - conformal prediction with LLMs on multiple-choice QA

**APA 7.** Kumar, B., Lu, C., Gupta, G., Palepu, A., Bellamy, D., Raskar, R., & Beam, A. (2023). Conformal prediction with large language models for multi-choice question answering. *arXiv preprint* arXiv:2305.18404. https://arxiv.org/abs/2305.18404

**Year / venue / grade.** 2023, preprint (ICML 2023 workshop version), PP/WS.

**Relevance.** Shows split conformal (LAC score) works on LLaMA-13B softmax over MCQ options and that set size tracks accuracy, i.e. the same machinery applies to an option scorer.

**Key measured findings (Sec. 4).** 16 MMLU subjects; at alpha=0.1 empirical coverage 91-94% (e.g., professional accounting 91+/-3%, marketing 91+/-1%, computer security 94+/-3%); average set size from 2.4+/-0.1 (marketing) to 3.7+/-0.1 (formal logic) out of 4 options; strong negative correlation between set size and top-1 accuracy; singleton sets are markedly more accurate than the overall top-1.

**Methodology.** Softmax over option letters; LAC non-conformity; 50/50 calibration/test splits with repeats.

**Limitations.** Preprint; 4-option MCQ only; a 13B decoder, not a small encoder.

**Contribution.** First demonstration of conformal sets over LLM option probabilities.

### 3c.9 [T1] Geifman & El-Yaniv (2017) - selective classification with guaranteed risk (SR baseline)

**APA 7.** Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *Advances in Neural Information Processing Systems, 30*. https://arxiv.org/abs/1705.08500

**Year / venue / grade.** 2017, NeurIPS 2017, PR.

**Relevance.** Defines the risk-coverage framework and the SGR algorithm that picks a confidence threshold to *guarantee* a target risk with probability 1-delta, i.e. exactly "coverage at 5% error".

**Key measured findings (Secs. 4-5).** ImageNet VGG-16: 2% top-5 error guaranteed with probability 99.9% at "almost 60%" coverage. Softmax response (SR) vs MC-dropout: nearly identical on CIFAR-10/100; on ImageNet top-1 at 60% coverage SR has ~10% error vs >20% for MC-dropout. CIFAR-10 VGG-16: risk-coverage curves allow e.g. ~1% error at ~80% coverage (Fig. 2).

**Methodology.** Binary search over threshold with a Gascuel-Caraux tail bound on a held-out set; SR and MC-dropout as confidence-rate functions.

**Limitations.** Post-hoc only; guarantee holds for i.i.d. test data; vision.

**Contribution.** The coverage-at-risk formulation and a proof that thresholding SR gives a certified selective classifier.

### 3c.10 [T1] Geifman & El-Yaniv (2019) - SelectiveNet (learned selection head)

**APA 7.** Geifman, Y., & El-Yaniv, R. (2019). SelectiveNet: A deep neural network with an integrated reject option. *Proceedings of the 36th International Conference on Machine Learning, PMLR 97*, 2151-2159. https://arxiv.org/abs/1901.09192

**Year / venue / grade.** 2019, ICML 2019, PR.

**Relevance.** The reference "learned abstention head" trained end-to-end for a target coverage.

**Key measured findings (Tables 2-4, selective risk %, 0-1 loss).** CIFAR-10 at coverage 0.90: SelectiveNet 2.43+/-0.08 vs SR 2.89+/-0.03 vs MC-dropout 2.92; at 0.80: 0.86 vs 1.05 vs 1.08; at 0.70: 0.32 vs 0.42 vs 0.43. SVHN at 0.80: 0.53 vs 0.61 vs 0.61. Cats vs Dogs at 0.80: 0.35 vs 0.68 vs 0.55. Coverage violation before calibration (Table 1): SR 11.98% average vs SelectiveNet 3.625%.

**Methodology.** Three heads (prediction, selection, auxiliary); interior-point penalty on coverage constraint; one model per target coverage.

**Limitations.** Requires retraining per coverage; later shown (3c.13) that much of the gain is a better classifier rather than a better selector.

**Contribution.** First end-to-end selective network and coverage-calibration procedure.

### 3c.11 [T1] Liu et al. (2019) - Deep Gamblers (abstention logit)

**APA 7.** Liu, Z., Wang, Z., Liang, P. P., Salakhutdinov, R., Morency, L.-P., & Ueda, M. (2019). Deep Gamblers: Learning to abstain with portfolio theory. *Advances in Neural Information Processing Systems, 32*. https://arxiv.org/abs/1907.00208

**Year / venue / grade.** 2019, NeurIPS 2019, PR.

**Relevance.** The (m+1)-class abstention-logit formulation; one model serves all coverages by thresholding the abstention score.

**Key measured findings (Table 4, CIFAR-10, error % on covered set; single best model o=2.2).** Coverage 1.00: 6.12+/-0.09 (SR/SelectiveNet 6.79); 0.95: 3.49 vs SR 4.55, SN 4.16; 0.90: 2.19 vs 2.89 / 2.43; 0.85: 1.09 vs 1.78 / 1.43; 0.80: 0.66 vs 1.05 / 0.86; 0.70: 0.43 vs 0.42 / 0.32. SVHN and Cats vs Dogs show the same pattern (Tables 3, 5).

**Methodology.** Loss = -log(o * p_y + p_abstain) derived from the doubling rate of a horse race; grid over payoff o.

**Limitations.** Sensitive to o; at low coverage the abstention score is numerically small; full-coverage accuracy differs from baselines, confounding comparisons (noted by the authors).

**Contribution.** Simple loss-only abstention mechanism.

### 3c.12 [T1] Huang, Zhang & Zhang (2020) - Self-Adaptive Training (SAT) for selective classification

**APA 7.** Huang, L., Zhang, C., & Zhang, H. (2020). Self-adaptive training: Beyond empirical risk minimization. *Advances in Neural Information Processing Systems, 33*. https://arxiv.org/abs/2002.10319

**Year / venue / grade.** 2020, NeurIPS 2020, PR.

**Relevance.** Strongest of the abstention-head family before Feng et al.; uses EMA of the model's own predictions as soft targets and routes uncertain samples to the abstention class.

**Key measured findings (Table 4, error % on covered set, 3 trials).** CIFAR-10: coverage 0.90 SAT 1.93+/-0.09 vs Deep Gamblers 2.19, SelectiveNet 2.43, SR 2.89; 0.80: 0.67 vs 0.66 / 0.86 / 1.05; 0.70: 0.34 vs 0.43 / 0.32 / 0.42. Dogs vs Cats 0.80: 0.15 vs 0.46 / 0.35 / 0.68. "Up to 50% relative improvement" claimed.

**Methodology.** Built on the Deep Gamblers code; soft targets = EMA of predictions; abstention class trained on high-uncertainty samples.

**Limitations.** Vision; CIFAR scale; 3 trials.

**Contribution.** Shows self-generated soft targets help both label-noise robustness and selective accuracy.

### 3c.13 [T1] Feng et al. (2023) - Towards better selective classification (SR beats learned heads)

**APA 7.** Feng, L., Ahmed, M. O., Hajimirsadeghi, H., & Abdi, A. H. (2023). Towards better selective classification. *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2206.09034

**Year / venue / grade.** 2023, ICLR 2023, PR.

**Relevance.** Directly tests whether the abstention heads of 3c.10-3c.12 beat plain max-softmax on the *same* trained classifier; answers the "learned abstention head vs SR" question.

**Key measured findings (Tables 1-2, ImageNet100, error % on covered set).** Full-coverage accuracy: vanilla 85.68, SelectiveNet 86.23, Deep Gamblers 86.51, SAT 86.40. Replacing each method's own selector with SR on the same network: SelectiveNet at coverage 0.90 9.44 -> 7.89; 0.80 6.00 -> 4.47; 0.70 3.38 -> 2.21; Deep Gamblers 0.80 5.21 -> 4.52; SAT 0.80 5.20 -> 4.46, 0.70 2.71 -> 2.33. SelectiveNet's own head collapses at low coverage (48.87% error at 0.20, 99.00% at 0.10). Adding an entropy-minimization regularizer (EM) to SAT+SR gives 3.90 at 0.80 and 1.81 at 0.70 (Table 4); CIFAR-10 SAT+EM+SR 0.23 at 0.70 vs 0.27.

**Methodology.** Same trained weights, swap the selection function; 5 datasets (CIFAR-10, Food101, StanfordCars, ImageNet100, ImageNet subset).

**Limitations.** Vision; the EM regularizer is semi-supervised-style and may interact with calibration.

**Contribution.** Shows the gains of abstention heads come from training a better classifier, not from the head; SR remains the best selector.

### 3c.14 [T1] Geifman, Uziel & El-Yaniv (2019); Jaeger et al. (2023); Traub et al. (2024) - AURC, E-AURC, FD-Shifts and AUGRC

**APA 7.**
- Geifman, Y., Uziel, G., & El-Yaniv, R. (2019). Bias-reduced uncertainty estimation for deep neural classifiers. *International Conference on Learning Representations (ICLR 2019)*. https://arxiv.org/abs/1805.08206
- Jaeger, P. F., Lüth, C. T., Klein, L., & Bungert, T. J. (2023). A call to reflect on evaluation practices for failure detection in image classification. *International Conference on Learning Representations (ICLR 2023, oral)*. https://arxiv.org/abs/2211.15259
- Traub, J., Bungert, T. J., Lüth, C. T., Baumgartner, M., Maier-Hein, K. H., Maier-Hein, L., & Jaeger, P. F. (2024). Overcoming common flaws in the evaluation of selective classification systems. *Advances in Neural Information Processing Systems, 37*. https://arxiv.org/abs/2407.01032

**Year / venue / grade.** 2019 ICLR; 2023 ICLR; 2024 NeurIPS; PR.

**Relevance.** Defines and critiques the coverage-at-risk summary metrics the project will report.

**Key measured findings.**
- Geifman et al. (2019) define AURC = area under the empirical risk-coverage curve and E-AURC = AURC minus the AURC of the optimal (oracle) confidence for that classifier (Eq. 3; e.g. SR E-AURC x10^3 = 4.78 on CIFAR-10 in their Table 1).
- Jaeger et al. (2023), FD-Shifts benchmark (CIFAR-10/100, SVHN, iWildCam, BREEDS, CAMELYON; CNN and ViT; i.i.d. plus corruption/semantic/domain shifts): "None of the evaluated methods from literature beats the simple Maximum Softmax Response baseline across a realistic range of failure sources"; ConfidNet, Deep Gamblers' reservation score and DeVries' confidence branch fail to generalize beyond their original settings; MC-dropout MSR is best or close to best on i.i.d. data (Table 1, AURC x10^3). AURC recommended as the primary metric because it jointly scores classifier accuracy and ranking.
- Traub et al. (2024) show AURC over-weights high-confidence failures and violates monotonicity/ranking-interpretability; propose AUGRC ("average risk of undetected failures"); on 6 datasets x 13 confidence functions, switching to AUGRC "changes metric rankings on 5 out of the 6 data sets".

**Methodology.** Large-scale re-implementation with 5 seeds (Jaeger); formal requirement analysis (Traub).

**Limitations.** Vision benchmarks; AUGRC adoption is recent.

**Contribution.** Gives the metric definitions and evidence that simple SR is a hard baseline under a unified protocol.

### 3c.15 [T1] Galil, Dabbah & El-Yaniv (2023) - 523 ImageNet classifiers: distillation and TS improve selective prediction

**APA 7.** Galil, I., Dabbah, M., & El-Yaniv, R. (2023). What can we learn from the selective prediction and uncertainty estimation performance of 523 ImageNet classifiers? *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2302.11874

**Year / venue / grade.** 2023, ICLR 2023, PR.

**Relevance.** Largest observational study linking training regime (including knowledge distillation) to AURC/AUROC/ECE and coverage at a selective-accuracy constraint; also measures whether temperature scaling changes ranking.

**Key measured findings (Secs. 1, 5).** (1) "Training regimes incorporating any kind of knowledge distillation lead to DNNs with improved uncertainty estimation"; distillation gives the largest median improvement in AUROC and ECE of all regimes, even when the teacher's own uncertainty metrics are poor (Fig. 4a). (2) "Temperature scaling ... consistently and greatly improves AUROC and selective performance", i.e. in multiclass softmax a single T *does* change the max-probability ranking across samples (Fig. 4b, 523 models). (3) Correlation between accuracy and AUROC across all 523 models is ~0.03; within families it flips sign (XCiT +0.76, ResNet -0.74). (4) ViT-L/16-384 reaches 99% top-1 selective accuracy at 47% coverage and 95% at 80%; EfficientNet-V2-XL cannot reach 95% at any coverage despite similar AURC. Pretraining on ImageNet-21k alone slightly improves ECE but degrades AUROC.

**Methodology.** timm model zoo; AUROC of correct-vs-incorrect, ECE, AURC, selective accuracy constraint (SAC); TS fitted on 5k ImageNet val images.

**Limitations.** Observational (regimes confounded with architecture); vision only.

**Contribution.** Empirical support that (a) distillation improves selective performance and (b) post-hoc temperature can reorder max-softmax confidences in multiclass settings.

### 3c.16 [T1] Xin et al. (2021) - selective prediction for NLU: BERT-base vs large, SR vs MC-dropout, error regularization

**APA 7.** Xin, J., Tang, R., Yu, Y., & Lin, J. (2021). The art of abstention: Selective prediction and error regularization for natural language processing. *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing*, 1040-1051. https://aclanthology.org/2021.acl-long.84/

**Year / venue / grade.** 2021, ACL 2021, PR.

**Relevance.** The NLP analogue of 3c.9-3c.13 with encoder sizes close to the target; shows bigger pre-trained encoders improve both accuracy and confidence ranking, and that a cheap training-time regularizer helps further.

**Key measured findings (Table 1; AUC of risk-coverage curve, lower is better, units as reported; RPP %).** MRPC: LSTM SR 101.5 / MC 137.0; BERT-base SR 33.8 / MC 38.3; BERT-large SR 27.0 / MC 35.9; ALBERT-base SR 16.0 / MC 43.9. QNLI: LSTM 1539.1; BERT-base 111.9; BERT-large 105.6; ALBERT 122.8. MNLI-m: LSTM 1984.0; BERT-base 514.8; BERT-large 486.1; ALBERT 469.3. MC-dropout is consistently worse than SR (Sec. 5.2); the error-regularization trick further improves selective AUC without extra inference cost (Sec. 5.3).

**Methodology.** Fine-tune with Hugging Face defaults; SR vs MC-dropout confidence; risk-coverage AUC and reversed-pair proportion.

**Limitations.** No calibration (ECE) metrics; dropout rate sensitive.

**Contribution.** Establishes SR + pre-trained encoders as the NLP selective-prediction baseline and adds a regularizer.

### 3c.17 [T1] Varshney, Mishra & Baral (2022) - selective prediction across 17 NLP datasets (IID/OOD/adversarial)

**APA 7.** Varshney, N., Mishra, S., & Baral, C. (2022). Investigating selective prediction approaches across several tasks in IID, OOD, and adversarial settings. *Findings of the Association for Computational Linguistics: ACL 2022*, 1995-2002. https://arxiv.org/abs/2203.00211

**Year / venue / grade.** 2022, Findings of ACL, PR.

**Relevance.** Tests whether anything beats MaxProb for a BERT-base classifier on NLI, duplicate detection and QA; the negative result is directly relevant to whether a learned calibrator/abstention head is worth building.

**Key measured findings (Sec. 4, Table 3; AUC of risk-coverage, lower better).** IID: MC-dropout improves NLI AUC by only 0.28 on average; OOD: 0.08; adversarial: MC-dropout *degrades* duplicate-detection AUC by 1.76 and calibration-based methods degrade NLI AUC by 1.27. Examples: SNLI MaxProb 2.78 vs MCD(K=30) 2.47 vs Calib-C 2.57; MNLI 5.47 vs 4.92 vs 5.16; DNLI 7.36 vs 6.69 vs 3.88 (Calib-C helps here). Conclusion: "none of the existing approaches consistently and considerably outperforms MaxProb in all three settings".

**Methodology.** BERT-base; MaxProb, MC-dropout, label smoothing, three learned calibrators (random forest, regression, transformer); 17 datasets.

**Limitations.** Single model size; no ECE reported; Findings track.

**Contribution.** Strong null result for learned confidence estimators over MaxProb in NLP.

### 3c.18 [T2] Kamath, Jia & Liang (2020) - selective QA under domain shift with a trained calibrator

**APA 7.** Kamath, A., Jia, R., & Liang, P. (2020). Selective question answering under domain shift. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 5684-5696. https://aclanthology.org/2020.acl-main.503/

**Year / venue / grade.** 2020, ACL 2020, PR.

**Relevance.** Reports coverage at a fixed accuracy budget (80%) and shows a small learned calibrator with access to *known* OOD data beats MaxProb; the metric is the project's product metric.

**Key measured findings (Sec. 5, averaged over 20 OOD pairings; BERT QA trained on SQuAD 1.1).** Coverage at 80% accuracy: MaxProb 48.2%; MaxProb with QA model also trained on known-OOD 51.8%; calibrator trained on SQuAD only 53.7%; calibrator trained on SQuAD + known-OOD 56.1% (+4.3 vs MaxProb on the same model; +6.7 at 90% accuracy; 1.1 lower AUC).

**Methodology.** Random-forest calibrator over softmax and length features; MRQA OOD datasets (NewsQA, TriviaQA, SearchQA, HotpotQA, NQ).

**Limitations.** Extractive QA rather than classification; gains depend on having some OOD data.

**Contribution.** Coverage-at-accuracy evidence that MaxProb is overconfident under shift and that a tiny post-hoc calibrator can recover ~8 coverage points.

---

## 4. Numeric evidence table

Values are as reported in the cited tables/figures; ECE is in percent unless the source used a fraction. "Cov@r" = coverage at selective risk r. "SR" = softmax response / max-probability.

| # | Method / comparison | Student (teacher) size | Task / dataset | Metric | Value | Source |
|---|---|---|---|---|---|---|
| 1 | CE vs focal (FLSD-53), pre-TS ECE | ResNet-50 (n/a) | CIFAR-100 | ECE % | 17.52 vs 4.50 | Mukhoti 2020 T1 |
| 2 | CE vs focal, post-TS ECE | ResNet-50 | CIFAR-100 | ECE % | 3.42 vs 2.00 | Mukhoti 2020 T1 |
| 3 | CE vs Brier vs LS vs focal, pre-TS ECE | ResNet-50 | CIFAR-100 | ECE % | 17.52 / 6.52 / 7.81 / 4.50 | Mukhoti 2020 T1 |
| 4 | CE vs Brier vs LS vs focal, post-TS ECE | GP-CNN | 20 Newsgroups | ECE % | 2.39 / 3.22 / 2.54 / 2.19 | Mukhoti 2020 T1 |
| 5 | CE vs LS vs focal, test error | GP-CNN | 20 Newsgroups | error % | 26.68 / 26.03 / 27.98 | Mukhoti 2020 T1 |
| 6 | Hard targets vs TS vs LS 0.05, ECE | ResNet-56 | CIFAR-100 | ECE | 0.150 / 0.021 / 0.024 | Müller 2019 T2 |
| 7 | Teacher trained hard vs LS -> student error | small MLP (large MLP) | MNIST | error % | 0.74 vs 0.91 (teachers 0.67 vs 0.59) | Müller 2019 Sec. 5 |
| 8 | CE vs LS 0.1/0.2/0.3 vs LS 0.2 + logit norm | ResNet-50 | ImageNet | Cov@1% risk | 15.66 / 0.05 / 0.06 / 0.02 / 22.31 % | Xia 2025 Fig. 1 |
| 9 | Square (Brier) vs CE accuracy | BERT fine-tuned | MRPC / SST-2 / QNLI / QQP | acc % | 83.8 vs 82.1 / 94.0 vs 93.9 / 90.6 tie / 88.9 tie | Hui & Belkin 2021 T2 |
| 10 | Square >= CE in tasks | many | NLP + ASR | count | 22 of 28 | Hui & Belkin 2021 |
| 11 | Log vs Brier vs Spherical training | Transformer | WMT14 En-De | BLEU | 27.61 / 28.01 / 28.07 | Shao 2024 T3 |
| 12 | Log vs Brier vs Spherical fine-tune | LLaMA-7B | WMT22 En-De | BLEU | 25.42 / 29.15 / 29.07 | Shao 2024 T5 |
| 13 | CE vs CORAL vs CORN | ResNet-34 | MORPH-2 age | MAE | 3.73 / 2.99 / 2.98 | Shi 2023 T3 |
| 14 | CE vs CORAL vs CORN | ResNet-34 | AFAD age | MAE | 3.28 / 2.99 / 2.81 | Shi 2023 T3 |
| 15 | CNN-POR vs SORD | VGG-16 | Adience age | acc % / MAE | 57.4 / 0.55 vs 59.6 / 0.49 | Diaz 2019 T2 |
| 16 | Hard vs soft targets (T=20) | 2x800 MLP (large MLP) | MNIST | test errors | 146 vs 74 (teacher 67) | Hinton 2015 |
| 17 | DistilBERT vs BERT-base | 66M (110M) | GLUE dev macro | score | 77.0 vs 79.5 (97%) | Sanh 2019 |
| 18 | DistilBERT vs BERT-base | 66M (110M) | MNLI / RTE | acc | 82.2 vs 86.7 / 59.9 vs 69.3 | Sanh 2019 |
| 19 | TinyBERT4 vs teacher | 14.5M (109M) | GLUE test avg | score | 77.0 vs 79.5 (96.8%) | Jiao 2020 T2 |
| 20 | TinyBERT6 vs teacher | 67M (109M) | GLUE test avg | score | 79.4 vs 79.5 | Jiao 2020 T2 |
| 21 | TinyBERT4 ablation full / -GD / -TD / -DA | 14.5M | MNLI-m,MRPC,CoLA avg | score | 75.6 / 72.5 / 68.5 / 68.4 | Jiao 2020 T4 |
| 22 | MiniLM 6x768 | 66M (109M) | MNLI-m / SQuAD2 F1 | acc / F1 | 84.0 / 76.4 | Wang 2020 T2 |
| 23 | MiniLMv2 6x384 vs RoBERTa-base | 30M (355M) vs 125M | GLUE+SQuAD2 dev avg | score | 79.5 vs 85.4 | Wang 2021 T1 |
| 24 | MiniLMv2 6x768 vs RoBERTa-base | 81M (355M) vs 125M | MNLI-m / SQuAD2 | acc / F1 | 87.0 / 81.6 vs 87.6 / 83.7 | Wang 2021 T1 |
| 25 | MiniLMv2 12x768 vs RoBERTa-base | 125M (355M) | GLUE+SQuAD2 dev avg | score | 87.6 vs 86.1 | Wang 2021 T3 |
| 26 | MobileBERT vs BERT-base | 25.3M (IB-BERT-large) | GLUE test | score | 77.7 vs 78.3 | Sun 2020 T4 |
| 27 | MobileBERT-tiny vs TinyBERT | 15.1M / 14.5M | GLUE test | score | 75.8 / 75.4 | Sun 2020 T4 |
| 28 | TF vs PF vs PD (6/768 student) | 67M (110M) | GLUE test meta | score | 80.5 / 81.6 / 82.1 | Turc 2019 T3 |
| 29 | PD vs PF vs D, BERT-Tiny | 4.4M (340M) | MNLI | acc (fig.) | ~63 / ~61 / ~58 | Turc 2019 Fig. 7 |
| 30 | PD vs PF vs D, BERT-Mini | 11.3M (340M) | MNLI | acc (fig.) | ~70 / ~67 / ~65 | Turc 2019 Fig. 7 |
| 31 | Born-again vs teacher | DenseNet-80-120 (same) | CIFAR-100 | error % | 16.00 vs 16.87 | Furlanello 2018 T2 |
| 32 | Student-teacher top-1 agreement (self-distill) | ResNet-56 (ResNet-56) | CIFAR-100 | agreement % | 80-90 | Stanton 2021 Sec. 5 |
| 33 | Student-teacher agreement (ensemble teacher) | ResNet-56 (3x ens.) | CIFAR-100 | agreement % | < 80 | Stanton 2021 Sec. 5 |
| 34 | KD from R18 / R34 / R50 teachers vs scratch | ResNet-18 | ImageNet | top-1 error % | 30.57 / 30.79 / 30.95 vs 30.24 | Cho 2019 T1 |
| 35 | Early-stopped-teacher KD | ResNet-18 (R34) | ImageNet | top-1 error % | 29.16 | Cho 2019 T1 |
| 36 | Function-matching KD, 9,600 epochs | ResNet-50 (BiT R152x2) | ImageNet | top-1 acc % | 82.8 (teacher 83.0; prior R50 78.8) | Beyer 2022 |
| 37 | Vanilla KD vs Att-KL+Val-KL (task-specific) | 6-layer BERT (BERT-base) | GLUE dev avg | score | 71.8 vs 77.5 | Wang 2023 T1 |
| 38 | Vanilla KD vs Att-MSE (task-agnostic) | 6-layer (BERT-base) | GLUE dev avg | score | 79.3 vs 81.3 | Wang 2023 T2 |
| 39 | BiLSTM hard vs soft-logit distilled vs BERT-large | 0.96M (335M) | SST-2 | acc % | 86.7 vs 90.7 vs 94.9 | Tang 2019 T1 |
| 40 | Same | 0.96M (335M) | MNLI-m | acc % | 68.7 vs 73.0 vs 86.7 | Tang 2019 T1 |
| 41 | Std FT vs DSS (540B) vs DSS (20B) | T5 220M (PaLM 540B / NeoX 20B) | ANLI | acc % | 43.58 vs 49.58 vs 48.15 | Hsieh 2023 T1 |
| 42 | Std FT vs DSS | T5 220M (540B) | e-SNLI / CQA / SVAMP | acc % | 88.38 vs 89.51 / 62.19 vs 63.29 / 62.63 vs 65.50 | Hsieh 2023 T1 |
| 43 | Single-task vs multi-task rationale training | T5 220M | ANLI | acc % | 43.50 vs 49.58 | Hsieh 2023 T2 |
| 44 | GPT-3 vs human label cost for equal performance | RoBERTa / PEGASUS (175B) | SST-2 / Gigaword / others | cost saving | 96% / 93.8% / 50-75% | Wang 2021 |
| 45 | BERT on 1k human vs 1k GPT-4 labels | 110M (GPT-4) | 14 CSS tasks | median F1 | 0.624 vs 0.586 (GPT-4 few-shot 0.592) | Pangakis 2024 |
| 46 | BERT-base + PGKD | 110M (Claude-3 / LLaMA-3-8B) | AG-News / Yahoo / HuffPost / Amazon | acc | 0.895 / 0.685 / 0.519 / 0.443 | Di Palo 2024 |
| 47 | ZeroGen vs supervised | DistilBERT 66M (GPT2-XL 1.5B) | SST-2 / QNLI | acc % | 87.27 vs 89.68 / 71.19 vs 88.05 | Ye 2022 T1 |
| 48 | KD vs KD+CutMix, student ECE | ShuffleNetV1 (WRN-40-2) | CIFAR-100 | ECE | 0.108 vs 0.046 (scratch 0.121) | Mishra 2023 T1 |
| 49 | R^2 teacher-ACE vs student-acc; teacher-acc vs student-acc | WRN-16-2 (17 teachers) | CIFAR-100 | R^2 | 0.9229 vs 0.6751 | Kim 2025 Fig. 1 |
| 50 | Uncalibrated vs TS, ECE | ResNet-110 | CIFAR-100 | ECE % | 16.53 vs 1.26 | Guo 2017 T1 |
| 51 | Uncalibrated vs TS, ECE | TreeLSTM | SST binary / fine | ECE % | 6.63 vs 1.84 / 6.71 vs 2.56 | Guo 2017 T1 |
| 52 | Matrix scaling overfits | ResNet-110 | CIFAR-100 | ECE % | 25.49 | Guo 2017 T1 |
| 53 | Dirichlet-L2 vs TS vs uncalibrated, average rank | 11 classifier types | 21 datasets (231 tasks) | rank (p-cw-ECE) | 2.34 vs 4.37 vs 5.02 | Kull 2019 T1 |
| 54 | BERT / RoBERTa in-domain ECE, out-of-box | 110M / 355M | SNLI | ECE % | 2.54 / 1.93 | Desai 2020 T2 |
| 55 | BERT / RoBERTa OOD ECE, out-of-box -> TS | 110M / 355M | SWAG->HellaSWAG | ECE % | 12.62 -> 12.83 / 11.93 -> 11.22 | Desai 2020 T2-3 |
| 56 | BERT in-domain ECE after TS | 110M | SNLI / QQP / SWAG | ECE % | 1.14 / 0.97 / 0.85 | Desai 2020 T3 |
| 57 | T5-base vanilla vs TS vs LS vs ensemble | 220M | MNLI (ID) | ECE % | 8.35 / 2.75 / 3.41 / 8.28 | Chen 2023 T1 |
| 58 | T5-base vanilla vs TS | 220M | HANS (OOD) | ECE % | 37.30 / 28.93 | Chen 2023 T1 |
| 59 | ECE vs scale (E-MLP calibrator) | T5-small / base / large | SST-5 (OOD from Amazon) | ECE % | 15.23 / 15.45 / 10.24 | Chen 2023 T5 |
| 60 | ECE vs scale | T5-small / base / large | Amazon (ID) | ECE % / acc | 4.78 / 4.35 / 4.70 ; 87.65 / 91.00 / 91.58 | Chen 2023 T5 |
| 61 | ECE vs model size, MC | 800M-52B decoders | BIG-Bench MC | ECE (fig.) | monotone decrease with size | Kadavath 2022 Fig. 4 |
| 62 | Single temperature for RLHF model | 52B | several evals | T | 2.5 | Kadavath 2022 Sec. 3 |
| 63 | RAPS vs APS vs naive vs top-k set size | ResNet-50 | ImageNet, 90% cov | avg set size | 2.57 / 12.3 / 11.8 / 3.14 | Angelopoulos 2021 T1 |
| 64 | RAPS vs APS set size | ResNet-152 | ImageNet, 90% cov | avg set size | 2.11 / 10.4 | Angelopoulos 2021 T1 |
| 65 | Conformal LAC on LLM MCQ | LLaMA-13B | MMLU 16 subjects, alpha=0.1 | coverage / set size | 91-94% / 2.4-3.7 of 4 | Kumar 2023 |
| 66 | Guaranteed risk | VGG-16 | ImageNet top-5 | Cov@2% risk | ~60% (delta=0.001) | Geifman 2017 |
| 67 | SR vs MC-dropout at 60% coverage | VGG-16 | ImageNet top-1 | error % | ~10 vs >20 | Geifman 2017 Fig. 2 |
| 68 | SR vs SelectiveNet vs DG vs SAT at 0.90 cov | VGG-16 variants | CIFAR-10 | sel. error % | 2.89 / 2.43 / 2.19 / 1.93 | SelectiveNet T2; DG T4; SAT T4 |
| 69 | Same at 0.80 cov | | CIFAR-10 | sel. error % | 1.05 / 0.86 / 0.66 / 0.67 | same |
| 70 | Same at 0.70 cov | | CIFAR-10 | sel. error % | 0.42 / 0.32 / 0.43 / 0.34 | same |
| 71 | SelectiveNet own head vs SR on same net, 0.80 cov | ResNet | ImageNet100 | sel. error % | 6.00 vs 4.47 | Feng 2023 T2 |
| 72 | Deep Gamblers own head vs SR, 0.80 cov | ResNet | ImageNet100 | sel. error % | 5.21 vs 4.52 | Feng 2023 T2 |
| 73 | SAT own head vs SR vs SR+EM, 0.80 cov | ResNet | ImageNet100 | sel. error % | 5.20 / 4.46 / 3.90 | Feng 2023 T2, T4 |
| 74 | Coverage violation before calibration | VGG | CIFAR-10 | avg % | SR 11.98 vs SelectiveNet 3.625 | SelectiveNet T1 |
| 75 | MSR vs learned confidence methods | CNN / ViT | FD-Shifts (6 datasets, shifts) | AURC | MSR best or near-best; ConfidNet/DG-Res/DeVries fail to generalize | Jaeger 2023 T1 |
| 76 | AURC -> AUGRC changes rankings | 13 CSFs | 6 datasets | count | 5 of 6 | Traub 2024 |
| 77 | Distillation vs other regimes, uncertainty metrics | 523 ImageNet models | ImageNet | AUROC, ECE | largest median improvement from KD | Galil 2023 Fig. 4 |
| 78 | Temperature scaling effect on ranking | 523 models | ImageNet | AUROC | consistently improves | Galil 2023 Sec. 5 |
| 79 | Selective accuracy constraint | ViT-L/16-384 | ImageNet | Cov@99% acc / Cov@95% | 47% / 80% | Galil 2023 |
| 80 | Accuracy vs AUROC correlation | 523 models | ImageNet | Spearman | 0.03 | Galil 2023 |
| 81 | SR risk-coverage AUC, LSTM vs BERT-base vs BERT-large | LSTM / 110M / 340M | MRPC | AUC (as reported) | 101.5 / 33.8 / 27.0 | Xin 2021 T1 |
| 82 | Same | | MNLI-m | AUC | 1984.0 / 514.8 / 486.1 | Xin 2021 T1 |
| 83 | SR vs MC-dropout | BERT-base | MRPC / QNLI | AUC | 33.8 vs 38.3 / 111.9 vs 130.1 | Xin 2021 T1 |
| 84 | MaxProb vs MC-dropout vs Calib-C | BERT-base | SNLI / MNLI / DNLI (IID) | RC-AUC | 2.78 / 2.47 / 2.57 ; 5.47 / 4.92 / 5.16 ; 7.36 / 6.69 / 3.88 | Varshney 2022 T3 |
| 85 | MC-dropout gain over MaxProb | BERT-base | NLI IID / OOD / ADV | delta AUC | +0.28 / +0.08 / -1.76 (DD) | Varshney 2022 Sec. 4 |
| 86 | Cov@80% accuracy: MaxProb vs calibrator (+known OOD) | BERT QA | SQuAD -> MRQA OOD mix | coverage % | 48.2 vs 56.1 | Kamath 2020 Sec. 5 |

---

## 5. Verification log

Landing pages fetched and PDFs converted for: 1706.04599, 2002.09437, 1906.02629, 2403.14715, 2006.07322, 2405.18906, 2111.08851, 1901.07884, 1705.05278, CVPR-2019 Diaz, 2606.24959, Crossref 10.1198/016214506000001437, 1503.02531, 1910.01108, 1909.10351, 2002.10957, 2012.15828, 2004.02984, 1908.08962, 1805.04770, 2106.05945, 1910.01348, 2106.05237, 2305.15032, 1903.12136, 2305.02301, 2108.13487, 2406.17633, 2411.05045, 2202.07922, 2104.07540, 2606.24747, 2302.11472, 2508.20224, 1910.12656, 2003.07892, 2211.00151, 2207.05221, 2106.07998, 2107.07511, 2006.02544, 2009.14193, 2305.18404, 1705.08500, 1901.09192, 1907.00208, 2002.10319, 2206.09034, 1805.08206, 2211.15259, 2407.01032, 2302.11874, ACL 2021.acl-long.84, ACL 2022.findings-acl.158, ACL 2020.acl-main.503.

Known extraction caveats: (i) an initial ar5iv extraction of Hsieh et al. Table 1 returned implausible duplicated values; the numbers above were re-read from the PDF text and are correct as printed. (ii) An initial extraction of Hui & Belkin read "2222 of 2828 tasks"; the paper says "22 out of 28 tasks". (iii) Kadavath et al. MC-ECE and Turc et al. per-size accuracies are figure-only and are marked (fig.).

## 6. Gaps noted during search (for Phase 3; not synthesized here)

- No source was found that trains a 100-600M *encoder* with Brier / spherical / RPS losses and reports ECE and coverage-at-risk; the closest are Mukhoti (CNN on 20NG), Hui & Belkin (BERT accuracy only) and Shao (generation BLEU only).
- No source measures calibration or selective-prediction metrics for DistilBERT / TinyBERT / MiniLM students; the KD-and-calibration evidence (Galil; Mishra; Kim) is vision-only.
- No source reports soft-target (teacher-probability) distillation from a decoder LLM into an encoder classifier with an ECE or coverage metric; LLM->encoder work found (Wang 2021; Pangakis 2024; Di Palo 2024; Hsieh 2023) uses hard labels, synthetic text or rationales.
- Evidence on whether a single temperature can reorder confidences is indirect: Guo et al. state TS preserves the argmax; Galil et al. measure that TS changes AUROC (ranking) across 523 multiclass models; no source examines the binary case where a single T is provably monotone in the logit margin.
