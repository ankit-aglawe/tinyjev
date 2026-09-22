# Literature stream A: backbone and head for a <=0.6B typed-decision model

Phase 2 annotated bibliography. Scope is limited to two sub-questions:

1. **BACKBONE** - encoder vs decoder for option-scoring, classification, multiple choice and cross-encoder reranking at <=0.6B parameters.
2. **HEAD** - multiple-choice / set-scoring architectures, label-set generalisation, high-cardinality option handling, permutation and option-ID bias, and decision heads that score discrete action sets.

This document records what the literature measured. It does not synthesise or recommend.

Compiled 2026-09-22.

---

## 1. Search strategy

**Tools.** WebSearch (discovery), WebFetch on arXiv abstract pages and arXiv HTML full text, ACL Anthology, ACM DL, Hugging Face model cards and blogs, Sentence-Transformers documentation, Google Developers Blog, Mixedbread blog, say-can.github.io. Three PDFs that the HTML fetcher could not parse (PET, GLiNER, RankT5) were downloaded and read page-by-page; the BERT PDF was downloaded and read for the SWAG table.

**Date range.** 2018-2026, preference for 2023+. Older items (BERT 2019, Set Transformer 2019, DRRN 2016, Wolpertinger 2015, Yin et al. 2019) are included only because they are the primary source for a head design still used by the replicas under study.

**Queries run (2026-09-22).**

- encoder vs decoder small language model classification comparison matched training 2025
- "Encoder vs Decoder" comparative analysis multilingual NLU ScandEval
- small encoder vs decoder cross-encoder reranker comparison MS MARCO latency 0.5B ModernBERT Qwen reranker
- mmBERT modern multilingual encoder arXiv 2025
- GLiClass generalist lightweight model sequence classification
- SmolLM2 paper 135M 360M benchmarks
- Gemma 3 270M release blog fine-tuning classification benchmarks
- mxbai-rerank-v2 Qwen2.5 0.5B reranker BEIR latency
- "FIRST: Faster Improved Listwise Reranking" EMNLP 2024
- "Smarter, Better, Faster, Longer" ModernBERT ACL 2025
- "Seq vs Seq" paired encoders decoders accepted conference
- "RankT5" SIGIR 2023 proceedings

Direct landing-page fetches (arXiv abs/HTML) for: 2412.13663, 2507.11412, 2507.00994, 2506.05176, 2404.06912, 2405.07920, 2309.03882, 2308.11483, 2407.15018, 2210.12353, 2210.08590, 2406.13469, 2506.21288, 2507.09973, 2509.06888, 2508.07662, 2502.02737, 2505.09388, 2111.09543, 2503.05500, 2406.08660, 2005.00700, 1810.00825, 2210.10634, 2003.06713, 2304.09542, 1909.00161, 2104.08315, 2009.07118, 2406.15657, 2311.08526, 2204.01691, 1511.04636, 1512.07679, 2010.02903, 1810.04805, 2102.09690, 2310.01208.

**Verification rule.** A source is included only if its landing page (arXiv abs, ACL Anthology, ACM DL, HF card, or publisher page) was fetched and the title/authors matched. Every source below passed. Numbers were taken from full text (arXiv HTML or PDF), model cards, or documentation tables; where a number could only be obtained from a fetch summary that looked internally inconsistent it is flagged in the entry.

## 2. Inclusion and exclusion criteria

**Included if all of:**
- Existence verified by fetching the landing page.
- Reports measured numbers on a named benchmark for a model <=~1B parameters, OR is the primary source for a head/scoring design that the open Jev replicas (Kev, NanoJev, Laya, option-letter-logit readout) reuse, OR directly measures the bias phenomenon the head must handle.
- Published 2018-2026; 2023+ preferred.

**Excluded:**
- Surfaced by search but landing page not fetched/verified: "ModernBERT or DeBERTaV3?" (IJCNLP 2025), Querit-Reranker, KaLM-Reranker-V1, MICE, MrBERT, T5Gemma, Lion-vs-AdamW cross-encoder study. Not cited anywhere below.
- Verified but redundant with a stronger, size-explicit source: Bucher & Martini (2024, arXiv 2406.08660; fine-tuned BERT-style vs zero-shot GPT-4/Claude on classification) - superseded here by Abbes et al. (2025), which reports sizes, FLOPs and fine-tuned-decoder controls.
- Secondary write-ups (alphaXiv, aimodels.fyi, liner, Medium tutorials) - never used as evidence.

**Evidence grades used.** `PR` peer-reviewed venue; `WS` workshop (light review); `PP` preprint; `MC` model card / official docs; `BL` vendor or community blog.

---

## 3. Annotated bibliography - Sub-question 1: BACKBONE

Tier 1 = core measured comparisons. Tier 2 = per-model reference numbers and supporting evidence.

### Tier 1

#### B1. Ettin / Seq vs Seq - the only matched-recipe encoder vs decoder suite at 17M-1B

**Citation.** Weller, O., Ricci, K., Marone, M., Chaffin, A., Lawrie, D., & Van Durme, B. (2026). Seq vs Seq: An open suite of paired encoders and decoders. *ICLR 2026*. https://arxiv.org/abs/2507.11412
**Year / venue / grade.** 2025 preprint, published ICLR 2026. `PR`.
**Relevance.** Directly answers "encoder vs decoder at matched size and matched data" for 17M, 32M, 68M, 150M, 400M, 1B - the exact range of interest.
**Key findings (numbers).**
- Encoder GLUE average by size: 17M 79.2, 32M 83.5, 68M 87.2, 150M 88.9, 400M 90.8, 1B 91.6. Encoder MNLI: 79.5 / 83.4 / 87.0 / 89.2 / 91.3 / 91.8.
- Dense retrieval, MS MARCO dev nDCG@10 (Table 9), encoder vs decoder at each size: 17M 30.93 vs 29.11; 32M 35.13 vs 32.93; 68M 38.17 vs 36.12; 150M 39.97 vs 37.71; 400M 42.24 vs 39.93; 1B 43.35 vs 41.70. Encoder leads by 1.6-2.3 points at every size; a decoder needs roughly one size step up to match the encoder below it.
- Cross-objective continued training (50B extra tokens), 150M: native encoder MNLI 89.2 vs decoder-continued-as-encoder 85.8; native decoder generative avg 46.2 vs encoder-continued-as-decoder 43.6. Abstract: "a 400M encoder outperforms a 1B decoder on MNLI" - note this refers to a 1B decoder *adapted* to MLM then fine-tuned, not a decoder fine-tuned with a classification head directly.
- Encoders beat ModernBERT at equal size; decoders beat Llama 3.2 / SmolLM2 at equal size.
**Methodology.** Same data (up to 2T tokens), architecture family (ModernBERT-style) and schedule for both objectives; 200+ checkpoints released.
**Limitations.** Decoders are **not** fine-tuned on GLUE with a classification head - they are evaluated generatively via the Eleuther harness, so the paper gives no decoder GLUE row. The encoder-vs-decoder retrieval numbers are bi-encoder dense retrieval, not cross-encoder reranking. No latency measurements.
**Contribution to sub-question.** The cleanest controlled evidence that, at matched data and size, the MLM/bidirectional backbone wins on representation tasks by a margin worth ~one size step; and that post-hoc adaptation of a decoder does not close the gap.

#### B2. Should we still pretrain encoders with MLM? - CLM vs MLM at 210M-1B

**Citation.** Gisserot-Boukhlef, H., Boizard, N., Faysse, M., Alves, D. M., Malherbe, E., Martins, A. F. T., Hudelot, C., & Colombo, P. (2025). Should we still pretrain encoders with masked language modeling? *arXiv*. https://arxiv.org/abs/2507.00994
**Year / venue / grade.** 2025 (v4 May 2026). `PP`.
**Relevance.** Separates the *objective* (CLM vs MLM) from the *architecture* question, at exactly the target sizes, on sequence classification and retrieval.
**Key findings (numbers; 100B tokens, average over datasets).**
- Sequence classification (SST-2, MNLI, QQP): 210M CLM 82.80 vs MLM 84.89; 610M 83.58 vs 87.00; 1B 82.68 vs 88.23. Gap widens with size.
- Token classification (CoNLL, OntoNotes, UNER): 210M 92.18 vs 92.13; 610M 92.69 vs 92.21; 1B 92.46 vs 92.51 - parity.
- QA (SQuAD, SQuAD-v2, ReCoRD): 210M 39.26 vs 49.43; 610M 42.09 vs 62.77; 1B 44.48 vs 70.28.
- IR (MS MARCO, NQ, MLDR): 210M 72.83 vs 75.78; 610M 76.20 vs 79.55; 1B 78.68 vs 79.95.
- Biphasic CLM-then-MLM with 25%/75% split "reliably surpasses the MLM baseline"; MLM continued-pretraining on a CLM checkpoint beats MLM from scratch; CLM pretraining gives lower learning-rate sensitivity at fine-tuning.
**Methodology.** 38 models, 3 sizes, 15k+ fine-tuning runs, 110k GPU-hours; same architecture, only objective and schedule vary.
**Limitations.** 100B-token budget is small versus production models; benchmarks are English; no MC or reranking-specific evaluation; no latency.
**Contribution.** Quantifies the classification penalty of a causal-only backbone (2-5.5 points SC at 210M-1B) and shows the cheapest route to a strong small encoder is MLM continued-pretraining of an existing small decoder.

#### B3. ModernBERT - reference modern encoder and its throughput

**Citation.** Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., & Poli, I. (2025). Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. *Proceedings of ACL 2025 (Long Papers)*, 2526-2547. https://arxiv.org/abs/2412.13663
**Year / venue / grade.** 2024 preprint, ACL 2025. `PR`.
**Relevance.** Laya's backbone; the encoder baseline every 2025 comparison uses.
**Key findings (numbers).**
- Params: base 149M, large 395M; 2T tokens; 8192 context.
- GLUE avg, base: BERT 84.7, RoBERTa 86.4, DeBERTa-v3 88.1, NomicBERT 84.0, GTE-en-MLM 85.6, ModernBERT 88.4. Large: BERT 85.2, RoBERTa 88.9, DeBERTa-v3 91.4, GTE-en-MLM 87.6, ModernBERT 90.4. DeBERTa-v3-large still wins GLUE at large.
- BEIR nDCG@10: base DPR 41.6 (GTE 41.4), ColBERT 51.3 (GTE 48.2); large DPR 44.0 (42.5), ColBERT 52.4 (50.7).
- Throughput (RTX 4090): ModernBERT-base 123.7k tokens/s on 8192-length variable-length input vs GTE-en-MLM 47.5k.
**Methodology.** Standard GLUE fine-tuning, DPR/ColBERT retrieval fine-tuning on MS MARCO, synthetic throughput benchmark.
**Limitations.** English only; GLUE numbers for DeBERTa-v3 are the reported ones, not re-run; throughput is tokens/s on a consumer GPU, not per-query latency.
**Contribution.** Supplies encoder accuracy and tokens/s at 149M and 395M, and the finding that DeBERTa-v3 remains the accuracy-per-parameter leader among encoders while ModernBERT leads on speed.

#### B4. Encoder vs decoder on multilingual NLU (ScandEval)

**Citation.** Nielsen, D. S., Enevoldsen, K., & Schneider-Kamp, P. (2025). Encoder vs decoder: Comparative analysis of encoder and decoder language models on multilingual NLU tasks. *Proceedings of NoDaLiDa/Baltic-HLT 2025*, 561-572. https://aclanthology.org/2025.nodalida-1.60/ (arXiv: https://arxiv.org/abs/2406.13469)
**Year / venue / grade.** 2024 preprint, NoDaLiDa 2025. `PR`.
**Relevance.** Large-N comparison of fine-tuned encoders against few-shot decoders on classification-style NLU.
**Key findings (numbers).**
- English leaderboard (mean rank, lower is better): DeBERTa-v3-large 1.09, DeBERTa-v3-base 1.29, ELECTRA-base 1.39 vs GPT-4-0613 1.44, GPT-4-1106 1.54, GPT-4o 1.64. GPT-4-0613 is on average 0.44 SD worse than the best model.
- Task pattern: decoders are strongly favoured on reading comprehension/QA; encoders on NER and linguistic acceptability; sentiment mixed. Danish/Swedish leaderboards are decoder-dominated; Icelandic parity.
**Methodology.** Fine-tuned encoders vs few-shot (not fine-tuned) decoders, 10 bootstrapped few-shot draws, 8 Germanic languages.
**Limitations.** Decoders are not fine-tuned, so the comparison mixes "architecture" with "training regime"; scores are rank-based, not absolute accuracy; decoder param counts not tabulated in one place.
**Contribution.** Evidence that a 184M-435M DeBERTa-v3 beats frontier decoders on fine-tuned classification-type tasks; also evidence that the encoder advantage is task-dependent and disappears on QA-like tasks.

#### B5. Small encoders vs large decoders for groundedness classification (with FLOPs)

**Citation.** Abbes, I., Prato, G., Fournier, Q., Rodriguez, F., Boukhary, A., Elwood, A., & Chandar, S. (2025). Small encoders can rival large decoders in detecting groundedness. *arXiv*. https://arxiv.org/abs/2506.21288
**Year / venue / grade.** 2025. `PP`.
**Relevance.** Binary classification of (context, claim) pairs - structurally the same as scoring one option against a state - with fine-tuned decoders as a control.
**Key findings (numbers).**
- Fine-tuned RoBERTa-large: 90.2% SQuAD-v2 groundedness, 88.5% NewsQA.
- Llama-3.1-8B-Instruct zero-shot: 81.9% / 79.4%; fine-tuned: 91.1% on SQuAD-v2. GPT-4o zero-shot 95.5% / 98.1%.
- Inference cost: RoBERTa-large 3.7e11 FLOPs vs Llama-3.1-8B 1.6e13 FLOPs (43x). The paper additionally claims roughly 1000x wall-clock speedup; that figure was not reproduced here and exceeds the FLOPs ratio - treat as unverified.
**Methodology.** Fine-tune BERT/RoBERTa/NomicBERT/ModernBERT/NeoBERT; zero-shot and fine-tuned Llama-3.2-1B/3B and 3.1-8B; four datasets.
**Limitations.** No 0.5B decoder fine-tuned comparison (smallest decoder is 1B, and its fine-tuned number was not extracted); accuracy of ModernBERT/NeoBERT variants not reported above.
**Contribution.** Shows a fine-tuned 355M encoder within ~1 point of a fine-tuned 8B decoder on pair classification at ~43x fewer FLOPs.

#### B6. Tiny Reward Models - ModernBERT preference scorers vs a 70B decoder RM

**Citation.** Pan, S. (2025). Tiny reward models. *ICML 2025 Workshop on Efficient Systems for Foundation Models*. https://arxiv.org/abs/2507.09973
**Year / venue / grade.** 2025. `WS`.
**Relevance.** Pairwise preference scoring with both candidates visible in one forward pass - the same "see all options, output a score" regime as a typed-decision head.
**Key findings (RewardBench, chat / reasoning / safety / overall).**
- ModernBERT-large specialists (400M): 78.8 / 91.2 / 89.3 / 86.4.
- ModernBERT-base specialists (150M): 73.5 / 83.3 / 78.4 / 78.4.
- Llama3-SteerLM-RM (70B): 89.7 / 90.6 / 92.8 / 91.0.
- Reasoning: 400M encoder 91.2 vs 70B decoder 90.6.
**Methodology.** FLAN-style prompting, DoRA, layer freezing; per-domain specialists.
**Limitations.** Options are visible jointly in context, which is not the official RewardBench protocol, so numbers are not comparable to the leaderboard; single author, workshop review; chat domain lags badly (-11 points).
**Contribution.** Direct evidence that a 150M-400M encoder that sees candidates jointly can match a 70B decoder on reasoning/safety preference scoring but not on open-ended chat.

#### B7. Qwen3 Embedding / Qwen3-Reranker-0.6B - a 0.6B decoder cross-encoder vs 0.3-0.6B encoder rerankers

**Citation.** Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., Huang, F., & Zhou, J. (2025). Qwen3 Embedding: Advancing text embedding and reranking through foundation models. *arXiv*. https://arxiv.org/abs/2506.05176 . Model card: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
**Year / venue / grade.** 2025. `PP` + `MC`.
**Relevance.** The best-documented 0.6B *decoder* cross-encoder, scored by yes/no token logits - the "read option logits off a stock LM" head, fine-tuned.
**Key findings (nDCG@10; MTEB-R / CMTEB-R / MMTEB-R / MLDR / MTEB-Code / FollowIR).**
- Qwen3-Reranker-0.6B (0.6B decoder): 65.80 / 71.31 / 66.36 / 67.28 / 73.42 / 5.41.
- bge-reranker-v2-m3 (0.6B XLM-R encoder): 57.03 / 72.16 / 58.36 / 59.51 / 41.38 / -0.01.
- gte-multilingual-reranker-base (0.3B encoder): 59.51 / 74.08 / 59.44 / 66.33 / 54.18 / -1.64.
- Jina-multilingual-reranker-v2-base (0.3B): 58.22 / 63.37 / 63.73 / 39.66 / 58.98 / -0.68.
- Qwen3-Reranker-4B: 69.76 / 75.94 / 72.74 / 69.97 / 81.20 / 14.84. 8B: 69.02 / 77.45 / 72.94 / 70.19 / 81.22 / 8.05.
- 0.6B-to-4B gap: 4.0 (MTEB-R), 6.4 (MMTEB-R), 7.8 (Code), 9.4 (FollowIR).
- Scoring: score(q,d) = softmax over logits of "yes" vs "no" at the next token; SFT with -log p(label); no weakly-supervised stage for the reranker.
**Methodology.** Supervised fine-tuning of Qwen3-0.6B/4B/8B; baselines are published checkpoints (different training data).
**Limitations.** Not a matched-training comparison; encoder baselines are older and trained on less data; no latency reported.
**Contribution.** Gives the reference gap between a 0.6B and a 4B decoder scorer on the same task (4-9 points) and shows a well-trained 0.6B decoder beating older 0.3-0.6B encoders on English/multilingual reranking while an encoder still wins on Chinese (CMTEB-R).

#### B8. Ettin reranker family - encoder rerankers 17M-1B with throughput on H100 / 3090 / CPU

**Citation.** Aarsen, T., et al. (2026, May 19). Introducing the Ettin reranker family. *Hugging Face Blog*. https://huggingface.co/blog/ettin-reranker
**Year / venue / grade.** 2026. `BL` (Hugging Face / LightOn / JHU authors; numbers from their own MTEB runs).
**Relevance.** The only source with encoder cross-encoders across six sizes measured against a 0.6B decoder reranker on the same benchmark with throughput on three hardware classes.
**Key findings (MTEB(eng, v2) retrieval, top-100 rerank, nDCG@10).**
- 17M 0.5576; 32M 0.5779; 68M 0.5915; 150M 0.5994; 400M 0.6091; 1B 0.6114.
- Qwen3-Reranker-0.6B (596M decoder) 0.5940 - matched by the 68M encoder, beaten by the 150M encoder by +0.005. bge-reranker-v2-m3 (568M) is 0.025 below the 32M encoder (~0.553). ms-marco-MiniLM-L12-v2 (33M) is 0.051 below the 17M (~0.507). Teacher mxbai-rerank-large-v2 (1.54B decoder) ~0.6115.
- Throughput, H100 bf16 + FA2, pairs/s: 17M 7,517; 32M 6,602; 68M 4,913; 150M 3,237 (architectural peers 1,404-1,418); 400M 1,738; 1B 928 (teacher 387). RTX 3090: 17M 9,008; 1B 189. CPU i7-13700K: 17M 267.4; 1B 2.1.
- Training: pointwise MSE distillation of raw teacher logits from mxbai-rerank-large-v2 on ~143M (query, doc, score) triples.
**Methodology.** Distillation, not supervised training; evaluation by the authors.
**Limitations.** Vendor blog; students inherit the teacher's ceiling; the decoder comparison points are third-party checkpoints trained differently; throughput is for pairs at reranking lengths, not for one state + N options in one pass.
**Contribution.** Concrete accuracy-per-parameter and pairs-per-second curve for encoders 17M-1B, and the data point that a 150M distilled encoder equals a 596M supervised decoder on English reranking at ~2.3x the throughput of same-size peers.

### Tier 2

#### B9. DeBERTaV3

**Citation.** He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. *ICLR 2023*. https://arxiv.org/abs/2111.09543 . Cards: https://huggingface.co/microsoft/deberta-v3-base , https://huggingface.co/microsoft/deberta-v3-large
**Grade.** `PR` + `MC`.
**Numbers.** Base: 86M backbone + 98M embeddings (128K vocab); MNLI-m/mm 90.6/90.7; SQuAD 2.0 F1/EM 88.4/85.4 (RoBERTa-base 87.6 MNLI, ELECTRA-base 88.8). Large: 304M backbone + 131M embeddings; MNLI 91.8/91.9; SQuAD 2.0 91.5/89.0; GLUE avg 91.37. mDeBERTa-base XNLI zero-shot 79.8 (+3.6 over XLM-R base).
**Limitations.** 512 context; slow relative to ModernBERT (see B3); numbers are the authors' own.
**Contribution.** Highest accuracy-per-parameter encoder at 184M/435M; the backbone GLiNER, GLiClass and the NLI zero-shot classifiers (H4-H6) all chose over ModernBERT on accuracy grounds.

#### B10. mmBERT

**Citation.** Marone, M., Weller, O., Fleshman, W., Yang, E., Lawrie, D., & Van Durme, B. (2025). mmBERT: A modern multilingual encoder with annealed language learning. *arXiv*. https://arxiv.org/abs/2509.06888
**Grade.** `PP`.
**Numbers.** Small 140M total / 42M non-embedding; base 307M / 110M non-embedding; 3T tokens, 1,833 languages. GLUE avg: ModernBERT-base 87.4, mmBERT-base 86.3, mmBERT-small 84.7, XLM-R-base 83.3, EuroBERT-210m 81.2. XTREME: mmBERT-base 72.8, mGTE-base 71.1, XLM-R-base 70.4, mmBERT-small 68.6. MTEB v2 English: mmBERT-base 53.9 vs ModernBERT-base 53.8. Claimed >2x faster than prior multilingual encoders on variable-length input, ~4x at long context.
**Contribution.** Shows a 42M-non-embedding encoder at 84.7 GLUE; puts EuroBERT-210m at 81.2 GLUE on the same harness (English).

#### B11. EuroBERT

**Citation.** Boizard, N., Gisserot-Boukhlef, H., Alves, D. M., et al. (2025). EuroBERT: Scaling multilingual encoders for European languages. *arXiv*. https://arxiv.org/abs/2503.05500
**Grade.** `PP`.
**Numbers.** 210M / 610M / 2.1B; 5T tokens; 8,192 context. MIRACL nDCG@10: 210M 90.82, 610M 92.62 vs XLM-R-280M 85.44, XLM-R-560M 89.43, mGTE-MLM-305M 91.22. XNLI acc: 210M 81.93, 610M 84.11 vs XLM-R-280M 74.16, XLM-R-560M 81.72. CodeSearchNet: 210M 58.94, 610M 69.92 vs ModernBERT-150M 53.95, ModernBERT-395M 65.83.
**Limitations.** Decoder-style architecture trained with MLM; English GLUE is weak relative to ModernBERT (81.2 per B10).
**Contribution.** A 610M encoder option with strong multilingual classification/retrieval; shows the 210M-to-610M step is worth ~2 XNLI points and ~1.8 MIRACL points.

#### B12. SmolLM2 (paper + model cards)

**Citation.** Ben Allal, L., Lozhkov, A., Bakouch, E., et al. (2025). SmolLM2: When smol goes big - data-centric training of a small language model. *arXiv*. https://arxiv.org/abs/2502.02737 . Cards: https://huggingface.co/HuggingFaceTB/SmolLM2-360M , https://huggingface.co/HuggingFaceTB/SmolLM2-135M
**Grade.** `PP` + `MC`. The paper defers 135M/360M numbers to the cards.
**Numbers (base, zero-shot unless stated).** 360M (4T tokens) vs Qwen2.5-0.5B vs SmolLM-360M: HellaSwag 54.5 / 51.2 / 51.8; ARC avg 53.0 / 45.4 / 50.1; PIQA 71.7 / 69.9 / 71.6; MMLU-cloze 35.8 / 33.7 / 34.4; CommonsenseQA 38.0 / 31.6 / 35.3; TriviaQA 16.9 / 4.3 / 9.1; Winogrande 52.5 / 54.1 / 52.8; OpenBookQA 37.4 / 37.4 / 37.2; GSM8K 5-shot 3.2 / 33.4 / 1.6. 135M (2T tokens): HellaSwag 42.1, ARC 43.9, PIQA 68.4, MMLU-cloze 31.5, CSQA 33.9, TriviaQA 4.1, Winogrande 51.3, OBQA 34.6.
**Limitations.** MMLU is reported in *cloze* form because letter-format MMLU is near chance at this scale (see H15/H18); these are likelihood-scored MC, not a trained head.
**Contribution.** Reference zero-shot MC numbers for 135M/360M decoders and Qwen2.5-0.5B on one harness.

#### B13. Qwen3 Technical Report (0.6B base)

**Citation.** Yang, A., Li, A., Yang, B., et al. (Qwen Team). (2025). Qwen3 technical report. *arXiv*. https://arxiv.org/abs/2505.09388
**Grade.** `PP`.
**Numbers.** 36T pretraining tokens. Qwen3-0.6B-Base vs Qwen2.5-0.5B-Base: MMLU 52.81 vs 47.50; MMLU-Redux 51.26 vs 45.10; BBH 41.47 vs 20.30; GSM8K 59.59 vs 41.62. Qwen3-1.7B-Base vs Qwen2.5-1.5B: MMLU 62.63 vs 60.90; BBH 54.47 vs 45.10.
**Contribution.** Reference numbers for the decoder backbone Kev and NanoJev use; 0.6B-to-1.7B MMLU gap is ~10 points.

#### B14. Gemma 3 270M (blog + model card)

**Citation.** Lacombe, O., Kenealy, K., Black, K., Kumar, R., Visin, F., & Zhang, J. (2025, August 14). Introducing Gemma 3 270M: The compact model for hyper-efficient AI. *Google Developers Blog*. https://developers.googleblog.com/en/introducing-gemma-3-270m/ . Card: https://huggingface.co/google/gemma-3-270m
**Grade.** `BL` + `MC`.
**Numbers.** 270M = 170M embedding + 100M transformer; 256k vocab; 6T tokens; 32K context. PT: HellaSwag (10-shot) 40.9, BoolQ 61.4, PIQA 67.7, TriviaQA (5-shot) 15.4, ARC-c (25-shot) 29.0, ARC-e 57.7, WinoGrande (5-shot) 52.0. IT: IFEval 51.2, BBH 26.7, HellaSwag 0-shot 37.7, ARC-c 28.2. INT4 on Pixel 9 Pro: 0.75% battery for 25 conversations. Intended use stated as fine-tuning for classification, routing, extraction.
**Limitations.** Only 100M non-embedding parameters; benchmark numbers are Google's own; no classification fine-tune numbers published by Google.
**Contribution.** Defines the smallest decoder candidate; its ARC-c/HellaSwag are below SmolLM2-360M and roughly at SmolLM2-135M level despite 270M total params, because most params are embeddings.

#### B15. Sentence-Transformers MS MARCO cross-encoders (accuracy vs docs/s at 4M-110M)

**Citation.** Reimers, N., & Sentence-Transformers contributors. (n.d.). Pretrained cross-encoders: MS MARCO. *Sentence-Transformers documentation*. https://www.sbert.net/docs/cross_encoder/pretrained_models.html
**Grade.** `MC`.
**Numbers (TREC DL19 nDCG@10 / MS MARCO dev MRR@10 / docs per second).** TinyBERT-L2 69.84 / 32.56 / 9,000; MiniLM-L2 71.01 / 34.85 / 4,100; MiniLM-L4 73.04 / 37.70 / 2,500; MiniLM-L6 74.30 / 39.01 / 1,800; MiniLM-L12 74.31 / 39.02 / 960; electra-base 71.99 / 36.41 / 340.
**Limitations.** Hardware for docs/s not stated on the current page; models are 2021-era.
**Contribution.** The classic encoder accuracy-vs-throughput knee: L6 to L12 doubles cost for +0.01 nDCG.

#### B16. mxbai-rerank-v2 (Qwen2.5-0.5B decoder reranker)

**Citation.** Mixedbread. (2025, March 13). Baked-in brilliance: Reranking meets RL with mxbai-rerank-v2. *Mixedbread Blog*. https://www.mixedbread.com/blog/mxbai-rerank-v2
**Grade.** `BL`.
**Numbers (BEIR nDCG@10).** mxbai-rerank-base-v2 (Qwen2.5-0.5B) 55.57; large-v2 (1.5B) 57.49; cohere-rerank-3.5 55.39; bge-reranker-v2-gemma (2.5B) 55.38; voyage-rerank-2 54.54; jina-reranker-v2-base-multilingual 54.35; bge-reranker-v2-m3 (568M encoder) 53.94. Latency (A100, s/query): base 0.67, large 0.89, bge-reranker-v2-gemma 7.20. Training: GRPO -> contrastive -> preference.
**Limitations.** Vendor numbers; per-query latency without stated candidate count.
**Contribution.** A 0.5B decoder reranker 1.6 nDCG above a 568M encoder reranker on BEIR (different training); the 0.5B-to-1.5B step is worth +1.9.

#### B17. Label Supervised LLaMA Finetuning (design precedent, out of scale)

**Citation.** Li, Z., Li, X., Liu, Y., et al. (2023). Label supervised LLaMA finetuning. *arXiv*. https://arxiv.org/abs/2310.01208
**Grade.** `PP`. 7B model - included only as the primary source for "decoder + label head, drop the causal mask".
**Findings.** Final-layer hidden state projected to label space with cross-entropy + LoRA beats BERT-large / RoBERTa-large on text classification; removing the causal mask (LS-unLLaMA) gives SOTA on NER. No numbers extracted at <=0.6B.
**Contribution.** Documents that bidirectionalising a decoder at fine-tune time helps token-level tasks; motivates NanoJev-style pooling choices but gives no small-scale evidence.

---

## 4. Annotated bibliography - Sub-question 2: HEAD

### 4a. Multiple-choice heads on encoders (shared scorer over (context, option) pairs; [MASK]-per-option)

#### H1. BERT SWAG head - the original shared-linear-head multiple-choice scorer

**Citation.** Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171-4186. https://aclanthology.org/N19-1423/ (arXiv: https://arxiv.org/abs/1810.04805)
**Grade.** `PR`.
**Design (Sec. 4.4).** Build one sequence per option (context ++ option); the only new parameter is a vector whose dot product with each sequence's [CLS] gives that option's score; softmax across options. This is the `*ForMultipleChoice` head.
**Numbers (SWAG, 4-way).** BERT-base dev 81.6; BERT-large dev 86.6 / test 86.3; ESIM+ELMo 59.1 / 59.2; OpenAI GPT test 78.0. BERT-base 110M, large 340M.
**Limitations.** N forward passes for N options; no inter-option attention; 2019 baselines.
**Contribution.** Primary source for the per-option shared scorer and the 110M-encoder vs 117M-decoder (GPT) comparison on MC: 86.6 vs 78.0 at similar size (different pretraining).

#### H2. PET / iPET - cloze [MASK] head on a 223M encoder beats GPT-3 175B few-shot

**Citation.** Schick, T., & Schütze, H. (2021). It's not just size that matters: Small language models are also few-shot learners. *Proceedings of NAACL-HLT 2021*. https://arxiv.org/abs/2009.07118
**Grade.** `PR`.
**Numbers (SuperGLUE, 32 training examples; Table 1).** Test avg: GPT-3 175B 71.8; PET (ALBERT-xxlarge-v2, 223M) 74.0; iPET 75.4. Dev avg: GPT-3 73.2; PET 74.1; iPET 76.8; GPT-3 Medium (350M) 56.2; GPT-3 Large (760M) 56.8. Per-task test (GPT-3 / PET / iPET): BoolQ 76.4 / 79.1 / 81.2; CB acc 75.6 / 87.2 / 88.8; COPA 92.0 / 90.8 / 90.8; RTE 69.0 / 67.2 / 70.8; WiC 49.4 / 50.7 / 49.3; WSC 80.1 / 88.4 / 88.4; MultiRC F1a 75.4 / 76.6 / 74.1; ReCoRD F1 91.1 / 85.9 / 85.9.
**Methodology.** Pattern-verbalizer pairs; multi-token verbalizers for COPA/WSC/ReCoRD; ensembling + distillation.
**Limitations.** Requires unlabeled data for distillation; ALBERT is slow; gradient-based, so not zero-shot.
**Contribution.** Strongest early evidence that a [MASK]-verbalizer head on a ~0.2B encoder outperforms 350M-760M decoders by ~18 points and a 175B decoder by 2-4 points on MC-style tasks.

#### H3. UniMC - one [O-MASK] per option, option-masked attention, 235M

**Citation.** Yang, P., Wang, J., Gan, R., Zhu, X., Zhang, L., Wu, Z., Gao, X., Zhang, J., & Sakai, T. (2022). Zero-shot learners for natural language understanding via a unified multiple choice perspective. *Proceedings of EMNLP 2022*. https://arxiv.org/abs/2210.08590
**Grade.** `PR`.
**Design.** Every task is cast as passage + question + options; each option is prefixed with an [O-MASK] token predicting yes/no; an attention mask **prevents options from attending to each other**; losses = MLM + option-MLM + option-prediction. Backbone ALBERT-xxlarge-v2, 235M. This is the closest published ancestor of Laya's one-[MASK]-per-option encoder.
**Numbers (zero-shot; ANLI R1 / R2 / R3 / CB).** UniMC 235M: 52.0 / 44.4 / 47.8 / 75.7. PaLM 540B: 48.4 / 44.2 / 45.7 / 51.8. FLAN 137B: 47.7 / 43.9 / 47.0 / 64.1. T0 11B: 43.6 / 38.7 / 41.3 / 70.1.
**Limitations.** Zero-shot transfer after MC pre-finetuning on many datasets; the design choice to block inter-option attention is asserted, not ablated against set-attention.
**Contribution.** Shows a 235M encoder with per-option mask tokens and a shared yes/no scorer beating 11B-540B decoders zero-shot on NLI-as-MC; establishes the independence-between-options design as a documented alternative to set attention (contrast H5, H10).

#### H4. GLiNER - one marker token per label type, matched against spans

**Citation.** Zaratiana, U., Tomeh, N., Holat, P., & Charnois, T. (2024). GLiNER: Generalist model for named entity recognition using bidirectional transformer. *Proceedings of NAACL 2024*, 5364-5376. https://aclanthology.org/2024.naacl-long.300/ (arXiv: https://arxiv.org/abs/2311.08526)
**Grade.** `PR`.
**Design.** Input = `[ENT] type_1 [ENT] type_2 ... [SEP] text`; the [ENT] representations pass through an FFN to give label embeddings; spans get embeddings; score = sigmoid(span . label); BCE training. All labels scored in one forward pass; labels attend to each other and to the text. DeBERTa-v3 backbone.
**Numbers (zero-shot F1).** OOD NER benchmark avg: GLiNER-S (50M) 52.7, GLiNER-M (90M) 55.4, GLiNER-L (0.3B) 60.9 vs ChatGPT 47.5, UniNER-7B 53.7, UniNER-13B 55.6, InstructUIE-11B 47.2, GoLLIE-7B 58.0, USM (0.3B) 37.8. 20-dataset avg: GLiNER-L 47.8 vs UniNER-7B 45.7 vs ChatGPT 36.5.
**Limitations.** Token/span labelling, not option selection; label-count scaling not measured here (see H5).
**Contribution.** Primary source for the "marker token per label in the encoder input" pattern Laya reuses; a 90M model matches a 13B decoder.

#### H5. GLiClass - label-token encoder for classification with measured label-count scaling

**Citation.** Stepanov, I., et al. (2025). GLiClass: Generalist lightweight model for sequence classification tasks. *arXiv*. https://arxiv.org/abs/2508.07662
**Grade.** `PP`.
**Design.** Each label prefixed with `<<LABEL>>`, concatenated with the text, one bidirectional pass; label tokens pooled (first-token / mean / attention) and scored by dot product or MLP against the text representation; explicit label-label interaction. DeBERTa-v3 backbones: edge 32.7M, base 187M, large 439M. Authors report DeBERTa-based variants "consistently outperform" ModernBERT-based ones.
**Numbers.** Zero-shot avg F1 over their benchmark suite: gliclass-large-v3.0 0.7193 vs deberta-v3-large-zeroshot-v2.0 (NLI cross-encoder) 0.6821; gliclass-base 0.6764. Throughput (A6000): edge 97.29 ex/s, base 51.61, large 25.22; 2.3x-16x over cross-encoders. Label-count scaling 1 -> 128 labels: edge 103.81 -> 82.64 ex/s (-20%), base 49.42 -> 45.94 (-7%), large 19.05 -> 17.60 (-7.6%) vs deberta-v3-base-zeroshot-v2.0 24.55 -> 0.47 ex/s (52x slower). Few-shot with 8 examples/label: edge +50% relative.
**Limitations.** Preprint; benchmark suite is the authors' own selection; "ex/s" at unstated sequence length.
**Contribution.** Only source measuring how a one-pass label-token head scales with option count versus a per-option cross-encoder; also the DeBERTa-vs-ModernBERT backbone note.

#### H6. Zero-shot classification via NLI cross-encoders (label-set generalisation)

**Citation.** Yin, W., Hay, J., & Roth, D. (2019). Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach. *Proceedings of EMNLP-IJCNLP 2019*. https://arxiv.org/abs/1909.00161 . Reference implementation and numbers: Laurer, M. (2024). deberta-v3-large-zeroshot-v2.0 [model card]. https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0
**Grade.** `PR` (Yin) + `MC` (Laurer).
**Design.** Each candidate label becomes a hypothesis ("This text is about X"); an NLI cross-encoder scores entailment for each (text, hypothesis) pair - one forward pass per label; label-fully-unseen setting.
**Numbers (Laurer card, mean F1-macro over 28 tasks).** deberta-v3-large-zeroshot-v2.0 (0.4B) 0.676; -c variant 0.673; deberta-v3-base-zeroshot-v2.0-c 0.619; bge-m3-zeroshot-v2.0-c 0.590; bart-large-mnli 0.497.
**Limitations.** Cost linear in label count (see H5 for the 52x collapse at 128 labels); Yin et al. numbers are 2019-era BERT.
**Contribution.** The baseline label-set-generalising head; the number GLiClass beats by +0.037.

### 4b. Set / listwise scoring and permutation invariance

#### H7. Set Transformer

**Citation.** Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., & Teh, Y. W. (2019). Set Transformer: A framework for attention-based permutation-invariant neural networks. *Proceedings of ICML 2019*. https://arxiv.org/abs/1810.00825
**Grade.** `PR`.
**Design.** Self-attention blocks over set elements (SAB); Induced Set Attention Block (ISAB) with m inducing points makes cost linear in set size; Pooling by Multihead Attention (PMA) for permutation-invariant readout.
**Contribution.** Theoretical basis for NanoJev's set-attention-over-candidates layer; gives the ISAB trick for large candidate sets. No NLP numbers.

#### H8. Set-Encoder - permutation-invariant listwise cross-encoder at 110M/330M

**Citation.** Schlatt, F., Fröbe, M., Scells, H., Zhuang, S., Koopman, B., Zuccon, G., Stein, B., Potthast, M., & Hagen, M. (2025). Set-Encoder: Permutation-invariant inter-passage attention for listwise passage re-ranking with cross-encoders. *Proceedings of ECIR 2025*. https://arxiv.org/abs/2404.06912
**Grade.** `PR`.
**Design.** Each candidate carries an [INT] interaction token; tokens attend within their own candidate plus to all [INT] tokens; positional encodings restart at 0 per candidate, so the model is permutation-invariant by construction; ELECTRA-base (110M) / large (330M).
**Numbers (nDCG@10; first stage BM25 / ColBERTv2).** TREC DL19: Set-Encoder-330M 0.733 / 0.765; monoELECTRA-330M 0.720 / 0.768; monoT5-3B 0.705 / 0.745; RankGPT-4o 0.725 / 0.784; RankZephyr-7B 0.719 / 0.749. DL20: Set-Encoder 0.727 / 0.799; monoELECTRA 0.711 / 0.770; monoT5-3B 0.715 / 0.757; RankGPT-4o 0.719 / 0.793. Authors: differences to other cross-encoders not significant. Reverse-ideal input order: Set-Encoder stays above RankGPT-4o and LiT5-Distill. Efficiency for 100 passages: Set-Encoder-330M 0.219 s / 2.60 GB; RankZephyr 24.047 s / 15.48 GB; RankGPT-4o 18.773 s (85x and 110x faster).
**Methodology.** Stage 1 MS MARCO InfoNCE; stage 2 RankNet on RankZephyr distillation scores.
**Limitations.** Inter-candidate attention gives no significant relevance gain over pointwise; gains appear on novelty-aware ranking; candidates are passages, not short options.
**Contribution.** The direct precedent for NanoJev's set attention: shows a 330M encoder with fused-token inter-candidate attention matches 7B listwise decoders at ~100x lower latency and is immune to ordering.

#### H9. Rank-DistiLLM - distilled 110M/330M cross-encoders match 7B / GPT-4 listwise rerankers

**Citation.** Schlatt, F., Fröbe, M., Scells, H., Zhuang, S., Koopman, B., Zuccon, G., Stein, B., Potthast, M., & Hagen, M. (2025). Rank-DistiLLM: Closing the effectiveness gap between cross-encoders and LLMs for passage re-ranking. *Proceedings of ECIR 2025*. https://arxiv.org/abs/2405.07920
**Grade.** `PR`.
**Numbers (nDCG@10, DL19 / DL20).** monoELECTRA-base 110M distilled: 0.720 / 0.711; monoELECTRA-large 330M distilled: 0.733 / 0.727; teachers RankGPT-4 0.713 / 0.713, RankZephyr 0.719 / 0.720; MS MARCO-only monoELECTRA-base 0.687 / 0.698. Latency (100 passages): monoELECTRA-large 0.215 s vs RankZephyr 24.047 s, RankGPT-4 20.234 s; abstract: up to 173x faster, 24x less memory.
**Contribution.** Quantifies what distillation from a 7B decoder buys a 110M encoder: +3.3 / +1.3 nDCG over supervised training, ending at or above the teacher.

#### H10. RankT5 - encoder-only vs encoder-decoder score heads, listwise vs pointwise losses

**Citation.** Zhuang, H., Qin, Z., Jagerman, R., Hui, K., Ma, J., Lu, J., Ni, J., Wang, X., & Bendersky, M. (2023). RankT5: Fine-tuning T5 for text ranking with ranking losses. *Proceedings of SIGIR 2023*, 2308-2313. https://doi.org/10.1145/3539618.3592047 (arXiv: https://arxiv.org/abs/2210.10634)
**Grade.** `PR`.
**Design.** RankT5-Enc: T5 encoder + linear head on first token; RankT5-EncDec: decoder emits a score. Losses: pointwise CE, pairwise logistic, listwise softmax CE, Poly-1.
**Numbers (T5-Large, MS MARCO dev MRR@10 / NQ MRR@10).** BERT-large PointCE 0.3867 / 0.5157, Softmax 0.3928 / 0.5213; monoT5 0.4156 / 0.5406; RankT5-EncDec PointCE 0.4209 / 0.5403, Softmax 0.4278 / 0.5687, Poly1 0.4343 / 0.5647; RankT5-Enc PointCE 0.4216 / 0.5441, Softmax 0.4305 / 0.5620, Poly1 0.4296 / 0.5689. BEIR zero-shot avg nDCG@10 (RankT5-Enc): PointCE 0.5024 vs Softmax 0.5241. Authors: no consistent winner between Enc and EncDec; listwise softmax beats pointwise at every size; training list size must be >=20-30 for softmax to beat pointwise.
**Contribution.** Evidence that an encoder-only head with a listwise softmax over candidates equals a generative head, and that the listwise loss is what generalises out-of-domain.

#### H11. monoT5 - relevance from true/false token logits

**Citation.** Nogueira, R., Jiang, Z., Pradeep, R., & Lin, J. (2020). Document ranking with a pretrained sequence-to-sequence model. *Findings of EMNLP 2020*, 708-718. https://aclanthology.org/2020.findings-emnlp.63/ (arXiv: https://arxiv.org/abs/2003.06713)
**Grade.** `PR`.
**Design.** Prompt "Query: q Document: d Relevant:", score = softmax over logits of "true" vs "false" - the ancestor of the Qwen3-Reranker yes/no head (B7).
**Contribution.** Primary source for reading a decision off a generative LM's target-token logits; RankT5 (H10) shows +1.8 MRR@10 from replacing it with a direct score head.

#### H12. RankGPT - listwise permutation generation, and distillation into a 435M encoder

**Citation.** Sun, W., Yan, L., Ma, X., Wang, S., Ren, P., Chen, Z., Yin, D., & Ren, Z. (2023). Is ChatGPT good at search? Investigating large language models as re-ranking agents. *Proceedings of EMNLP 2023*. https://arxiv.org/abs/2304.09542
**Grade.** `PR`.
**Numbers (nDCG@10, DL19 / DL20).** GPT-4 listwise 75.59 / 70.56; gpt-3.5-turbo 65.80 / 62.91; monoT5-3B 71.83 / 68.89; monoBERT 70.50 / 67.28. BEIR avg: GPT-4 53.68; gpt-3.5 49.37; monoT5-3B 51.36. Permutation-distilled DeBERTa-v3-large (435M): DL19 70.66, DL20 67.15, BEIR 53.03 - above monoT5-3B on BEIR. Sliding window 20, step 10.
**Contribution.** Shows the listwise signal from a large decoder can be compressed into a 435M encoder that beats a 3B pointwise model.

#### H13. FIRST - rank from the first-token logits over candidate identifiers

**Citation.** Reddy, R. G., Doo, J., Xu, Y., Sultan, M. A., Swain, D., Sil, A., & Ji, H. (2024). FIRST: Faster improved listwise reranking with single token decoding. *Proceedings of EMNLP 2024*. https://aclanthology.org/2024.emnlp-main.491/ (arXiv: https://arxiv.org/abs/2406.15657)
**Grade.** `PR`.
**Design.** Zephyr-7B listwise reranker; instead of generating a permutation, read the logits over candidate identifiers at the first generated position and sort; train with weighted RankNet (weight 1/(i+j)) + LM loss (lambda = 10).
**Numbers.** Reported nDCG@10 averages (as extracted from the main table; these values are far above typical BEIR averages, so confirm the exact evaluation subset before reuse): FIRST 78.8, RankZephyr 78.4, RankVicuna 71.3, cross-encoder 71.0. Latency per 20-passage window: 1.2 s generation vs 0.6 s FIRST (50% reduction).
**Limitations.** 7B; numbers above flagged.
**Contribution.** The one-forward-pass "option-identifier logits as scores" head, with a learning-to-rank loss that makes those logits calibrated for ranking rather than for generation.

### 4c. Option-ID / position bias and calibration in multiple choice

#### H14. LLMs are not robust multiple choice selectors (PriDe)

**Citation.** Zheng, C., Zhou, H., Meng, F., Zhou, J., & Huang, M. (2024). Large language models are not robust multiple choice selectors. *ICLR 2024 (spotlight)*. https://arxiv.org/abs/2309.03882
**Grade.** `PR`.
**Numbers.** 20 LLMs; MMLU (4-way), ARC-Challenge (4-way), CommonsenseQA (5-way). Selection-bias metric RStd (std of per-option recall), 0-shot: llama-30B MMLU 8.5; gpt-3.5-turbo MMLU 5.5, ARC 3.3. PriDe (prior estimated on 5% of test samples by permuting option contents, then divided out): +2.6 MMLU, +2.9 ARC, +4.0 CSQA average accuracy across 20 models; cost x1.15 vs x4 for full cyclic permutation; 40% estimation x2.2. Removing option IDs reduces bias markedly; shuffling IDs barely changes it -> the bias is **token bias** on the ID symbols, not position bias.
**Contribution.** Establishes that the letter-logit readout head carries an ID-token prior that must be estimated and divided out; gives a label-free, cheap debiasing recipe.

#### H15. Multiple choice symbol binding (MCSB) - small models cannot bind letters to options

**Citation.** Robinson, J., Rytting, C. M., & Wingate, D. (2023). Leveraging large language models for multiple choice question answering. *ICLR 2023*. https://arxiv.org/abs/2210.12353
**Grade.** `PR`.
**Numbers.** Proportion of plurality agreement (PPA) on OpenBookQA under option permutation (random = 25%): Codex ~75%, Instruct-davinci ~77%, GPT-3 davinci ~50%, Jurassic-1 Jumbo ~40%, Instruct-Curie ~27%, GPT-2 ~26%, CodeParrot ~25%. With a high-MCSB model (Codex), multiple-choice prompting beats cloze scoring by +8.3 / +12.2 / +9.7 points (0 / 1 / few-shot) averaged over 20 datasets; up to +32.5 / +37.8 / +44.3 on CosmosQA.
**Contribution.** Direct evidence that sub-billion decoders (GPT-2 scale, Curie 6.7B even) sit at chance on symbol binding, so "read option-letter logits off a stock small LM" is not a viable zero-shot head at <=0.6B; cloze/likelihood scoring is the fallback.

#### H16. Sensitivity to option order

**Citation.** Pezeshkpour, P., & Hruschka, E. (2024). Large language models sensitivity to the order of options in multiple-choice questions. *Findings of NAACL 2024*, 2006-2017. https://aclanthology.org/2024.findings-naacl.130/ (arXiv: https://arxiv.org/abs/2308.11483)
**Grade.** `PR`.
**Numbers.** Reordering options moves accuracy by "approximately 13% to 75%" (arXiv abstract) / "13% to 85%" (Anthology abstract) across models and benchmarks; calibration gives up to +8 points; placing the top-two candidates first-and-last amplifies bias, adjacent placement reduces it.
**Contribution.** Magnitude of positional fragility in MC readout; complements H14 (which attributes most of it to ID tokens).

#### H17. Answer, Assemble, Ace - where symbol binding lives, including Qwen2.5-0.5B

**Citation.** Wiegreffe, S., Tafjord, O., Belinkov, Y., Hajishirzi, H., & Sabharwal, A. (2025). Answer, assemble, ace: Understanding how LMs answer multiple choice questions. *ICLR 2025 (spotlight)*. https://arxiv.org/abs/2407.15018
**Grade.** `PR`.
**Numbers / findings.** Models: Llama 3.1 8B, OLMo 0724 7B, Qwen2.5 0.5B and 1.5B (base + instruct). Answer-symbol promotion is causally attributed to a few middle layers (layer 24 of OLMo-7B-Instruct for A/B/C/D; layer 29 for unusual symbols Q/Z/R/X); self-attention dominates over MLPs; 1-4 heads per layer (of 32) carry the signal. On a synthetic task, OLMo-7B goes from near-random to near-100% between 80k and 100k pretraining steps; poorly performing checkpoints cannot separate answer symbols in vocabulary space.
**Contribution.** Mechanistic support for H15: symbol binding is a late-emerging, sparse-head capability; explains why small/early models fail letter-format MC and why MMLU is reported as cloze for 135M-360M models (B12).

#### H18. Surface form competition (PMI scoring for likelihood-based MC)

**Citation.** Holtzman, A., West, P., Shwartz, V., Choi, Y., & Zettlemoyer, L. (2021). Surface form competition: Why the highest probability answer isn't always right. *Proceedings of EMNLP 2021*, 7038-7051. https://aclanthology.org/2021.emnlp-main.564/ (arXiv: https://arxiv.org/abs/2104.08315)
**Grade.** `PR`.
**Design.** Domain-conditional PMI: divide option likelihood by its likelihood under a task-generic prompt; "consistent gains" over raw and calibrated likelihood on all GPT-2 and GPT-3 sizes across MC datasets.
**Contribution.** The standard fix when the head is "score each option string's likelihood under a stock LM" (the cloze fallback for models without MCSB).

#### H19. Calibrate before use (contextual calibration)

**Citation.** Zhao, T. Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). Calibrate before use: Improving few-shot performance of language models. *Proceedings of ICML 2021*. https://arxiv.org/abs/2102.09690
**Grade.** `PR`.
**Design / numbers.** Feed a content-free input ("N/A") to estimate the label prior, fit an affine correction so it becomes uniform; up to +30.0 absolute accuracy on GPT-3/GPT-2 and reduced variance across prompt formats.
**Contribution.** The generic prior-removal step that PriDe (H14) specialises to option IDs.

### 4d. Decision heads over discrete action sets (RL / behavioural cloning)

#### H20. DRRN - Q(s, a) from separate state and action encoders plus an interaction function

**Citation.** He, J., Chen, J., He, X., Gao, J., Li, L., Deng, L., & Ostendorf, M. (2016). Deep reinforcement learning with a natural language action space. *Proceedings of ACL 2016*. https://arxiv.org/abs/1511.04636
**Grade.** `PR`.
**Design.** State text and each candidate action text are embedded separately; Q-value = interaction (inner product) of the two; softmax/argmax over the candidate set; generalises to paraphrased actions.
**Contribution.** The canonical "score a variable set of text actions against a text state" head; the late-interaction alternative to a cross-encoder.

#### H21. Wolpertinger - two-stage shortlist for up to 1M discrete actions

**Citation.** Dulac-Arnold, G., Evans, R., van Hasselt, H., Sunehag, P., Lillicrap, T., Hunt, J., Mann, T., Weber, T., Degris, T., & Coppin, B. (2015). Deep reinforcement learning in large discrete action spaces. *arXiv*. https://arxiv.org/abs/1512.07679
**Grade.** `PP`.
**Design.** Actor emits a proto-action in embedding space; k-nearest-neighbour lookup shortlists k real actions (logarithmic time); critic re-scores the shortlist. Demonstrated up to one million actions.
**Contribution.** Primary source for the retrieve-then-score pattern for high-cardinality option sets.

#### H22. CALM - LM generates candidates, RL re-ranker scores them

**Citation.** Yao, S., Rao, R., Hausknecht, M., & Narasimhan, K. (2020). Keep CALM and explore: Language models for action generation in text-based games. *Proceedings of EMNLP 2020*. https://arxiv.org/abs/2010.02903
**Grade.** `PR`.
**Numbers.** A fine-tuned LM proposes admissible actions; a DRRN-style re-ranker scores them; +69% relative average game score over prior SOTA on Jericho; competitive with agents given ground-truth admissible actions on half the games.
**Contribution.** Two-stage generate-then-score with a small scorer, the text-game analogue of shortlist + typed decision.

#### H23. SayCan - LM likelihood x affordance over a fixed skill set

**Citation.** Ahn, M., Brohan, A., Brown, N., et al. (2022). Do as I can, not as I say: Grounding language in robotic affordances. *arXiv*. https://arxiv.org/abs/2204.01691 (project page: https://say-can.github.io/)
**Grade.** `PP` (later CoRL 2022; not verified here).
**Design / numbers.** For each skill in a fixed set, score = p_LM(skill text | instruction, history) x value-function affordance; pick argmax; 101 instructions, PaLM-SayCan plan success 84%, execution 74%; PaLM 540B roughly halves errors vs FLAN.
**Contribution.** The "read option likelihoods off a stock LM, then multiply by a learned feasibility prior" decision head, with evidence that scorer quality tracks LM scale.

---

## 5. Numeric evidence table

Metric conventions: GLUE/MNLI = accuracy (avg for GLUE); nDCG@10 and MRR@10 as stated; F1 macro for zero-shot classification; throughput units as stated per row. "Enc"/"Dec" = architecture. Sources reference the entry IDs above.

| Model | Params | Arch | Task / benchmark | Metric | Value | Source |
|---|---|---|---|---|---|---|
| Ettin encoder | 17M | Enc | GLUE avg / MNLI | acc | 79.2 / 79.5 | B1 |
| Ettin encoder | 32M | Enc | GLUE avg / MNLI | acc | 83.5 / 83.4 | B1 |
| Ettin encoder | 68M | Enc | GLUE avg / MNLI | acc | 87.2 / 87.0 | B1 |
| Ettin encoder | 150M | Enc | GLUE avg / MNLI | acc | 88.9 / 89.2 | B1 |
| Ettin encoder | 400M | Enc | GLUE avg / MNLI | acc | 90.8 / 91.3 | B1 |
| Ettin encoder | 1B | Enc | GLUE avg / MNLI | acc | 91.6 / 91.8 | B1 |
| Ettin enc vs dec | 17M | Enc / Dec | MS MARCO dev dense retrieval | nDCG@10 | 30.93 / 29.11 | B1 |
| Ettin enc vs dec | 68M | Enc / Dec | MS MARCO dev dense retrieval | nDCG@10 | 38.17 / 36.12 | B1 |
| Ettin enc vs dec | 150M | Enc / Dec | MS MARCO dev dense retrieval | nDCG@10 | 39.97 / 37.71 | B1 |
| Ettin enc vs dec | 400M | Enc / Dec | MS MARCO dev dense retrieval | nDCG@10 | 42.24 / 39.93 | B1 |
| Ettin enc vs dec | 1B | Enc / Dec | MS MARCO dev dense retrieval | nDCG@10 | 43.35 / 41.70 | B1 |
| Ettin native enc vs dec-adapted-to-enc | 150M | Enc / Dec->Enc | MNLI | acc | 89.2 / 85.8 | B1 |
| CLM vs MLM (same arch) | 210M | Dec-obj / Enc-obj | Seq. classification avg (SST-2, MNLI, QQP) | acc | 82.80 / 84.89 | B2 |
| CLM vs MLM (same arch) | 610M | Dec-obj / Enc-obj | Seq. classification avg | acc | 83.58 / 87.00 | B2 |
| CLM vs MLM (same arch) | 1B | Dec-obj / Enc-obj | Seq. classification avg | acc | 82.68 / 88.23 | B2 |
| CLM vs MLM (same arch) | 610M | Dec-obj / Enc-obj | IR avg (MS MARCO, NQ, MLDR) | score | 76.20 / 79.55 | B2 |
| CLM vs MLM (same arch) | 610M | Dec-obj / Enc-obj | QA avg | score | 42.09 / 62.77 | B2 |
| ModernBERT-base | 149M | Enc | GLUE avg | acc | 88.4 | B3 |
| ModernBERT-large | 395M | Enc | GLUE avg | acc | 90.4 | B3 |
| DeBERTa-v3-base | 184M (86M bb) | Enc | GLUE avg (ModernBERT harness) / MNLI-m/mm | acc | 88.1 / 90.6 / 90.7 | B3, B9 |
| DeBERTa-v3-large | 435M (304M bb) | Enc | GLUE avg (ModernBERT harness) / MNLI-m/mm | acc | 91.4 / 91.8 / 91.9 | B3, B9 |
| RoBERTa-base / large | 125M / 355M | Enc | GLUE avg | acc | 86.4 / 88.9 | B3 |
| BERT-base / large | 110M / 340M | Enc | GLUE avg | acc | 84.7 / 85.2 | B3 |
| ModernBERT-base | 149M | Enc | Throughput, 8192-len variable, RTX 4090 | k tokens/s | 123.7 (GTE-en-MLM 47.5) | B3 |
| ModernBERT-base / large | 149M / 395M | Enc | BEIR ColBERT | nDCG@10 | 51.3 / 52.4 | B3 |
| mmBERT-small | 140M (42M non-emb) | Enc | GLUE avg | acc | 84.7 | B10 |
| mmBERT-base | 307M (110M non-emb) | Enc | GLUE avg / XTREME / MTEB-v2-eng | score | 86.3 / 72.8 / 53.9 | B10 |
| XLM-R-base | 278M | Enc | GLUE avg / XTREME | score | 83.3 / 70.4 | B10 |
| EuroBERT-210m | 210M | Enc | GLUE avg (mmBERT harness) | acc | 81.2 | B10 |
| EuroBERT-210m / 610m | 210M / 610M | Enc | XNLI | acc | 81.93 / 84.11 | B11 |
| EuroBERT-210m / 610m | 210M / 610M | Enc | MIRACL | nDCG@10 | 90.82 / 92.62 | B11 |
| DeBERTa-v3-large (fine-tuned) | 435M | Enc | ScandEval English mean rank | rank (lower better) | 1.09 (GPT-4-0613 few-shot 1.44) | B4 |
| RoBERTa-large (fine-tuned) | 355M | Enc | Groundedness, SQuAD-v2 / NewsQA | acc | 90.2 / 88.5 | B5 |
| Llama-3.1-8B-Instruct zero-shot / fine-tuned | 8B | Dec | Groundedness, SQuAD-v2 | acc | 81.9 / 91.1 | B5 |
| RoBERTa-large vs Llama-3.1-8B | 355M / 8B | Enc / Dec | Inference FLOPs | FLOPs | 3.7e11 / 1.6e13 | B5 |
| ModernBERT-large specialists | 400M | Enc | RewardBench overall / reasoning | acc | 86.4 / 91.2 | B6 |
| ModernBERT-base specialists | 150M | Enc | RewardBench overall / reasoning | acc | 78.4 / 83.3 | B6 |
| Llama3-SteerLM-RM | 70B | Dec | RewardBench overall / reasoning | acc | 91.0 / 90.6 | B6 |
| Qwen3-Reranker-0.6B | 0.6B | Dec (yes/no logits) | MTEB-R / MMTEB-R / MLDR / Code | nDCG@10 | 65.80 / 66.36 / 67.28 / 73.42 | B7 |
| Qwen3-Reranker-4B | 4B | Dec | MTEB-R / MMTEB-R / MLDR / Code | nDCG@10 | 69.76 / 72.74 / 69.97 / 81.20 | B7 |
| bge-reranker-v2-m3 | 568M | Enc | MTEB-R / CMTEB-R / MMTEB-R / MLDR | nDCG@10 | 57.03 / 72.16 / 58.36 / 59.51 | B7 |
| gte-multilingual-reranker-base | 0.3B | Enc | MTEB-R / CMTEB-R / MLDR | nDCG@10 | 59.51 / 74.08 / 66.33 | B7 |
| Ettin reranker | 17M | Enc | MTEB(eng,v2) retrieval rerank | nDCG@10 | 0.5576 | B8 |
| Ettin reranker | 32M | Enc | MTEB(eng,v2) retrieval rerank | nDCG@10 | 0.5779 | B8 |
| Ettin reranker | 68M | Enc | MTEB(eng,v2) retrieval rerank | nDCG@10 | 0.5915 | B8 |
| Ettin reranker | 150M | Enc | MTEB(eng,v2) retrieval rerank | nDCG@10 | 0.5994 | B8 |
| Ettin reranker | 400M | Enc | MTEB(eng,v2) retrieval rerank | nDCG@10 | 0.6091 | B8 |
| Ettin reranker | 1B | Enc | MTEB(eng,v2) retrieval rerank | nDCG@10 | 0.6114 | B8 |
| Qwen3-Reranker-0.6B | 596M | Dec | MTEB(eng,v2) retrieval rerank (Ettin harness) | nDCG@10 | 0.5940 | B8 |
| bge-reranker-v2-m3 | 568M | Enc | MTEB(eng,v2) retrieval rerank (Ettin harness) | nDCG@10 | ~0.553 (32M - 0.025) | B8 |
| mxbai-rerank-large-v2 | 1.54B | Dec | MTEB(eng,v2) retrieval rerank (Ettin harness) | nDCG@10 | ~0.6115 | B8 |
| Ettin reranker | 17M / 150M / 1B | Enc | Throughput H100 bf16 FA2 | pairs/s | 7,517 / 3,237 / 928 | B8 |
| Ettin reranker | 17M / 1B | Enc | Throughput RTX 3090 | pairs/s | 9,008 / 189 | B8 |
| Ettin reranker | 17M / 1B | Enc | Throughput CPU i7-13700K | pairs/s | 267.4 / 2.1 | B8 |
| mxbai-rerank-large-v2 (teacher) | 1.54B | Dec | Throughput H100 | pairs/s | 387 | B8 |
| ms-marco-MiniLM-L6-v2 | ~22M | Enc | TREC DL19 / MS MARCO dev / throughput | nDCG@10 / MRR@10 / docs/s | 74.30 / 39.01 / 1,800 | B15 |
| ms-marco-MiniLM-L12-v2 | 33M | Enc | TREC DL19 / MS MARCO dev / throughput | nDCG@10 / MRR@10 / docs/s | 74.31 / 39.02 / 960 | B15 |
| ms-marco-TinyBERT-L2-v2 | ~4M | Enc | TREC DL19 / MS MARCO dev / throughput | nDCG@10 / MRR@10 / docs/s | 69.84 / 32.56 / 9,000 | B15 |
| ms-marco-electra-base | 110M | Enc | TREC DL19 / MS MARCO dev / throughput | nDCG@10 / MRR@10 / docs/s | 71.99 / 36.41 / 340 | B15 |
| mxbai-rerank-base-v2 | 0.5B (Qwen2.5) | Dec | BEIR / latency A100 | nDCG@10 / s per query | 55.57 / 0.67 | B16 |
| mxbai-rerank-large-v2 | 1.5B | Dec | BEIR / latency A100 | nDCG@10 / s per query | 57.49 / 0.89 | B16 |
| bge-reranker-v2-m3 | 568M | Enc | BEIR (mxbai harness) | nDCG@10 | 53.94 | B16 |
| bge-reranker-v2-gemma | 2.5B | Dec | BEIR / latency A100 | nDCG@10 / s per query | 55.38 / 7.20 | B16 |
| SmolLM2-360M | 360M | Dec | HellaSwag / ARC / MMLU-cloze / CSQA | acc | 54.5 / 53.0 / 35.8 / 38.0 | B12 |
| Qwen2.5-0.5B | 0.5B | Dec | HellaSwag / ARC / MMLU-cloze / CSQA (SmolLM2 harness) | acc | 51.2 / 45.4 / 33.7 / 31.6 | B12 |
| SmolLM2-135M | 135M | Dec | HellaSwag / ARC / MMLU-cloze / CSQA | acc | 42.1 / 43.9 / 31.5 / 33.9 | B12 |
| Qwen3-0.6B-Base | 0.6B | Dec | MMLU / MMLU-Redux / BBH | acc | 52.81 / 51.26 / 41.47 | B13 |
| Qwen2.5-0.5B-Base | 0.5B | Dec | MMLU / MMLU-Redux / BBH (Qwen3 harness) | acc | 47.50 / 45.10 / 20.30 | B13 |
| Qwen3-1.7B-Base | 1.7B | Dec | MMLU / BBH | acc | 62.63 / 54.47 | B13 |
| Gemma 3 270M PT | 270M (100M non-emb) | Dec | HellaSwag 10-shot / ARC-c 25-shot / PIQA / WinoGrande | acc | 40.9 / 29.0 / 67.7 / 52.0 | B14 |
| Gemma 3 270M IT | 270M | Dec | IFEval / BBH | acc | 51.2 / 26.7 | B14 |
| BERT-large (MC head) | 340M | Enc | SWAG test | acc | 86.3 (OpenAI GPT 78.0) | H1 |
| BERT-base (MC head) | 110M | Enc | SWAG dev | acc | 81.6 | H1 |
| PET / iPET (ALBERT-xxlarge-v2) | 223M | Enc ([MASK] head) | SuperGLUE test avg, 32 examples | score | 74.0 / 75.4 | H2 |
| GPT-3 175B / Large 760M / Medium 350M | - | Dec | SuperGLUE test avg (175B) / dev avg (760M, 350M), 32 examples | score | 71.8 / 56.8 / 56.2 | H2 |
| UniMC (ALBERT-xxlarge-v2) | 235M | Enc ([O-MASK] per option) | ANLI R1 / R2 / R3 / CB zero-shot | acc | 52.0 / 44.4 / 47.8 / 75.7 | H3 |
| PaLM / FLAN / T0 | 540B / 137B / 11B | Dec | ANLI R1 zero-shot | acc | 48.4 / 47.7 / 43.6 | H3 |
| GLiNER-S / M / L | 50M / 90M / 0.3B | Enc (label markers) | OOD NER zero-shot avg | F1 | 52.7 / 55.4 / 60.9 | H4 |
| UniNER-7B / 13B, ChatGPT | 7B / 13B / - | Dec | OOD NER zero-shot avg | F1 | 53.7 / 55.6 / 47.5 | H4 |
| gliclass-large-v3.0 | 439M | Enc (label tokens, one pass) | Zero-shot classification avg | F1 | 0.7193 | H5 |
| gliclass-base-v3.0 | 187M | Enc | Zero-shot classification avg | F1 | 0.6764 | H5 |
| deberta-v3-large-zeroshot-v2.0 | 0.4B | Enc (NLI, one pass per label) | Zero-shot classification avg (GLiClass suite) / 28-task mean | F1 | 0.6821 / 0.676 | H5, H6 |
| gliclass edge / base / large | 32.7M / 187M / 439M | Enc | Throughput A6000, 1 -> 128 labels | ex/s | 103.81->82.64 / 49.42->45.94 / 19.05->17.60 | H5 |
| deberta-v3-base-zeroshot-v2.0 | 184M | Enc (per-label pass) | Throughput A6000, 1 -> 128 labels | ex/s | 24.55 -> 0.47 | H5 |
| Set-Encoder | 330M | Enc (set attention) | TREC DL19 / DL20, ColBERTv2 first stage | nDCG@10 | 0.765 / 0.799 | H8 |
| monoELECTRA | 330M | Enc (pointwise) | TREC DL19 / DL20, ColBERTv2 first stage | nDCG@10 | 0.768 / 0.770 | H8 |
| RankGPT-4o / RankZephyr-7B | - / 7B | Dec (listwise) | TREC DL19, ColBERTv2 first stage | nDCG@10 | 0.784 / 0.749 | H8 |
| Set-Encoder / RankZephyr / RankGPT-4o | 330M / 7B / - | Enc / Dec / Dec | Latency, 100 passages | s | 0.219 / 24.047 / 18.773 | H8 |
| monoELECTRA-base distilled | 110M | Enc | TREC DL19 / DL20 | nDCG@10 | 0.720 / 0.711 | H9 |
| monoELECTRA-large distilled | 330M | Enc | TREC DL19 / DL20 | nDCG@10 | 0.733 / 0.727 | H9 |
| RankZephyr / RankGPT-4 (teachers) | 7B / - | Dec | TREC DL19 / DL20 | nDCG@10 | 0.719 / 0.720 ; 0.713 / 0.713 | H9 |
| RankT5-Enc, Softmax vs PointCE | T5-Large enc | Enc (score head) | MS MARCO dev / BEIR zero-shot avg | MRR@10 / nDCG@10 | 0.4305 vs 0.4216 / 0.5241 vs 0.5024 | H10 |
| RankT5-EncDec Softmax vs monoT5 | T5-Large | EncDec | MS MARCO dev | MRR@10 | 0.4278 vs 0.4156 | H10 |
| DeBERTa-v3-large distilled from ChatGPT permutations | 435M | Enc | BEIR avg / DL19 / DL20 | nDCG@10 | 53.03 / 70.66 / 67.15 | H12 |
| monoT5-3B / GPT-4 listwise | 3B / - | EncDec / Dec | BEIR avg | nDCG@10 | 51.36 / 53.68 | H12 |
| FIRST vs RankZephyr | 7B | Dec (first-token logits) | Latency per 20-passage window | s | 0.6 vs 1.2 | H13 |
| PriDe (5% estimation) | 20 LLMs | Dec (letter logits) | MMLU / ARC / CSQA accuracy gain | pts | +2.6 / +2.9 / +4.0 | H14 |
| llama-30B / gpt-3.5-turbo | 30B / - | Dec | MMLU selection bias | RStd | 8.5 / 5.5 | H14 |
| GPT-2 / CodeParrot / Instruct-Curie / GPT-3 davinci / Codex | 1.5B / 1.5B / 6.7B / 175B / 175B | Dec | OpenBookQA symbol binding (random 25%) | PPA % | ~26 / ~25 / ~27 / ~50 / ~75 | H15 |
| Codex, MCP vs cloze | 175B | Dec | 20-dataset avg, 0 / 1 / few-shot | pts gain | +8.3 / +12.2 / +9.7 | H15 |
| Various LLMs | - | Dec | Accuracy swing under option reordering | range | 13%-75% (arXiv) / 13%-85% (Anthology) | H16 |
| PaLM-SayCan | 540B LM | Dec (likelihood x affordance) | 101 instructions, plan / execution | success | 84% / 74% | H23 |
| CALM + DRRN re-ranker | GPT-2 + small scorer | Dec gen + scorer | Jericho avg score vs prior SOTA | relative | +69% | H22 |

---

## 6. Gaps found in the literature (recorded, not interpreted)

- No paper trains a small encoder and a small decoder **as cross-encoder rerankers or MC scorers** on the same data and reports both accuracy and latency. Ettin (B1) matches training but compares dense bi-encoders and generative eval; the Ettin reranker blog (B8) is encoder-only; Qwen3-Reranker vs bge/gte (B7) and mxbai vs bge (B16) compare differently trained checkpoints.
- No source reports fine-tuned decoders with a classification head at 135M-600M on GLUE/MC alongside encoders; the nearest is B2 (same architecture, CLM vs MLM objective) and B5 (1B-8B decoders fine-tuned).
- No accuracy-per-millisecond table exists across architectures on one hardware setup. Throughput figures are scattered over RTX 4090 tokens/s (B3), H100/3090/CPU pairs/s (B8), A6000 ex/s (H5), A100 s/query (B16), unstated-hardware docs/s (B15), and FLOPs (B5).
- Label-count scaling of one-pass heads is measured only by GLiClass (H5), and only up to 128 labels.
- Option-ID bias and symbol binding are measured on >=1.5B decoders (H14-H17); the only sub-1B data points are Qwen2.5-0.5B/1.5B in H17 (mechanistic, no accuracy table extracted) and the near-chance PPA of GPT-2-scale models in H15.
