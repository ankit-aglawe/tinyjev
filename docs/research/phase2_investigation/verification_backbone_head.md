# Verification report: lit_backbone_head.md (Phase 2, stream A: backbone and head)

Verified 2026-09-22 against the live sources. Input: `lit_backbone_head.md` (40 entries B1-B17, H1-H23; ~100-row numeric table). This report grades sources and checks numbers; it does not synthesise.

**Method.** All 36 arXiv ids were resolved in one arXiv-API query (title, author list, dates, `comments`, DOI). Venues were confirmed independently of arXiv: ACL Anthology pages (B3, B4, H1, H4, H11, H13, H16, H18), Crossref (H10), Springer DOIs (H8, H9), OpenReview / ICLR proceedings / ML Anthology (B1, B2, B9, H14, H15, H17), PMLR (H23), arXiv `comments` only where nothing else exists (B6, H2, H3, H6, H7, H12, H19, H20, H22). Numbers were re-read from arXiv HTML (text-stripped and grepped), from PDFs via `pdftotext` (B9, H1, H2, H3, H10, H12, H15), and from the model cards, docs and blogs cited. Retrieved content was treated as data. Gray-zone existence = FAIL (none occurred).

---

## (a) Source quality matrix

Tier 1 = peer-reviewed, venue confirmed. Tier 2 = preprint (or workshop, light review, marked W). Tier 3 = model card / official docs / leaderboard. Tier 4 = vendor or community blog. COI = source reports its own model or product.

| ID | Source | Existence | Venue in bibliography -> confirmed | Tier | COI | Notes |
|---|---|---|---|---|---|---|
| B1 | Weller et al., Seq vs Seq / Ettin, 2507.11412 | PASS | ICLR 2026 -> confirmed (arXiv comment "Accepted to ICLR'26"; ML Anthology; OpenReview PDF) | 1 | own models (JHU/LightOn) | v2 2026-03-12 |
| B2 | Gisserot-Boukhlef et al., 2507.00994 | PASS | arXiv `PP` -> **actually ICLR 2026** (proceedings.iclr.cc, ML Anthology). Grade must be raised to `PR` | 1 | own models | v4 2026-05-05 |
| B3 | Warner et al., ModernBERT, 2412.13663 | PASS | ACL 2025 Long, pp. 2526-2547 -> confirmed (2025.acl-long.127) | 1 | own model (Answer.AI/LightOn) | |
| B4 | Nielsen et al., Encoder vs Decoder, 2406.13469 | PASS | NoDaLiDa/Baltic-HLT 2025, pp. 561-572 -> confirmed (2025.nodalida-1.60) | 1 | ScandEval maintainers evaluate their own benchmark | |
| B5 | Abbes et al., 2506.21288 | PASS | arXiv -> preprint, no venue found | 2 | none material | v1 only |
| B6 | Pan, Tiny Reward Models, 2507.09973 | PASS | ICML 2025 ES-FoMo workshop -> confirmed (arXiv comment "2025 ICML Efficient Systems for Foundation Models Workshop") | 2-W | author affiliated with answer.ai (ModernBERT vendor) | single author |
| B7 | Zhang et al., Qwen3 Embedding, 2506.05176 + card | PASS | arXiv -> preprint | 2 (+3 card) | **yes: Qwen team scoring Qwen3-Reranker** | baselines are third-party checkpoints |
| B8 | Aarsen, Ettin reranker blog (HF) | PASS | blog 2026-05-19 -> confirmed | 4 | **yes: author trained and released the models; own MTEB runs and own throughput harness** | Single author (Tom Aarsen). Bibliography's "Aarsen, T., et al. (HF/LightOn/JHU authors)" is wrong: the blog is one HF author |
| B9 | He et al., DeBERTaV3, 2111.09543 + cards | PASS | ICLR 2023 -> confirmed (arXiv comment; iclr.cc poster) | 1 (+3 cards) | own model (Microsoft) | |
| B10 | Marone et al., mmBERT, 2509.06888 | PASS | arXiv -> preprint | 2 | own model | |
| B11 | Boizard et al., EuroBERT, 2503.05500 | PASS | arXiv -> preprint | 2 | own model | v3 2026-06-01 |
| B12 | Ben Allal et al., SmolLM2, 2502.02737 + cards | PASS | arXiv -> preprint | 2 (+3 cards) | **yes: HF cards scoring HF models vs Qwen** | paper confirmed to defer 135M/360M numbers to cards |
| B13 | Qwen Team, Qwen3 Technical Report, 2505.09388 | PASS | arXiv -> preprint | 2 | **yes** | |
| B14 | Google Developers Blog + gemma-3-270m card | PASS | blog 2025-08-14 -> confirmed | 4 (+3 card) | **yes: Google** | |
| B15 | Sentence-Transformers docs, MS MARCO cross-encoders | PASS | docs -> confirmed | 3 | yes (own models); hardware for docs/s not stated | |
| B16 | Mixedbread blog, mxbai-rerank-v2 | PASS | blog 2025-03-13 -> confirmed (Lee, Huang, Shakir + 1) | 4 | **yes: vendor** | no candidate count for latency |
| B17 | Li et al., Label Supervised LLaMA, 2310.01208 | PASS | arXiv -> preprint | 2 | none | 7B only |
| H1 | Devlin et al., BERT, 1810.04805 | PASS | NAACL 2019, pp. 4171-4186 -> confirmed (N19-1423) | 1 | own model | |
| H2 | Schick & Schütze, PET/iPET, 2009.07118 | PASS | NAACL 2021 -> confirmed (arXiv comment) | 1 | none | |
| H3 | Yang et al., UniMC, 2210.08590 | PASS | EMNLP 2022 -> confirmed (2022.emnlp-main.474, pp. 7042-7055) | 1 | own model | |
| H4 | Zaratiana et al., GLiNER, 2311.08526 | PASS | NAACL 2024, pp. 5364-5376 -> confirmed (2024.naacl-long.300); arXiv comment still says "Work in progress" | 1 | own model | |
| H5 | Stepanov et al., GLiClass, 2508.07662 | PASS | arXiv -> preprint | 2 | **yes: Knowledgator evaluating its own product on its own suite** | 6 authors, all Knowledgator |
| H6 | Yin et al., 1909.00161 + Laurer card | PASS | EMNLP-IJCNLP 2019 -> confirmed (arXiv comment "EMNLP2019 camera-ready") | 1 (+3 card) | card: **yes (author scoring own models)** | |
| H7 | Lee et al., Set Transformer, 1810.00825 | PASS | ICML 2019 -> confirmed (arXiv comment) | 1 | none | no NLP numbers |
| H8 | Schlatt et al., Set-Encoder, 2404.06912 | PASS | ECIR 2025 -> confirmed (DOI 10.1007/978-3-031-88711-6_1) | 1 | own model | v5 |
| H9 | Schlatt et al., Rank-DistiLLM, 2405.07920 | PASS | ECIR 2025 -> confirmed (DOI 10.1007/978-3-031-88714-7_31) | 1 | own dataset | v4 |
| H10 | Zhuang et al., RankT5, 2210.10634 | PASS | SIGIR 2023, pp. 2308-2313 -> confirmed (Crossref, DOI 10.1145/3539618.3592047) | 1 | own model (Google) | |
| H11 | Nogueira et al., monoT5, 2003.06713 | PASS | Findings EMNLP 2020, pp. 708-718 -> confirmed | 1 | own model | |
| H12 | Sun et al., RankGPT, 2304.09542 | PASS | EMNLP 2023 -> confirmed (2023.emnlp-main.923; Outstanding Paper) | 1 | none | |
| H13 | Reddy et al., FIRST, 2406.15657 | PASS | EMNLP 2024 -> confirmed (2024.emnlp-main.491, pp. 8642-8652); arXiv comment says "Preprint" | 1 | none | |
| H14 | Zheng et al., PriDe, 2309.03882 | PASS | ICLR 2024 spotlight -> confirmed (arXiv comment; OpenReview shr9PXz7T0) | 1 | none | |
| H15 | Robinson et al., MCSB, 2210.12353 | PASS | ICLR 2023 -> confirmed (arXiv comment; iclr.cc poster 10737) | 1 | none | |
| H16 | Pezeshkpour & Hruschka, 2308.11483 | PASS | Findings NAACL 2024, pp. 2006-2017 -> confirmed | 1 | none | |
| H17 | Wiegreffe et al., 2407.15018 | PASS | ICLR 2025 spotlight -> confirmed (arXiv comment; OpenReview 6NNA0MxhCH; ICLR proceedings) | 1 | none | |
| H18 | Holtzman et al., Surface form competition, 2104.08315 | PASS | EMNLP 2021, pp. 7038-7051 -> confirmed | 1 | none | |
| H19 | Zhao et al., Calibrate before use, 2102.09690 | PASS | ICML 2021 -> confirmed (arXiv comment) | 1 | none | |
| H20 | He et al., DRRN, 1511.04636 | PASS | ACL 2016 -> confirmed (arXiv comment) | 1 | none | |
| H21 | Dulac-Arnold et al., Wolpertinger, 1512.07679 | PASS | arXiv -> preprint | 2 | none | |
| H22 | Yao et al., CALM, 2010.02903 | PASS | EMNLP 2020 -> confirmed (arXiv comment) | 1 | none | |
| H23 | Ahn et al., SayCan, 2204.01691 | PASS | "`PP`, later CoRL 2022 not verified" -> **CoRL 2022 confirmed** (PMLR v205, `ichter23a`). Note the proceedings list Brian Ichter as first author; arXiv lists Michael Ahn (alphabetical) | 1 | own system (Google) | 84%/74%/101 confirmed on project page |

Totals: 40/40 exist; 0 DOI mismatches; 0 claimed venues that are not real; 2 grade corrections (B2 up to Tier 1; H23 up to Tier 1); 1 authorship correction (B8).

---

## (b) Claim spot-check table

Verdicts: CONFIRMED / MISQUOTED (correct value) / UNVERIFIABLE. Conditions are stated where they change the reading. 36 checks.

| # | Source | Claim as written in the bibliography | Verdict | Correct value and conditions |
|---|---|---|---|---|
| 1 | B1 | Encoder GLUE avg 79.2/83.5/87.2/88.9/90.8/91.6; MNLI 79.5/83.4/87.0/89.2/91.3/91.8 | CONFIRMED | Table 7 (GLUE, encoders) and Table 9 |
| 2 | B1 | MS MARCO dev nDCG@10 enc vs dec 30.93/29.11 ... 43.35/41.70 | CONFIRMED | Table 9; dense bi-encoder retrieval, not reranking |
| 3 | B1 | 150M: native encoder MNLI 89.2 vs decoder-continued-as-encoder 85.8 ("3.4 points") | CONFIRMED (arithmetic, not stated verbatim) | Table 9: Enc-150m 89.2, Enc-from-Dec-150m 85.8. **Omitted:** Table 9 also reports the native decoder fine-tuned on MNLI at every size: 77.6 / 80.4 / 83.9 / **85.6** / 88.2 / 89.9 (17M-1B). Gap at 150M vs native decoder = 3.6. Text: "the 150M encoder scoring 89.2 compared to the 400M decoder's 88.2" |
| 4 | B1 | "Decoders are not fine-tuned on GLUE with a classification head ... the paper gives no decoder GLUE row" | MISQUOTED (limitation is wrong for MNLI) | No full-GLUE decoder rows, but decoders (and Dec-from-Enc) have fine-tuned MNLI accuracies in Table 9 (see #3). Generative avg 46.2 (Dec-150m) vs 43.6 (Dec-from-Enc-150m) confirmed |
| 5 | B1 | Abstract: "a 400M encoder outperforms a 1B decoder on MNLI" refers to an MLM-adapted decoder | CONFIRMED | Intro: "a 400M encoder outperforms a 1B decoder continue-trained with MLM on MNLI". Holds for both readings: Enc-400m 91.3 > Enc-from-Dec-1b 89.0 and > native Dec-1b 89.9 |
| 6 | B2 | SC avg CLM vs MLM 82.80/84.89 (210M), 83.58/87.00 (610M), 82.68/88.23 (1B) | CONFIRMED, with condition | Appendix Table 2; the MLM figures are the **40% masking** rows (paper's default). Best-ratio MLM: 85.12 (30%, 210M), 87.56 (20%, 610M), 88.23 (40%, 1B). Gap range 2.1-5.6 at 40% |
| 7 | B2 | QA 42.09/62.77, IR 76.20/79.55 (610M); TC 92.69/92.21 | CONFIRMED | Same 40%-mask condition |
| 8 | B2 | Biphasic 25/75 "reliably surpasses the MLM baseline"; 38 models, 15k runs, 110k GPU-h | CONFIRMED | Section 4; "110k MI250X GPU hours" |
| 9 | B3 | GLUE base: BERT 84.7, RoBERTa 86.4, DeBERTa-v3 88.1, NomicBERT 84.0, GTE 85.6, ModernBERT 88.4; large 85.2/88.9/91.4/87.6/90.4 | CONFIRMED | Table 1 |
| 10 | B3 | "ModernBERT-base 123.7k tokens/s on 8192-length variable-length input vs GTE-en-MLM 47.5k" | MISQUOTED (columns mislabelled) | Table 2 (RTX 4090, thousands of tokens/s): ModernBERT-base long **fixed** 123.7, long **variable** 133.8; GTE-en-MLM long fixed 46.8 / long variable 23.4; GTE-en-MLM-xformers long fixed 47.5 / long variable 67.3. Correct variable-length pair is 133.8 vs 23.4 (plain) or 67.3 (xformers). Evidence-table row must be corrected |
| 11 | B3 | BEIR DPR 41.6/44.0, ColBERT 51.3/52.4 | CONFIRMED | |
| 12 | B9 | Base 86M backbone + 98M embedding, 128K vocab; MNLI 90.6/90.7; SQuAD2 88.4/85.4; large 304M+131M, 91.8/91.9, 91.5/89.0, GLUE 91.37; XNLI 79.8 (+3.6) | CONFIRMED | Paper Tables 2-3 + HF cards (embedding splits are only on the cards) |
| 13 | H15 | PPA on OpenBookQA: Codex ~75, Instruct-davinci ~77, davinci ~50, Jurassic ~40, Instruct-Curie ~27, GPT-2 ~26, CodeParrot ~25; random 25 | CONFIRMED qualitatively; exact values UNVERIFIABLE (figure-read) | Text: "GPT-2, CodeParrot, and Instruct (Curie) all have PPA scores close to the 25% baseline"; "GPT-3 seems to perform about half as well [as Codex/Instruct Davinci], interestingly outperforming the larger Jurassic-1". Values come from Figure 2. **"Curie 6.7B" is not in the paper** (it gives only the 1.5B-178B span); 6.7B is an external attribution |
| 14 | H15 | MCP over cloze +8.3/+12.2/+9.7 (0/1/few-shot, 20 datasets); CosmosQA +32.5/+37.8/+44.3 | CONFIRMED | Verbatim |
| 15 | H17 | Layer 24 (A/B/C/D) and 29 (Q/Z/R/X) in OLMo-7B-Instruct; attention over MLPs; 1-4 heads/layer of 32; OLMo-7B near-random to near-100% between 80k-100k steps; Qwen2.5-0.5B/1.5B studied | CONFIRMED | Colors synthetic task; models also include OLMo 1B. No 0.5B accuracy table in the paper |
| 16 | H14 | PriDe (5% samples) +2.6 MMLU, +2.9 ARC, +4.0 CSQA; cost x1.15 vs x4; 40% -> x2.2 | MISQUOTED (wrong column) | Table 3 (0-shot, Δ average over 20 models): **PriDe (5%) = +1.2 / +1.3 / +1.7** (cost x1.15); +2.6 / +2.9 / +4.0 are **PriDe (40%)** (cost x2.2); PriDe (80%) +4.1/+4.9/+6.6 (x3.4); Cyclic Perm +4.9/+5.7/+7.9 (x4). RStd llama-30B MMLU 8.5, gpt-3.5 5.5/3.3, and the token-bias conclusion, confirmed |
| 17 | H16 | Reordering moves accuracy "13% to 75%" (arXiv) / "13% to 85%" (Anthology); calibration up to +8 | CONFIRMED | Both abstracts differ exactly as stated |
| 18 | H8 | Set-Encoder-330M DL19 0.733/0.765, DL20 0.727/0.799; monoELECTRA-330M DL19 0.720/0.768, DL20 0.711/0.770 | MISQUOTED (rows shifted by one) | Table 1 (BM25 / ColBERTv2): **Set-Encoder-330M DL19 0.727 / 0.789, DL20 0.735 / 0.790**; monoELECTRA-330M DL19 0.733 / 0.765, DL20 0.727 / 0.799; monoELECTRA-110M 0.720 / 0.768, 0.711 / 0.770; Set-Encoder-110M 0.724 / 0.788, 0.710 / 0.777. The bibliography's "Set-Encoder" numbers are monoELECTRA-330M's and its "monoELECTRA" numbers are the 110M's. monoT5-3B, RankGPT-4o, RankZephyr rows correct |
| 19 | H8 | Latency 100 passages: 0.219 s / 2.60 GB vs RankZephyr 24.047 s / 15.48 GB, RankGPT-4o 18.773 s; 85x and 110x; "differences to other cross-encoders not significant" | CONFIRMED | Verbatim; also "Interestingly, the pointwise monoELECTRA model is similarly effective. This suggests that passage interactions are unnecessary" and, in the ablation, inter-passage attention helps only when primed with duplicate-aware DA-InfoNCE (novelty ranking) |
| 20 | H8 | Reverse-ideal order: Set-Encoder stays above RankGPT-4o and LiT5-Distill | CONFIRMED | "reaches a higher nDCG@10 than the other cross-encoders on the inverse ideal, on the random, and on the original BM25 rankings" |
| 21 | H3 | UniMC 235M ANLI R1/R2/R3/CB 52.0/44.4/47.8/75.7 vs PaLM-540B 48.4/44.2/45.7/51.8, FLAN-137B 47.7/43.9/47.0/64.1, T0-11B 43.6/38.7/41.3/70.1; options cannot attend to each other | CONFIRMED | Table 1; GLaM-60B row also present (40.9/38.2/40.9/33.9) |
| 22 | H5 | gliclass-large 0.7193 vs deberta-v3-large-zeroshot-v2.0 0.6821; base 0.6764; throughput 97.29/51.61/25.22 ex/s (A6000); 2.3-16x | CONFIRMED | Batch size 1; cross-encoder baselines 10.63 (base) and 6.03 (large) ex/s |
| 23 | H5 | 1 -> 128 labels: 103.81->82.64, 49.42->45.94, 19.05->17.60 vs deberta-v3-base-zeroshot 24.55 -> 0.47 (52x) | CONFIRMED | Paper states verbatim "drops from 24.55 to 0.47 ex/s (≈52× slower)" |
| 24 | H10 | T5-Large MS MARCO / NQ MRR@10 table (BERT PointCE 0.3867/0.5157 ... RankT5-Enc Poly1 0.4296/0.5689); BEIR PointCE 0.5024 vs Softmax 0.5241; "no consistent winner"; list size >=20-30 | CONFIRMED | All nine rows verbatim. BEIR table is RankT5-Enc fine-tuned on MS MARCO. Verbatim: "the list size needs to be at least around 20 to 30 for the Softmax loss to beat the PointCE loss" |
| 25 | H11 | "RankT5 shows +1.8 MRR@10 from replacing [monoT5] with a direct score head" | MISQUOTED (imprecise) | vs monoT5 0.4156: RankT5-EncDec Softmax +1.2, Enc Softmax +1.5, EncDec Poly1 +1.9, Enc Poly1 +1.4. No configuration gives +1.8 |
| 26 | H9 | monoELECTRA-base distilled 0.720/0.711 vs RankZephyr 0.719/0.720, RankGPT-4 0.713/0.713; large 0.733/0.727; MS MARCO-only 0.687/0.698; 0.215 s; 173x / 24x | CONFIRMED | ColBERTv2 first stage for the distilled rows |
| 27 | H13 | "FIRST 78.8, RankZephyr 78.4, RankVicuna 71.3, cross-encoder 71.0" (flagged as suspicious in the bibliography) | MISQUOTED | Those are the **TREC-COVID column** of Table 1. BEIR (11 datasets, top-100 Contriever) **averages: FIRST 54.3, RankZephyr 53.7, RankVicuna 50.7, cross-encoder 50.7** |
| 28 | H13 | Latency per 20-passage window 1.2 s vs 0.6 s | UNVERIFIABLE | Text states only "50%" (A100 40GB, 200 queries); numbers exist only in Figure 4. A figure read suggests roughly 0.8 vs 0.4 s at m=20. Keep as "~50%", drop the absolute seconds |
| 29 | B8 | Ettin rerankers 0.5576/0.5779/0.5915/0.5994/0.6091/0.6114; Qwen3-Reranker-0.6B 0.5940; bge-v2-m3 ~0.553; MiniLM-L12 ~0.507; mxbai-large-v2 ~0.6115 | CONFIRMED | Exact: bge 0.5526, MiniLM-L12 0.5066, mxbai 0.6115. MTEB(eng, v2) retrieval, top-100 rerank. Author's own runs |
| 30 | B8 | H100 pairs/s 7,517/6,602/4,913/3,237/1,738/928; 150M peers 1,404-1,418; teacher 387; 3090 9,008/189; CPU 267.4/2.1 | CONFIRMED | Peers = gte-reranker-modernbert-base (1418) and granite-embedding-reranker-english-r2 (1404), both padded inputs; 3090 mid-table: 1B 189 is below mxbai-rerank-base-v2 (221) |
| 31 | B7 | Reranking table (Qwen3-Reranker-0.6B 65.80/71.31/66.36/67.28/73.42/5.41; 4B; 8B; bge-v2-m3; gte; jina); yes/no logits; no weak-supervision stage | CONFIRMED | Paper Table 4 and model card agree; top-100 candidates from Qwen3-Embedding-0.6B |
| 32 | B5 | RoBERTa-large 90.2/88.5; Llama-3.1-8B 0-shot 81.9/79.4, FT 91.1; GPT-4o 95.5/98.1; "RoBERTa-large 3.7e11 vs Llama-8B 1.6e13 FLOPs (43x)"; "~1000x wall-clock" | Accuracies CONFIRMED; FLOPs MISQUOTED | Table 1 inference FLOPs: **RoBERTa-base 3.7e11, RoBERTa-large 1.1e12**, Llama-3.1-8B 1.6e13. 43x is base-vs-8B; large-vs-8B is ~15x. The 1,000x is stated as "reducing inference costs by up to 1,000x" (contributions bullet), not wall-clock. **Omitted rows that bear directly on the <=1B decoder question:** fine-tuned Llama-3.2-1B 56.8 (SQuAD-v2) / 84.0 (NewsQA), inference 2.9e12; fine-tuned Llama-3.2-3B 82.2 / 86.4, 7.0e12; Llama-8B fine-tuned NewsQA 92.3 (beats RoBERTa-large by 3.8, so "within ~1 point" holds only for SQuAD-v2) |
| 33 | B6 | RewardBench 78.8/91.2/89.3/86.4 (400M), 73.5/83.3/78.4/78.4 (150M), 89.7/90.6/92.8/91.0 (70B); both options visible, not official protocol | CONFIRMED | Reasoning numbers are the DoRA variants |
| 34 | H2 | SuperGLUE test avg GPT-3 71.8, PET 74.0, iPET 75.4; dev 73.2/74.1/76.8; GPT-3 Med 56.2, Large 56.8; all per-task test values | CONFIRMED | Table 1 verbatim; "18 points better compared to GPT-3 Med" is the paper's own phrasing |
| 35 | H6 | Laurer card: deberta-v3-large-zeroshot-v2.0 0.676; "-c variant 0.673" | MISQUOTED (swapped) | Card: **-c = 0.676, non-c = 0.673**; base-c 0.619, bge-m3-c 0.590, bart-large-mnli 0.497 confirmed |
| 36 | Misc | H1 SWAG (81.6; 86.6/86.3; 59.1/59.2; GPT 78.0); H12 RankGPT table incl. DeBERTa-large 70.66/67.15/53.03, window 20 step 10; H4 GLiNER table; B10 mmBERT; B11 EuroBERT; B12 cards; B13 Qwen3; B14 Gemma; B15 sbert; B16 mxbai; B4 ranks 1.09/1.29/1.39/1.44/1.54/1.64 and 0.44 SD; H19 "up to 30.0"; H22 "+69%"; H21 "one million"; H23 84%/74%/101 | CONFIRMED | Conditions: B11 values are reported to **one decimal** in every arXiv version (90.8/92.6/85.4/89.4; mGTE 91.2) - the two-decimal figures in the bibliography are false precision; B12 135M column is labelled "SmolLM2-135M-8k"; H1 compares BERT-large **dev** 86.6 with GPT **test** 78.0 (test-to-test is 86.3 vs 78.0); B14 cross-harness comparison with SmolLM2 (10-/25-shot vs 0-shot) |

Tally: 26 CONFIRMED outright (5 with conditions), 2 mixed (#13 confirmed qualitatively / values figure-read; #32 accuracies confirmed / FLOPs misquoted), 7 MISQUOTED (#4, #10, #16, #18, #25, #27, #35), 1 UNVERIFIABLE (#28).

---

## (c) Relevance-transfer flags

The sub-questions are option scoring, classification, multiple choice and typed decisions. The following cited results are on **retrieval or passage reranking**; using them for a backbone/head recommendation is an inference the synthesis must label as such.

| Entry | What was measured | Why the transfer is an inference |
|---|---|---|
| B1 (Table 9 retrieval column) | Dense **bi-encoder** retrieval on MS MARCO dev | Not cross-encoder scoring; the encoder lead (1.6-2.3 nDCG) is a representation-quality result. The MNLI column (fine-tuned classification, incl. native decoders) is the directly relevant one and is under-used |
| B7 | Multilingual/code **reranking** (MTEB-R etc.), yes/no-logit head, top-100 candidates | Long-passage relevance; option sets in a typed decision are short and mutually exclusive |
| B8 | English passage **reranking** accuracy and pairs/s at reranking lengths | Throughput is per (query, passage) pair, not per state with N short options in one pass; the distilled student inherits a reranking teacher |
| B15, B16 | MS MARCO / BEIR **reranking** and docs/s or s/query | Same; B16 latency has no candidate count |
| B3 (BEIR rows), B10 (MTEB), B11 (MIRACL) | Retrieval | Only the GLUE/XNLI/XTREME rows transfer directly |
| H8 | Listwise passage **reranking**; "inter-candidate attention gives no significant gain" | The authors attribute the null result to TREC-style relevance labels lacking any inter-passage signal, and show set attention **does** help once the training signal requires comparison (duplicate-aware loss, novelty ranking). A typed decision is inherently comparative, so the null result must not be read as "set attention is useless for option scoring" |
| H9, H12, H13 | Reranking (distillation from LLM rankers; identifier-logit head) | Ranking metrics (nDCG@10) reward partial orderings; a typed decision is top-1 over a small set. H13's identifier-logit head is the closest analogue to a letter-logit MC head but was only trained/measured for ranking |
| H10 | RankT5 loss comparison on MS MARCO/NQ/BEIR | Listwise-softmax > pointwise is a ranking finding; the "list size >= 20-30" condition is about training lists of passages |
| H11 | monoT5 true/false head | Design precedent only |
| B5 | Binary (context, claim) groundedness classification | Closest fine-tuned decoder-vs-encoder classification evidence; still pairwise, not N-way |
| B6 | Pairwise preference (RewardBench) with both candidates visible | Two-option scoring, non-standard protocol |
| H4 | Span labelling (NER) | Head pattern only, not option selection |
| H20-H23 | Text-game / robotics action scoring | Different scale (GPT-2/PaLM); reward-driven, not supervised typed decisions |

Direct classification/MC evidence (no transfer needed): B1 MNLI/GLUE, B2, B4, B9, H1-H3, H5-H6, H14-H19.

---

## (d) Corrections required before synthesis

Ordered by effect on a backbone/head recommendation.

1. **B1 limitation is wrong and hides the paper's most relevant row.** Replace "no decoder GLUE row" with: Table 9 reports fine-tuned MNLI for native decoders at every size (17M 77.6, 32M 80.4, 68M 83.9, 150M 85.6, 400M 88.2, 1B 89.9). Add the row "Ettin enc vs native dec, 150M, MNLI 89.2 / 85.6" and "400M enc 91.3 vs 1B dec 89.9" to the evidence table. The "3.4 points" (vs enc-from-dec) becomes 3.6 vs the native decoder; the "one size step" reading is supported by the paper's own sentence (150M encoder 89.2 vs 400M decoder 88.2). No full GLUE decoder rows exist; that part stands.
2. **B5: add the fine-tuned Llama-3.2-1B and 3B rows** (56.8 / 84.0 and 82.2 / 86.4 on SQuAD-v2 / NewsQA; RoBERTa-large 90.2 / 88.5). This is the only fine-tuned <=3B decoder classification comparison in the corpus and it was left out. Correct the FLOPs: RoBERTa-large inference 1.1e12 (not 3.7e11, which is RoBERTa-base); 43x applies to base, ~15x to large. Change "1000x wall-clock" to the paper's "inference costs by up to 1,000x". Note NewsQA: fine-tuned 8B 92.3 beats RoBERTa-large by 3.8.
3. **H14: PriDe gains at 5% are +1.2 / +1.3 / +1.7, not +2.6 / +2.9 / +4.0** (those are the 40% column at x2.2 cost). Fix entry text and evidence-table row "PriDe (5% estimation)".
4. **H8 effectiveness rows are shifted.** Set-Encoder-330M: DL19 0.727 / 0.789, DL20 0.735 / 0.790 (BM25 / ColBERTv2); monoELECTRA-330M: 0.733 / 0.765, 0.727 / 0.799. Fix both evidence-table rows ("0.765 / 0.799" -> "0.789 / 0.790"; "0.768 / 0.770" -> "0.765 / 0.799"). The qualitative conclusion (not significant) is unchanged, but also record the authors' caveat that inter-passage attention helps once the loss requires comparison.
5. **B3 throughput row.** "123.7k vs 47.5k, 8192 variable-length" -> long variable-length: ModernBERT-base 133.8k vs GTE-en-MLM 23.4k (67.3k with xformers); long fixed-length: 123.7k vs 46.8k (47.5k xformers). State units as thousands of tokens/s, RTX 4090.
6. **H13: replace 78.8/78.4/71.3/71.0 with BEIR averages 54.3 / 53.7 / 50.7 / 50.7**; drop "1.2 s vs 0.6 s" and keep "~50% per-window latency reduction (A100 40GB)".
7. **B2 grade `PP` -> `PR` (ICLR 2026)**; state that all MLM numbers are the 40%-masking configuration (610M at 20% masking reaches 87.56).
8. **H23 grade `PP` -> `PR` (CoRL 2022, PMLR v205)**; note first-author discrepancy (arXiv Ahn / proceedings Ichter).
9. **B8 citation**: single author, Tom Aarsen (Hugging Face); remove "et al." and the "LightOn / JHU authors" attribution; mark COI (author trained and benchmarked the models).
10. **H6 card values**: -c variant 0.676, plain 0.673.
11. **H11**: replace "+1.8 MRR@10" with "+1.2 to +1.9 depending on head and loss (Enc Softmax +1.5)".
12. **H15**: mark the PPA percentages as figure-read approximations and drop "6.7B" for Curie (not in the source) or attribute it externally.
13. **B11**: report MIRACL/XNLI at the paper's one-decimal precision (210M 90.8, 610M 92.6, XLM-R 85.4 / 89.4, mGTE 91.2; XNLI 81.9 / 84.1 / 74.1 / 81.7).
14. **B9 contribution sentence**: GLiNER (Nov 2023) and Laurer's zeroshot-v2.0 (Apr 2024) predate ModernBERT (Dec 2024); only GLiClass actually compared DeBERTa-v3 against ModernBERT. Rewrite "all chose over ModernBERT on accuracy grounds".
15. Minor: H1 use test-to-test (86.3 vs 78.0); B12 label the 135M column "SmolLM2-135M-8k"; B14 flag the SmolLM2 comparison as cross-harness (10-/25-shot vs 0-shot); B6 note the DoRA condition on the reasoning numbers.

---

## (e) Confidence statement

Existence and venue are solid: all 40 sources resolve, every claimed acceptance (ICLR 2023/2024/2025/2026, ECIR 2025, ACL 2025, NoDaLiDa 2025, SIGIR 2023, the EMNLP/NAACL entries, ICML 2019/2021, CoRL 2022) was confirmed against proceedings or anthology pages rather than arXiv alone, and two entries were under-graded rather than over-graded. The numeric table is mostly faithful (26 of 36 spot-checks confirmed outright, two more in part), but the errors cluster exactly where a recommendation would lean: the PriDe gain is reported from the wrong cost column (2x overstated), the Set-Encoder rows are shifted so the set-attention model is credited with the pointwise model's scores, the ModernBERT throughput pairs fixed-length and variable-length columns, the FIRST "averages" are one dataset, and the Abbes FLOPs ratio is RoBERTa-base's. Two omissions matter more than any misquote: Ettin does report fine-tuned MNLI for native decoders at every size (150M decoder 85.6 vs encoder 89.2), and Abbes does report fine-tuned Llama-3.2-1B/3B (56.8 / 82.2 on SQuAD-v2 vs RoBERTa-large 90.2), which together are the only matched-size, fine-tuned decoder-vs-encoder classification numbers in the corpus. Roughly half of the head-side evidence (B7, B8, B15, B16, H8-H13) is passage reranking; after the corrections above the backbone conclusion (encoder wins fine-tuned classification at matched size by about one size step) rests on Tier 1 sources and is high-confidence, while any claim about set attention, listwise losses or throughput for typed decisions is a transfer from reranking and should be presented as medium-confidence inference. Vendor-reported numbers (B7, B8, B12-B16, H5, H6 card) are internally consistent with their sources but were not independently reproduced.
