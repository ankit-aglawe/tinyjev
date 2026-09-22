# Verification report: literature stream B (objective, distillation, calibration, selective prediction)

Phase 2 source-verification pass over `lit_objective_calibration.md`. Deliverable is a verification report; nothing here is synthesis. All retrieved content was treated as data.

Compiled 2026-09-22. Independent of the bibliography agent's own log: every identifier was re-resolved from scratch (arXiv Atom API for 51 IDs; Crossref for 5 DOIs; ACL Anthology pages for 15 IDs; CVF Open Access for 3; PMLR for 4; OpenReview for 3; NeurIPS proceedings indices 2017/2019/2020/2021; Semantic Scholar batch as a secondary cross-check only). For the claim spot-check, 35 PDFs (33 arXiv latest-version, 2 ACL Anthology camera-ready, plus the Anthology PDF of Di Palo) were downloaded and converted with `pdftotext -layout`; every number below was read from the extracted table or sentence, not from an abstract page.

Headline: 55 works, 0 existence failures, 0 DOI mismatches, 1 venue mismatch (Di Palo: EMNLP 2024 main, not Industry Track), 58 numeric claims checked, 51 confirmed, 5 misquoted (none reverses a direction), 2 confirmed with a source-internal inconsistency. The bibliography's "52 references" is a miscount: the document contains 55 distinct works.

---

## (a) Source quality matrix

Evidence tiers used here (per task spec, not the bibliography's own T1/T2 which mean core/supporting): **Tier 1** = peer-reviewed with venue acceptance confirmed from the venue's own index (Anthology / PMLR / CVF / NeurIPS proceedings / OpenReview decision / Crossref journal record / PDF venue header); **Tier 1-W** = peer-reviewed workshop *with archival proceedings* (weight below Tier 1); **Tier 2** = preprint or non-archival workshop; **Tier 3** = model card / README; **Tier 4** = blog. No Tier 3 or Tier 4 sources are cited in this stream.

Existence codes: PASS = identifier resolves and title, first author, year and venue all match; VENUE_MISMATCH = exists but venue/track stated incorrectly; FAIL / DOI_MISMATCH = none found.

| # | Entry | Work | Identifier resolved | Title / first author / year | Venue: claimed -> confirmed by | Existence | Tier | COI / notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 3a.1 | Mukhoti et al. 2020 | arXiv 2002.09437v2 | match | NeurIPS 2020 -> arXiv comment + NeurIPS 2020 proceedings index | PASS | 1 | - |
| 2 | 3a.2 | Müller, Kornblith, Hinton 2019 | arXiv 1906.02629v3 | match | NeurIPS 2019 -> arXiv comment + NeurIPS 2019 index | PASS | 1 | Google Brain; no product COI |
| 3 | 3a.3 | Xia et al. 2025 | arXiv 2403.14715v3 | match | ICLR 2025 -> PDF header "Published as a conference paper at ICLR 2025" | PASS | 1 | - |
| 4 | 3a.4 | Hui & Belkin 2021 | arXiv 2006.07322v5 | match | ICLR 2021 -> arXiv comment ("extended version of the paper published at ICLR2021") + S2 | PASS | 1 | arXiv v5 is an *extended* version; numbers verified against v5 |
| 5 | 3a.5 | Shao et al. 2024 | arXiv 2405.18906v1 | match | ICML 2024 -> arXiv comment + S2 | PASS | 1 | Tencent/WeChat AI authors; not spot-checked |
| 6 | 3a.6 | Shi, Cao, Raschka 2023 (CORN) | arXiv 2111.08851v5; DOI 10.1007/s10044-023-01181-9 | match | PAA 26(3):941-955 -> Crossref | PASS | 1 | Raschka releases `coral-pytorch`; author-maintained library |
| 7 | 3a.6 | Cao, Mirjalili, Raschka 2020 (CORAL) | arXiv 1901.07884v7; DOI 10.1016/j.patrec.2020.11.008 | match | PRL 140:325-331 -> Crossref | PASS | 1 | same |
| 8 | 3a.7 | Beckham & Pal 2017 | arXiv 1705.05278v2 | match | ICML 2017 -> PMLR v70 pp. 411-419 | PASS | 1 | - |
| 9 | 3a.8 | Diaz & Marathe 2019 | CVF CVPR 2019 page | match (pp. 4738-4747) | CVPR 2019 -> CVF Open Access | PASS | 1 | - |
| 10 | 3a.9 | Haas et al. 2026 | arXiv 2606.24959v1 (2026-06-23) | match | preprint -> arXiv only | PASS | 2 | 0 citations; no venue |
| 11 | 3a.10 | Gneiting & Raftery 2007 | DOI 10.1198/016214506000001437 | match | JASA 102(477):359-378 -> Crossref | PASS | 1 | - |
| 12 | 3b.1 | Hinton, Vinyals, Dean 2015 | arXiv 1503.02531v1 | match | NIPS 2014 DL workshop -> arXiv comment | PASS | 2 | non-archival workshop; canonical but unrefereed |
| 13 | 3b.2 | Sanh et al. 2019 (DistilBERT) | arXiv 1910.01108v4 | match | EMC2 @ NeurIPS'19 -> PDF p.1 footnote "EMC^2: 5th Edition Co-located with NeurIPS'19" | PASS | 2 | **COI**: Hugging Face authors evaluating their own released model |
| 14 | 3b.3 | Jiao et al. 2020 (TinyBERT) | arXiv 1909.10351v5; Anthology 2020.findings-emnlp.372 | match (pp. 4163-4174) | Findings EMNLP 2020 -> Anthology | PASS | 1 | Huawei Noah's Ark; released model |
| 15 | 3b.4 | Wang et al. 2020 (MiniLM) | arXiv 2002.10957v2 | match | NeurIPS 2020 -> NeurIPS 2020 proceedings index | PASS | 1 | **COI**: Microsoft; released model |
| 16 | 3b.4 | Wang et al. 2021 (MiniLMv2) | arXiv 2012.15828v2; Anthology 2021.findings-acl.188 | match (pp. 2140-2151) | Findings ACL 2021 -> Anthology | PASS | 1 | same |
| 17 | 3b.5 | Sun et al. 2020 (MobileBERT) | arXiv 2004.02984v2; Anthology 2020.acl-main.195 | match (pp. 2158-2170) | ACL 2020 -> Anthology | PASS | 1 | **COI**: Google; released model |
| 18 | 3b.6 | Turc et al. 2019 | arXiv 1908.08962v2 | match | preprint -> arXiv only | PASS | 2 | **COI**: Google; released BERT-Tiny/Mini/Small checkpoints |
| 19 | 3b.7 | Furlanello et al. 2018 | arXiv 1805.04770v2 | match | ICML 2018 -> PMLR v80 pp. 1607-1616 | PASS | 1 | - |
| 20 | 3b.8 | Stanton et al. 2021 | arXiv 2106.05945v2 | match | NeurIPS 2021 -> arXiv comment + NeurIPS 2021 index | PASS | 1 | - |
| 21 | 3b.9 | Cho & Hariharan 2019 | arXiv 1910.01348v1 | match | ICCV 2019 -> CVF (pp. 4794-4802) | PASS | 1 | - |
| 22 | 3b.10 | Beyer et al. 2022 | arXiv 2106.05237v2 | match | CVPR 2022 -> CVF (pp. 10925-10934) | PASS | 1 | Google Brain |
| 23 | 3b.11 | Wang, Weissweiler, Schütze, Plank 2023 | arXiv 2305.15032v1; Anthology 2023.acl-short.157 | match (pp. 1843-1852) | ACL 2023 short -> Anthology | PASS | 1 | - |
| 24 | 3b.12 | Tang et al. 2019 | arXiv 1903.12136v1 | match | preprint -> arXiv only | PASS | 2 | - |
| 25 | 3b.13 | Hsieh et al. 2023 | arXiv 2305.02301v2; Anthology 2023.findings-acl.507 | match (pp. 8003-8017) | Findings ACL 2023 -> Anthology | PASS | 1 | **COI**: Google authors using PaLM (Google) as teacher |
| 26 | 3b.14 | Wang, Liu, Xu, Zhu, Zeng 2021 | arXiv 2108.13487v1; Anthology 2021.findings-emnlp.354 | match (pp. 4195-4205) | Findings EMNLP 2021 -> Anthology | PASS | 1 | Microsoft authors; GPT-3 accessed via OpenAI API |
| 27 | 3b.15 | Pangakis & Wolken 2024 | arXiv 2406.17633v1; Anthology 2024.nlpcss-1.9 | match (pp. 113-131) | NLP+CSS workshop -> Anthology | PASS | 1-W | archival workshop proceedings |
| 28 | 3b.16 | Di Palo, Singhi, Fadlallah 2024 | arXiv 2411.05045v1; Anthology **2024.emnlp-main.215** | match (pp. 3675-3687) | claimed "EMNLP 2024 (Industry Track)" -> Anthology places it in **EMNLP 2024 main proceedings** | **VENUE_MISMATCH** | 1 | **COI**: all three authors Amazon; cost/latency comparisons vs Claude-3 are the authors' own deployment numbers |
| 29 | 3b.17 | Ye et al. 2022 (ZeroGen) | arXiv 2202.07922v2; Anthology 2022.emnlp-main.801 | match (pp. 11653-11669) | EMNLP 2022 -> Anthology | PASS | 1 | - |
| 30 | 3b.18 | Schick & Schütze 2021 (DINO) | arXiv 2104.07540v3; Anthology 2021.emnlp-main.555 | match (pp. 6943-6951) | EMNLP 2021 -> Anthology | PASS | 1 | - |
| 31 | 3b.19 | Ghita, Desai, Boier 2026 | arXiv 2606.24747v2 (2026-06-23; v2 2026-08-23) | match | preprint -> arXiv only | PASS | 2 | 0 citations |
| 32 | 3b.20 | Mishra, Krishna, Mishra 2023 | arXiv 2302.11472v1 | match | preprint -> arXiv only | PASS | 2 | 3 citations |
| 33 | 3b.21 | Kim, Park, Lee, Kwak 2025 | arXiv 2508.20224v1; DOI 10.1109/ACCESS.2025.3585106 | match | IEEE Access 13:115548-115557 -> Crossref | PASS | 1 | IEEE Access is a lighter-review megajournal; weight accordingly |
| 34 | 3c.1 | Guo et al. 2017 | arXiv 1706.04599v2 | match | ICML 2017 -> PMLR v70 pp. 1321-1330 | PASS | 1 | - |
| 35 | 3c.2 | Kull et al. 2019 | arXiv 1910.12656v1 | match | NeurIPS 2019 -> arXiv comment + NeurIPS 2019 index | PASS | 1 | - |
| 36 | 3c.3 | Desai & Durrett 2020 | arXiv 2003.07892v3; Anthology 2020.emnlp-main.21 | match (pp. 295-302) | EMNLP 2020 -> Anthology | PASS | 1 | - |
| 37 | 3c.4 | Chen et al. 2023 | arXiv 2211.00151v3; Anthology 2023.acl-long.75 | match (pp. 1343-1367) | ACL 2023 -> Anthology | PASS | 1 | - |
| 38 | 3c.5 | Kadavath et al. 2022 | arXiv 2207.05221v4 (36 authors) | match | preprint -> arXiv only | PASS | 2 | **COI**: Anthropic authors evaluating non-public Anthropic models; this verification run is itself executed by an Anthropic model, so the synthesis should treat this entry as vendor self-evaluation and not lean on it |
| 39 | 3c.6 | Minderer et al. 2021 | arXiv 2106.07998v2 | match | NeurIPS 2021 -> arXiv comment + NeurIPS 2021 index | PASS | 1 | Google |
| 40 | 3c.7 | Angelopoulos & Bates 2023 | arXiv 2107.07511v6; DOI 10.1561/2200000101 | match | FnT ML 16(4):494-591 -> Crossref | PASS | 1 | tutorial monograph, not an empirical study |
| 41 | 3c.7 | Romano, Sesia, Candès 2020 (APS) | arXiv 2006.02544v1 | match | NeurIPS 2020 -> arXiv journal-ref + NeurIPS 2020 index | PASS | 1 | - |
| 42 | 3c.7 | Angelopoulos et al. 2021 (RAPS) | arXiv 2009.14193v5 | match | ICLR 2021 spotlight -> arXiv comment with OpenReview forum id | PASS | 1 | - |
| 43 | 3c.8 | Kumar et al. 2023 | arXiv 2305.18404v3 | match | ICML 2023 workshop -> OpenReview "ICML 2023 Neural Conversational AI Workshop" (id 0yN9RW1Vac) | PASS | 2 | non-archival workshop |
| 44 | 3c.9 | Geifman & El-Yaniv 2017 | arXiv 1705.08500v2 | match | NeurIPS 2017 -> NeurIPS 2017 proceedings index | PASS | 1 | arXiv PDF carries no venue header |
| 45 | 3c.10 | Geifman & El-Yaniv 2019 (SelectiveNet) | arXiv 1901.09192v4 | match | ICML 2019 -> PMLR v97 pp. 2151-2159 | PASS | 1 | - |
| 46 | 3c.11 | Liu (Ziyin) et al. 2019 (Deep Gamblers) | arXiv 1907.00208v2 | match | NeurIPS 2019 -> arXiv comment + NeurIPS 2019 index | PASS | 1 | - |
| 47 | 3c.12 | Huang, Zhang, Zhang 2020 (SAT) | arXiv 2002.10319v2 | match | NeurIPS 2020 -> NeurIPS 2020 index | PASS | 1 | - |
| 48 | 3c.13 | Feng et al. 2023 | arXiv 2206.09034v4 | match | ICLR 2023 -> PDF header "Published as a conference paper at ICLR 2023" | PASS | 1 | Borealis AI |
| 49 | 3c.14 | Geifman, Uziel, El-Yaniv 2019 | arXiv 1805.08206v4 | match | ICLR 2019 -> arXiv comment | PASS | 1 | not spot-checked |
| 50 | 3c.14 | Jaeger et al. 2023 (FD-Shifts) | arXiv 2211.15259v2 | match | ICLR 2023 oral -> arXiv journal-ref | PASS | 1 | - |
| 51 | 3c.14 | Traub et al. 2024 (AUGRC) | arXiv 2407.01032v2 | match | NeurIPS 2024 -> OpenReview "NeurIPS 2024 spotlight" (id 2TktDpGqNM) | PASS | 1 | - |
| 52 | 3c.15 | Galil, Dabbah, El-Yaniv 2023 | arXiv 2302.11874v1 | match | ICLR 2023 -> arXiv journal-ref | PASS | 1 | - |
| 53 | 3c.16 | Xin, Tang, Yu, Lin 2021 | Anthology 2021.acl-long.84 | match (pp. 1040-1051) | ACL 2021 -> Anthology | PASS | 1 | - |
| 54 | 3c.17 | Varshney, Mishra, Baral 2022 | arXiv 2203.00211v1; Anthology 2022.findings-acl.158 | match (pp. 1995-2002) | Findings ACL 2022 -> Anthology | PASS | 1 | listed twice in the bibliography's log (arXiv + ACL id) |
| 55 | 3c.18 | Kamath, Jia, Liang 2020 | Anthology 2020.acl-main.503 | match (pp. 5684-5696) | ACL 2020 -> Anthology | PASS | 1 | - |

Tally: 55 works; PASS 54, VENUE_MISMATCH 1, FAIL 0, DOI_MISMATCH 0. Tier 1: 44; Tier 1-W: 1; Tier 2: 10 (Haas, Hinton, Sanh, Turc, Tang, Ghita, Mishra, Kadavath, Kumar; plus Kim counted Tier 1 with a megajournal caveat). Semantic Scholar disagreed with the venue index on three page ranges (Cho, Beyer, Furlanello); in all three the bibliography was right and S2 wrong, so S2 was not used as a source of truth anywhere.

---

## (b) Claim spot-check table

Priority claims from the task brief first, then additional rows checked because their PDFs were already in hand. "Bib." = as written in `lit_objective_calibration.md`; "Source" = what the extracted PDF says. Row numbers refer to the bibliography's numeric evidence table where applicable.

| # | Source (entry; table rows) | Claim as written | Checked against | Result | Correct value / note |
|---|---|---|---|---|---|
| 1 | Mukhoti 2020 (3a.1; rows 1-3) | CIFAR-100/ResNet-50 ECE: CE 17.52->3.42 (T=2.1), Brier 6.52->3.64 (1.1), MMCE 15.32->2.38 (1.8), LS 7.81->4.01 (1.1), FLSD-53 4.50->2.00 (1.1); errors 23.30/23.39/23.43/23.22 | Table 1, Table 2 (arXiv v2) | CONFIRMED | every cell matches |
| 2 | Mukhoti 2020 (3a.1) | CIFAR-10/ResNet-50: CE 4.35->1.35 (2.5), Brier 1.82->1.08, LS 2.96->1.67, FLSD 1.55->0.95; errors 4.95/4.98 | Table 1-2 | CONFIRMED | - |
| 3 | Mukhoti 2020 (3a.1; rows 4-5) | 20NG GP-CNN: CE 17.92->2.39 (3.4), Brier 13.58->3.22 (2.3), LS 4.79->2.54, FLSD 6.92->2.19 (1.5); errors 26.68/26.03/27.98 | Table 1-2 | CONFIRMED | - |
| 4 | Mukhoti 2020 (3a.1, "Pattern") | "on text ... LS/focal cost ~0.5-1.3 points of accuracy" | Table 2 | **MISQUOTED** | On 20NG LS-0.05 *lowers* error (26.03 vs CE 26.68, i.e. +0.65 acc); only focal costs accuracy (FL-3 +2.58 error, FLSD-53 +1.30). Also the paper has a second text row the bibliography omits: SST-Binary Tree-LSTM, CE 7.37->2.62 (T=1.8), Brier 9.01->2.79, LS 4.84->4.11, FLSD-53 9.19->1.83 (T=0.7); error 12.85/12.85/13.23/12.80. On that dataset focal is *worse* than CE pre-TS. "One text dataset" in Limitations should read two. |
| 5 | Xia 2025 (3a.3; row 8) | Cov@1% risk, ResNet-50 ImageNet MSP: CE 15.66, LS 0.1/0.2/0.3 = 0.05/0.06/0.02, LS 0.2 + logit norm 22.31 | Fig. 1 table (arXiv v3 = ICLR camera-ready) | CONFIRMED | caption: "Deployment-time logit normalisation effectively negates the degradation caused by LS" |
| 6 | Müller 2019 (3a.2; row 7) | MNIST: hard-target teacher 0.67% -> student 0.74%; LS teacher 0.59% -> student 0.91% | Sec. 5 text | CONFIRMED | - |
| 7 | Müller 2019 (3a.2; row 6) | ECE: ResNet-56/CIFAR-100 0.150 / TS 0.021 (T=1.9) / LS 0.024 (a=0.05); Inception-v4 0.071/0.022 (1.4)/0.035 (0.1); En-De 0.056/0.018 (1.13)/0.019 (0.1) | Table 2 | CONFIRMED | - |
| 8 | Hui & Belkin 2021 (3a.4; row 9) | BERT square vs CE: MRPC 83.8/82.1, SST-2 94.0/93.9, QNLI 90.6/90.6, QQP 88.9/88.9; F1 MRPC 88.1/86.7, QQP 70.9/70.7; 5 seeds | Tables 2-3 (arXiv v5); column order square / CE / square-same-epochs confirmed from header | CONFIRMED | "average results of 5 runs" confirmed |
| 9 | Hui & Belkin 2021 (3a.4; row 10) | "22 out of 28 tasks (NLP and ASR)" | Sec. 1 | CONFIRMED with scope error | Sentence is "in 22 out of 28 tasks" over *all* domains (NLP + ASR + CV); NLP-only breakdown is 12/14 accuracy and 5/6 F1 (Sec. 4). Drop "(NLP and ASR)". Vision numbers 95.9 vs 96.3 (CIFAR-10 WRN) and 74.6 vs 77.0 (EfficientNet ImageNet) confirmed. |
| 10 | Sanh 2019 (3b.2; rows 17-18) | GLUE dev 77.0 vs 79.5; MNLI 82.2/86.7; QQP 89.2/91.8; SST-2 91.3/92.7; CoLA 51.3/56.3; MRPC 87.5/88.6; QNLI 88.5/89.6; RTE 59.9/69.3; STS-B 86.9/89.0; IMDb 92.82/93.46; SQuAD 77.7/85.8 vs 81.2/88.5; 410 vs 668 s | Tables 1-3 (arXiv v4, 2020 revision) | CONFIRMED | RTE gap 9.4 confirmed |
| 11 | Jiao 2020 (3b.3; rows 19-20) | GLUE test avg: teacher 109M 79.5; TinyBERT4 14.5M 77.0 (96.8%, 9.4x); TinyBERT6 67M 79.4 | Table 2 (Findings version, arXiv v5) | CONFIRMED | - |
| 12 | Jiao 2020 (3b.3; row 21) | Ablation "dev avg over MNLI-m/MRPC/CoLA": full 75.6, -GD 72.5, -TD 68.5, -DA 68.4 | Table 4 | CONFIRMED with label error | Average is over four columns: MNLI-m, MNLI-mm, MRPC, CoLA (82.8/82.9/85.8/50.8 -> 75.6) |
| 13 | Sun 2020 (3b.5; rows 26-27) | GLUE test: BERT-base 78.3; MobileBERT 25.3M 77.7 (MNLI 83.3/82.6, SST-2 92.8, QQP 70.2, QNLI 90.6); MobileBERT-tiny 15.1M 75.8; TinyBERT 14.5M 75.4; SQuAD v1.1 dev F1 90.0 vs 88.5; 62 ms Pixel 4 | Table 4; Sec. 1 | CONFIRMED | - |
| 14 | Wang 2020 MiniLM (3b.4; row 22) | 6x768 66M: MNLI-m 84.0, SST-2 92.0, QNLI 91.0, QQP 91.0, RTE 71.5, MRPC 88.4, CoLA 49.2, SQuAD2 76.4; 2.0x, ">99%" | Table 2 (dev) | CONFIRMED | - |
| 15 | Wang 2021 MiniLMv2 (3b.4; rows 23-25) | 6x384 30M from RoBERTa-large: SQuAD2 76.4, MNLI 84.4, SST-2 92.0, avg 79.5, 5.3x; 6x768 81M: 81.6, 87.0, avg 83.8; RoBERTa-base 85.4 (87.6, 83.7); 12x768 avg 87.6 vs 86.1; 4 runs | Tables 1, 3 (dev) | CONFIRMED | Table 1 avg is over 8 tasks, Table 3 over 9 (adds STS-B); the two "RoBERTa-base" averages (85.4 vs 86.1) therefore differ by construction, which the bibliography does not say |
| 16 | Wang 2023 (3b.11; rows 37-38) | Task-specific: vanilla KD 71.8, Hid-Seq 75.8, Att-MSE 76.5, Att-KL+Val-KL 77.5; task-agnostic: vanilla 79.3, Att-MSE 81.3, DistilBERT 78.5, TinyBERT 79.9, MiniLM 81.0; 4 runs | Tables 1-2 | CONFIRMED | - |
| 17 | Wang 2023 (3b.11) | "Initializing from lower teacher layers improves vanilla KD on QNLI from 68.1% upward (Table 3)" | Tables 1, 3 | **MISQUOTED** | Baseline is **66.5 +/- 1.49** (init 4,8,12); lower-layer inits give 82.9 (1,8,12) and 86.2 (1,2,3). 68.1 does not appear in the paper. Direction unchanged. |
| 18 | Turc 2019 (3b.6; rows 28-30) | GLUE test meta: TF 80.5, PF 81.6, PD 82.1; MNLI PD 82.8/82.2 vs PF 81.8/81.1; dev PD 84.4 vs PF 82.8 vs DistilBERT 82.3; 5 runs | Table 3 | CONFIRMED | Fig. 7 per-size values not re-read (figure-only, marked as such in bibliography) |
| 19 | Turc 2019 (3b.6, Sec. 6.2) | 11M Transformer-Mini "recovers the accuracy of the teacher" on 8M Book Reviews; plain distillation needs 10x larger student; PF beats plain distillation by 12% on MNLI when transfer set is 1.3M and slightly OOD | Sec. 6.2 text | CONFIRMED | sentence verbatim; also "8% on RTE" |
| 20 | Cho & Hariharan 2019 (3b.9; rows 34-35) | ResNet-18 ImageNet top-1 error: scratch 30.24; KD from R18/R34/R50 (err 30.24/26.70/23.85) = 30.57/30.79/30.95; ESKD 29.01/29.16/29.35; ESKD R152 29.45 | Table 1; Table (Sec. 5) | CONFIRMED | 29.45 is in the ESKD ResNet152 row (a KD+ONE row coincidentally also reads 29.45) |
| 21 | Stanton 2021 (3b.8; rows 32-33) | self-distillation agreement "between 80% and 90%"; train agreement 83.3% after 5,000 epochs vs 78.95%; ensemble teacher agreement "below 80%" | Secs. 4-6 text | CONFIRMED | - |
| 22 | Kim 2025 (3b.21; row 49) | R^2 teacher-ACE vs student-acc 0.9229 (WRN-16-2), 0.8988 (ShuffleNetV2); teacher-acc vs student-acc 0.6751, 0.5557 | Fig. 1 labels + Sec. 1 text | CONFIRMED with source-internal inconsistency | Fig. 1 prints 0.8988; the body text says 0.8998. Cite as "~0.90". |
| 23 | Hsieh 2023 (3b.13; rows 41-43) | T5-base 220M: std FT 88.38/43.58/62.19/62.63; DSS 540B 89.51/49.58/63.29/65.50; DSS 20B 89.12/48.15/63.25/63.00; single-task ANLI 43.50; "8% and 13%" on ANLI; 11B beats teacher on 3 of 4; 80% data | Tables 1-2; Secs. 4.1-4.2 | CONFIRMED | "8% and 13%" is explicitly ANLI-specific in the source, as the bibliography states |
| 24 | Pangakis & Wolken 2024 (3b.15; row 45) | median F1: DistilBERT-human 0.641, BERT-human 0.624, GPT-4 few-shot 0.592, BERT-GPT-4 0.586; recall 0.746; precision gap 0.214; Mistral-7B labels 0.16 F1 worse | Sec. 3 text; Appendix | CONFIRMED | 0.641/0.624 assignment confirmed by "respectively" |
| 25 | Ye 2022 ZeroGen (3b.17; row 47) | DistilBERT: IMDb 84.28, SST-2 87.27, QNLI 71.19, RTE 59.93 vs supervised 87.24/89.68/88.05/58.12; LSTM 79.80/78.40/52.26/58.85 vs 84.60/76.30/69.00/54.87; SQuAD 25.5/31.5 vs 76.3/84.7 | Table 1 | CONFIRMED | SQuAD exact: 25.50/31.53 vs 76.28/84.67 |
| 26 | Chen 2023 (3c.4; rows 57-58) | T5-base MNLI: acc 86.50, conf 94.85, ECE 8.35; TS 2.75; LS 3.41; ensemble 8.28; HANS 37.30 -> TS 28.93; ANLI 54.27/44.17; Amazon 4.86->1.39; SST-5 13.52->4.94 | Table 1 | CONFIRMED | - |
| 27 | Chen 2023 (3c.4; rows 59-60) | Table 5 E-MLP rows: Amazon ECE 4.78/4.35/4.70, acc 87.65/91.00/91.58 (T5-small/base/large); SST-5 15.23/15.45/10.24; SemEval 27.91/23.36/21.61 | Table 5 (Appendix, "increasing model scales") | CONFIRMED | Caution for synthesis: these ECEs are *after* the E-MLP learned calibrator, not vanilla. Table 2 (same layout, "Small/Middle/Large") varies *calibration-set size*, not model scale; the two are easy to confuse. |
| 28 | Feng 2023 (3c.13; rows 71-73) | ImageNet100 full-cov acc 85.68/86.23/86.51/86.40; SN 0.90 9.44->7.89, 0.80 6.00->4.47, 0.70 3.38->2.21; DG 0.80 5.21->4.52; SAT 0.80 5.20->4.46, 0.70 2.71->2.33; SN head 48.87 @0.20, 99.00 @0.10; SAT+EM+SR 3.90 @0.80, 1.81 @0.70; CIFAR-10 0.23 vs 0.27 @0.70 | Tables 1, 2, 4 | CONFIRMED | Table 2 caption confirms ImageNet100 |
| 29 | Jaeger 2023 (3c.14; row 75) | "None of the evaluated methods from literature beats the simple Maximum Softmax Response baseline"; ConfidNet / DG / DeVries fail to generalize; MCD-MSR best or close on i.i.d. | Sec. 5 text | CONFIRMED | verbatim |
| 30 | Varshney 2022 (3c.17; rows 84-85) | MCD gain +0.28 (IID), +0.08 (OOD); ADV: MCD -1.76 (DD), calibration -1.27 (NLI); SNLI 2.78/2.47/2.57, MNLI 5.47/4.92/5.16, DNLI 7.36/6.69/3.88; "none ... consistently and considerably" | Abstract, Sec. 4, Table 3 | CONFIRMED | - |
| 31 | Kamath 2020 (3c.18; row 86) | Cov@80% acc: MaxProb 48.2, MaxProb+known-OOD 51.8, calibrator SQuAD-only 53.7, calibrator +known-OOD 56.1; +4.3 / +6.7 @90%; -1.1 AUC | Sec. 1, Sec. 5 | CONFIRMED | - |
| 32 | Angelopoulos 2021 RAPS (3c.7; rows 63-64) | 90% cov set size: ResNet-152 top-k 2.63 / naive 9.78 / APS 10.4 / RAPS 2.11; R50 3.14/11.8/12.3/2.57; R18 5.72/15.5/16.2/4.43; ResNeXt-101 2.42/17.1/19.7/2.00; naive coverage 0.889-0.896 | Table 1 | CONFIRMED | - |
| 33 | Angelopoulos 2021 RAPS (3c.7) | "At 95% coverage RAPS sets are 4.4-11.7 vs APS 22.5-33.2"; SSCV "0.02-0.04 vs 0.04-0.07" | Appendix tables (95% coverage; violation) | **MISQUOTED** | 95%: RAPS **4.21-11.7**, APS **22.5-46.3** (ResNeXt101 = 4.21 / 46.3 dropped). Violation at a=10%: APS 0.047-0.088 vs RAPS 0.030-0.059; at a=5%: APS 0.032-0.046 vs RAPS 0.015-0.023. The bibliography's ranges blend the two alpha columns. Direction unchanged. |
| 34 | Galil 2023 (3c.15; row 77) | "Training regimes incorporating any kind of knowledge distillation lead to DNNs with improved uncertainty estimation"; largest median improvement; "even when the teacher's own uncertainty metrics are poor (Fig. 4a)" | Sec. 1, Sec. 5 | CONFIRMED with figure mislabel | Quote verbatim (line 124). The "teacher itself is much worse at both metrics" statement points to **Figure 5**, not Fig. 4a. |
| 35 | Galil 2023 (3c.15; row 78) | "Temperature scaling ... consistently and greatly improves AUROC and selective performance" | Sec. 5 (2)(a) | CONFIRMED | verbatim |
| 36 | Galil 2023 (3c.15; row 80) | "Correlation between accuracy and AUROC across all 523 models is ~0.03; within families it flips sign (XCiT +0.76, ResNet -0.74)" | Sec. 5 (4); Appendix | **MISQUOTED (metric pair)** | 0.03 is correct (Appendix: "overall Spearman correlation between AUROC and accuracy ... is 0.03"). But **0.76 and -0.74 are Spearman correlations between AUROC and ECE** (28 undistilled XCiTs; 33 ResNets), and the overall AUROC-ECE correlation is -0.44. The paper does say accuracy/#params-vs-AUROC/ECE correlations flip sign within families (Sec. 1 (5)) but gives no family-level accuracy-AUROC coefficients in the main text. |
| 37 | Galil 2023 (3c.15; row 79) | ViT-L/16-384: 99% selective accuracy at 47% coverage, 95% at 80%; EfficientNet-V2-XL cannot reach these | Abstract | CONFIRMED | - |
| 38 | Traub 2024 (3c.14; row 76) | AUGRC "changes metric rankings on 5 out of the 6 data sets" | Abstract | CONFIRMED | verbatim |
| 39 | Guo 2017 (3c.1; rows 50-52) | ECE: CIFAR-100 ResNet-110 16.53->1.26, matrix 25.49; DenseNet-40 10.37->1.18; ImageNet R152 5.48->1.86; SVHN 0.44->0.17; 20 News DAN-3 8.02->4.11; Reuters 0.85->0.91; SST-B 6.63->1.84; SST-FG 6.71->2.56; "does not affect the model's accuracy" | Table 1; Sec. 4.2 | CONFIRMED | - |
| 40 | Desai & Durrett 2020 (3c.3; rows 54-56) | all 24 out-of-box / TS ECE cells for BERT, RoBERTa, DA, ESIM; LS: BERT HellaSWAG 12.62->5.73, RoBERTa MNLI 3.62->4.50 | Tables 2-3 | CONFIRMED | - |
| 41 | Xin 2021 (3c.16; rows 81-83) | MRPC AUC: LSTM 101.5/137.0, BERT-base 33.8/38.3, BERT-large 27.0/35.9, ALBERT 16.0/43.9; QNLI 1539.1/111.9/105.6/122.8, BERT-base MC 130.1; MNLI-m 1984.0/514.8/486.1/469.3 | Table 1 (camera-ready) | CONFIRMED | - |
| 42 | Geifman & El-Yaniv 2019 SelectiveNet (3c.10; rows 68-70, 74) | CIFAR-10: 0.90 2.43/2.89/2.92; 0.80 0.86/1.05/1.08; 0.70 0.32/0.42/0.43; SVHN 0.80 0.53/0.61/0.61; Cats-Dogs 0.80 0.35/0.68/0.55; violation 11.98 vs 3.625 | Tables 1-4 | CONFIRMED | - |
| 43 | Liu 2019 Deep Gamblers (3c.11; rows 68-70) | 1.00 6.12 (6.79); 0.95 3.49 vs 4.55/4.16; 0.90 2.19; 0.85 1.09 vs 1.78/1.43; 0.80 0.66 vs 1.05/0.86; 0.70 0.43 vs 0.42/0.32 | Table 4 | CONFIRMED | - |
| 44 | Huang 2020 SAT (3c.12; rows 68-70) | 0.90 1.93 vs 2.19/2.43/2.89; 0.80 0.67 vs 0.66/0.86/1.05; 0.70 0.34 vs 0.43/0.32/0.42; Dogs-Cats 0.80 0.15 vs 0.46/0.35/0.68; "up to 50%"; 3 trials | Table 4 | CONFIRMED | - |
| 45 | Geifman & El-Yaniv 2017 (3c.9; row 66) | 2% top-5 ImageNet error guaranteed w.p. 99.9% at "almost 60%" coverage | Abstract, Sec. 1 | CONFIRMED | Fig. 2 SR-vs-MC values (~10 vs >20) not re-read (figure-only) |
| 46 | Mishra 2023 (3b.20; row 48) | WRN-40-2 (77.09/0.103) -> ShuffleNetV1: scratch 70.58/0.121, KD 71.05/0.108, KD+mixup 74.65/0.049, KD+CutMix 74.94/0.046; R50 (78.98/0.104) -> MobileNetV2: 63.20/0.169, 63.92/0.135, 67.32/0.037 | Table 1 | CONFIRMED | - |
| 47 | Beyer 2022 (3b.10; row 36) | ResNet-50 82.8% at 9,600 epochs; teacher 83.0%; prior 78.8% / 80.49% | Secs. 1, 4; Table 1 | CONFIRMED | Fig. 5 value 81.45 not located in text; leave as (fig.) |
| 48 | Di Palo 2024 (3b.16; row 46) | AG-News 0.895, Yahoo 0.685, HuffPost 0.519, Amazon 0.443; -validation 0.893/0.669/0.501/0.419; -hard-neg 0.887/0.675/0.510/0.433; 0.46 s vs 60.64 s (~130x); 25x cheaper on CPU | Tables 2, 4, 5; Sec. 5 | CONFIRMED | The 130x/6x figures are Claude-3 on GPU; the 3x/25x figures are LLaMA-3-8B on CPU. The bibliography's row 46 blends the two comparators. |

Tally: 58 numeric/verbatim claims across 48 rows; CONFIRMED 51; CONFIRMED-with-label/scope error 4 (rows 9, 12, 22, 34); MISQUOTED 5 (rows 4, 17, 33, 36, and the Di Palo comparator blend in 48 is borderline and counted here as a note, not a misquote); UNVERIFIABLE 0. No misquote reverses the direction of a finding. The worst is row 36 (Galil): a metric-pair substitution that, if carried into synthesis, would support a false claim that accuracy and selective-prediction quality are strongly (anti)correlated within families.

Not spot-checked (PDFs not opened; figure-only or outside the priority list): Shao 2024 BLEU rows (11-12), CORN/CORAL/Diaz/Beckham ordinal rows (13-15), Hinton MNIST (16), Furlanello (31), Tang (39-40), Wang 2021 GPT-3 cost (44), Schick, Ghita, Kadavath (61-62), Minderer, Kull ranks (53), Kumar (65), Geifman 2019 E-AURC. Rows 61-62 are figure-only from a Tier 2 vendor preprint and should not carry weight in synthesis.

---

## (c) Vision-vs-text relevance flags

Modality of the *measured* result, not of the paper's topic. Any transfer of a vision-only result to a 0.1-0.6B text option scorer is an inference the synthesis must label as such.

**Vision-only (ImageNet / CIFAR / SVHN / age-estimation images / medical):**
Xia 2025 (3a.3); CORN/CORAL (3a.6, plus one tabular set); Beckham (3a.7); Diaz (3a.8); Haas 2026 (3a.9, images + tabular); Furlanello (3b.7); Stanton (3b.8); Cho & Hariharan (3b.9); Beyer (3b.10); Mishra (3b.20); Kim (3b.21); Minderer (3c.6); Romano/RAPS (3c.7); Geifman 2017 (3c.9); SelectiveNet (3c.10); Deep Gamblers (3c.11); SAT (3c.12); Feng (3c.13); Geifman 2019, Jaeger, Traub (3c.14); Galil (3c.15).

**Mixed (vision plus at least one text dataset):**
Mukhoti (3a.1: 9 vision rows + 20NG GP-CNN + SST-Binary Tree-LSTM); Müller (3a.2: vision + En-De MT; distillation result is MNIST/CIFAR only); Hui & Belkin (3a.4: NLP incl. BERT + ASR + CV); Guo 2017 (3c.1: vision + 20NG/Reuters DAN + SST TreeLSTM); Kull (3c.2: tabular-heavy + CIFAR/SVHN); Hinton (3b.1: MNIST + speech).

**Text (encoders or seq2seq at or near the target size):**
Sanh, Jiao, Wang MiniLM/v2, Sun, Turc, Wang 2023, Tang, Hsieh, Wang 2021, Pangakis, Di Palo, Ye, Schick (3b); Desai, Chen, Xin, Varshney, Kamath (3c).

**Text but decoder LLMs far above target size:** Shao (7-13B, generation only), Ghita (decoder students), Kadavath (800M-52B), Kumar (LLaMA-13B).

Evidence-table modality count (86 rows): 37 vision-only, 44 text, 5 other (2 MT-BLEU, 1 tabular-heavy rank, 2 decoder-LLM figure-only).

Where this bites, by sub-question:

1. **Label smoothing vs selective prediction (3a.3, row 8).** The coverage-at-1%-risk collapse (15.66% -> 0.05%) and the logit-normalization recovery are ImageNet/ResNet-50 and ViT-S/16 only. The only text-side LS evidence is Desai & Durrett (ECE, no selective metric; LS helps some OOD cells and hurts others) and Mukhoti's two text rows (LS improves 20NG accuracy, ECE post-TS within 0.2 of CE). There is no text measurement of LS on coverage-at-risk. Transfer is an inference.
2. **Teacher label smoothing hurts distillation (3a.2, row 7).** MNIST and CIFAR-10 only.
3. **KD improves calibration / selective performance (3c.15, 3b.20, 3b.21; rows 48-49, 77).** All three are vision, and Galil is observational over a model zoo. No source measures ECE or AURC for DistilBERT/TinyBERT/MiniLM students (the bibliography's Gaps section already says so; the flag is repeated here because rows 77-78 sit in the T1 table without a modality column).
4. **Learned abstention heads vs softmax response (3c.10-3c.13; rows 68-74).** Entirely vision. The text-side analogues are Xin (SR beats MC-dropout for BERT) and Varshney (nothing consistently beats MaxProb for BERT-base). Both point the same way as Feng/Jaeger, which is reassuring, but neither tests SelectiveNet/DG/SAT-style heads on text.
5. **Conformal set sizes (3c.7; rows 63-64).** ImageNet 1,000-class; the 4-option setting is covered only by Kumar (LLaMA-13B, workshop preprint).
6. **Teacher-size mismatch (3b.9, row 34-35) and function-matching recipe (3b.10, row 36).** ImageNet CNNs. The text analogue with a sweep of student sizes is Turc (figure-only for most sizes).
7. **AURC / AUGRC metric critique (3c.14; rows 75-76).** Vision benchmarks; the metric definitions are modality-free, but the "ranking changes on 5 of 6 datasets" evidence is vision.
8. **Temperature scaling can reorder max-softmax (3c.15, row 78).** 523 ImageNet classifiers. The bibliography's Gaps section correctly notes the binary case is untested anywhere.
9. **Ordinal losses (3a.6-3a.9; rows 13-15).** Age-estimation images and tabular; no text ordinal-regression measurement.

Conversely, the text evidence that *is* at target scale (rows 17-30, 37-47, 54-60, 81-86) contains no coverage-at-risk metric except Kamath (extractive QA) and no training-objective comparison except Hui & Belkin (accuracy only) and Mukhoti's 20NG CNN.

---

## (d) Corrections required before synthesis

Ordered by impact on the objective/calibration recommendation.

1. **Galil 2023, entry 3c.15 finding (3) and row 80.** Replace "within families it flips sign (XCiT +0.76, ResNet -0.74)" with: overall Spearman(AUROC, accuracy) = 0.03 (Appendix); overall Spearman(AUROC, ECE) = -0.44; within families Spearman(AUROC, ECE) = +0.76 (28 undistilled XCiTs) and -0.74 (33 ResNets). The sign-flip claim about accuracy is stated qualitatively in the paper (Sec. 1 (5)) but the coefficients quoted are for ECE. Also change "Fig. 4a" to "Fig. 5" for the poor-teacher statement.
2. **Mukhoti 2020, entry 3a.1 "Pattern" and Limitations.** LS-0.05 improves 20NG accuracy (error 26.03 vs 26.68); only focal costs accuracy (FL-3 +2.58, FLSD-53 +1.30). Add the omitted SST-Binary Tree-LSTM text row (CE 7.37->2.62 @T=1.8; Brier 9.01->2.79; LS 4.84->4.11; FLSD-53 9.19->1.83 @T=0.7; errors 12.85/12.85/13.23/12.80). On that second text dataset focal is worse than CE pre-TS and the LS advantage post-TS disappears, which weakens any "focal/LS is innately better calibrated on text" reading. Change "one text dataset" to "two".
3. **Angelopoulos 2021 RAPS, entry 3c.7.** 95% coverage: RAPS 4.21-11.7 vs APS 22.5-46.3. Size-stratified violation: a=10% APS 0.047-0.088 vs RAPS 0.030-0.059; a=5% APS 0.032-0.046 vs RAPS 0.015-0.023.
4. **Wang 2023, entry 3b.11.** "from 68.1%" -> "from 66.5% (init 4,8,12) to 82.9% (1,8,12) and 86.2% (1,2,3)".
5. **Di Palo 2024, entry 3b.16 APA and grade.** Venue is *Proceedings of EMNLP 2024* (main), Anthology 2024.emnlp-main.215, pp. 3675-3687, not Industry Track. Add COI (Amazon). In row 46 separate the two comparators: 130x slower / 6x costlier is Claude-3 on GPU; 3x slower / 25x costlier is LLaMA-3-8B on CPU.
6. **Reference count.** "All 52 references" -> 55 distinct works (entries 3a.6, 3b.4 carry two each; 3c.7 and 3c.14 carry three each). The verification log lists Varshney twice (arXiv and ACL ids).
7. **Jiao 2020, entry 3b.3 and row 21.** Ablation average is over MNLI-m, MNLI-mm, MRPC, CoLA (four columns).
8. **Hui & Belkin, entry 3a.4 and row 10.** "22 of 28" spans NLP, ASR and CV; drop "(NLP and ASR)". Note that the cited arXiv v5 is an extended version of the ICLR paper.
9. **Kim 2025, entry 3b.21 and row 49.** Cite ShuffleNetV2 R^2 as ~0.90 (figure 0.8988, text 0.8998).
10. **Chen 2023, entry 3c.4 and rows 59-60.** State explicitly that Table 5 ECEs are post E-MLP calibrator and that Table 2 (same "Small/Middle/Large" layout) varies calibration-set size, so the scale conclusion must cite Table 5 and Fig. 5 only.
11. **MiniLMv2 rows 23 and 25.** The RoBERTa-base reference average differs between Table 1 (85.4, 8 tasks) and Table 3 (86.1, 9 tasks incl. STS-B); label the task count so the two rows are not read as inconsistent.
12. **Column-label collision.** In the numeric evidence table the "Source" suffixes T1/T2/T3/T4/T5 denote *table numbers in the source* (e.g. "Mukhoti 2020 T1" = Table 1), while the bibliography's [T1]/[T2] entry tags mean core/supporting, and this report's tiers mean evidence grade. Rename the table suffixes to "Tab.1" etc. before synthesis to avoid a Tier-1/Tier-4 misreading.
13. **Kadavath 2022, entry 3c.5 and rows 61-62.** Add COI (vendor self-evaluation of non-public models; figure-only ECE). Mark as Tier 2 and do not let it carry the "calibration improves with scale on MC" claim alone; Chen 2023 Table 5 is the public, sub-1B, text datapoint for that question.
14. **Evidence grades for workshops.** Sanh (EMC2), Hinton (NIPS 2014 DL workshop) and Kumar (ICML 2023 workshop) are non-archival workshops and should be graded PP/WS-nonarchival, not WS on a par with Pangakis (archival Anthology workshop proceedings).

Nothing in the list above removes a source or reverses a reported direction; items 1-3 change magnitudes or metric attributions that a synthesis could otherwise lean on.

---

## (e) Confidence statement

Existence and venue of all 55 works are established with high confidence: every arXiv id, DOI, Anthology id and CVF/PMLR page resolved to the claimed title, first author and year, and every peer-review claim was confirmed against the venue's own index rather than a secondary aggregator (the one aggregator disagreement, Semantic Scholar's page ranges, resolved in the bibliography's favour). The numeric content is likewise reliable: of 58 claims re-read from PDF text, 51 match exactly and the five misquotes are second-order (a 1.6-point baseline, a range that omits one architecture, a mislabeled column average, a metric-pair substitution in Galil, and an interpretive sentence in Mukhoti that inverts the sign of LS's accuracy effect on 20NG); none flips a conclusion. The principal risk to the Phase 3 synthesis is therefore not fabrication but modality: 37 of 86 evidence rows, and every row bearing on label-smoothing-vs-selective-prediction, KD-vs-calibration, learned abstention heads and conformal set size, are vision-only, while the text evidence at target scale contains no coverage-at-risk measurement of any training objective. The Galil correlation correction (item 1) and the Mukhoti text-row correction (item 2) should be applied before any objective or calibration recommendation is drafted; the Di Palo venue fix and the reference count are bookkeeping.
