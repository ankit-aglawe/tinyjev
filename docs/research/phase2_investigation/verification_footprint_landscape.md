# Verification report: literature stream C (footprint and landscape)

Phase 2 source verification. Input: `lit_footprint_landscape.md` (bibliography agent, search date 2026-09-22). Verification date: 2026-09-22. This file records what was checked and what was found; it does not synthesise.

Method. Every cited URL was resolved by HTTP (arXiv export API for all 22 arXiv ids; Hugging Face model API + raw `README.md` for every model card; GitHub API + raw `README.md`, `PLAN.md`, `runs/leaderboard.md` for every repository; direct fetch for the IJCAI PDF, ACL Anthology, Apple, Google, archerhume.com and benchmarkheaven.com pages; a browser-class fetch for the Zenodo record, which returns 403 to plain `curl`). All 22 arXiv PDFs were converted to text and their tables read directly; the IJCAI PDF likewise. Every number in the annotated bibliography, the landscape table and the numeric evidence table was grepped against the fetched artefact, and for the priority claims the surrounding table row was read to confirm benchmark and conditions. Retrieved content was treated as data, not instructions.

Tier scheme used below (as specified for this task): **T1** peer-reviewed, venue named and confirmed; **T2** preprint (arXiv, Zenodo); **T3** model card / README / leaderboard (self-reported unless stated); **T4** blog or post. "COI" = conflict of interest between who produced the number and whose model it describes.

---

## (a) Source quality matrix

### Papers and preprints (sections 3, 5)

| ID | Source | Exists? | Tier (verified) | COI flag | Currency / notes |
|---|---|---|---|---|---|
| A1 | Zafrir et al. 2019, Q8BERT, arXiv:1910.06188 | PASS (v2, title matches) | T1, workshop (EMC2 @ NeurIPS 2019, per arXiv comment; lightly reviewed) | none | stable |
| A2 | Kim et al. 2021, I-BERT, arXiv:2101.01321 | PASS (jref "ICML 2021 (Oral)") | T1 | none | stable |
| A3 | Bondarenko et al. 2021, arXiv:2109.12948 | PASS; venue confirmed at aclanthology.org/2021.emnlp-main.627 | T1 | none | stable |
| A4 | Bai et al. 2021, BinaryBERT, aclanthology 2021.acl-long.334 / arXiv:2012.15701 | PASS (both resolve, same title) | T1 | none | stable |
| A5 | Dettmers et al. 2022, LLM.int8(), arXiv:2208.07339 | PASS (comment "Published at NeurIPS 2022") | T1 | none | stable |
| A6 | Zheng et al. 2025, Qwen3 quantization, arXiv:2505.02214 | PASS (v1, 2025-05-04) | T2 | none (not Qwen team) | stable; abstract says "1.8B", tables say "1.7B" |
| A7 | Lee et al. 2025, IJCAI-25 pp. 8113-8121 | PASS (PDF header and proceedings page confirm pages) | T1 | none | stable |
| A8 | Srivastava et al. 2026, arXiv:2507.04023v3 | PASS (v3 2026-04-23) | **T1, not T2**: arXiv comment "Accepted to ACL 2026 Findings" | none | bibliography grades PP; upgrade to Findings-of-ACL |
| A9 | Husom et al. 2025, arXiv:2504.03360 | PASS (v1) | T2 | none | stable |
| A10 | Melton 2026, Zenodo 22698290 | PASS (DOI 10.5281/zenodo.22698290; 403 to plain curl, resolves in a browser-class fetch) | T2 (single author, "American Code Labs", no review) | none | published 2026-09-10; 12 days old |
| A11 | ZeroDegress, NanoJev-mlx-4bit (HF card) | PASS | T3 | third-party quantiser of C-Tianyu's model; evaluated through the upstream harness | created and last modified 2026-09-18; 4 days old |
| A12 | yzfly/edgejev (GitHub README) | PASS | T3 | edgejev author converts and measures other people's models; has an interest in the INT8 path working | created 2026-09-20, pushed 2026-09-21; 2 days old |
| B1 | Turc et al. 2019, arXiv:1908.08962 | PASS | T2 | none | stable |
| B2 | Jiao et al. 2020, TinyBERT, arXiv:1909.10351 | PASS (comment "Findings of EMNLP 2020") | T1 | none | stable |
| B3 | Sun et al. 2020, MobileBERT, arXiv:2004.02984 | PASS (comment "Accepted to ACL 2020") | T1 | none | stable |
| B4 | Wang et al. 2020, MiniLM, arXiv:2002.10957 | PASS; venue confirmed at papers.nips.cc (NeurIPS 2020) | T1 | none | stable |
| B5 | Xia et al. 2022, CoFi, arXiv:2204.00408 | PASS (comment "Accepted to ACL 2022") | T1 | none | stable |
| B6 | Men et al. 2024, ShortGPT, arXiv:2403.03853 | PASS (v3) | T2 | none | stable |
| C1 | Kusupati et al. 2022, MRL, arXiv:2205.13147 | PASS (paper header "NeurIPS 2022") | T1 | none | stable |
| C2 | Devvrit et al. 2024, MatFormer, arXiv:2310.07707 | PASS (comment "NeurIPS, 2024") | T1 | none | stable |
| C3 | Google, Gemma 3n developer guide (blog) | PASS (dated June 26, 2025) | T4 (vendor) | vendor describing own product | stable |
| C4 | Fan et al. 2020, LayerDrop, arXiv:1909.11556 | PASS | T1 per bibliography (ICLR 2020); OpenReview page is bot-blocked, so the venue was **not independently confirmed** | none | stable |
| C5 | Zhou et al. 2020, PABEE, arXiv:2006.04152 | PASS (comment "NeurIPS 2020") | T1 | none | stable |
| C6 | Xin et al. 2020, DeeBERT, arXiv:2004.12993 | PASS (comment "Accepted at ACL 2020") | T1 | none | stable |
| D1 | Apple ML Research, ANE transformers (2022) | PASS | T4 (vendor, with released code) | vendor measuring own hardware | stable |
| D2 | Liu et al. 2024, MobileLLM, arXiv:2402.14905 | PASS (comment "ICML 2024") | T1 | Meta measuring on iPhone; no product COI | stable |
| D3 | Zhang & Huang 2025, arXiv:2505.06461 | PASS (v1) | T2 | none | stable |
| D4 | john-rocky/apple-silicon-llm-bench (GitHub) | PASS | T3 (reproducible harness; raw JSONL per run) | single author; no model COI | repo created 2026-05-02, pushed 2026-09-08; cells span June-Aug 2026 sessions and **the summary table now disagrees with the section table** (see spot-check) |
| D5 | Warner et al. 2024, ModernBERT, arXiv:2412.13663 | PASS | **T1, not T2**: aclanthology.org/2025.acl-long.127 (ACL 2025) | authors are the model's creators (throughput numbers are self-measured) | bibliography grades PP; upgrade |
| D6 | jaredpalmer/kev README; HF kev-0.6b; HF kev-0.8b | PASS (all three) | T3 | author reports own model; Jev numbers on Kev suites measured by Kev's author via Vercel AI Gateway | kev-0.6b card modified 2026-09-20; kev-0.8b 2026-09-21; repo pushed 2026-09-22 00:44 (13 h after the leaderboard was generated) |
| D7 | FluidInference/laya-coreml (HF card) | PASS | T3 | third-party converter of Laya; parity gates are its own | created 2026-09-21, modified 2026-09-22 04:43; **hours old** |
| D8 | mpnikhil/dev-0.4b (HF card) | PASS | T3 | self-report | created 2026-09-20, modified 2026-09-22 00:00 |
| D9 | Heman10x-NGU/Verdict-open-jev (GitHub); heman10x/rlcd-modernbert-151m (HF) | PASS (both) | T3 | self-report; "external TypeSafe eval" run by the author | card modified 2026-09-20; repo pushed 2026-09-20; Verdict 2.0 weights not yet downloadable per benchmarkheaven |
| D10 | iapp/OpenThai-SystemOne (HF card) | PASS | T3 | self-report; Nimble-9B and Jev columns "as published by Bespoke Labs", not measured by iapp | card v0.3 dated 2026-09-22; macro score moved 63.2 -> 74.3 between v0.2 and v0.3 within days |
| D11 | lostargon/Tiny-Jev (HF card) | PASS | T3 | self-report | created and modified 2026-09-21 |

### Landscape sources (section 4) not already listed

| Source | Exists? | Tier | COI flag | Currency / notes |
|---|---|---|---|---|
| benchmarkheaven.com/jev-models (jevbench v1.3.0) | PASS (5.3 MB page; rows read from `aria-label` attributes) | T3, **independently measured** by the benchmark author (fstandhartinger) | independent of every model; self-hosted endpoints carry a x2 + 0.15 s latency *assumption*; kev family run in BF16 on the author's RTX 3090 | jevbench repo pushed 2026-09-21 22:59; v1.3.0 |
| fstandhartinger/jevbench README, RESULTS-v1.2.md | PASS | T3 (IM) | as above | 3 days old |
| jaredpalmer/kev PLAN.md, runs/leaderboard.md | PASS | T3 | author's own research log and trial table | leaderboard "Generated 2026-09-21T11:34+00:00"; PLAN.md carries a 2026-09-20 release note and 2026-09-21 deltas |
| HF datasets/jaredpalmer/kev-suites | PASS (API 200); datasets-server viewer returns "The dataset generation failed ... Couldn't cast array" (the schema error the bibliography describes is real) | T3 | - | github.com/jaredpalmer/kev-suites returns 404 as stated |
| jaredpalmer/kev-0.5b (HF card) | PASS | T3 | self-report; card credits Hume's post for the architecture | modified 2026-09-20 |
| Mapika/decider-0.8b | PASS | T3 | self-report; teacher labels from Qwen3.5-27B | created 2026-09-19 |
| C-Tianyu/NanoJev | PASS | T3 | self-report | created 2026-09-17, modified 2026-09-20; **weights carry no declared licence** (code MIT) |
| rongxinzy/LightJev-0.6B-v0.1 | PASS | T3 | self-report; single seed | created 2026-09-18; 0 downloads |
| anthonym21/qwen3-0.6b-rlcd-decision | PASS | T3 | self-report | created 2026-09-19 |
| samatv256/mini-Jev | PASS | T3 | self-report | created 2026-09-21 |
| dwidlee/systemone-lite-0.5b | PASS | T3 | self-report | modified 2026-09-20 |
| com-kotobalabs/open-jev-deberta-v3-large | PASS | T3 | self-report | created 2026-09-18 |
| convaiinnovations/laya; convaiinnovations/laya-multilingual | PASS (both) | T3 | self-report; the Jev columns in Laya's comparison table are AbdelStark's third-party numbers, not Laya's own measurements (PLAN.md also notes this) | laya modified 2026-09-20; multilingual 2026-09-19 |
| logan-markewich/jeff | PASS | T3 (wrapper) | none | created 2026-09-19 |
| kaivoss/system-one-270m | PASS | T3 | self-report | created 2026-09-21 |
| akash-kamat/system-one-gemma (GitHub) | PASS | T3 | self-report | pushed 2026-09-18; GitHub reports **no licence file** |
| mithalouni/system-one-open (GitHub) | PASS | T3 | self-report | pushed 2026-09-17; README says MIT; 270M weights "HF upload pending" |
| aaroncool9/laya-vision-smolvlm-256m | PASS | T3 | self-report | created 2026-09-20 |
| nampham1106/laya-flash | PASS | T3 | - | created 2026-09-22 05:15 UTC (hours old); head is "freshly initialized and untrained" as stated |
| rupeshpoojary9/poorjev | PASS | T3 | self-report | pushed 2026-09-21 |
| us/jev-local | PASS | T3 (wrapper) | - | pushed 2026-09-18 |
| xingwudao/OpenJev | PASS | T3 (mock) | - | pushed 2026-09-18; no licence detected |
| kshetrajna12/reflex; vagmi/jev-lite | PASS (both) | T3 | self-report | scale-only rows |
| AbdelStark/jev-benchmarks | PASS | T3 (IM, 300-example pilot) | independent | pushed 2026-09-17 |
| heyjunpenn/awesome-jev | PASS | T3 (list) | - | pushed 2026-09-22 09:28 |
| archerhume.com, "Jev's Architecture Unmasked" | PASS (dated 17 September 2026) | T4 (independent black-box study with released evidence bundle) | none | 5 days old |
| rockyshikoku.medium.com (D4 write-up) | 403 (already excluded by the bibliography) | - | - | correctly excluded |
| Radexito/kev; Xubqpanda/nanojev | 401 / 301 (both already excluded) | - | - | correctly excluded |

**Existence summary.** 66 distinct cited sources (74 URLs) checked. 0 failures among sources that the bibliography actually cites. 2 gray-zone resolutions: Zenodo (A10) blocks plain HTTP clients but resolves normally; LayerDrop (C4) exists on arXiv but its ICLR 2020 venue could not be independently confirmed because OpenReview is bot-blocked. No DOI/title mismatches. Two grades are too low (A8 and D5 are peer-reviewed).

---

## (b) Claim spot-check table

Verdicts: **CONFIRMED** (number, benchmark and conditions match); **CONFIRMED + CAVEAT** (number matches, a material condition is missing from the bibliography); **MISQUOTED** (number or attribution wrong; correct value given); **UNVERIFIABLE** (not in the cited artefact).

### Quantization (priority)

| # | Claim as cited | Source | Verdict | Detail |
|---|---|---|---|---|
| 1 | Qwen3-0.6B FP16 MMLU 47.1, zero-shot avg 52.9, Wiki2 ppl 20.9; 8-bit RTN/GPTQ/AWQ 47.0/47.0/46.9; 4-bit RTN/GPTQ/AWQ 37.3/40.0/43.1, ppl 37.5/33.0/25.8; 3-bit AWQ 26.4; Qwen3-1.7B 60.0 -> AWQ 53.9 / GPTQ 52.8 / RTN 47.9 | A6, Table 2 | **CONFIRMED + CAVEAT** | Every number matches Table 2 ("2 to 8-bits **per-channel** PTQ results of Qwen3 Models", i.e. the post-trained model, no weight grouping). The bibliography does not say per-channel. Table 4 (per-group, g=128) gives materially better 4-bit numbers for 0.6B: GPTQ MMLU **44.0** (ppl 25.3), AWQ **42.1** (ppl 26.9); 3-bit AWQ g128 collapses to 25.0. Since MLX's default is g=64, the per-group table is the relevant one for a deployment estimate and should be cited alongside. |
| 2 | Q8BERT QAT: CoLA 58.48->58.48, MRPC 90.00->89.56, QNLI 90.30->90.62, QQP 87.84->87.96, RTE 69.70->68.78, SST-2 92.36->92.24, STS-B 89.62->89.04, SQuAD F1 88.46->87.74; dynamic PTQ SQuAD 80.02, RTE 63.32, QQP 84.98 | A1, Table 1 | **CONFIRMED** | Exact (paper prints 90, 90.3, 69.7 without trailing zeros). Standard deviations are in the table (RTE QAT sd 3.52, DQ sd 4.58), which supports the bibliography's "small-task variance not controlled" caveat. |
| 3 | I-BERT RoBERTa-base GLUE avg 86.0->86.3; MNLI-m 87.8->87.5; SST-2 94.6->95.2; RTE 78.0->79.4; large 89.0->89.5; T4 INT8 speedup 2.42-3.39x, avg 3.08x | A2, Tables 2/3 | **CONFIRMED** | Exact. |
| 4 | Bondarenko BERT-base GLUE avg 83.06; W8A8 per-tensor PTQ 71.03; per-embedding-group 82.45; W8A8 QAT 83.26; W4A8 QAT 82.64; W4A32 QAT 82.95 | A3 | **CONFIRMED** | Exact. |
| 5 | BinaryBERT MNLI-m 84.6 -> 83.5 (TernaryBERT) -> 84.2 (W1A8) -> 83.9 (W1A4); SQuAD 80.8/88.5 -> 80.8/88.3 -> 79.3/87.2; 418 MB -> 17 MB (24x) | A4, Table 4 | **CONFIRMED** | Exact; paper's notation is "1-1-8" / "1-1-4" (weight-embedding-activation bits) and the ratio is 24.6x. |
| 6 | LLM.int8(): OPT-125M C4 ppl 25.65 / 87.76 (absmax) / 25.83; 2.7B 14.43 / 15.11 / 14.44 | A5, Table 1 | **CONFIRMED** | Exact. |
| 7 | IJCAI-25: Llama-3.2-1B-it FP16 42.06; FP8 41.98; SmoothQuant 42.06; AWQ 38.48 (-3.58); GPTQ 34.90 (-7.16); 3B 52.13 -> 50.26 / 50.51; GPTQ-1B -25.32 GSM8K, -16.01 IFEval; pp. 8113-8121 | A7, Table 1 | **CONFIRMED** | Exact, including page range. |
| 8 | Qwen2.5-0.5B 21.31% -> 12.77% at 4-bit (~40% rel.); 1.5B 43.03%, 3B 45.75% "modest" | A8, Table 2 | **CONFIRMED + CAVEAT** | Numbers exact. The models are Qwen2.5-**Instruct** ("Qwen2.5 (I)"); 8-bit is lossless (21.29). The paper's word "modest" refers to 7B-14B (3-5% relative); 1.5B and 3B lose ~8% relative at 4-bit (43.03->39.42, 45.75->41.94), which the bibliography's phrasing understates. |
| 9 | Husom, Raspberry Pi 4: Qwen2.5-0.5B avg accuracy FP16 0.32, Q8_0 0.28, Q4_1 0.35, Q4_0 0.30, Q3_K_M 0.33 (sd ~0.40); energy 4.42 / 2.14 / 2.44 / 2.30 J/token | A9 | **CONFIRMED** | Accuracy row: 0.32+-0.40, 0.28+-0.40, 0.35+-0.42, 0.30+-0.39, 0.33+-0.41 (instruct GGUFs via Ollama). Energy averages exact. |
| 10 | Melton: 220 discordant pairs favour bf16 vs 125; +0.52 pt per billion params; calibrated AWQ arm does not recover the gap; five code models | A10 abstract | **CONFIRMED** | Abstract matches verbatim (HumanEval+ / MBPP+, McNemar p < 1e-6). Model list and any sub-1B row remain unconfirmed, as the bibliography says. |
| 11 | NanoJev MLX affine 4-bit g64, head fp16; 335.86 MB (14.1% of fp32); argmax agreement 99.17% (357/360); mean KL 2.91e-4; mean TV 0.0066; 0 boolean flips /120; teacher CE 0.4981 vs 0.4992; 4-bit matmul 2.96 vs 2.99/3.60 TFLOPS | A11 card | **CONFIRMED + CAVEAT** | All numbers exact (335,857,708 bytes; "3 of 360 questions flip"; 0.498122 vs 0.499212). **Condition the bibliography omits**: the card states the artefact was *dequantized back to fp32 and evaluated through the unmodified upstream PyTorch inference path* on an M4, so the figures isolate weight-quantization error and exclude half-precision activation error of a real MLX runtime; the split is the frozen `stage2/dev.jsonl` (a development split, 120 states). Calling it "the only found MLX 4-bit measurement" overstates it: it is an MLX-format *weight* measurement, not an MLX-runtime measurement. |
| 12 | edgejev, 4 vCPU Cascade Lake: Laya 322M FP32 32.1 ms / 92.8% / 54.0%; INT8 15.6 ms / 91.2% / 48.2%; 1,290 -> 324 MB; kev FP32 44 ms / 90% / 44% (n=100), INT8 26 ms / 84% / 21%; NanoJev 155 -> 69 ms | A12 README | **CONFIRMED + CAVEAT** | All figures exact (n=400 for Laya, n=100 for kev, batch 1). Caveat: the README's own INT8-variant ablation table reports the *same* Laya fp32 weights at Emotion **47.0%** and int8 per-tensor at **52.2%** (and 90.5% AG News), i.e. the Emotion INT8 delta reverses sign between two runs and is inside noise; the AG News delta (-1.6 to -2.3 pts) is stable. On the n=100 kev subset, Laya scored fp32 94.0/57.0 and int8 91.0/53.0. |
| 13 | Laya-multilingual CoreML FP16, M5 Pro CPU+ANE: L128 3.6 ms, L256 9.9, L512 27.5, L1024 80.1; all units L512 9.0 ms; 16/16 argmax; max prob error 0.0021; AG News 0.935 vs 0.930; int8-embedding variants 448-453 MB within 0.5 pt | D7 card | **CONFIRMED + OMISSION** | Exact (0.0021 is for `ALL` compute units; `CPU_AND_NE` max error is 0.0126). **Omitted negative result that bears directly on SQ5**: the card states "Encoder-weight int8 and 6-/4-bit palettes fail the parity gates (the ANE in particular), so they are not published." Only the int8 *embedding-table* variant passed. |
| 14 | Verdict FP16 vs FP32 accuracy delta 0.00% | D9 README, E8 | **CONFIRMED** | "FP32 test Acc 94.50% / FP16 94.50%; max delta across all splits 0.00%". |

### Pruning, nesting, early exit

| # | Claim as cited | Source | Verdict | Detail |
|---|---|---|---|---|
| 15 | CoFi ~95% sparsity: SST-2 90.6 (12.0x), QNLI 86.1 (12.1x), MNLI 80.6 (12.1x vs 84.8), QQP 90.1 (11.0x), RTE 64.7, STS-B 83.1, MRPC 82.6, SQuAD 82.6 F1 (8.7x); <=20 vs ~350 GPU-hours | B5, Table 1 | **CONFIRMED** | Exact. |
| 16 | MatFormer: 850M MatLM yields 582M-850M submodels "each with better validation loss and one-shot downstream" than independently trained models | C2 | **CONFIRMED (wording)** | Paper: "match or even perform better than the independently trained baselines". "Each with better" slightly overstates; "match or better" is the paper's claim. |
| 17 | LayerDrop: RoBERTa 12 -> 6 layers MNLI-m 82.9, MRPC 85.3, QNLI 89.4, SST-2 92.5 vs DistilBERT 81.6/82.4/85.5/92.7; WikiText-103 16 -> 8 layers ppl 20.78 / 20.56 | C4 | **CONFIRMED** | Exact. |
| 18 | PABEE: ALBERT-base 1.57x, macro 84.4 -> 85.1; BERT-base 1.62x, MNLI 84.5 -> 83.6, SST-2 92.1 -> 92.0, STS-B 88.9 -> 88.7 | C5, Tables 1/2 | **CONFIRMED** | Exact (dev set). |
| 19 | DeeBERT: BERT-base savings SST-2 21%, QQP 24%, MNLI 14% (9-24%); RoBERTa 19-32%; ~40% at 2-4 pt loss | C6, Table 1 | **CONFIRMED** | Exact. |
| 20 | Turc: PD 84.4 dev / 82.1 test vs PF 82.8 / 81.6 vs DistilBERT 82.3 vs PKD 81.7; sizes 4.4M-110.1M | B1 | **CONFIRMED** | Exact. |
| 21 | ShortGPT ~25% layers removed retains 86.3 / 91.6 / 85.1 / 90.4% | B6 | **CONFIRMED** | 86.31 (Llama-2-7B, 27.1%), 91.64 (13B, 24.6%), 85.10 (Baichuan2-7B, 24.2%), 90.42 (Mamba-2.8B, 25%). |
| 22 | TinyBERT4 14.5M >96.8%, 7.5x smaller, 9.4x faster; TinyBERT6 67M on par | B2 | **CONFIRMED** | Exact (67.0M). |
| 23 | MobileBERT 25.3M, 4.3x smaller, GLUE 77.7 (-0.6), SQuAD 90.0/79.2, 62 ms Pixel 4 | B3 | **CONFIRMED** | Exact. |
| 24 | MiniLM 12x384 33M, >99% of teacher, 50% params/FLOPs; MRL up to 14x, +2% long-tail | B4, C1 | **CONFIRMED** | Exact. |

### On-device latency (priority)

| # | Claim as cited | Source | Verdict | Detail |
|---|---|---|---|---|
| 25 | DistilBERT-SST-2, FP16, seq 128, batch 1, iPhone 13: 3.47 ms @ 0.454 W; 9.44 ms @ 0.072 W; up to 10x faster, 14x less memory | D1 | **CONFIRMED** | Exact ("the iPhone 13 ANE achieves an average latency of 3.47 ms at 0.454 W and 9.44 ms at 0.072 W"). |
| 26 | MobileLLM-125M, iPhone 13, ExecuTorch+MPS: load 39.2 ms, init 1361.7 ms, 15.6 ms/token (~64 tok/s); LS +2.6%; zero-shot 46.3 (125M) / 51.3 (350M) vs OPT 42.6 / 43.9 | D2, Tables 5/3 | **CONFIRMED + CLARIFICATION** | Latency and accuracy exact. "+2.6%" is the **execution-time overhead** of the layer-sharing variant, not an accuracy gain (LS-125M accuracy is 47.0, +0.7 pt). |
| 27 | iPhone 15 Pro, llama.cpp: Llama-3.2-1B F16 CPU 2 threads 17 tok/s vs GPU 12.8; Qwen2-0.5B / Llama-3.2-1B Q4_1/F16 CPU 1.31-1.33x GPU; matmul 87.6% of prefill | D3 abstract | **CONFIRMED** | Exact. |
| 28 | apple-silicon-llm-bench, Qwen3-0.6B, iPhone 17 Pro, short-chat: Core AI GPU 193.3, MLX 158.8, LiteRT-LM 120.4, CoreML/ANE 37.7 tok/s; numeric table labels all four "4-bit (per repo tags)" | D4 README | **MISQUOTED (conditions) + CURRENCY** | Numbers exist, but not under one condition. Per the repo's provenance list: Core AI 193.3 = INT4 dynamic, **cold**, June 2026 session; MLX 158.8 = Q4, warm, 2026-07-13 session; LiteRT 120.4 = INT4 blockwise, warm; **CoreML/ANE 37.7 = INT8 palettized, cold, single launch** (not 4-bit). The repo's summary table now shows MLX **178.8** and LiteRT **122.1** from a 2026-08-26 session and Core AI ANE variants at 122.4 / 116.9; the bibliography quotes the older section table. Use "iPhone 17 Pro, mixed sessions June-Aug 2026, per-cell quantization" and correct the CoreML cell to INT8. |
| 29 | Qwen2.5-0.5B, M4 Max: MLX-Swift 531.1 tok/s, 21 ms TTFT, 390 MB; llama.cpp Q4_K_M 297.1, 22 ms, 538 MB; CoreML/ANE 181.2, 171 ms, 962 MB. Qwen3.5-0.8B: 421.1 / 36 / 600; 201.1 / 22 / 752; 58.2 / 405 / 221 | D4 README | **CONFIRMED + CAVEAT** | All numbers exact. The Qwen2.5-0.5B CoreML cell is **FP16** (hence 962 MB) and the Qwen3.5-0.8B CoreML cell is **INT8**; the evidence table's "CoreML-ANE" column should carry those precisions. |
| 30 | Gemma 4 E2B, iPhone 17 Pro: LiteRT-LM 61.1 tok/s / 497 MB; MLX-Swift 49.1 / 3,010 MB; llama.cpp Q4 38.8 / 191 MB | D4 README | **CONFIRMED** | Exact; LiteRT arm is wNa8o8 QAT, MLX is PTQ 4-bit, llama.cpp Q4_K_M. |
| 31 | Kev on Apple M5, bf16, five 3-option questions, ~230-token state: Kev-0.8B 329 ms, 4B 779 ms, 9B ~2 s; Qwen3 generation 123 / 174 / ~300 ms; Kev-0.6B card 0.12 s | D6 README, kev-0.6b card | **CONFIRMED** | Exact ("about 2 s", "about 300 ms"; card: "0.12 s per five-question request vs 0.33 s for Kev-0.8B"). |
| 32 | dev-0.4b: M1 Max MPS ~27.6 ms; CUDA FP16 SDPA ~10 ms; Banking77 (300) 91.33 / BoolQ (500) 85.2 / Yelp (300) 62.7; ECE 5.5 / 7.7 / 15.5% temp-scaled | D8 card | **CONFIRMED** | Exact (0.055 / 0.077 / 0.155 calibrated; raw 0.075 / 0.103 / 0.318). |
| 33 | Verdict WASM single-thread K=5: p50 35.58 ms, p95 39.81 ms | D9 card | **CONFIRMED** | Card table; p95 is in the card, not the README. |
| 34 | OpenThai: M3 Max ~154 ms (3-question, 166-token ticket); H100 ~44 ms (255 options); ~41 ms/decision Doom; "framework on Mac unstated" | D10 card | **CONFIRMED; one correction** | Numbers exact. The card *does* state the Mac framework: "154 ms on a MacBook M3 Max (**MPS**)". |
| 35 | Tiny-Jev ~5 ms on a consumer GPU, ~20-50 ms on Apple M-series | D11 card | **MISQUOTED (minor)** | Card: "On a consumer GPU a call takes **a few milliseconds**; on an Apple M-series laptop ~20-50 ms." Replace "~5 ms" with "a few ms". |
| 36 | ModernBERT-base 149M / large 395M; GLUE 88.4 / 90.4 (DeBERTaV3 88.1 / 91.4); RTX 4090 147.3K / 52.9K tok/s at 512 tokens; max batch 1,604 / 770 | D5 | **CONFIRMED** | Exact. |
| 37 | Laya (ModernBERT-large, 421M) on T4: "32.8-39.5 ms / 72.3-158.6 ms" (evidence table) | convaiinnovations/laya card | **MISQUOTED (attribution)** | The card gives two backbones: ModernBERT-large **39.5 ms** (1 q) / **158.6 ms** (10 q) and mmBERT-base multilingual (322M) **32.8 ms** / **72.3 ms**. The evidence row assigns both to the 421M model; split into two rows. |

### Landscape table (priority: jevbench, Kev suites, Kev-0.5B/0.6B)

| # | Claim as cited | Source | Verdict | Detail |
|---|---|---|---|---|
| 38 | Jev 1.13.0 jevbench v1.3.0: score 74.4, Int 85.7, Cal 82.7 | benchmarkheaven | **CONFIRMED** | "Jev 1.13.0 (TypeSafe AI): 74.4, rank 1. Intelligence 85.7, calibration 82.7". |
| 39 | Kev-0.6B jevbench rank 19, 62.5 (Int 51.9, Cal 51.1) | benchmarkheaven | **CONFIRMED** | Exact; run "BF16 on an RTX 3090 ... measured serially from Sandy over the internet", with the x2 + 0.15 s adjustment. |
| 40 | Kev-0.5B jevbench rank 38, 33.2 (Int 38.2, Cal 47.4); Laya rank 33, 54.4 (45.8 / 62.5); jeff rank 32, 54.4 (46.9 / 64.6); Verdict 1.4 rank 36, 38.9 (38.6 / 74.1); Verdict 1.0 rank 37, 38.1 (39.8 / 51.3); GLiNER2.5 multi rank 43, 16.6 (27.7 / 56.1); GLiNER2.5 small rank 44, 13.8 (25.6 / 47.2); reflex 70.3; SemIf 73.1 | benchmarkheaven | **CONFIRMED** | All exact. jeff is "GLiFormer 400M" (knowledgator/gliformer-large-v1), consistent with the table. |
| 41 | Kev-0.6B decision-v7 dev 0.801 / locked test 0.808; ECE "0.086-0.089 (dev)" | kev-0.6b card | **CONFIRMED; one label error** | 0.801 and 0.808 exact. ECE 0.086 is dev; **0.089 is the locked test** (Brier 0.266), not dev. Also note the card's own model-index names the 1,204-record dev set "**decision-v4** development" while the kev-0.8b card's table calls the same set "decision-v7 dev"; the training suite is decision-v7. Source-internal inconsistency; carry as "decision-v4/v7 dev (1,204 records)". |
| 42 | Kev-0.6B transfer-v4 dev 0.613 / 0.605 / 0.620 (seeds), locked 0.642; Brier 0.536; conf-err 0.108 dev, ~7.9% test | kev-0.6b card; leaderboard row v7-06b/02-trial-2 | **CONFIRMED** | Exact (0.108 is the leaderboard's conf-err column; card says "11%" rounded; locked test 7.9%, Brier 0.483, ECE 0.128). |
| 43 | Kev-0.8B decision dev 0.825 / locked 0.834; Brier 0.268, ECE 0.100; transfer dev 0.652 / locked 0.684; Brier 0.499 / 0.460; ECE 0.154; LoRA r=16, 11.3M | kev-0.8b card | **CONFIRMED** | Exact; 0.684 is post-delta (pre-delta locked 0.668, which is the figure the kev-0.6b card header quotes). |
| 44 | Kev-0.5B v0.1: 0.799; ECE 0.065 (0.031 after T=1.47); 13,500 questions; 9.3M trainable; 896->256 head; transfer-v4 dev 0.561 | kev-0.5b card | **CONFIRMED** | Exact (9,000 records / 13,500 questions). |
| 45 | Leaderboard ablation-v2/04 (Qwen2.5-0.5B): 0.742 / 0.605 / Brier 0.501; leaderboard generated 2026-09-21; only Qwen bases | runs/leaderboard.md | **CONFIRMED** | Exact; "Generated 2026-09-21T11:34+00:00"; no encoder rows. |
| 46 | Jev on Kev suites: decision dev 0.845, transfer-v4 dev 0.857, Brier 0.211; transfer-v9 knowable 0.854 / MMLU-Pro 0.840; SUITES_REVISION a957287d; Laya 39.5 / 158.6 ms; SemIf 0.747 | PLAN.md; cards | **CONFIRMED** | All present in PLAN.md (transfer-v9 row "Jev (live, $0.02) 0.854 / 0.840"). Note: 0.845 / 0.857 are **not** in `runs/leaderboard.md` (it lists local trials only); they are in the cards and PLAN.md. PLAN.md is internally ambiguous on whether the 0.747 SemIf probe used Qwen3.5-4B (run path) or 9B (text). |
| 47 | Note 2: "PLAN.md's family table lists Kev-0.6B transfer-v4 dev at 0.598, while the card lists 0.613/0.605/0.620" (flagged as a discrepancy) | PLAN.md release note | **MISQUOTED (misread)** | Not a discrepancy. PLAN.md: "three v7 seeds 0.613 / 0.605 / 0.620, all above **the previous 0.598**", and elsewhere "Kev-0.6B dev 0.805/0.598 -> test 0.819/0.631" refers to the superseded `v4-06b-hardened/00-trial-0` checkpoint. 0.598 is the prior checkpoint, not the shipped one. Delete note 2 or reword. |
| 48 | Laya own typed decisions (2,000) 0.766; Brier 0.062; ECE 0.213; Banking77 0.425 / AG News 0.950 / Emotion 0.595 / XNLI-en 0.860 / MASSIVE-en 0.783; ECE 0.081 post-temperature | laya card | **CONFIRMED + CAVEAT** | Exact. The 0.766 / 0.062 / 0.213 row belongs to the separate `laya-typed-decisions` checkpoint (1024 ctx), not the root `laya` repo; the Jev comparison column is AbdelStark's data. |
| 49 | Laya-multilingual: "MASSIVE (13 langs) / XNLI (14 langs): 0.451 / 0.731" | laya-multilingual card | **MISQUOTED** | XNLI "14 other languages" 0.731 is exact. The MASSIVE figure in the card is **macro accuracy 0.366 across all 51 MASSIVE languages** (English checkpoint 0.227; 45/51 languages above 3x random). No "0.451" and no 13-language set appears in the card. |
| 50 | Verdict: 231-task jevbench public run easy/standard/hard 87.5 / 69.4 / 36.9 (v1.4); hard ECE 0.118; own held-out 1,000 K=5 95.00%, Brier 0.0785, ECE 3.35%; TypeSafe external 337 cases 48.07 vs Jev 90.80 | D9 README + card | **CONFIRMED** | Exact (48 / 72 / 111 tasks). |
| 51 | OpenThai own 13-subset bench, 3,881 items, macro 74.3 (Nimble-9B 74.8); ECE 0.035-0.371 | D10 card | **CONFIRMED** | Per-subset n sums to exactly 3,881; ECE range spans paws 0.035 to helpsteer2 0.371; Jev column 76.0. |
| 52 | Tiny-Jev SST-2 / AG News / Emotion / Banking77 90.4 / 90.5 / 82.2 / 81.2; ECE 0.023 / 0.031 / 0.054 / 0.036; held-out 830: 53.1, ECE 0.299; data "~100k synthetic states" | D11 card | **CONFIRMED + CAVEAT** | Numbers exact. **Condition omitted**: "Tiny-Jev was additionally fitted on 2,000 items per dataset that are disjoint from the evaluation items" (open-system-one protocol allowance). The public-benchmark row is therefore *few-shot-fitted*, not a synthetic-only zero-shot result; the landscape "Data" cell should say so. |
| 53 | decider-0.8b 0.776 / 0.707; ECE 0.032 / 0.096; 69 / 24 tasks; 1.47M examples; teacher Qwen3.5-27B | card | **CONFIRMED** | Exact. |
| 54 | LightJev test (848) 0.7957 / OOD (448) 0.7244; Brier 0.2662 / 0.3037; ECE 0.0265 / 0.0641; 2,312 records; "3,073-param scorer" | card | **CONFIRMED except head size** | Metrics exact and are "hard accuracy" on the 656 / 352 hard questions (raw, CE arm). The **3,073-parameter** head figure is not in the card (UNVERIFIABLE from the cited artefact). |
| 55 | qwen3-0.6b-rlcd-decision 0.807 [0.798, 0.816]; ECE 0.021; Brier 0.268; 8,000-row test; 26-row letter head; REINFORCE | card | **CONFIRMED** | Exact. |
| 56 | mini-Jev 72.97 / 67.64; 50k synthetic; frozen Qwen3-0.6B; "263k head" | card | **CONFIRMED except head size** | Metrics and data exact; **263k** head parameter count not in the card (UNVERIFIABLE). |
| 57 | systemone-lite 43,200 rows, 4 domains; IID 0.781 / Hard 0.733; no ECE | card | **CONFIRMED** | "43 200 rows; 10 800 per gym" (ticket / alloc / debate + chess); iid n=3,600, hard n=5,400. |
| 58 | open-jev-deberta 0.854 / 0.690; Brier 0.213 / 0.399; ECE 0.022 / 0.035; 42k questions; M1 Max 1.8 s (4 q) / H100 28 ms (10 q) / 518 q/s | card | **CONFIRMED** | Exact (M1 Max is CPU fp32; H100 bf16). |
| 59 | system-one-270m 0.6574 (baseline 0.4204); Brier 0.4110; ECE 0.1311 (0.0374 at T=2.0); 25,002 decisions; 2,493 held-out | card | **CONFIRMED** | Exact. |
| 60 | system-one-gemma 64.4; ECE 0.047; Brier 0.454; 12,913 Q; 2.6M / 268M; ~50 ms; "Apache-2.0 / CC-BY-NC part" | README | **CONFIRMED except licence** | Metrics exact. Licence split **UNVERIFIABLE**: GitHub reports no licence file and the README grep finds no licence statement. |
| 61 | system-one-open: 76.7% on TypeSafe strict 343-pair eval is the Gemma 4 E2B variant; 270M has no numbers; "HF upload pending" | README | **CONFIRMED** | Exact (Jev 86.9% on the same subset); repo is MIT. |
| 62 | laya-vision A-OKVQA / ScienceQA / VQAv2 63.1 / 89.0 / 73.2 (75.9 combined); ECE 0.035 (0.108 raw); ~150M trainable; CC BY-NC-SA | card | **CONFIRMED** | Exact. |
| 63 | NanoJev game test Maze 4/10, Snake 8/8, Basic 128/128, PredictPos 27/128 (274 cases); "18,760 game decisions"; "Yes, MIT" | C-Tianyu card | **CONFIRMED except data count and licence** | Per-game results exact; 274 = 10+8+128+128. **18,760** training decisions not in the card (UNVERIFIABLE). Licence: card says "Source code carries the included MIT license"; the MLX card states the upstream **weights carry no declared licence**. Landscape "Yes, MIT" should read "weights: no licence declared (code MIT)". |
| 64 | poorjev 0.781 / 0.656; ECE 0.071 (raw 0.170) / 0.414; n=160 / 154; 55 labelled items; ~400 MB NLI model | README | **CONFIRMED** | Exact. |
| 65 | AbdelStark pilot: AG News 0.910 vs 0.700; Banking77 0.870 vs 0.610; Emotion 0.480 vs 0.440; Brier 0.846 vs 0.668; 300 examples | README | **CONFIRMED + CLARIFICATION** | Exact; on Emotion **Jev** is the one with Brier 0.846 ("Jev is substantially worse calibrated: Brier 0.846 versus 0.668"). The note's ordering is correct but should say so explicitly. |
| 66 | jevbench caveats: x2 (+0.15 s) latency adjustment; "small models are very sensitive to option order"; 534 decisions; "cost is hypothetical provider pricing" | jevbench README | **CONFIRMED (paraphrase)** | First three verbatim. Cost is computed from public tariffs / list prices and labelled "estimated" on the site; "hypothetical" is the bibliography's word. |
| 67 | kev-suites: HF dataset exists but viewer fails with a schema error; GitHub repo 404 | HF datasets-server; GitHub API | **CONFIRMED** | datasets-server: "The dataset generation failed ... Couldn't cast array of type struct<...> to {'row': Value('int64') ...}". |

### Section 6 (Hume post)

| # | Claim as cited | Verdict | Detail |
|---|---|---|---|
| 68 | ~10,000 API calls; 1,029 instrumented; 6,800 benchmark; "541 follow-up" | **MISQUOTED (541)** | 10,000 / 1,029 / 6,800 exact. The post lists follow-up studies of 146 + 311 + 445 + 192 + 148 + 181 + 105 + 35 = **1,563** requests; no "541" appears. |
| 69 | ~30k tokens in ~160 ms -> ~10B active, "an inference, not a measurement"; MoE "can't be observed"; readout final-position vs pointer undecidable; ECE 0.0313 on an MMLU sample | **CONFIRMED** | ECE 0.0313 is on a **1,200-item** MMLU sample (ten-bin). MoE is stated as "I expect"; the 10B figure is qualified exactly as the bibliography says. Post dated 17 September 2026. |

---

## (c) Sources or claims that must be removed or corrected before synthesis

Ordered by how much they could move a recommendation.

1. **A6 (Qwen3-0.6B 4-bit) - add the per-group numbers.** The cited 4-bit losses (RTN 37.3, GPTQ 40.0, AWQ 43.1 MMLU from 47.1) are *per-channel* PTQ. Table 4 (g=128) gives GPTQ 44.0 / AWQ 42.1, roughly halving the GPTQ loss. Any recommendation about "how much 4-bit costs a 0.6B decoder" must quote the grouped numbers, since MLX/llama.cpp deployments use groups.
2. **A11 (NanoJev MLX 4-bit) - relabel the condition.** The 99.17% / KL 2.91e-4 figures were obtained by dequantizing to fp32 and running the PyTorch upstream path on a dev split; they bound weight-quantization error only. Do not describe them as an MLX-runtime measurement. Also correct the NanoJev landscape row: weights have no declared licence (code is MIT).
3. **D7 (laya-coreml) - add the omitted negative finding.** Encoder-weight int8 and 4/6-bit palettes *failed* the ANE parity gates and were not shipped; only the int8 embedding table passed. This is the only direct evidence in the bibliography on sub-8-bit encoders on the ANE and it points the other way from the A1-A4 GPU/CPU results.
4. **D4 (apple-silicon-llm-bench) - fix the precision labels and the session mix.** The CoreML/ANE Qwen3-0.6B cell (37.7 tok/s) is INT8 palettized, cold-start, single launch, not 4-bit; the Qwen2.5-0.5B CoreML cell on M4 Max is FP16 and the Qwen3.5-0.8B CoreML cell is INT8. The iPhone 17 Pro cells come from June-August 2026 sessions and the repo's current summary table shows MLX 178.8 (not 158.8) and LiteRT 122.1. Cite as a range with per-cell provenance, or re-read the summary table.
5. **Landscape row "Laya-multilingual MASSIVE (13 langs) 0.451" - replace** with "MASSIVE, all 51 languages, macro 0.366 (45/51 above 3x random)". The 0.451 figure is not in the card.
6. **Evidence row "Laya (ModernBERT-large) 421M ... 32.8-39.5 ms / 72.3-158.6 ms" - split.** 39.5 / 158.6 ms is ModernBERT-large 421M; 32.8 / 72.3 ms is mmBERT-base 322M. The current row makes the large model look faster than it is.
7. **Tiny-Jev landscape row - add the fit-set condition.** The SST-2 / AG News / Emotion / Banking77 numbers were obtained after fitting on 2,000 items per benchmark dataset. As written ("~100k synthetic states"), the row invites a false zero-shot comparison against Laya's Banking77 0.425 in the direct answer to SQ6.
8. **Note 2 ("PLAN.md lists Kev-0.6B at 0.598") - delete or reword.** 0.598 is the explicitly superseded previous checkpoint; there is no discrepancy between PLAN.md and the card.
9. **Kev-0.6B landscape row - "ECE 0.086-0.089 (dev)"** should read "ECE 0.086 dev / 0.089 locked test"; and the dev-suite label should acknowledge the card's own "decision-v4 development (1,204 records)" naming.
10. **Section 6 - "541 follow-up"** should be "~1,560 follow-up requests across eight studies".
11. **Tiny-Jev "~5 ms on a consumer GPU"** should be "a few milliseconds" (card wording).
12. **D10 OpenThai limitation "framework on Mac unstated"** is wrong; the card says MPS.
13. **A8 "modest degradation" for 1.5B/3B** should be "~8% relative at 4-bit" (the paper's "modest" applies to 7B-14B); note the models are Qwen2.5-Instruct.
14. **Grades.** A8 -> peer-reviewed (Findings of ACL 2026). D5 -> peer-reviewed (ACL 2025). Keep C4 at ICLR 2020 but mark the venue as not independently confirmed here.
15. **Unverifiable figures to drop or source elsewhere:** LightJev "3,073-param scorer"; mini-Jev "263k head"; NanoJev "18,760 game decisions"; system-one-gemma "Apache-2.0 / CC-BY-NC part". None appears in the cited card or README.
16. **Currency banner for section 4.** Every model card in the landscape table was created between 2026-09-17 and 2026-09-22; OpenThai's headline score moved 63.2 -> 74.3 in one version bump; Verdict has a 2.0 checkpoint whose weights are not yet downloadable; the kev repository was pushed 13 hours after its leaderboard was generated. The landscape should be dated to the hour and treated as a snapshot.

No source needs to be removed for non-existence. No DOI or title mismatches were found.

---

## (d) Overall confidence statement

The bibliography is unusually accurate at the level of individual numbers: of 69 spot-checked claims (well above the 12 required), 57 are confirmed exactly against the primary table or card, 8 are misquoted or mislabelled (one wrong number - Hume's "541"; one figure not in its source - Laya-multilingual "0.451"; one attribution conflation - the Laya latency row; four condition/label errors - D4's CoreML precision, Kev-0.6B's ECE split, the NanoJev licence, the "0.598 discrepancy" misread; one over-precision - Tiny-Jev "~5 ms"), 4 are confirmed but missing a condition that changes their meaning (A6 per-channel vs per-group, A11 dequantized-fp32 evaluation, Tiny-Jev's 2,000-item fit set, laya-coreml's failed int8/4-bit parity gates), and 4 sub-figures are unverifiable from the cited artefact. All 66 sources exist and resolve to the described content; two grades are too conservative. The peer-reviewed encoder-quantization evidence (A1-A5, B5, C4-C6) is stable and can be used as-is. The evidence that would actually drive a footprint recommendation for a 0.1-0.6B decision model - the Qwen3-0.6B 4-bit deltas, the NanoJev MLX agreement figure, the laya-coreml ANE parity result and the iPhone 17 Pro decode numbers - is Tier 2-3, days old, and each item carries a condition the bibliography omitted, so the synthesis should quote those four with their conditions attached and should not treat any single-card figure as settled. The landscape answer to SQ6 ("no sub-0.5B open model beats Kev-0.6B on Kev's suites or on jevbench") is fully supported by the verified numbers, with the caveat that the strongest self-reported sub-0.5B in-domain figures (Tiny-Jev, Verdict, dev-0.4b) all come from cards created within the last five days and, in Tiny-Jev's case, after fitting on the benchmark's own distribution.
