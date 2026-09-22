# Literature stream C: footprint and landscape (sub-questions 5 and 6)

Phase 2 annotated bibliography. Date of search: 2026-09-22. Scope: a <=0.6B (target 0.1-0.3B) typed-decision model in the Jev / System One category, deployable on Apple Silicon (MLX), CPU/CUDA (PyTorch) and plausibly iPhone. This file records evidence only; it does not synthesise or recommend.

Evidence grades used throughout:

| Grade | Meaning |
|---|---|
| PR | peer-reviewed venue |
| PP | preprint / technical report (arXiv, Zenodo) |
| MC | Hugging Face model card (author self-report) |
| RC | GitHub README claim (author self-report) |
| IM | independently measured by a third party (e.g. the jevbench author running someone else's model) |
| BL | blog post |

Every URL below was fetched on 2026-09-22 and confirmed to resolve to the described content. Items that could not be confirmed are listed in section 7 and are not cited anywhere else.

---

## 1. Search strategy

Sources of candidates:

1. Named seeds from the brief (Q8BERT, I-BERT, BinaryBERT, LLM.int8, GPTQ/AWQ small-scale studies, MLX quantization reports, MiniLM/TinyBERT/MobileBERT, MRL/MatFormer/LayerDrop/early-exit BERT, ModernBERT/DeBERTa/SmolLM2/Qwen-0.5B on phones).
2. WebSearch (11 queries) for the seed terms plus "quantization hurts small models", "iPhone llama.cpp tokens per second", "MLX Swift iPhone", "CoreML Neural Engine BERT", "Gemma 3n MatFormer".
3. For each hit, WebFetch of the arXiv abstract page, then the ar5iv or arXiv-HTML full text to extract tables. One PDF (IJCAI-25) was read page-by-page.
4. Landscape: WebFetch of `heyjunpenn/awesome-jev` (raw README), `jaredpalmer/kev` (README, PLAN.md, runs/leaderboard.md, release page), HF model cards for kev-0.5b/0.6b/0.8b, `fstandhartinger/jevbench` (README, RESULTS-v1.2.md, benchmarkheaven.com/jev-models), `AbdelStark/jev-benchmarks`, archerhume.com.
5. Hugging Face API sweeps: `?search=jev`, `?search=system-one`, `?filter=decision-model`, `?search=decider`, `?search=nanojev`, `?search=laya`. Every candidate under 1B with a model card was fetched.
6. GitHub READMEs for every "Jev-like model" entry in awesome-jev that plausibly ships weights under 1B (edgejev, poorjev, jev-local, OpenJev, system-one-open, system-one-gemma, Verdict-open-jev, NanoJev, reflex).

## 2. Inclusion / exclusion criteria

Included if all of: (a) the reference resolves at a public URL; (b) it reports at least one *measured* number (accuracy, calibration, latency, memory) on a named model, benchmark and hardware, or it defines a suite/leaderboard used by such numbers; (c) for footprint (SQ5) the model is <=1B or the method is directly applicable to <=1B encoders/decoders; (d) for landscape (SQ6) the model is open-weight (or explicitly not, in which case it is listed to close the gap) and under 1B, or it is the reference point (Jev, Kev-0.6B).

Excluded: pages that returned 403/404; models with no numbers at all; models whose card says the head is untrained (laya-flash); mock servers with no weights (OpenJev); pure API wrappers that add nothing measurable; secondary blog re-reporting when the primary card/README was reachable.

---

## 3. Annotated bibliography, sub-question 5 (footprint)

### 5a. Quantization of sub-1B encoders and decoders to 8-bit and 4-bit

**A1.** Zafrir, O., Boudoukh, G., Izsak, P., & Wasserblat, M. (2019). Q8BERT: Quantized 8bit BERT. *5th Workshop on Energy Efficient Machine Learning and Cognitive Computing (EMC2), NeurIPS 2019*. arXiv:1910.06188. https://arxiv.org/abs/1910.06188
Grade: PR (workshop). Relevance: first systematic INT8 result on a 110M encoder for classification/NLI.
Measured: BERT-base, FP32 vs QAT-INT8 vs dynamic PTQ-INT8. QAT: CoLA 58.48 -> 58.48, MRPC 90.00 -> 89.56, QNLI 90.30 -> 90.62, QQP 87.84 -> 87.96, RTE 69.70 -> 68.78, SST-2 92.36 -> 92.24, STS-B 89.62 -> 89.04, SQuAD-1.1 F1 88.46 -> 87.74. Dynamic PTQ: SQuAD 88.46 -> 80.02, RTE 69.70 -> 63.32, QQP 87.84 -> 84.98. 4x compression.
Limitations: 8-bit only; no 4-bit; no latency on a named device; small-task variance (RTE) not controlled by seeds.

**A2.** Kim, S., Gholami, A., Yao, Z., Mahoney, M. W., & Keutzer, K. (2021). I-BERT: Integer-only BERT quantization. *Proceedings of the 38th International Conference on Machine Learning (ICML 2021)*, PMLR 139. arXiv:2101.01321. https://arxiv.org/abs/2101.01321
Grade: PR (oral). Relevance: fully integer INT8 encoder (no FP for GELU/Softmax/LayerNorm), the regime CoreML/ANE and ONNX-int8 CPUs reward.
Measured: RoBERTa-base GLUE avg 86.0 -> 86.3 (INT8); per task MNLI-m 87.8 -> 87.5, SST-2 94.6 -> 95.2, RTE 78.0 -> 79.4. RoBERTa-large 89.0 -> 89.5. T4 GPU INT8 speedup 2.42-3.39x (base, seq 128/256), average 3.08x.
Limitations: INT8 only; QAT required; speedups on T4 tensor cores, not on ANE/Metal.

**A3.** Bondarenko, Y., Nagel, M., & Blankevoort, T. (2021). Understanding and overcoming the challenges of efficient transformer quantization. *Proceedings of EMNLP 2021*. arXiv:2109.12948. https://arxiv.org/abs/2109.12948
Grade: PR. Relevance: the clearest published evidence that naive per-tensor W8A8 PTQ breaks a 110M encoder, and why (structured residual outliers), plus W4A8.
Measured: BERT-base GLUE avg FP32 83.06; W8A8 PTQ per-tensor 71.03; W8A8 PTQ per-embedding-group 82.45; W8A8 QAT 83.26; W4A8 QAT 82.64; W4A32 QAT 82.95.
Limitations: BERT-base only; no ModernBERT/DeBERTa-v3; QAT needed for 4-bit weights.

**A4.** Bai, H., Zhang, W., Hou, L., Shang, L., Jin, J., Jiang, X., Liu, Q., Lyu, M. R., & King, I. (2021). BinaryBERT: Pushing the limit of BERT quantization. *Proceedings of ACL-IJCNLP 2021*. https://aclanthology.org/2021.acl-long.334/ (arXiv:2012.15701)
Grade: PR. Relevance: lower bound of what a 110M encoder tolerates on NLI.
Measured: BERT-base MNLI-m dev 84.6 (FP32) -> 83.5 (TernaryBERT W2A8) -> 84.2 (BinaryBERT W1A8) -> 83.9 (W1A4). SQuAD-1.1 EM/F1 80.8/88.5 -> 80.8/88.3 (W1A8) -> 79.3/87.2 (W1A4). Model size 418 MB -> 17 MB (24x).
Limitations: needs distillation-aware QAT and ternary-weight splitting; no real-device latency.

**A5.** Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2208.07339. https://arxiv.org/abs/2208.07339
Grade: PR. Relevance: shows that naive absmax INT8 *also* fails at 125M decoders (not only at 6.7B), and that vector-wise + outlier decomposition repairs it.
Measured: OPT-125M C4 perplexity 25.65 (16-bit) vs 87.76 (int8 absmax) vs 25.83 (LLM.int8()). 2.7B: 14.43 / 15.11 / 14.44. Phase shift in outlier features at 6.7B.
Limitations: perplexity, not classification; 8-bit only.

**A6.** Zheng, X., Li, Y., Chu, H., Feng, Y., Ma, X., Luo, J., Guo, J., Qin, H., Magno, M., & Liu, X. (2025). An empirical study of Qwen3 quantization. arXiv:2505.02214. https://arxiv.org/abs/2505.02214
Grade: PP. Relevance: the most directly relevant decoder result: Qwen3-0.6B (the Kev-0.6B / NanoJev / Tiny-Jev backbone) under RTN/GPTQ/AWQ at 8/4/3/2 bits on MMLU.
Measured: Qwen3-0.6B FP16 MMLU 47.1, zero-shot avg 52.9, WikiText2 ppl 20.9. 8-bit: 47.0 (RTN), 47.0 (GPTQ), 46.9 (AWQ). 4-bit: 37.3 (RTN), 40.0 (GPTQ), 43.1 (AWQ); ppl 37.5/33.0/25.8. 3-bit AWQ: 26.4 MMLU (chance). Qwen3-1.7B: MMLU 60.0 -> 4-bit AWQ 53.9, GPTQ 52.8, RTN 47.9.
Limitations: generative/MC benchmarks, not a decision head; no calibration metrics; no MLX quantizer.

**A7.** Lee, J., Park, S., Kwon, J., Oh, J., & Kwon, Y. (2025). Exploring the trade-offs: Quantization methods, task difficulty, and model size in large language models from edge to giant. *Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI-25)*, 8113-8121. https://www.ijcai.org/proceedings/2025/0902.pdf
Grade: PR. Relevance: controlled 1B-405B sweep (GPTQ, AWQ, SmoothQuant, FP8) with explicit "small models suffer more at 4-bit" finding.
Measured (Table 1, OpenLLM-v1 avg): Llama-3.2-1B-it FP16 42.06; FP8 41.98; SmoothQuant W8A8 42.06; AWQ 4-bit 38.48 (-3.58); GPTQ 4-bit 34.90 (-7.16). Llama-3.2-3B-it 52.13 -> GPTQ 50.26, AWQ 50.51. GPTQ-4bit on 1B: -25.32% GSM8K, -16.01% IFEval. AWQ > GPTQ throughout; FP8 most stable.
Limitations: smallest model is 1B; instruction-tuned generative tasks; no 0.1-0.6B rows.

**A8.** Srivastava, G., Hussain, A., Srinivasan, S., & Wang, X. (2026). Do LLMs overthink basic math reasoning? Benchmarking the accuracy-efficiency tradeoff in language models. arXiv:2507.04023v3. https://arxiv.org/abs/2507.04023
Grade: PP. Relevance: gives the widely quoted "Qwen2.5-0.5B loses ~40% at 4-bit" number.
Measured: Qwen2.5-0.5B 21.31% (full precision) -> 12.77% (4-bit), ~40% relative loss; 1.5B 43.03% and 3B 45.75% with "modest" degradation.
Limitations: arithmetic reasoning only; quantizer and calibration details thin; absolute accuracies near floor, so relative loss is inflated.

**A9.** Husom, E. J., et al. (2025). Sustainable LLM inference for edge AI: Evaluating quantized LLMs for energy efficiency, output accuracy, and inference latency. arXiv:2504.03360. https://arxiv.org/abs/2504.03360
Grade: PP. Relevance: only paper found with Qwen2.5-0.5B GGUF Q8/Q4/Q3 measured on a real edge board with energy.
Measured (Raspberry Pi 4, 4 GB, Ollama/llama.cpp): Qwen2.5-0.5B avg accuracy FP16 0.32, Q8_0 0.28, Q4_1 0.35, Q4_0 0.30, Q3_K_M 0.33 (SD ~0.40 on all); energy/token 4.42 J (FP16) -> 2.14 J (Q8_0) -> 2.44 J (Q4_1).
Limitations: accuracy differences are inside noise (SD > mean); generative tasks; ARM CPU, not Apple.

**A10.** Melton, J. (2026). What does MLX 4-bit cost? A controlled audit of quantization for code generation on Apple Silicon. Zenodo. https://zenodo.org/records/22698290
Grade: PP. Relevance: only controlled study found of MLX's default affine 4-bit quantizer.
Measured: across five code models, 220 discordant pairs favour bf16 vs 125 favour 4-bit (significant); loss shrinks with model size at +0.52 points per billion parameters; a calibrated AWQ arm does not recover the gap.
Limitations: code generation only; model list not in the abstract; no sub-1B row confirmed.

**A11.** ZeroDegress. (2026). NanoJev-mlx-4bit [Model card]. Hugging Face. https://huggingface.co/ZeroDegress/NanoJev-mlx-4bit
Grade: MC. Relevance: the only found MLX 4-bit measurement on an actual 0.6B *decision* model with a scoring head.
Measured: Qwen3-0.6B backbone + heads, MLX affine 4-bit group 64, head kept fp16; 335.86 MB (14.1% of fp32). Argmax agreement with fp32 99.17% (357/360 questions); mean KL 2.91e-4; mean TV 0.0066; zero boolean polarity flips over 120 states; teacher-target cross-entropy 0.4981 vs 0.4992 (fp32). Notes 4-bit matmul at parity with fp32/bf16 throughput (2.96 vs 2.99/3.60 TFLOPS), i.e. footprint win, not speed win.
Limitations: 360-question game-task eval; self-reported; no calibration (ECE) reported.

**A12.** yzfly. (2026). edgejev: Local Jev-like typed-decision runtime using ONNX and quantized models [README]. GitHub. https://github.com/yzfly/edgejev
Grade: RC. Relevance: INT8 ONNX numbers on two sub-1B decision models on CPU.
Measured (4 vCPU Xeon Cascade Lake, AVX512-VNNI, batch 1): Laya mmBERT-base 322M FP32 32.1 ms / AG News 92.8% / Emotion 54.0%; INT8 15.6 ms / 91.2% / 48.2%; file 1,290 MB -> 324 MB. kev (Qwen2.5-0.5B+LoRA) FP32 44 ms / AG News 90% / Emotion 44% (n=100); INT8 26 ms / 84% / 21%. NanoJev Qwen3-0.6B FP32 155 ms, INT8 69 ms (no accuracy; game-only checkpoint).
Limitations: n=400 (Laya) and n=100 (kev); single machine; the decoder INT8 drop (-6 / -23 points) is large and unexplained.

### 5b. Pruning and distillation for footprint

**B1.** Turc, I., Chang, M.-W., Lee, K., & Toutanova, K. (2019). Well-read students learn better: On the importance of pre-training compact models. arXiv:1908.08962. https://arxiv.org/abs/1908.08962
Grade: PP (widely cited; released the 24 BERT-miniatures). Relevance: parameter ladder 4.4M-110M with GLUE numbers; the reference for "how small can an encoder go".
Measured: BERT-Tiny 4.4M, Mini 11.3M, Small 29.1M, Medium 41.7M, Base 110.1M. 6-layer/768 student: Pre-trained Distillation 84.4 dev / 82.1 test GLUE meta-score vs pre-train+fine-tune 82.8 / 81.6 vs DistilBERT 82.3 vs PKD 81.7.
Limitations: 2019 BERT recipe; no latency.

**B2.** Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2020). TinyBERT: Distilling BERT for natural language understanding. *Findings of EMNLP 2020*. arXiv:1909.10351. https://arxiv.org/abs/1909.10351
Grade: PR. Measured: TinyBERT4 (14.5M) >96.8% of BERT-base GLUE, 7.5x smaller, 9.4x faster; TinyBERT6 (67M) on par with BERT-base.
Limitations: two-stage distillation costs ~350 GPU-hours of general distillation (per Xia et al. 2022).

**B3.** Sun, Z., Yu, H., Song, X., Liu, R., Yang, Y., & Zhou, D. (2020). MobileBERT: A compact task-agnostic BERT for resource-limited devices. *Proceedings of ACL 2020*. arXiv:2004.02984. https://arxiv.org/abs/2004.02984
Grade: PR. Relevance: only classic compact encoder with a measured phone latency.
Measured: 25.3M params, 4.3x smaller than BERT-base, GLUE 77.7 (-0.6 vs BERT-base), SQuAD-1.1/2.0 dev F1 90.0/79.2, 62 ms on Pixel 4 (seq 128).
Limitations: Android CPU (TFLite), not iOS/ANE; bottleneck architecture needs a custom teacher.

**B4.** Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. *Advances in Neural Information Processing Systems 33*. arXiv:2002.10957. https://arxiv.org/abs/2002.10957
Grade: PR. Measured: 12x384 (33M) student retains >99% of teacher accuracy on SQuAD 2.0 and several GLUE tasks with 50% of parameters and FLOPs.
Limitations: numbers are relative; teacher is BERT-base.

**B5.** Xia, M., Zhong, Z., & Chen, D. (2022). Structured pruning learns compact and accurate models. *Proceedings of ACL 2022*. arXiv:2204.00408. https://arxiv.org/abs/2204.00408
Grade: PR. Relevance: structured (layer + head + hidden) pruning of a 110M encoder to ~5M with real speedup, without unlabeled data.
Measured: CoFi at ~95% sparsity (4.4M encoder params): SST-2 90.6 (12.0x), QNLI 86.1 (12.1x), MNLI 80.6 (12.1x, vs 84.8 dense), QQP 90.1 (11.0x), RTE 64.7, STS-B 83.1, MRPC 82.6, SQuAD 82.6 F1 (8.7x). Trains in <=20 GPU-hours vs ~350 for TinyBERT4.
Limitations: BERT-base only; speedups on GPU.

**B6.** Men, X., Xu, M., Zhang, Q., Wang, B., Lin, H., Lu, Y., Han, X., & Chen, W. (2024). ShortGPT: Layers in large language models are more redundant than you expect. arXiv:2403.03853. https://arxiv.org/abs/2403.03853
Grade: PP. Relevance: layer dropping as a footprint lever for decoders.
Measured: removing ~25% of layers retains 86.3% (Llama-2-7B), 91.6% (13B), 85.1% (Baichuan2-7B), 90.4% (Mamba-2.8B) of the dense average.
Limitations: smallest model 2.8B; no sub-1B result, so transfer to 0.6B is untested.

### 5c. Matryoshka / nested / early-exit (many sizes from one run)

**C1.** Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., Howard-Snyder, W., Chen, K., Kakade, S., Jain, P., & Farhadi, A. (2022). Matryoshka representation learning. *Advances in Neural Information Processing Systems 35*. arXiv:2205.13147. https://arxiv.org/abs/2205.13147
Grade: PR. Measured: up to 14x smaller embedding at equal ImageNet-1K accuracy; up to 14x retrieval speed-up; +2% long-tail few-shot. Applied to ViT, ResNet, BERT and ALIGN.
Limitations: shrinks the *representation*, not the backbone; a nested classification head is implied, not benchmarked for calibration.

**C2.** Devvrit, Kudugunta, S., Kusupati, A., Dettmers, T., Chen, K., Dhillon, I., Tsvetkov, Y., Hajishirzi, H., Kakade, S., Farhadi, A., & Jain, P. (2024). MatFormer: Nested transformer for elastic inference. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. arXiv:2310.07707. https://arxiv.org/abs/2310.07707
Grade: PR. Relevance: one training run yields a family of FFN-nested submodels in exactly the 0.1-0.85B band.
Measured: MatLM-850M yields extractable 582M-850M submodels each with better validation loss and one-shot downstream than independently trained models of the same size; MatViT encoders preserve metric space for retrieval; consistent submodels enable speculative decoding.
Limitations: no classification/calibration benchmark; requires training the largest granularity.

**C3.** Google. (2025, June 26). Introducing Gemma 3n: The developer guide. *Google Developers Blog*. https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/
Grade: BL (vendor). Relevance: production use of MatFormer; E2B is a nested submodel of E4B; Mix-n-Match slices FFN width and skips layers; Per-Layer Embeddings move embedding params off the accelerator (E2B runs in ~2 GB).
Measured: E2B/E4B raw 5B/8B params; E4B LMArena >1300. MMLU for Mix-n-Match sizes shown as a plot, not numbers.
Limitations: vendor blog; the effective-parameter accounting is not the same as a 0.3B dense model.

**C4.** Fan, A., Grave, E., & Joulin, A. (2020). Reducing transformer depth on demand with structured dropout. *ICLR 2020*. arXiv:1909.11556. https://arxiv.org/abs/1909.11556
Grade: PR. Relevance: LayerDrop lets one trained encoder be served at any depth without fine-tuning.
Measured: RoBERTa-base 12-layer -> 6-layer pruned (LayerDrop): MNLI-m 82.9, MRPC 85.3, QNLI 89.4, SST-2 92.5 vs DistilBERT 81.6/82.4/85.5/92.7. WikiText-103 16 -> 8 layers: ppl 20.78 without finetune, 20.56 with.
Limitations: 2019 RoBERTa; halving depth is the only granularity reported.

**C5.** Zhou, W., Xu, C., Ge, T., McAuley, J., Xu, K., & Wei, F. (2020). BERT loses patience: Fast and robust inference with early exit. *Advances in Neural Information Processing Systems 33*. arXiv:2006.04152. https://arxiv.org/abs/2006.04152
Grade: PR. Measured: ALBERT-base + PABEE 1.57x speed-up with GLUE macro 84.4 -> 85.1; BERT-base 1.62x with MNLI 84.5 -> 83.6, SST-2 92.1 -> 92.0, STS-B 88.9 -> 88.7.
Limitations: dynamic depth breaks single-batch determinism; extra per-layer classifiers.

**C6.** Xin, J., Tang, R., Lee, J., Yu, Y., & Lin, J. (2020). DeeBERT: Dynamic early exiting for accelerating BERT inference. *Proceedings of ACL 2020*. arXiv:2004.12993. https://arxiv.org/abs/2004.12993
Grade: PR. Measured: BERT-base at <=1-point loss saves 9-24% runtime (SST-2 21%, QQP 24%, MNLI 14%); RoBERTa-base 19-32%; up to ~40% at 2-4 point loss.
Limitations: entropy threshold interacts with calibration (confident-wrong exits early).

### 5d. On-device inference for 100-600M models

**D1.** Apple Machine Learning Research. (2022, June). Deploying transformers on the Apple Neural Engine. https://machinelearning.apple.com/research/neural-engine-transformers
Grade: BL (vendor, with reproducible code). Relevance: the canonical ANE number for a 66M encoder on iPhone.
Measured: DistilBERT-SST-2, FP16, seq 128, batch 1, iPhone 13: 3.47 ms at 0.454 W (or 9.44 ms at 0.072 W); up to 10x faster and 14x less memory than the baseline CoreML export.
Limitations: iPhone 13 / iOS 16 era; requires the ANE-friendly re-implementation; no 4-bit.

**D2.** Liu, Z., Zhao, C., Iandola, F., Lai, C., Tian, Y., Fedorov, I., Xiong, Y., Chang, E., Shi, Y., Krishnamoorthi, R., Lai, L., & Chandra, V. (2024). MobileLLM: Optimizing sub-billion parameter language models for on-device use cases. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*. arXiv:2402.14905. https://arxiv.org/abs/2402.14905
Grade: PR. Relevance: 125M/350M decoders with iPhone latency.
Measured: iPhone 13, ExecuTorch + MPS: MobileLLM-125M load 39.2 ms, init 1361.7 ms, 15.6 ms/token (~64 tok/s); LS variant +2.6%. Zero-shot common-sense avg 46.3 (125M) and 51.3 (350M) vs OPT 42.6 / 43.9.
Limitations: 350M latency not separately reported; 2024 hardware.

**D3.** Zhang, H., & Huang, J. (2025). Challenging GPU dominance: When CPUs outperform for on-device LLM inference. arXiv:2505.06461. https://arxiv.org/abs/2505.06461
Grade: PP. Measured: iPhone 15 Pro, llama.cpp: Llama-3.2-1B F16 CPU (2 threads) 17 tok/s vs GPU 12.8 tok/s; Qwen2-0.5B and Llama-3.2-1B at Q4_1/F16 CPU 1.31-1.33x faster than GPU; matmul 87.6% of prefill time.
Limitations: no absolute Qwen2-0.5B tok/s in the abstract/HTML we could read; no energy.

**D4.** Majima, D. (john-rocky). (2026). apple-silicon-llm-bench: Reproducible on-device LLM benchmarks for Apple Silicon [README and result tables]. GitHub. https://github.com/john-rocky/apple-silicon-llm-bench
Grade: RC (reproducible harness, every number tagged with quantization and capture session). Relevance: the only source found with Qwen3-0.6B / Qwen2.5-0.5B / Qwen3.5-0.8B on iPhone 17 Pro and M4 Max across MLX-Swift, llama.cpp, CoreML/ANE, LiteRT-LM, Core AI.
Measured: Qwen3-0.6B, iPhone 17 Pro, short-chat: Core AI GPU 193.3 tok/s, MLX 158.8, LiteRT-LM 120.4, CoreML/ANE 37.7. Qwen2.5-0.5B, M4 Max: MLX-Swift 531.1 tok/s, 21 ms TTFT, 390 MB; llama.cpp Q4_K_M 297.1 tok/s, 538 MB; CoreML/ANE 181.2 tok/s, 171 ms TTFT, 962 MB. Qwen3.5-0.8B, M4 Max: MLX-Swift 421.1 tok/s, 600 MB; CoreML/ANE 58.2 tok/s, 405 ms TTFT, 221 MB. Gemma 4 E2B, iPhone 17 Pro: LiteRT-LM 61.1 tok/s / 497 MB; MLX-Swift 49.1 / 3,010 MB; llama.cpp Q4 38.8 / 191 MB.
Limitations: decode tok/s for chat, not one-shot prefill+readout; no encoder models; single author.

**D5.** Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., & Poli, I. (2024). Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. arXiv:2412.13663. https://arxiv.org/abs/2412.13663
Grade: PP. Relevance: the backbone of Verdict (base) and Laya / dev-0.4b (large); establishes that *no* phone or Apple-Silicon numbers are published by the authors.
Measured: ModernBERT-base 149M, -large 395M; GLUE dev avg 88.4 / 90.4 (DeBERTaV3-base 88.1, -large 91.4). RTX 4090: base 147.3K tok/s at 512 tokens (max batch 1,604); large 52.9K tok/s. No mobile numbers.
Limitations: GPU only; unpadded FlashAttention path not available in CoreML/MLX out of the box.

**D6.** Palmer, J. (2026). Kev [README; model cards kev-0.6b, kev-0.8b]. GitHub / Hugging Face. https://github.com/jaredpalmer/kev ; https://huggingface.co/jaredpalmer/kev-0.6b ; https://huggingface.co/jaredpalmer/kev-0.8b
Grade: RC / MC. Relevance: the only Apple-Silicon latency numbers for a sub-1B *decision* model with typed questions.
Measured: bf16 on Apple M5, five 3-option questions on a ~230-token state: Kev-0.8B 329 ms, Kev-4B 779 ms, Kev-9B ~2 s; previous Qwen3-generation models 123 ms / 174 ms / ~300 ms. Kev-0.6B card: 0.12 s per five-question request on Apple Silicon.
Limitations: PyTorch/MPS path, not MLX; single machine; no memory figure.

**D7.** FluidInference. (2026). laya-coreml [Model card]. Hugging Face. https://huggingface.co/FluidInference/laya-coreml
Grade: MC. Relevance: CoreML/ANE numbers for a 322M encoder decision model.
Measured: Laya-multilingual (mmBERT-base 322M), FP16, Apple M5 Pro, CPU+ANE: L128 3.6 ms, L256 9.9 ms, L512 27.5 ms, L1024 80.1 ms; all compute units L512 9.0 ms. 16/16 argmax agreement per bucket, max prob error 0.0021; AG News 0.935 vs 0.930 PyTorch; int8-embedding variants 448-453 MB within 0.5 pt.
Limitations: Mac, not iPhone; fixed-length buckets.

**D8.** mpnikhil. (2026). dev-0.4b [Model card]. Hugging Face. https://huggingface.co/mpnikhil/dev-0.4b
Grade: MC. Measured: ModernBERT-large 399M decision model, Apple M1 Max via MPS ~27.6 ms per forward pass; CUDA FP16 SDPA ~10 ms.
Limitations: no MLX/CoreML; batch and sequence length unstated.

**D9.** Heman10x-NGU. (2026). Verdict-open-jev [README]; heman10x. (2026). rlcd-modernbert-151m [Model card]. https://github.com/Heman10x-NGU/Verdict-open-jev ; https://huggingface.co/heman10x/rlcd-modernbert-151m
Grade: RC / MC. Measured: ModernBERT-base 151M, single-thread WASM in browser, K=5: p50 35.58 ms, p95 39.81 ms; FP16 vs FP32 accuracy delta 0.00%.
Limitations: browser CPU, not ANE/Metal; no phone.

**D10.** iapp. (2026). OpenThai-SystemOne [Model card]. Hugging Face. https://huggingface.co/iapp/OpenThai-SystemOne
Grade: MC. Measured: Qwen3.5-0.8B text tower + 256-way slot head; MacBook M3 Max ~154 ms for a 3-question, 166-token Thai ticket; H100 ~44 ms (255 options); ~41 ms/decision in real-time Doom.
Limitations: 0.8B (above target); framework on Mac unstated.

**D11.** lostargon. (2026). Tiny-Jev [Model card]. Hugging Face. https://huggingface.co/lostargon/Tiny-Jev
Grade: MC. Measured: Qwen3-0.6B (596M) + 1-d decision head: ~5 ms on a consumer GPU, ~20-50 ms on Apple M-series.
Limitations: no device model, framework or sequence length given.

Gap statement for 5d: no source was found that reports ModernBERT, DeBERTa-v3 or SmolLM2 latency on an iPhone. The closest are D1 (DistilBERT on iPhone 13 ANE), D7 (mmBERT-base on M5 Pro ANE) and D4 (Qwen3-0.6B on iPhone 17 Pro). The Medium write-up of D4 returned HTTP 403 and is excluded; the GitHub repository is the primary source.

---

## 4. Landscape table, sub-question 6 (as of 2026-09-22)

Reference points first, then open models under 1B sorted by parameter count. "Suite / split" names the benchmark exactly as the source does. "Self-rep." = self-reported by the model author; "IM" = measured by the jevbench author or by Kev's harness. Kev suite results marked "dev" come from `runs/leaderboard.md` (selection on development partitions; the locked test is "never read here"); "locked test" values come from the model cards' single confirmatory read.

| Model | Params | Backbone | Head | Objective | Data | Suite | Split | Accuracy | Calibration | Self-rep.? | Weights public? | URL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Jev 1.13.0** (TypeSafe, reference) | undisclosed | undisclosed (Hume infers causal transformer, ~10B active, prefill-only) | undisclosed | undisclosed (RLCD per Hume) | undisclosed | Kev decision-v7 | dev | 0.845 | Brier 0.211 (transfer-v4) | IM (Kev harness) | No (API) | https://github.com/jaredpalmer/kev |
| | | | | | | Kev transfer-v4 | dev | 0.857 | | IM | | |
| | | | | | | Kev transfer-v9 knowable / MMLU-Pro | dev | 0.854 / 0.840 | | IM | | https://github.com/jaredpalmer/kev/blob/main/PLAN.md |
| | | | | | | jevbench v1.3.0 (534 decisions) | public+held-out | score 74.4 (Int 85.7) | Cal 82.7 | IM | | https://benchmarkheaven.com/jev-models |
| **Kev-0.6B** (reference for this brief) | ~596M | Qwen3-0.6B-Base | pointer head over option boundary tokens | CE over options | decision-v7 (12,576 records) | decision-v7 | dev / locked test | 0.801 / 0.808 | ECE 0.086-0.089 (dev) | self-rep. (harness public) | Yes, Apache-2.0 | https://huggingface.co/jaredpalmer/kev-0.6b |
| | | | | | | transfer-v4 | dev (seeds 0/1/2) / locked test | 0.613 / 0.605 / 0.620 ; 0.642 | Brier 0.536 (seed 2); conf-err 0.108 (dev), ~7.9% (test) | self-rep. | | https://raw.githubusercontent.com/jaredpalmer/kev/main/runs/leaderboard.md |
| | | | | | | jevbench v1.3.0 | rank 19 | score 62.5 (Int 51.9) | Cal 51.1 | IM | | https://benchmarkheaven.com/jev-models |
| Kev-0.8B | 0.8B | Qwen3.5-0.8B-Base | pointer head, LoRA r=16 (11.3M trainable) | CE over options | decision-v7 + delta fine-tune | decision-v7 | dev / locked test | 0.825 / 0.834 | Brier 0.268, ECE 0.100 (test) | self-rep. | Yes, Apache-2.0 | https://huggingface.co/jaredpalmer/kev-0.8b |
| | | | | | | transfer-v4 | dev / locked test | 0.652 / 0.684 | Brier 0.499 / 0.460; ECE 0.154 (test) | self-rep. | | |
| OpenThai-SystemOne | 0.8B | Qwen3.5-0.8B-Base (text tower) | 256-way slot head | SFT + Brier calibration stage | ~5B Thai CPT tokens; ~2-3M decision examples | own 13-subset public bench (3,881 items) | test | macro 74.3 (Nimble-9B 74.8) | per-subset ECE 0.035-0.371 | self-rep. | Yes, Apache-2.0 | https://huggingface.co/iapp/OpenThai-SystemOne |
| decider-0.8b | 0.8B | Qwen3.5-0.8B-Base | distribution readout (card) | supervised, teacher labels from Qwen3.5-27B | 1.47M examples | own 69 in-task / 24 held-out tasks | in-task / held-out | 0.776 / 0.707 | ECE 0.032 / 0.096 | self-rep. | Yes, Apache-2.0 | https://huggingface.co/Mapika/decider-0.8b |
| NanoJev | 596M | Qwen3-0.6B | set-attention Choice head + sigmoid Boolean + Score | complete-question CE | 18,760 game decisions (Maze/Snake/ViZDoom) | own game test (274 cases) | test | Maze 4/10, Snake 8/8, Basic 128/128, PredictPos 27/128 | none | self-rep. | Yes, MIT | https://huggingface.co/C-Tianyu/NanoJev |
| LightJev-0.6B-v0.1 | 596M | Qwen3-0.6B | shared 3,073-param candidate scorer | CE (and Brier arm) | NanoJev-Data stage1 (2,312 records) | own frozen splits | test (848) / OOD (448) | 0.7957 / 0.7244 | Brier 0.2662 / 0.3037; ECE 0.0265 / 0.0641 | self-rep. | Yes, Apache-2.0 | https://huggingface.co/rongxinzy/LightJev-0.6B-v0.1 |
| Tiny-Jev | 596M | Qwen3-0.6B | marker token + 1-d head + temperature | soft CE on probability targets | ~100k synthetic states | SST-2 / AG News / Emotion / Banking77 | public test | 90.4 / 90.5 / 82.2 / 81.2 | ECE 0.023 / 0.031 / 0.054 / 0.036 | self-rep. | Yes, Apache-2.0 | https://huggingface.co/lostargon/Tiny-Jev |
| | | | | | | own held-out domains (830) | held-out | 53.1 | ECE 0.299 | self-rep. | | |
| qwen3-0.6b-rlcd-decision | 0.6B | Qwen3-0.6B-Base | 26-row letter head | RLCD (REINFORCE, reward = outcome - p(chosen)) | Banking77, AG News, MNLI, SST-5, Yelp, BoolQ, Bitext, synthetic | own 8,000-row test | test | 0.807 [0.798, 0.816] | ECE 0.021; Brier 0.268 | self-rep. | Yes, Apache-2.0 | https://huggingface.co/anthonym21/qwen3-0.6b-rlcd-decision |
| mini-Jev (ODM Mini v1) | 0.6B frozen + 263k head | Qwen3-0.6B (frozen) | Linear-GELU-Linear grouped softmax | grouped CE | 50k synthetic decisions | own synthetic held-out | held-out | 72.97 (semantic) / 67.64 (stress) | none | self-rep. | Yes (head), Apache-2.0 | https://huggingface.co/samatv256/mini-Jev |
| systemone-lite-0.5b | 0.5B | Qwen2.5-0.5B-Instruct | next-token scoring over option aliases | SFT | 43,200 synthetic rows (4 domains) | own IID / Hard | test | 0.781 / 0.733 | none | self-rep. | Yes, Apache-2.0 | https://huggingface.co/dwidlee/systemone-lite-0.5b |
| **Kev-0.5B v0.1** | 494M | Qwen2.5-0.5B (frozen) | 896->256 attention pointer head, LoRA r=16 (9.3M trainable) | CE over options | 6 public sources, 13,500 questions | own 6-source held-out | test | 0.799 | ECE 0.065 (0.031 after T=1.47) | self-rep. | Yes, Apache-2.0 | https://huggingface.co/jaredpalmer/kev-0.5b |
| | | | | | | Kev transfer-v4 (card remark) | dev | 0.561 | | self-rep. | | |
| | | | | | | Kev leaderboard row ablation-v2/04 (Qwen2.5-0.5B) | dev | 0.742 (decision-v7) / 0.605 (transfer) | Brier 0.501 | self-rep. | | https://raw.githubusercontent.com/jaredpalmer/kev/main/runs/leaderboard.md |
| | | | | | | jevbench v1.3.0 | rank 38 | score 33.2 (Int 38.2) | Cal 47.4 | IM | | https://benchmarkheaven.com/jev-models |
| open-jev-deberta-v3-large | ~0.4B (card) | DeBERTa-v3-large | 3-layer scorer over [Q]/[OPT] marker means | CE + Brier | Banking77, SST-5, BoolQ (42k questions) | own in-domain / OOD | test | 0.854 / 0.690 | Brier 0.213 / 0.399; ECE 0.022 / 0.035 | self-rep. | Yes, Apache-2.0 | https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large |
| **Laya** (convaiinnovations) | 421M | ModernBERT-large | 2 transformer layers + option-marker scorer + act/escalate head | RLCD (log + spherical + RPS) | not specified | jevbench v1.3.0 | rank 33 | score 54.4 (Int 45.8) | Cal 62.5 | IM | Yes, Apache-2.0 | https://huggingface.co/convaiinnovations/laya |
| | | | | | | own typed-decisions (2,000) | test | 0.766 | Brier 0.062; ECE 0.213 | self-rep. | | |
| | | | | | | Banking77 / AG News / Emotion / XNLI-en / MASSIVE-en | test | 0.425 / 0.950 / 0.595 / 0.860 / 0.783 | ECE 0.081 post-temperature | self-rep. | | |
| jeff (GLiFormer) | ~400M | GLiNER-family | GLiNER-style | n/a (wrapper server) | n/a | jevbench v1.3.0 | rank 32 | score 54.4 (Int 46.9) | Cal 64.6 | IM | Yes (upstream GLiNER) | https://github.com/logan-markewich/jeff |
| dev-0.4b | 399M | ModernBERT-large | unified 2-layer dynamic-candidate head | single-pass CE | BoolQ, CodeSearchNet, Yelp, Banking77 | Banking77 (300) / BoolQ (500) / Yelp (300) | test | 91.33 / 85.2 / 62.7 | ECE 5.5% / 7.7% / 15.5% (temp-scaled) | self-rep. | Yes, Apache-2.0 | https://huggingface.co/mpnikhil/dev-0.4b |
| Laya-multilingual | 322M | mmBERT-base | as Laya | RLCD | not specified | MASSIVE (13 langs) / XNLI (14 langs) | test | 0.451 / 0.731 | none stated | self-rep. | Yes, Apache-2.0 | https://huggingface.co/convaiinnovations/laya-multilingual |
| | | | | | | edgejev AG News / Emotion (n=400) FP32 -> INT8 | test | 92.8 -> 91.2 / 54.0 -> 48.2 | none | IM (edgejev author) | | https://github.com/yzfly/edgejev |
| GLiNER2.5 multi | 287M | GLiNER2.5 | GLiNER | n/a | n/a | jevbench v1.3.0 | rank 43 | score 16.6 (Int 27.7) | Cal 56.1 | IM | Yes | https://benchmarkheaven.com/jev-models |
| system-one-270m (kaivoss) | 270M | Gemma-3-270M-IT | letter-token logit gather + softmax | log score vs soft ensemble targets | 25,002 synthetic decisions, 28 domains | own held-out (2,493 Q) | held-out | 0.6574 (baseline 0.4204) | Brier 0.4110; ECE 0.1311 (0.0374 at T=2.0) | self-rep. | Yes, Gemma licence | https://huggingface.co/kaivoss/system-one-270m |
| system-one-gemma (akash-kamat) | 268M (2.6M trainable) | Gemma-3-270M | single linear scorer on last-token state | softmax over per-option logits | Banking77, GoEmotions, AG News, MMLU, Yelp, support tickets (12,913 Q) | own held-out | test | 64.4 | ECE 0.047; Brier 0.454 | self-rep. | Yes (LoRA in repo), Apache-2.0 / CC-BY-NC part | https://github.com/akash-kamat/system-one-gemma |
| system-one-open (mithalouni), Gemma 3 270M variant | 270M | Gemma-3-270M | choice/score/noul heads | CE + Brier + temperature | 92 public HF decision datasets + synthetic | none published for the 270M variant (76.7% on TypeSafe strict 343-pair eval is the Gemma 4 E2B variant) | - | - | - | self-rep. | "HF upload pending" (Modal volume only) | https://github.com/mithalouni/system-one-open |
| laya-vision-smolvlm-256m | ~256M (150M trainable) | SmolVLM-256M-Instruct | Laya decision head | RLCD | A-OKVQA, ScienceQA, VQAv2 (~72k) | A-OKVQA / ScienceQA / VQAv2 | test | 63.1 / 89.0 / 73.2 (combined 75.9) | ECE 0.035 calibrated (0.108 raw) | self-rep. | Yes, CC BY-NC-SA 4.0 | https://huggingface.co/aaroncool9/laya-vision-smolvlm-256m |
| laya-flash | 230M | LFM2.5-Encoder-230M | randomly initialised head | none (format baseline) | none | none | - | - | - | - | Yes, LFM Open Licence | https://huggingface.co/nampham1106/laya-flash |
| **Verdict-open-jev** (rlcd-modernbert-151m) | 151M | ModernBERT-base (gliclass-modern-base-v2.0) | GLiClass bi-encoder head, 25 slots (24 + abstain) | CE + multiclass Brier + L-BFGS temperature | Banking77, CLINC150 | jevbench v1.3.0 | rank 36 (v1.4) / 37 (v1.0) | score 38.9 / 38.1 (Int 38.6 / 39.8) | Cal 74.1 / 51.3 | IM | Yes, Apache-2.0 | https://huggingface.co/heman10x/rlcd-modernbert-151m |
| | | | | | | jevbench public (231 tasks), own run | easy / standard / hard | 87.5 / 69.4 / 36.9 (v1.4) | hard-tier ECE 0.118 | self-rep. | | https://github.com/Heman10x-NGU/Verdict-open-jev |
| | | | | | | own held-out Banking77+CLINC150 (1,000, K=5) | test | 95.00 | Brier 0.0785; ECE 3.35% | self-rep. | | |
| | | | | | | "TypeSafe external benchmark" (337 cases) | - | 48.07 (Jev 90.80) | | self-rep. | | |
| GLiNER2.5 small | 74M | GLiNER2.5 | GLiNER | n/a | n/a | jevbench v1.3.0 | rank 44 | score 13.8 (Int 25.6) | Cal 47.2 | IM | Yes | https://benchmarkheaven.com/jev-models |
| poorjev (wrapper) | ~400 MB NLI model, unnamed | small NLI model | NLI entailment scores | none (temperature + conformal on frozen model) | 55 labelled items / 160 decisions | own multi-primitive (160) / Banking77 (154) | test | 0.781 / 0.656 | ECE 0.071 (raw 0.170) / 0.414 | self-rep. | No own weights; MIT code | https://github.com/rupeshpoojary9/poorjev |
| edgejev (wrapper/runtime) | wraps Laya 322M, kev 0.5B, NanoJev 0.6B, PlayJev 0.8B | as wrapped | as wrapped | none (ONNX INT8 conversion) | n/a | see 5a A12 | test | see A12 | none | IM for INT8 deltas | No own weights; Apache-2.0 | https://github.com/yzfly/edgejev |
| jev-local (wrapper) | frozen Qwen3-0.6B ... 9B | Qwen | logprob scorer | none | n/a | own set1/set3 (numbers only for 3B/9B) | - | 0.6B: none published | - | self-rep. | No own weights | https://github.com/us/jev-local |
| OpenJev (xingwudao) | none | none | none | none | none | none (mock server, synthetic probabilities) | - | - | - | - | No weights; no licence | https://github.com/xingwudao/OpenJev |
| For scale only: reflex 4B / SemIf 4B / jev-lite (Gemma 4 E4B, 8B) | 4B / 4B / 8B | Qwen3.5-4B / Qwen3.5-4B / Gemma-4-E4B | LoRA / frozen prompt / QLoRA letter head | - | - | jevbench v1.3.0 / own 1,898 rows | - | 70.3 / 73.1 / 0.816 | ECE 0.019 (jev-lite) | IM / IM / self-rep. | Yes | https://github.com/kshetrajna12/reflex |

Notes on the table.

1. Kev's own suites. `decision-v7` = 10,000 records from ten public sources + 896 policy minimal pairs + 1,680 generated rule-structure records. `transfer-v4` = 764 dev records from never-trained sources (MMLU 80, PAWS, SciQ, QNLI, TweetEval, Emotion, authorization/deadline/rule families). `transfer-v9` = transfer-v4 + 200 MMLU-Pro (10-way) + 80 buried-state + 110 unknowable + 110 intact controls per partition (revision `a957287d`). Source: PLAN.md. The HF dataset `jaredpalmer/kev-suites` exists but its viewer fails with a schema error, so no numbers were read from it; the GitHub repo `jaredpalmer/kev-suites` returns 404.
2. Discrepancy flagged: PLAN.md's family table lists Kev-0.6B transfer-v4 dev at 0.598, while the model card and leaderboard list 0.613/0.605/0.620 across seeds (0.620 for the shipped seed). Both are self-reported by the same author; the card is the later document.
3. Kev's `runs/leaderboard.md` (generated 2026-09-21) contains only Qwen2.5/Qwen3/Qwen3.5 bases. No ModernBERT, DeBERTa, Gemma, Laya or Verdict row exists. PLAN.md mentions Laya only for latency ("39.5 ms for 1 question, 158.6 ms for 10") and explicitly does not evaluate it on the suites; SemIf (untrained Qwen3.5-4B) is the only external system measured on transfer-v4 dev (0.747).
4. jevbench caveats from its own README: latency of self-hosted endpoints is "adjusted x2 (+0.15 s)" as an assumption; "small models are very sensitive to option order"; 534 English decisions measured from one origin; cost is hypothetical provider pricing.
5. `AbdelStark/jev-benchmarks` is a 300-example pilot comparing TypeSafe Jev and GLiNER2.5 only (AG News 0.910 vs 0.700; Banking77 0.870 vs 0.610; Emotion 0.480 vs 0.440, Brier 0.846 vs 0.668). It contains no open Jev-style model and is not a leaderboard.
6. awesome-jev's "Jev-like models" section (18 entries) lists no parameter counts; of its entries only kev, SemIf, OpenThai-SystemOne, poorjev, edgejev and jev-local were relevant to <1B and all are covered above. Verdict, Laya, NanoJev, decider, system-one-270m and the DeBERTa/ModernBERT cards were found via the HF API, not via awesome-jev.

### Direct answer to sub-question 6

As of 2026-09-22:

- **On Kev's frozen suites (decision-v7 / transfer-v4 / transfer-v9): no external sub-0.5B open model has published any number.** The only sub-0.5B rows are Kev's own: the Qwen2.5-0.5B ablation (transfer-v4 dev 0.605, Brier 0.501) and Kev-0.5B v0.1 (transfer 0.561 per its card). Both are below Kev-0.6B (0.620 dev / 0.642 locked test). Therefore **no sub-0.5B open model beats Kev-0.6B on Kev's suites**, and none has even been measured there.
- **On jevbench v1.3.0 (independently measured):** the best sub-0.5B open models are Laya 421M (54.4) and jeff/GLiFormer ~400M (54.4), then Verdict 151M (38.9), Kev-0.5B (33.2), GLiNER2.5 multi 287M (16.6), GLiNER2.5 small 74M (13.8). All are below Kev-0.6B (62.5). Laya's calibration axis (62.5) and Verdict v1.4's (74.1) exceed Kev-0.6B's (51.1), but their intelligence axes (45.8, 38.6) are well below Kev-0.6B's (51.9).
- The sub-0.5B models with the strongest *self-reported* in-domain numbers (Verdict 95.0% on its own Banking77+CLINC150 test; dev-0.4b 91.3% Banking77; open-jev-deberta 0.854 in-domain) have no cross-suite numbers, and where an external suite exists their scores drop sharply (Verdict 48.07% on the 337-case TypeSafe set vs Jev 90.80%; Laya 0.425 on Banking77).

---

## 5. Numeric evidence table: quantization and on-device

| Model | Params | Precision / method | Device or setting | Metric | Value | Source (grade) |
|---|---|---|---|---|---|---|
| BERT-base | 110M | INT8 QAT | GLUE dev | SST-2 / MRPC / QNLI / RTE | 92.36->92.24 / 90.00->89.56 / 90.30->90.62 / 69.70->68.78 | A1 Zafrir 2019 (PR) |
| BERT-base | 110M | INT8 dynamic PTQ | GLUE / SQuAD dev | SQuAD F1 / RTE / QQP | 88.46->80.02 / 69.70->63.32 / 87.84->84.98 | A1 (PR) |
| RoBERTa-base | 125M | INT8 integer-only (I-BERT) | GLUE dev; T4 GPU | avg / speedup | 86.0->86.3 ; 2.42-3.39x | A2 Kim 2021 (PR) |
| BERT-base | 110M | W8A8 PTQ per-tensor / per-embedding-group / QAT | GLUE dev avg | score | 83.06 -> 71.03 / 82.45 / 83.26 | A3 Bondarenko 2021 (PR) |
| BERT-base | 110M | W4A8 QAT / W4A32 QAT | GLUE dev avg | score | 82.64 / 82.95 | A3 (PR) |
| BERT-base | 110M | W1A8 / W1A4 (BinaryBERT) | MNLI-m dev; size | acc ; MB | 84.6->84.2 / 83.9 ; 418->17 | A4 Bai 2021 (PR) |
| OPT-125M | 125M | int8 absmax / LLM.int8() | C4 | perplexity | 25.65 -> 87.76 / 25.83 | A5 Dettmers 2022 (PR) |
| Qwen3-0.6B | 0.6B | 8-bit RTN/GPTQ/AWQ | MMLU | acc | 47.1 -> 47.0 / 47.0 / 46.9 | A6 Zheng 2025 (PP) |
| Qwen3-0.6B | 0.6B | 4-bit RTN / GPTQ / AWQ | MMLU ; WikiText2 ppl | acc ; ppl | 37.3 / 40.0 / 43.1 ; 37.5 / 33.0 / 25.8 (FP16 20.9) | A6 (PP) |
| Qwen3-0.6B | 0.6B | 3-bit AWQ | MMLU | acc | 26.4 | A6 (PP) |
| Qwen3-1.7B | 1.7B | 4-bit AWQ / GPTQ / RTN | MMLU | acc | 60.0 -> 53.9 / 52.8 / 47.9 | A6 (PP) |
| Llama-3.2-1B-it | 1B | FP8 / W8A8 SmoothQuant / 4-bit AWQ / 4-bit GPTQ | OpenLLM-v1 avg | score | 42.06 -> 41.98 / 42.06 / 38.48 / 34.90 | A7 Lee 2025 (PR) |
| Llama-3.2-3B-it | 3B | 4-bit AWQ / GPTQ | OpenLLM-v1 avg | score | 52.13 -> 50.51 / 50.26 | A7 (PR) |
| Qwen2.5-0.5B | 0.5B | 4-bit | arithmetic reasoning | acc | 21.31 -> 12.77 (-40% rel.) | A8 Srivastava 2026 (PP) |
| Qwen2.5-0.5B | 0.5B | FP16 / Q8_0 / Q4_1 / Q4_0 GGUF | Raspberry Pi 4 (llama.cpp) | avg acc (SD~0.40) ; J/token | 0.32 / 0.28 / 0.35 / 0.30 ; 4.42 / 2.14 / 2.44 / 2.30 | A9 Husom 2025 (PP) |
| 5 code models | n/s | MLX default 4-bit vs bf16 | Apple Silicon | discordant pairs ; trend | 220 favour bf16 vs 125 ; +0.52 pt/B params | A10 Melton 2026 (PP) |
| NanoJev (Qwen3-0.6B + heads) | 596M | MLX affine 4-bit g64, head fp16 | Apple Silicon | argmax agreement ; KL ; size | 99.17% (357/360) ; 2.91e-4 ; 335.86 MB (14.1% of fp32) | A11 (MC) |
| Laya-multilingual (mmBERT-base) | 322M | ONNX FP32 -> INT8 | 4 vCPU Xeon, batch 1 | AG News / Emotion ; latency ; size | 92.8->91.2 / 54.0->48.2 ; 32.1->15.6 ms ; 1,290->324 MB | A12 edgejev (RC) |
| kev (Qwen2.5-0.5B + LoRA) | 494M | ONNX FP32 -> INT8 (n=100) | 4 vCPU Xeon | AG News / Emotion ; latency | 90->84 / 44->21 ; 44->26 ms | A12 (RC) |
| Laya-multilingual | 322M | CoreML FP16 / int8 embeddings | Apple M5 Pro | argmax agreement ; accuracy delta | 16/16 per bucket, max prob err 0.0021 ; within 0.5 pt | D7 (MC) |
| Verdict (ModernBERT-base) | 151M | FP16 vs FP32 | - | accuracy delta | 0.00% | D9 (RC) |
| DistilBERT-SST2 | 66M | FP16, seq 128 | iPhone 13 ANE | latency @ power | 3.47 ms @ 0.454 W ; 9.44 ms @ 0.072 W ; 10x faster, 14x less memory | D1 Apple 2022 (BL) |
| MobileLLM-125M | 125M | fp (ExecuTorch MPS) | iPhone 13 | load / init / per-token | 39.2 ms / 1361.7 ms / 15.6 ms (~64 tok/s) | D2 Liu 2024 (PR) |
| MobileBERT | 25M | fp (TFLite) | Pixel 4, seq 128 | latency ; GLUE | 62 ms ; 77.7 | B3 Sun 2020 (PR) |
| Llama-3.2-1B | 1B | F16 llama.cpp | iPhone 15 Pro CPU 2-thr / GPU | decode | 17 / 12.8 tok/s | D3 Zhang 2025 (PP) |
| Qwen3-0.6B | 0.6B | 4-bit (per repo tags) | iPhone 17 Pro: Core AI GPU / MLX / LiteRT-LM / CoreML-ANE | decode | 193.3 / 158.8 / 120.4 / 37.7 tok/s | D4 (RC) |
| Qwen2.5-0.5B | 0.5B | MLX-Swift 4-bit / llama.cpp Q4_K_M / CoreML-ANE | M4 Max | decode ; TTFT ; peak mem | 531.1 tok/s, 21 ms, 390 MB / 297.1, 22 ms, 538 MB / 181.2, 171 ms, 962 MB | D4 (RC) |
| Qwen3.5-0.8B | 0.8B | MLX-Swift / llama.cpp Q4_K_M / CoreML-ANE | M4 Max | decode ; TTFT ; peak mem | 421.1 tok/s, 36 ms, 600 MB / 201.1, 22 ms, 752 MB / 58.2, 405 ms, 221 MB | D4 (RC) |
| Gemma 4 E2B | ~2B eff. | LiteRT-LM / MLX-Swift / llama.cpp Q4_K_M | iPhone 17 Pro | decode ; memory | 61.1 tok/s, 497 MB / 49.1, 3,010 MB / 38.8, 191 MB | D4 (RC) |
| Kev-0.6B | 596M | bf16 PyTorch | Apple Silicon (M-series) | 5-question request | 0.12 s | D6 kev-0.6b card (MC) |
| Kev-0.8B / 4B / 9B | 0.8B-9B | bf16 PyTorch | Apple M5, 5 questions, ~230-token state | request latency | 329 ms / 779 ms / ~2 s (Qwen3 gen: 123 / 174 / ~300 ms) | D6 kev README (RC) |
| Laya-multilingual | 322M | CoreML FP16, CPU+ANE | Apple M5 Pro | latency L128 / L256 / L512 / L1024 | 3.6 / 9.9 / 27.5 / 80.1 ms (all units L512: 9.0 ms) | D7 (MC) |
| Laya (ModernBERT-large) | 421M | fp (PyTorch) | Tesla T4 | 1 question / 10 batched | 32.8-39.5 ms / 72.3-158.6 ms | convaiinnovations/laya (MC) |
| dev-0.4b (ModernBERT-large) | 399M | fp (MPS) / FP16 SDPA | M1 Max / CUDA | per forward | ~27.6 ms / ~10 ms | D8 (MC) |
| open-jev-deberta-v3-large | ~0.4B | fp32 / bf16 | M1 Max / H100 | 4 Q / 10 Q | 1.8 s / 28 ms (518 Q/s at batch 8) | com-kotobalabs card (MC) |
| Verdict (ModernBERT-base) | 151M | WASM single-thread, K=5 | browser CPU | p50 / p95 | 35.58 / 39.81 ms | D9 (MC) |
| OpenThai-SystemOne | 0.8B | fp | MacBook M3 Max / H100 | 3-Q 166-token ticket / 1 Q 255 options | ~154 ms / ~44 ms | D10 (MC) |
| Tiny-Jev (Qwen3-0.6B) | 596M | fp | consumer GPU / Apple M-series | per decision | ~5 ms / ~20-50 ms | D11 (MC) |
| system-one-gemma (Gemma-3-270M) | 268M | fp | unstated | per decision | ~50 ms | RC |
| ModernBERT-base / -large | 149M / 395M | bf16 unpadded | RTX 4090, 512 tokens | throughput ; max batch | 147.3K / 52.9K tok/s ; 1,604 / 770 | D5 Warner 2024 (PP) |

---

## 6. "Jev's Architecture Unmasked" (Archer Hume), graded as a blog claim

Hume, A. (2026, September 17). Jev's architecture unmasked. *archerhume.com*. https://archerhume.com/posts/jevs-architecture-unmasked/
Grade: BL (black-box inference, no weights, no vendor confirmation).

What it claims and how firmly:

| Claim | Basis given | Author's own confidence |
|---|---|---|
| Causal (decoder-style) transformer that ends after prefill plus a readout; no token-by-token decoding | latency signatures over ~10,000 API calls (1,029 instrumented, 6,800 benchmark, 541 follow-up) | "observable" |
| ~10B *active* parameters | ~30k tokens processed in ~160 ms | explicitly "an inference, not a measurement"; no total-parameter claim |
| Sparse mixture-of-experts | expectation from throughput | "can't be observed from outside"; least certain |
| Readout is either a final-position head or a pointer-style scorer | option-interaction probes | cannot distinguish the two |
| Calibration by outcome-based training (RLCD) with a proper scoring rule (log or Brier assumed) | public TypeSafe description + measured ECE 0.0313 on an MMLU sample | training objective assumed, ECE measured |

Relevance to this brief: the post is the public origin of the "prefill-only + pointer/final-position head + proper-scoring-rule calibration" recipe that Kev-0.5B's card describes as "the architecture inferred for TypeSafe's Jev" and that every sub-1B model in section 4 reproduces. Its ~10B-active estimate, if right, puts Jev roughly 20-100x above the 0.1-0.6B band, which is consistent with the transfer gap in Kev's suites (Jev 0.857 vs Kev-0.6B 0.620 on transfer-v4 dev). None of the size claims should be cited as fact.

---

## 7. Verification log and exclusions

Fetched and confirmed (used): all URLs in sections 3-6.

Fetched but excluded:
- `https://rockyshikoku.medium.com/local-llm-on-iphone-which-runtime-is-actually-fastest-58096685481e` returned HTTP 403; the same author's GitHub repository (D4) is used instead.
- `https://github.com/jaredpalmer/kev-suites` returns 404; suites live at `https://huggingface.co/datasets/jaredpalmer/kev-suites`, whose viewer currently fails (schema cast error on `row`), so suite definitions are taken from PLAN.md.
- `nampham1106/laya-flash` (230M): card states the decision head is randomly initialised; no numbers; listed in the table only to close the "Laya's smallest" question.
- `xingwudao/OpenJev`: mock server, synthetic probabilities, no weights, no licence.
- `mithalouni/system-one-open` Gemma 3 270M variant: no numbers for that size; weights not on HF ("upload pending").
- `jaswanthsanjay88/jev-0.5b`, `aaroncool9/kev-0.5b`, `Radexito/kev`: forks/re-uploads of Kev-0.5B; not separately counted.
- `vagmi/jev-lite` (Gemma 4 E4B, 7.98B total): above 1B; listed for scale only.
- `Xubqpanda/nanojev`: an agent loop that calls Jev/Laya; not a model.

Not found (searched, no resolvable source): published iPhone latency for ModernBERT-base/large, DeBERTa-v3, SmolLM2-135M/360M via MLX Swift or CoreML; a GPTQ/AWQ study with rows below 0.6B on classification/NLI tasks; any sub-0.5B model evaluated on Kev's transfer-v4/v9 by a third party.
