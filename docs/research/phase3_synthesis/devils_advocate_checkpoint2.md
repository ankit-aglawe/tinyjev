# Devil's Advocate — Checkpoint 2 (after analysis): tinyjev synthesis

Date: 2026-09-22. Object under review: `phase3_synthesis/synthesis_report.md` (453 lines). Spot-checked against `phase1_scoping/research_question_brief.md` and every `phase2_investigation/*.md`. Line numbers below are `synthesis_report.md` unless prefixed (LE = `local_evidence.md`, VLE = `verification_local_evidence.md`, LBH/VBH, LOC/VOC, LFL/VFL as in the synthesis).

## Verdict: REVISE

Two Critical, eight Major, ten Minor, five Observations. The synthesis is careful, its numbers reproduce at their cited lines, and §7 refuses the wanted answer. But its headline band (§4.3 #1, "Kev-0.6B ± 5 pp") is derived inconsistently with its own strongest source, and the experiment that is supposed to settle the frame (E1) is confounded in a way that would let a full-fine-tuned encoder "beat" a LoRA-adapted decoder without any architectural signal.

---

## Steel-man first

What the synthesis gets right, and should keep:

1. Every verification correction is applied before use (§0, line 25), including the ones that hurt the encoder case (Set-Encoder rows, PriDe cost, laya-coreml ANE failure, Tiny-Jev few-shot fitting).
2. Every cross-task, cross-size, cross-device step is labelled `[I]`; the A-chain arithmetic (line 351) reproduces: 0.47×(0..+5) + 0.30×(−1..−13) + 0.23×(±8) → −1 central, ±4 → 0.57–0.65. Anchor numbers reproduce at VLE #1, #5, #14, #27, #31, #37.
3. The block composition (line 40) sums to 764 and matches VLE (d) exactly (232/356/176).
4. §7 is the strongest section: it states the capacity null (0.6B→4B +17.5), refuses "distillation closes the gap to Jev", and refuses 0.80 at 150M. DA-1's two cautions are honoured.
5. E2 carries the right control (hard-label-only on the enlarged transfer set, line 383) — the one design that separates "more data" from "soft targets".
6. The contamination table (§6) is exact and the "start from raw checkpoints, never `gliclass-*`/`*-mnli`" rule is correct and non-obvious.
7. "Coverage tracks accuracy, not calibration" (§4.3 #4, C21) is a genuine synthesis result, not a restatement.
8. §8 names the KD gain as the least-supported number. Honest.

Now the attack.

---

## Critical issues

### CR-1. The headline band is derived inconsistently with the synthesis's own strongest source, and is skewed optimistic

- **Type:** internal inconsistency / false precision.
- **Location:** §4.2 A evidence chain (line 351); §4.3 #1–#2 (lines 368–369); §8 ordering (line 453); C20 (line 312).
- **Problem:** The classification-block delta for recipe A ("+0 to +5 on Kev-0.8B's 0.64 block") is taken from Ettin's *150M-vs-150M* gap (+3.6) and B2's 210M gap (+2.1). But A is a 149M encoder compared against a 596M decoder. Ettin's own decoder ladder (VBH #3: 150M 85.6, 400M 88.2, 1B 89.9 on MNLI) interpolates to ≈ 89.0 at 600M — i.e. Ettin-150M-enc (89.2) *ties* a 600M decoder at matched 2T-token pretraining. The synthesis then adds, correctly, that Qwen3-0.6B has 36T tokens (line 199). Read consistently, B1 predicts the classification block delta is **−2 to +1**, not +0 to +5. That alone moves A's centre ~1.5 pp down.
  The knowledge-block range "−1 to −13 on block" (line 351) has no stated basis for either endpoint. Kev-0.8B's MMLU 0.42 is essentially the base model's zero-shot 0.425 (LE #8) — the "knowledge the 36T decoder carries" is ~+17 pp over chance on MMLU and SciQ 0.91. An encoder at 0.28–0.35 on MMLU and 0.70–0.80 on SciQ (no passage; see O-5) gives a block of ~0.50–0.57, i.e. **−10 to −17 on block**, making the synthesis's *lower bound* roughly the central case.
  Together: A's honest band is ≈ **0.52–0.64 (centre ~0.58)**, not 0.57–0.65 (0.61); B ≈ 0.54–0.67. The symmetric "±5" (line 368) hides a downside that is ~2x the upside.
- **Impact:** §4.3 #1's "Kev-0.6B ± 5 pp" becomes "Kev-0.6B −10 to +4, skewed down"; §8's ordering "C ≈ A ≈ G ≥ E(SmolLM2) ≥ D(SmolLM2)" is not supported once A's centre is ≈ 0.58 (D(SmolLM2) is 0.57 with better on-suite anchoring, line 363 "MED"); §4.3 #2's "A ≈ Kev-0.6B at 25% of the parameters" is the conditional-optimistic branch presented as the central one.
- **Recommendation:** (a) Re-derive the classification delta from B1 at the actual size ratio (149M vs 596M) and state the 36T-token adjustment as a further negative term. (b) Before any training, run a zero-cost MLM-cloze probe (ModernBERT-base, DeBERTa-v3-base) on the 116 MMLU + 116 SciQ dev items, the encoder twin of Kev's `base_mmlu_probe.py` — this replaces the guessed knowledge-block range with a measurement. (c) Restate the band as asymmetric with the centre below the anchor; move F to the top of §8's ordering *explicitly* ("the only arm the evidence places above the anchor is the largest permitted decoder plus an unmeasured KD lift").

### CR-2. E1, the experiment that decides the frame, confounds architecture with fine-tuning regime

- **Type:** experimental design flaw in the ladder.
- **Location:** §5 E1 (line 382); §4.1 A/C vs D/F (lines 334–339); §4.2 "Params" row (line 356) and "Training cost" row (line 362).
- **Problem:** The decoder arms are "pointer **LoRA** r=16" (D, D', anchor) with ~5–9M trainable + 0.5M head. The encoder arms are unspecified in §4.1 but costed as full fine-tunes (line 362: "Laya 421M full FT … → 149M ≈ 0.3"), i.e. ~149M trainable. Ettin's decoder MNLI rows — the evidence E1 is meant to test — are full fine-tunes on both sides. E1 as written therefore compares a 149M fully-trained encoder against a 9M-parameter adapter on a decoder. The decision rule ("if the best encoder is within CI of the anchor … the encoder line continues") will promote the encoder on a trainable-parameter confound, and every downstream rung (E2–E6) inherits it. A second, smaller confound: head family differs by backbone (marker scorer with a 2-layer MLP vs bilinear q·k pointer) and E3 studies heads only on the already-promoted backbone.
- **Impact:** E1 cannot answer C8/C20/C23/C30 as claimed (line 382). A positive E1 for encoders is uninterpretable; a negative one is interpretable but the synthesis's own alternative explanation (M-7) says the LoRA anchor is itself under-tuned.
- **Recommendation:** Match the regime: either full-FT every arm (cheap at ≤0.6B — Kev's 0.6B LoRA trial was 4.5 min; NanoJev full-FT 0.6B was 600 updates on one A100) or LoRA every arm at matched trainable-parameter count. Add a **full-FT, lr 2e-5 Qwen3-0.6B** arm as the true anchor (see M-7). Add one cross arm (pointer head on ModernBERT, or marker scorer on Qwen3-0.6B) so head and backbone are separable before E3.

---

## Major issues

### M-1. Abbes (B5) is cherry-picked to the outlier row
- **Location:** §0 line 25 ("56.8 / 3B 82.2 on SQuAD-v2 groundedness are added"); SQ1 #1 line 189; C20 line 312; §7 row 1 line 431 ("A fine-tuned 1B decoder scored 56.8 vs 90.2").
- **Problem:** VBH #32 reports *both* of Abbes's columns: Llama-3.2-1B fine-tuned **56.8 (SQuAD-v2) / 84.0 (NewsQA)**; 3B 82.2 / 86.4; RoBERTa-large 90.2 / 88.5. On NewsQA the 1B decoder is −4.5 from a 355M encoder — consistent with Ettin's "one size step". The 56.8 is below the same family's 8B *zero-shot* (81.9) and looks like a failed run; the synthesis uses only it, four times, as a "warning". This is the one place the synthesis leans harder toward encoders than its verifier did.
- **Recommendation:** Quote both columns everywhere; characterise B5 as "one size step on NewsQA, one unexplained collapse on SQuAD-v2, preprint, recipe unstated".

### M-2. "Five sources, four peer-reviewed, all direct" overstates the matched-fine-tuned evidence
- **Location:** SQ1 converges #1 (line 189); matrix rows H2 (line 103) and H3 (line 104) marked `D`.
- **Problem:** H2 (PET 223M vs GPT-3 350M/760M) and H3 (UniMC 235M vs 11B–540B) compare *fine-tuned* encoders to *few-/zero-shot* decoders. B4 is marked `I` for exactly that reason (line 88) but H2/H3 are marked `D`. The matched, fine-tuned, peer-reviewed encoder-vs-decoder evidence is **B1 and B2** — two sources — plus B5 with M-1's caveat.
- **Recommendation:** Mark H2/H3 `I (few-/zero-shot decoders)`; rewrite #1 as "two matched sources (both ICLR 2026, both ≤2T tokens), three supporting comparisons against un-fine-tuned decoders".

### M-3. The block decomposition is on records; the accuracy is on clean questions
- **Location:** line 27 ("764 records / 656 clean questions"); line 40 (30/47/23 by source counts); line 351 (block model).
- **Problem:** Weights 232/356/176 are record counts (VLE (d)). Headline accuracies are on 656 clean questions (VLE #14, #37). Per-source rows are "80–116" n, i.e. records (LE:231). No per-block clean count exists in the corpus. If the 108 dropped items are concentrated (e.g. multi-question `composition_holdout` records, or the 1,100-zero policy that produced "clean n 656" for Jev, VLE #31), the weights move and so does every §4 estimate. LE:230 itself lists Kev-0.8B's 0.652 at n=764 while VLE #14 lists locked reads at n=656 — the corpus is not consistent about the denominator.
- **Recommendation:** Compute per-block clean counts from `evals/v4/transfer-v4/manifest.json` + the metric policy; re-weight; state the denominator once.

### M-4. The Kev-0.8B anchor row merges a delta-tuned release number with matched-recipe seeds
- **Location:** anchor table line 35 ("0.652 (seeds 0.622/0.634/0.643)"); F chain line 351 ("+1.4 dev / +4.8 locked"); §8 line 453.
- **Problem:** 0.652 is the released `night2-08b-du2` checkpoint (post dates+unknowable delta; LE #59, #79). The three seeds (mean 0.633) are the matched-recipe experiment `q35-08b.json` (LE #5). The F chain's "+1.4" is the seed median (0.634 − 0.620); the anchor table and §8 use 0.652. The per-source rows the block model rests on (MMLU 0.42, SciQ 0.91, …) are the delta-tuned release's (LE:231, `kev-0.8b.md`).
- **Recommendation:** Two rows: "Kev-0.8B released (delta-tuned) 0.652" and "Kev-0.8B matched recipe, 3 seeds, 0.633". Use the matched row wherever a Kev-0.6B comparison is drawn.

### M-5. The KD "+2 to +6" is hopeful, not justified
- **Location:** §4.2 B chain (line 351); §4.3 #3 (line 370: "seven peer-reviewed analogues"); §8 (line 453 acknowledges it is least supported).
- **Problem:** (i) Tang 2019 and Turc 2019 are preprints (VOC Tier 2), so "seven peer-reviewed" is wrong. (ii) Rank-DistiLLM's "≥ teacher" closes a **+3.2** nDCG gap (teacher 0.719/0.720 vs MS-MARCO-only 0.687/0.698, VBH #26); here the gap is +17.7. (iii) Tang's student is a BiLSTM without pretraining; TinyBERT's +7.1 is in-distribution. (iv) The one in-category read is split: NanoJev OOD +2.6 comes with in-distribution **−4.1** (LE #47) — the signature of a regulariser, not of knowledge transfer, and predicts the gain vanishes once the hard-label arm is regularised (more data, 1 epoch, low lr) — which is precisely what E2's control will do. (v) The corpus's own counter-evidence — Cho & Hariharan (bigger teacher worse), Stanton (80–90% agreement, "optimization is the bottleneck"), Müller, Mishra ("augmentation, not soft targets, drives student ECE"), Furlanello ("gain persists with non-argmax info removed") — is listed in §1c but never weighed against the chain in §4. (vi) Decoder→encoder KD is restricted to logits: Wang 2023's attention transfer (the larger 2–6 GLUE effect, 3b.11) is architecturally unavailable across a causal/bidirectional boundary. (vii) The teacher is weakest exactly where the encoder student is weakest: Kev-4B *loses* MMLU and PAWS to its own base (line 54, LE #7–8), so its soft targets on the knowledge block carry less than the base's zero-shot would.
- **Recommendation:** Restate as "0 to +4, sign uncertain in-distribution, mechanism (regularisation vs transfer) unknown"; keep E2 exactly as designed; add a teacher = *untrained Qwen3.5-4B zero-shot* arm on the knowledge sources only (it out-scores Kev-4B on MMLU per LE #7–8 and is free).

### M-6. Off-suite evidence is admitted for decoders and excluded for encoders
- **Location:** D chain line 351 (system-one-open 270M 0.585 → "0.50–0.58"); C30 line 322 ("Not evidence against A"); SQ6 #2 line 276.
- **Problem:** S20's 270M (own 23-task suite, IT base, CE+Brier, 92 datasets) is uncontrolled and off-suite, and is used to set D(Gemma)'s band. Laya's held-out themes (0.53–0.76), Verdict's TypeSafe (0.48), and JevBench (Laya 54.4 / Verdict 38.9 < kev 0.6B 62.5) are uncontrolled and off-suite, and are excluded from A's band as "data-breadth, not architecture". The exclusion argument is sound but it applies with equal force to S20 (data breadth: 92 datasets vs Kev's 10). Either both are weak priors or neither is.
- **Recommendation:** Apply one rule. If S20 stays, admit JevBench's encoder rows as an equally weak negative prior on A and say what it does to the band (it pulls the lower bound to ~0.50).

### M-7. "0.6B saturated" is a hyperparameter-regime finding presented as a capacity finding
- **Location:** S1 stance (line 52: "0.6B saturated 0.59–0.61 regardless of knobs"); SQ1 #2 (line 190: "saturated at 0.59–0.62 across eight single-factor mutations"); C10 (line 302); E1 (line 382, anchor is "pointer LoRA re-run").
- **Problem:** VLE #1: the capacity ladder ran at default lr 2e-4, one seed, on v3 data. VLE #10: the only ≤1B lr mutations went *up* (3e-4, 5e-4); "no clean low-lr test at 0.6B/0.8B". Kev never ran full FT at ≤0.8B, while every replica reporting a sub-1B held-out number (NanoJev, decider, system-one-open) is full FT. At 4B the drift mechanism was worth +4.7 [+0.4, +9.6] (VLE #3). If any of that transfers to 0.6B, the anchor is ~0.64–0.66 and the "saturation" was over the wrong knobs. The synthesis records C10 and puts an lr sweep in E1, but keeps "saturated" in the stance column and in SQ1's convergent finding, and E1's anchor is LoRA-only.
  *Does it change which recipe wins?* It moves **F** up (the anchor is F without KD), not D; it makes the encoder line's parity target harder by the same amount; and it means the 384-token state cap is irrelevant to transfer-v4 (short sources) but decisive on any long-policy product mix (O-3).
- **Recommendation:** Change S1's stance to "flat over eight high-lr mutations; low lr and full FT untested"; make the full-FT low-lr Qwen3-0.6B the E1 anchor (CR-2).

### M-8. §7 treats "unmeasured" as "contradicted" twice
- **Location:** §7 row "Set attention over options improves accuracy" (line 438); row "Ordinal accuracy can be fixed by an ordinal loss" (line 446); SQ2 #2 (line 207: "Set attention adds nothing measurable").
- **Problem:** Set attention: H8's null is under pointwise relevance labels; H8's authors report a gain under a comparative loss (VBH #19), typed decisions *are* comparative (C22, line 314, concedes this), NanoJev never ablated, jevbetter has no backbone. Kev's `head_dim`/special-embedding mutations (line 56) are not set attention. Ordinal loss: one seed, one source (Kev's RPS "did not help"), vision-only elsewhere. Both belong in "Silent", not in "the corpus contradicts it".
- **Recommendation:** Move both to a "not measured on typed decisions" row in §7, or delete from §7 and leave them in the SQ silent lists where they already are.

---

## Minor issues

- **m-1.** §0 line 25 "Kev-0.6B ECE 0.086 dev / 0.089 locked" is decision-v7 in-distribution; the anchor table's "0.15 dev raw" (line 36) is transfer-v4. Label the suite in both places.
- **m-2.** Coverage chain line 353: "Kamath-style calibrator +8 pts" — the like-for-like gain is +4.3 (calibrator vs MaxProb on the same known-OOD-trained model, VOC #31); +7.9 stacks two changes.
- **m-3.** E chain line 351 cites Ghita 2026 ("decoder students lose general knowledge first") — a 0-citation preprint that VOC did not spot-check (VOC line 136, 153). Flag or drop.
- **m-4.** SQ5 #3 line 261: "encoders are 3–5x cheaper per question than *same-accuracy* decoders" is stated as convergent; every pair is cross-device (C17) and "same-accuracy" is unmeasured. Say "per parameter" or label `[I]` at the sentence level.
- **m-5.** §6 line 418: the pretraining-contamination caveat names Qwen3/Gemma/SmolLM2 only. ModernBERT (2T web+code) and DeBERTa-v3 (160GB web) carry the same MMLU/SciQ/PAWS/QNLI risk and should be listed.
- **m-6.** Line 189: B2's "+2.1/+3.4/+5.6" are the 40%-mask rows; VBH #6 asked that the condition travel with the number.
- **m-7.** E1 decision rule (line 382): lr is selected and arms are promoted on the same dev split; 5 backbones × 3 lr under ±4–5 pp CIs with no multiple-comparison handling. Select lr on the calibration split (968 records exist, LE #51) and promote on dev.
- **m-8.** Line 110 / B chain: Rank-DistiLLM "≥ teacher" should carry "teacher is +3.2 over the un-distilled baseline" where it is used as a KD-lift analogue.
- **m-9.** SQ2 #2 line 207 "Set-Encoder ≈ monoELECTRA": with the corrected rows Set-Encoder-330M wins ColBERTv2-DL19 by +2.4 and loses BM25-DL19 by −0.6 (VBH #18); "n.s." (authors) is fine, "adds nothing" is not.
- **m-10.** Line 199 / C20: "the 36T-token pretraining gap … is not captured by Ettin's 2T-token matched pairs" is asserted as a mechanism; B1 also reports its 2T decoders *beat* SmolLM2 (4T+ tokens) at equal size (LBH line 70), so tokens are not a clean proxy for the gap. Keep the gap, weaken the mechanism claim.

---

## Observations

- **O-1.** The frame is ~80% one author's logs. Every number in §4's accuracy, coverage and ECE rows is Kev-relative (one to three seeds, ±4–5 pp CI on 656 items). This is what the brief's evidence policy asked for, but the synthesis should say it in one sentence at the top of §4, not only in §8.
- **O-2.** Is the encoder-favourable frame a search artifact? Partly no: LBH §6 records that no fine-tuned decoder-with-head at 135–600M exists in the literature alongside encoders, and the pre-LLM comparisons that do exist (BERT paper Table 1: OpenAI GPT 72.8 vs BERT-base 79.6 GLUE at ~110M — in the corpus as H1 but cited only for SWAG) point the same way as Ettin. Partly yes: the *direction* is robust, the *magnitude at a 4x size ratio and 18x token ratio* is not in any source, and the synthesis filled that hole with the 1x-size-ratio number (CR-1). The brief's own SQ1 wording ("what does the MC-QA / cross-encoder literature say about small encoders") pre-selected the search.
- **O-3.** The 30/47/23 mix is a property of transfer-v4, not of the product. Kev's 384-token state cap (LE #13) does not bite on transfer-v4 (short sources) but decides JevBench's hard tier. If tinyjev's real decision mix is policy-heavy with long states, ModernBERT's 8k context is worth more than anything in §4 and none of the transfer-v4 bands apply. The synthesis mentions the cap only to explain away JevBench (C5, C30).
- **O-4.** The document is framed encoder-first (A is the first recipe, "the encoder line continues" is E1's success branch) while its own table puts F at the top with the highest confidence. A reader skimming §4.1–4.2 takes away "build the 150M encoder"; the numbers say "the only arm expected above the anchor is the 0.6B decoder".
- **O-5.** SciQ (116 items, 15% of the suite, Kev-0.8B 0.91) ships with a supporting passage in `allenai/sciq`. Whether Kev renders it is not in the corpus. If it does, half the "knowledge block" is reading comprehension, where encoders are strong, and CR-1's knowledge-block pessimism halves. One `grep` in `kev/data.py` settles it.

---

## Strongest counter-argument to the synthesis

The synthesis has built a seven-arm comparative frame on a suite where the encoder arms have zero measurements on 53% of the items, and where its own strongest source (Ettin, B1) — read at the actual size ratio — predicts that a 149M encoder *ties* a 596M decoder on the other 47% at matched pretraining, before an 18x token gap that the synthesis itself calls "real". Read consistently, the corpus predicts every encoder arm lands 3–8 pp *below* Kev-0.6B, with the downside dominated by a knowledge block on which the only mechanism the synthesis offers (36T tokens of pretraining) is one no ≤400M encoder possesses. The only arm the evidence places above the anchor is F — the largest decoder the size budget allows — plus a KD lift whose one in-category read is a regulariser signature (OOD +2.6, in-distribution −4.1) and whose seven analogues close 3-pp gaps in other tasks. Meanwhile the anchor itself was measured under a recipe (LoRA r=16, lr ≥1e-4, one to three seeds) that at 4B was worth −4.7 pp, and has never been re-run at low lr or full FT below 0.8B. The honest one-line reading of this corpus is: *"A 0.1–0.3B model is expected to land 5–10 pp below Kev-0.6B on transfer-v4; parity with Kev-0.6B requires the largest permitted decoder; parity with Kev-4B is not supported by anything."* The synthesis says the third clause clearly (§7), the second only in §8's ordering, and never the first — its "±5" hides it.

---

## What's missing

1. **Per-block clean counts** (656) from the manifest and metric policy — zero cost, changes every §4 weight (M-3).
2. **SciQ rendering** (passage or not) from `kev/data.py` — zero cost, changes the sign on 15% of the suite (O-5).
3. **Zero-cost encoder knowledge probe**: MLM-cloze / zero-shot marker scoring of ModernBERT-base and DeBERTa-v3-base on the 116 MMLU + 116 SciQ dev items, mirroring `scripts/base_mmlu_probe.py`. Replaces the guessed "−1 to −13" with a number before E1 (CR-1).
4. **A full-FT, low-lr Qwen3-0.6B anchor** (0.1–0.5 H100-h) — the cheapest run in the ladder and the one that decides whether the target is 0.62 or 0.66 (M-7, CR-2).
5. **Regime-matched E1 arms** and one head×backbone cross arm (CR-2).
6. **Any fine-tuned decoder-with-head vs encoder comparison at a >2x size ratio on a ≥10T-token decoder.** None exists; say so in SQ1 "Silent" (line 201) rather than filling it with the 1x-ratio number.
7. **Ettin's decoder MNLI fine-tuning protocol** (head type, full FT?) — VBH #3 restored the row but not the recipe; it determines whether the 85.6 is comparable to a pointer-head LoRA decoder at all.
8. **Abbes's second column** (M-1) and **Rank-DistiLLM's baseline gap** (m-8) wherever those sources are load-bearing.
9. **A one-sentence statement of the flipped question** (see stress test row 4): "Is the 0.1–0.3B target reachable at Kev-0.6B parity?" — the synthesis's own numbers say "probably not without KD, and KD is unmeasured OOD"; it never says this in one place.
10. **Counter-evidence weighing in §4.2 B/E/F chains**: Cho, Stanton, Müller, Mishra, Furlanello are in §1c and absent from §4 (M-5).

---

## Stress-test table

| Test | What changes | Does the frame survive? |
|---|---|---|
| **Remove the strongest source — Kev's logs (S1/S8/S12/S13)** | Every anchor, every per-block row, the loss screen, the LS-coverage collapse, the temperature-transfer result, the coverage numbers, and all of §4's accuracy/coverage/ECE rows vanish. What remains is literature-only: encoder ≈ one size step at matched pretraining (B1/B2); pointer/marker heads and no zero-shot letter logits (H1–H3, H15–H17); KD helps in-distribution (3b.2–3b.5); **Desai (3c.3, PR-D) says TS does little under shift** — the opposite of the synthesis's post-hoc default. No ladder can be scored. | **No.** The frame is Kev-calibrated by construction; that is the brief's choice, but §4 should open by saying so. The objective/calibration conclusions partly *reverse* without Kev (TS under shift). |
| **Remove Ettin (B1) only** | Encoder-vs-decoder rests on B2 (+2.1 at 210M, CLM vs MLM at the same architecture), B5 (one collapse, one −4.5), and three few-/zero-shot comparisons (B4, H2, H3). The classification-block delta for A drops to "−2 to +2"; A's centre ≈ 0.59; A ≈ D(SmolLM2). | **Weakened.** The encoder line becomes a coin-flip against a 360M decoder, and §8's ordering collapses to "F, then everything else within noise". |
| **Remove the local replicas (S14, S15, S19–S21)** | Loses: the only sub-300M decoder held-out number (S20 0.585), the only KD-in-category read (S15), the K-scaling curve (S19), the full-FT letter-slot at 0.8B (S21). D(Gemma) loses its only anchor; B/E/F lose their only local KD prior; C23's "letter slot works at 0.8B after FT" goes silent. | **Mostly survives**, and gets *more* honest: D(Gemma) and the KD lift become "silent" rather than ranged. |
| **Flip the question: "Which recipe is expected to beat Kev-0.6B on transfer-v4?"** | Only F (0.62–0.68, centre 0.65, MED). Every 150–360M arm's centre is at or below the anchor even on the synthesis's own numbers (A 0.61, B 0.645, C 0.62, G 0.61, E-SmolLM2 0.60). Under CR-1 they all sit below. | The frame's answer to the flipped question is "build the 0.6B decoder with KD" — which contradicts the brief's *target* size (0.1–0.3B) but not its *ceiling* (0.6B). The synthesis never says this in one sentence. |
| **Flip the question: "Is the 0.1–0.3B target reachable at parity?"** | Reachable at Kev-0.6B parity: only if (i) the encoder knowledge probe (missing #3) comes back ≥0.35 MMLU / ≥0.85 SciQ and (ii) E2's KD lift is real OOD and (iii) the low-lr anchor does not move. Three unmeasured conditions, all required. Reachable at Kev-4B parity: no (§7). | The frame implies P(parity at 150M) is low but never estimates it; it should. |
| **Different context: policy-heavy, long-state product (JevBench-hard-like)** | transfer-v4's 30% knowledge block is irrelevant; the 384-token Kev cap becomes the dominant failure (VLE #5: Kev truncates; hard tier 40–47% for every Kev vs 74% Jev); ModernBERT's 8k context and Laya's 1024 become assets; C30's dismissal of JevBench flips into the most relevant evidence. | **Rankings invert**: encoders (or long-context decoders) first, and none of §4's numbers apply. The synthesis should state the product-mix assumption behind transfer-v4 once. |
| **Different context: classification-only product (tickets, routing, moderation)** | Knowledge block gone; encoder edge from B1/B2 applies directly; PAWS/Emotion/Tweet-style label noise dominates coverage (S1: 59% of errors on 37% of items). | Encoders win outright at 3–5x lower cost; F's advantage disappears; KD's value shrinks to calibration. |
| **Stricter inclusion: peer-reviewed + text-only** | Drops all local logs and replicas, all vendor cards, B5–B8, B10–B16, H5, A6, A8–A12, D1–D9, and every vision KD/calibration/selective source (3a.1–3a.3, 3b.7–3b.10, 3b.20–3b.21, 3c.7, 3c.9–3c.15). Survives: B1–B4, B9, H1–H3, H8–H17, 3a.4, 3b.3–3b.5, 3b.11, 3b.13–3b.16, 3c.1, 3c.3–3c.4, 3c.16–3c.18, A1–A4, A7, D2. | **Architecture and head conclusions survive and strengthen** (all Tier-1, text). **LS-kills-coverage has zero text evidence left.** **"Single T transfers OOD" reverses** (Desai). **Every number in §4 disappears.** The landscape section is empty. The synthesis would look like a 2023 NLP survey with no decision-model content — which is roughly the truth about how much of this frame is nine days old. |

---

## Summary line for the orchestrator

Verdict REVISE. 2 Critical / 8 Major / 10 Minor / 5 Observations. Strongest counter-argument: read at the real size ratio, the synthesis's own best source (Ettin) predicts every 150–360M encoder arm lands *below* Kev-0.6B, and the only arm the evidence ranks above the anchor is the largest permitted decoder plus an unmeasured KD lift — the "±5 pp" band hides a downside twice the size of the upside. Single change that most improves the frame: re-derive recipe A's block deltas consistently with B1 (149M vs 596M, not 150M vs 150M) and replace the guessed knowledge-block range with a zero-cost MLM-cloze probe on the 232 MMLU/SciQ items — then fix E1 so encoder and decoder arms share a fine-tuning regime and the anchor is a full-FT low-lr Qwen3-0.6B.
