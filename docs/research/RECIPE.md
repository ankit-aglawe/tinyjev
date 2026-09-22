# tinyjev — training recipe decision (v2, after Codex review)

Date: 2026-09-22. Derived from docs/research/phase2_investigation (4 verified evidence streams), phase3_synthesis (synthesis + devil's-advocate checkpoint 2), and a Codex second opinion (session 01a0c8d5). This file is the decision; the research files are the receipts.

## The call

Build **tinyjev-0.6b**: Qwen3-0.6B-Base + Kev-style pointer head, trained on Kev's public decision-v7 data, scored on Kev's frozen suites through our own `/v1/systemone` server with Kev's harness.

**First bet is the training regime, not distillation.** The public anchor (Kev-0.6B, 0.620 transfer-v4 dev) was only ever trained as LoRA r16 at lr 1e-4. Low LR was the largest recipe effect Kev measured (+4.7 pp at 4B, replicated) and full fine-tuning was never tried below 0.8B. That is the cheapest, best-supported lever. Soft-target distillation from Kev-4B is the second rung, conditional on the first, because its supporting evidence is vision-only.

Target: **+1 to +6 pp over 0.620** on transfer-v4 dev, confirmed once on the locked test, with calibration and ordinal quality reported alongside. Not 4B parity: the corpus does not support 4B accuracy below 0.6B.

The encoder size (ModernBERT-base, 149M) is a **separate footprint objective**, deferred until the 0.6b result is settled. It is not needed for the primary claim.

## Why each choice

| choice | pick | reason (evidence) |
|---|---|---|
| backbone | Qwen3-0.6B-Base | Only sub-1B arm the corpus ranks above the anchor (DA2). Attention-only: MLX-native, no DeltaNet. Read consistently, every 150–360M encoder arm lands 3–8 pp below Kev-0.6B. |
| head | Kev pointer head (`<decide>` · each `</opt>`) | 255 options, no option-letter binding (small decoders can't bind letters, chance even at 6.7B). No `option_isolation` (−3 to −4 pp, verified). Order handled by shuffling at train time. |
| regime (E1) | reproduce released config exactly; then LoRA vs full-FT at **matched** lr 5e-5; then full-FT lr 2e-5 | Isolates adaptation regime from LR. Codex: comparing full-FT 2e-5 against LoRA 1e-4 confounds the two. Pin epochs, effective batch, scheduler, head init, checkpoint selection. |
| objective | plain CE; label smoothing forbidden | Kev's 5-arm screen: nothing beats CE on coverage; smoothing collapses coverage@5% 0.576→0.006. |
| distillation (E2, conditional) | one teacher (Kev-4B, 1-epoch parent), one KD temperature, one weight; CE + α·KL | Teacher calibration predicts student accuracy better than teacher accuracy (vision, R² 0.92) — treat as hypothesis, not fact. Define KL direction, KD temperature, and reduction over variable K up front. Add 9B only if 4B shows a gain. |
| data | Kev decision-v7 train split, hash-pinned; nothing added for E1 | Same data as the anchor → clean comparison. "More data hurt" was measured under the old high-LR regime; revisit only after the regime is settled. |
| calibration | one temperature on decision-v7 dev; softmax-response for abstention; report ECE, Brier, coverage@5% and score MAE as acceptance criteria, not just accuracy | Learned abstention heads don't beat max-softmax (Feng, Jaeger, Varshney). TS transferred OOD on the Qwen3.5 generation; verify on ours. |
| eval | `kev.benchmark --remote` against our server | Comparable to every published number. Repeated selection on transfer-v4 dev makes it a tuning set: dev gains are selection-biased, the locked test (read once) confirms. |
| footprint | fp16 + INT8; 4-bit only if E5 shows < 1 pp AND calibration holds | 4-bit costs sub-1B decoders 3–10 MMLU points. "INT8 free" is unmeasured for these checkpoints on these backends: measure, including ECE. |

## Contamination rule

Never train on anything in transfer-v4 / v9 holdouts: `mmlu`, `emotion`, `tweet_offensive`, `qnli`, `paws`, `sciq`, `mmlu_pro`, plus Kev-synthetic `contrastive`, `legacy_holdout`, `composition_holdout`, `buried`, `unknowable`. Start only from raw base checkpoints — never GLUE-tuned, `gliclass-*`, or `*-zeroshot-*` variants.

## Teacher-target contract (Codex's catch)

Before any KD run:
- Cache targets keyed by (record id, question id, **ordered option identities**, rendered-input hash, teacher checkpoint revision).
- Option shuffling at train time must **remap** cached targets. The causal teacher is order-sensitive, so remapping is not the same as re-querying; decide once (remap) and record it.
- Use a **raw-logit path** for targets: `KevFamily.logits` divides by the serving temperature, so KD from served probabilities would double-scale.
- Validate boolean ordering, score-level indexing, and that teacher and student see identical visible content after token limits, before generating the cache.

## Ladder

| rung | what | arms | eval | decision rule | H100-h |
|---|---|---|---|---|---|
| **E0** local, free | (a) per-block clean counts from the manifest; (b) does Kev render SciQ with its support passage; (c) **export/backend smoke**: a Kev-format checkpoint round-trips through convert → MLX and torch → `/v1/systemone` → `kev.benchmark --remote` (done for kev-0.6b: 0.6204); (d) re-derive the anchor's exact released config (lr 1e-4, epochs, batch, scheduler) | transfer-v4 dev | nothing trains until (c) passes and (d) is pinned | 0 |
| **E1** regime | Qwen3-0.6B + pointer, decision-v7 | (a) released config, exact [expect ≈0.620]; (b) LoRA lr 5e-5; (c) full-FT lr 5e-5; (d) full-FT lr 2e-5 | v7 dev + transfer-v4 dev, per-source, plus ECE / coverage@5% / score MAE | (b) vs (c) isolates regime at matched LR; best arm becomes the base. If any arm beats (a) by ≥ 2 pp on transfer-v4 dev with ECE not worse, that is the candidate | ~2 (LoRA ≈ 20 min/run per PLAN; full-FT timing unmeasured, assume 2×) |
| **E2** distillation, conditional | E1-best regime + KD from Kev-4B-1ep | CE-only repeat; CE + α·KL at one (T, α); one extra repeat of each | same | ship KD only if it beats the CE repeat by ≥ 1 pp with calibration not worse; otherwise ship E1-best and record the null | ~2.5 (incl. ~0.3 teacher inference) |
| **E4** seeds + lock | winner **and** matched baseline × 3 seeds | — | mean ± paired, record-clustered CI on dev; then ONE locked-test read | a single run is not a result (Kev: 75.6 vs 69.5 on identical config) | ~1.5 |
| **E5** footprint | fp16 / INT8 / 4-bit on MLX + torch; CoreML export attempt | — | transfer-v4 dev accuracy AND ECE delta; M1 latency; iPhone if a device is available | ship INT8 only if < 1 pp and calibration holds; 4-bit same bar | 0 (local) |
| **E3** encoder, deferred | ModernBERT-base + [MASK]-marker scorer, full FT, same data, same teacher cache | lr {2e-5, 5e-5} × {CE, CE+KD} | same | separate footprint objective; run only after E4 lands; publish only with its own measured number | ~2 |

**GPU total ≈ 6 H100-hours for E1–E4 (E3 +2 if run).** At ~$4/H100-h (assumption, confirm against the Modal ledger) ≈ **$25–35 for the main ladder**. Full-FT timing, teacher throughput, evaluation overhead and reruns are the unmeasured part; **$50 is a budget cap, not an estimate.** Engineering time for the training script and the target cache is separate and larger than the GPU bill. Nothing launches until the ledger is read and the total stated.

## Expected outcome, stated before running

| model | params | transfer-v4 dev | vs anchors |
|---|---|---|---|
| tinyjev-0.6b | 596M | 0.620 + (1 to 6 pp); failure (≤ 0.620) is a real possibility and will be reported | Kev-0.6B 0.620 · Kev-4B 0.797 · Jev 0.857 |
| tinyjev-0.15b (deferred) | 149M | no number claimed until E3 runs | first sub-0.5B model with a number on Kev's suites, if it runs |

The claim, if it lands: "beats the same-size public anchor on its own frozen suite, with calibration reported, confirmed once on the locked test." Not "best sub-1B open model" — that needs a defined competitor set evaluated on the same suite, which does not exist yet.

## Risks and their detectors

| risk | detector | mitigation |
|---|---|---|
| low-LR / full-FT gain doesn't replicate at 0.6B | E1 (a) vs (b)/(c)/(d) | 2 H100-h to find out; record the null |
| KD only regularizes an under-tuned baseline (in-category signature: OOD +2.6, in-dist −4.1) | E2 CE-repeat vs KD at matched regime | KD is conditional on beating the CE repeat |
| dev split over-fitted by repeated selection | locked test read once; paired CI | pre-register the decision rules above |
| calibration regresses while accuracy rises | ECE / coverage@5% as acceptance criteria at every rung | reject arms that trade calibration for accuracy |
| teacher/student format mismatch or double temperature scaling | teacher-target contract checks before caching | raw-logit path; hash-keyed cache |
| single-run luck | E4 seeds on winner and baseline | paired CI, always |
| contamination | E0 manifest audit; raw bases only | holdout list above |
| Kev changes suites/leaderboard under us | pin dataset revisions + suite sha256 | cite the pins |
| INT8/4-bit hurts calibration even when accuracy holds | E5 measures ECE, not just accuracy | ship fp16 if needed |

## E1 outcome (2026-09-22) and precommitted E2 gates

E1: LoRA 0.614 (released config, reproduced) / 0.625 (lr 5e-5); full-FT 0.483 (5e-5) / 0.581 (2e-5) on
transfer-v4 dev. The regime lever is null at 0.6B; full fine-tuning forgets the base. Details: benchmarks/e1/.

E2 = KD via Kev's anchor loss to Kev-4B pointer-head targets, anchor_w 1.0 (a 50:50 hard-label / teacher
mix, Codex), on the LoRA-5e-5 regime; two KD seeds + one CE repeat, giving two matched CE/KD seed pairs
with E1(b). Before training: tools/check_targets.py (coverage, entropy, p(gold) by source). Gates on the
mean paired KD − CE transfer-v4 accuracy:

| E2 result | decision |
|---|---|
| ≤ 0 | stop tuning the 0.6B; ship anchor parity if it meets the product need, else pivot |
| 0 to < 1 pp | stop, unless selective prediction (cov@5%) improves materially |
| ≥ 1 pp, both seeds positive | fund one more matched CE/KD seed pair, then locked test once |
| ≥ 1 pp, seeds disagree | one tie-break pair, no alpha search |

Continuation threshold ≈ 0.635 on transfer-v4 dev. One point is ~7 of 656 questions: a screening
threshold, not proof. The 149M encoder is a footprint experiment, not an accuracy rescue.

