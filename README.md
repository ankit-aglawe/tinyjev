# tinyjev

**Tiny typed-decision models you can run on the laptop you already own.**

Give it a state and typed questions — `choice` over options, `noul` (yes/no probability), `score` (ordinal) — and get probabilities back in one forward pass. No text generation, nothing to parse, no label that wasn't in your list. The category is TypeSafe's Jev; these are open weights, trained by us, scored on the public frozen suites with their author's own harness, and served on MLX (Apple Silicon) or PyTorch (CPU / CUDA / MPS).

[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20weights-AnkitAI%2Ftinyjev--0.6b-blue)](https://huggingface.co/AnkitAI/tinyjev-0.6b)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Suites](https://img.shields.io/badge/scored%20on-Kev%20frozen%20suites-0a0a0a)](https://github.com/jaredpalmer/kev)

```bash
pip install 'tinyjev[mlx]'     # Apple Silicon
pip install 'tinyjev[torch]'   # everything else
```

```python
import tinyjev

agent = tinyjev.load("tinyjev-0.6b")          # ~1.2 GB fp16, downloads once
agent.predict({
    "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    "questions": {
        "team":     {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                  "shipping": "Delivery status, delays, lost packages",
                                  "billing": "Charges, invoices, payment problems"}},
        "escalate": {"type": "noul",   "instructions": "Does this need urgent human attention?"},
        "anger":    {"type": "score",  "instructions": "How angry is the customer?",
                     "criteria": ["calm", "frustrated", "very angry"]},
    }})
```

`tinyjev serve tinyjev-0.6b` gives you `POST /v1/systemone` on `127.0.0.1:8077`, the request shape the TypeSafe SDK, Kev's harness, Laya and the browser/agent wrappers in the ecosystem already speak. The Hub repo is a standard transformers layout: `AutoModel.from_pretrained("AnkitAI/tinyjev-0.6b")` loads the backbone on its own; `tinyjev` adds the decision head.

## The number

Scored on Kev's frozen `transfer-v4` suite (held out from fine-tuning) with Kev's own `kev.benchmark`, so the rows compare directly with the published Kev numbers.

| model | params | transfer-v4 **dev** | transfer-v4 **test** | ECE (test) | coverage@5% (test) |
|---|---|---|---|---|---|
| **tinyjev-0.6b** | 596M | **0.625** | **0.663** | 0.082 | 0.069 |
| Kev-0.6B, released config, reproduced by us | 596M | 0.614 | 0.657 | 0.108 | 0.078 |
| Kev-0.6B, published (Qwen3 generation) | 596M | 0.620 | 0.642 | — | — |
| Kev-4B, published | 4B | 0.797 | 0.837 | — | — |
| Jev (hosted), published by Kev | ? | 0.857 | — | — | — |

What that table does and does not say:

- Dev was used repeatedly for selection. Test results are exploratory, ungated reads, once per candidate; neither candidate passed Kev's promotion gates (nor do Kev's own published 0.6B reads). The planned three-seed confirmation was not run.
- Against our reproduced baseline, the test gain is 0.6 points, four net questions of 656. ECE improved (0.082 vs 0.108); coverage at a 5% error budget fell (0.069 vs 0.078). Against the published number the gain is 2.1 points, but that comparison crosses runs on different hardware.
- 0.6B is the category's standard small size. This is the anchor's level with the anchor's recipe, tuned slightly. It is not 4B-class, and the next section is the list of things that did not get it there.

**On a base M1 with 16 GB**, same suite, same harness, served through `tinyjev serve`:

| build | transfer-v4 dev | ECE | p50 latency | notes |
|---|---|---|---|---|
| H100, fp32 (the training run's own score) | 0.625 | 0.136 raw / 0.074 at T=1.46 | — | reference |
| MLX fp16 | 0.620 | 0.140 | 65 ms | three net questions below the H100 score; cause not isolated |
| MLX INT8 (Linear layers) | 0.620 | 0.136 | 58 ms | aggregate accuracy preserved in this run |
| MLX 4-bit (Linear layers) | 0.599 | 0.097 | 59 ms | −2.1 points |

Latencies are single-run medians over the 764 dev requests; repeated timing trials were not recorded. Embeddings and the fp32 decision head are never quantized, so on-disk size is not a bit-width multiple; measured resident sizes are on the to-do list. The shipped checkpoint applies the fitted temperature (1.46) to its outputs.

## What we tried at 0.6B, and what happened

All on Kev's `decision-v7` training partition, all scored on `transfer-v4` dev with the raw `result.json` files in [`benchmarks/`](benchmarks/). One seed per row unless stated. The two CE seeds we ran scored 0.6250 and 0.6235; that is two runs, not an uncertainty estimate.

| lever | transfer-v4 dev | what we saw |
|---|---|---|
| reproduce Kev's released config (LoRA r16, lr 1e-4) | 0.614 | within run noise of the published 0.620 |
| LoRA at half the LR (5e-5) | **0.625** | ships as tinyjev-0.6b |
| full fine-tune, lr 5e-5 | 0.483 | mmlu 0.21, emotion 0.49; well below LoRA |
| full fine-tune, lr 2e-5 | 0.581 | mmlu 0.34, emotion 0.40; rule-composition blocks improved, knowledge sources fell |
| distillation from Kev-4B (KL to its pointer-head targets, T=1 and T=3, two seeds each) | −1.5, −1.2, −1.1, −0.15 vs matched CE | this recipe did not help in four matched comparisons; the teacher's targets are near one-hot on its own training set, which is a plausible reason, not a demonstrated one |
| 4-bit MLX | −2.1 | ship INT8 |

The stopping rules, including the 0.55 bar for the encoder below, were written into [`docs/research/RECIPE.md`](docs/research/RECIPE.md) before the first GPU run (commit `836e6b1`) and applied as written; the deviations (no three-seed confirmation, no paired CIs) are recorded there. The research behind the recipe, 161 verified sources across four streams, is in [`docs/research/`](docs/research/).

## The tiny one did not make it

We also trained the model the name is for: ModernBERT-base, 149M, an encoder with a `[MASK]`-marker scorer, on the same data with the same batching and schedule as the 0.6B, scored the same way, two learning rates, one seed each. It scored **0.532 / 0.531** on transfer-v4 dev (0.71 / 0.74 in-distribution), nine points below the 0.6B, with MMLU near chance. The bar to ship it was 0.55. It does not ship. The runs are in [`benchmarks/e3/`](benchmarks/e3/) for reuse.

## How it works

Qwen3-0.6B-Base with a pointer head, Kev's design: the state, then per question `<q> instructions <opt> option </opt> … <decide>`; the hidden state at `<decide>` is dot-producted against each `</opt>` hidden state, softmax over the options. One causal row per question; the state's KV cache is computed once and shared across questions. Trained with plain cross-entropy, option order shuffled during training, one temperature fitted on the calibration partition afterwards. The decision head runs in fp32 numpy on every backend; the backbones differ (MLX vs torch), so answers match on our fixtures but probabilities differ at the third decimal.

The runtime also converts and serves two other open models in the same layout, as baselines: [NanoJev](https://github.com/TianyuCodings/NanoJev), whose inference entry point refuses to run without CUDA (MLX fp16 selected-answer agreement with its authors' published CUDA predictions: 2492/2496, max probability difference 0.014), and [Kev-0.6B](https://github.com/jaredpalmer/kev) (selected answers match `kev.model` on 14/14 fixtures; max probability difference 0.005 on MLX, 0.003 on torch).

```
tinyjev models                       # what's on the Hub
tinyjev serve tinyjev-0.6b           # /v1/systemone on :8077, --quantize 8 for INT8
tinyjev ask tinyjev-0.6b req.json    # one request
tinyjev play snake --model nanojev   # NanoJev playing its own game, one forward pass per move
```

## Reproduce

Training runs through Kev's own study runner on Modal with a small patch ([`experiments/kev-tinyjev.patch`](experiments/kev-tinyjev.patch), 79 added lines: full fine-tune support, a distillation temperature, teacher-target and encoder entry points; `tools/kev_teacher_targets.py` and `tools/encoder_train.py` are copied into Kev's `scripts/`). Plans are in [`experiments/`](experiments/). Recorded trial time for everything in this README is 2.2 H100-hours; with teacher inference, locked reads and container overhead, call it about 3.

```bash
git clone https://github.com/jaredpalmer/kev && cd kev && git apply ../tinyjev/experiments/kev-tinyjev.patch
cp ../tinyjev/tools/kev_teacher_targets.py ../tinyjev/tools/encoder_train.py scripts/
uv run modal run modal_app.py::study --suite evals/v7/decision-v7 --plan ../tinyjev/experiments/e1-lora.json --name e1 --gpu H100
```

Serving-side parity and latency: `tools/check_parity_kev.py`, `tools/bench.py`; the M1 rows above came from `kev.benchmark --remote` against a local `tinyjev serve`.

## Limits

- One seed per row except where stated. Kev's own runs spread several points on identical configs.
- Held-out accuracy of 0.62–0.66 means roughly one in three new-source questions is wrong. The coverage@5% figures are in-sample maxima over confidence thresholds on the suite, not a deployed error guarantee; validate a threshold on your own data before automating on it.
- The 149M encoder scored 0.532 and does not ship; the name is a target we did not hit at this data scale.
- Locked test read once, ungated, the same convention as the published Kev 0.6B reads.

## Credits

Data, suites, harness, training code and the pointer-head design are [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer (Apache-2.0), used as published; our changes are in the patch file. Qwen3-0.6B-Base is Apache-2.0 (Alibaba). NanoJev is MIT (OpenJev contributors). The category's interface is TypeSafe's Jev. tinyjev is MIT.
