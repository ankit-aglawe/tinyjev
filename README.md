# tinyjev

**Tiny typed-decision models you can run on the laptop you already own.**

Give it a state and typed questions — `choice` over options, `noul` (yes/no probability), `score` (ordinal) — and get calibrated probabilities back in one forward pass. No text generation, nothing to parse, no label that wasn't in your list. The category is TypeSafe's Jev; these are open weights, trained by us, scored on the public frozen suites everyone else in the category uses, and served on MLX (Apple Silicon) or PyTorch (CPU / CUDA / MPS).

[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20weights-AnkitAI%2Ftinyjev-blue)](https://huggingface.co/AnkitAI/tinyjev)
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

`tinyjev serve tinyjev-0.6b` gives you `POST /v1/systemone`, the same request shape TypeSafe, Kev and Laya clients already speak, on `127.0.0.1:8077`.

## The number

Every model in this category reports on its own chosen tasks. We report on **Kev's frozen suites**, scored with **Kev's own harness**, so the row can be compared with the published numbers directly.

| model | params | transfer-v4 **dev** (held-out sources) | transfer-v4 **test** (read once) | ECE (test) |
|---|---|---|---|---|
| **tinyjev-0.6b** | 596M | **0.625** | **0.663** | 0.082 |
| Kev-0.6B, published (Qwen3 generation) | 596M | 0.620 | 0.642 | — |
| Kev-0.6B, released config reproduced by us | 596M | 0.614 | 0.657 | 0.108 |
| Kev-4B | 4B | 0.797 | 0.837 | — |
| Jev (hosted, TypeSafe) | ? | 0.857 | — | — |

Read that honestly: at 0.6B we are at the public anchor's level, a couple of points ahead on the locked test with better calibration. We are not at 4B accuracy, and nothing we tried gets there at this size. The section after next says what we tried.

**Runs on a base M1 with 16 GB** (not a Pro, not a Max), scored on the same suite through the same harness:

| build | transfer-v4 dev | ECE | p50 latency | weights |
|---|---|---|---|---|
| MLX fp16 | 0.620 | 0.140 | 65 ms | 1.2 GB |
| MLX INT8 | 0.620 | 0.136 | 58 ms | ~0.6 GB |
| MLX 4-bit | 0.599 | 0.097 | 59 ms | ~0.35 GB |

INT8 is free. 4-bit costs two points. Quantization changes memory, not speed, on an M1.

## What we tried at 0.6B, and what happened

All on Kev's `decision-v7` training partition, all scored on `transfer-v4` dev, all in [`benchmarks/`](benchmarks/) with Kev's raw `result.json` files. One seed per row unless stated; seeds agree to 0.15 pp where we repeated.

| lever | result | verdict |
|---|---|---|
| reproduce Kev's released config (LoRA r16, lr 1e-4) | 0.614 | reproduces the published 0.620 within run noise |
| LoRA at half the LR (5e-5) | **0.625** | ships as tinyjev-0.6b |
| full fine-tune, lr 5e-5 | 0.483 | forgets the base: mmlu 0.34, emotion 0.40 |
| full fine-tune, lr 2e-5 | 0.581 | better, still −4 pp; learns rule composition, loses knowledge |
| distillation from Kev-4B (Hinton KD, T=1 and T=3, two seeds each) | −1.5, −1.2, −1.1, −0.15 pp vs matched CE | **hurts**; the 4B teacher is near-one-hot on its own training set |
| 4-bit MLX | −2.1 pp | ship INT8 |

The evidence going in (a 200-source verified literature pass, in [`docs/research/`](docs/research/)) already said the same thing from the outside: Kev's own ladder shows accuracy collapsing below 4B, and no sub-0.5B open model had a number on these suites at all. We pre-registered stop rules before spending and stopped when they fired ([`docs/research/RECIPE.md`](docs/research/RECIPE.md)).

## The tiny one did not make it

We also trained the model the name promises: ModernBERT-base, 149M, an encoder with a `[MASK]`-marker scorer, on the same data with the same batching and schedule as the 0.6B, scored the same way. It came in at **0.532** on transfer-v4 dev (0.714 in-distribution) — nine points below the 0.6B, with knowledge questions at chance. Our pre-registered bar to ship it was 0.55. It doesn't ship. The number is in [`benchmarks/e3/`](benchmarks/e3/) because it is the first sub-0.5B result on these suites and the next person should not have to spend the $1 to learn it.

## How it works

Qwen3-0.6B-Base with a pointer head, Kev's design: the state, then per question `<q> instructions <opt> option </opt> … <decide>`; the hidden state at `<decide>` is dot-producted against each `</opt>` hidden state, softmax over the options. One causal row per question; the state's KV cache is computed once and shared across questions. Trained with plain cross-entropy (label smoothing destroys selective-prediction coverage; we checked the literature and Kev's own screen), one temperature fitted post hoc, option order shuffled during training. The decision head runs in fp32 numpy on every backend so MLX and torch give identical answers.

The runtime also converts and serves two other open models in the same layout, useful as baselines: [NanoJev](https://github.com/TianyuCodings/NanoJev) (its inference code refuses to run without CUDA; ours matches its authors' CUDA predictions on 2492/2496 test questions) and [Kev-0.6B](https://github.com/jaredpalmer/kev) (14/14 parity with `kev.model`).

```
tinyjev models                       # what's on the Hub
tinyjev serve tinyjev-0.6b           # /v1/systemone on :8077, --quantize 8 for INT8
tinyjev ask tinyjev-0.6b req.json    # one request
tinyjev play snake --model nanojev   # NanoJev playing its own game, one forward pass per move
```

## Reproduce

Training runs through Kev's own study runner on Modal with a 100-line patch ([`experiments/kev-tinyjev.patch`](experiments/kev-tinyjev.patch): full fine-tune support, a distillation temperature, teacher-target and encoder entry points). Plans are in [`experiments/`](experiments/). Total GPU for everything in this README: about 4 H100-hours.

```bash
git clone https://github.com/jaredpalmer/kev && cd kev && git apply ../tinyjev/experiments/kev-tinyjev.patch
uv run modal run modal_app.py::study --suite evals/v7/decision-v7 --plan ../tinyjev/experiments/e1-lora.json --name e1 --gpu H100
```

Serving-side parity and latency: `tools/check_parity_kev.py`, `tools/bench.py`; the M1 numbers above came from `kev.benchmark --remote http://127.0.0.1:8077`.

## Limits

- 0.6B is the category's standard small size, not tiny. The 149M encoder we trained to be the tiny one scored 0.532 and does not ship; the name is a target we did not hit at this data scale.
- One seed per row except where stated. Kev's own runs spread several points on identical configs; treat single-run differences under 2 points as noise.
- Held-out accuracy of 0.62–0.66 means roughly one in three new-source questions is wrong. Use the probabilities: coverage at a 5% error budget is low at this size (0.07–0.08 on transfer-v4), so gate on confidence and escalate the rest.
- Locked test read once, ungated, same convention as the published Kev 0.6B reads.

## Credits

Data, suites, harness, training code and the pointer-head design are [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer (Apache-2.0), used as published; our changes are in the patch file. Qwen3-0.6B-Base is Apache-2.0 (Alibaba). NanoJev is MIT (OpenJev contributors). The category's interface is TypeSafe's Jev. tinyjev is MIT.
