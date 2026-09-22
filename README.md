<p align="center">
  <img src="https://raw.githubusercontent.com/ankit-aglawe/nanojev-mlx/main/assets/snake.gif" alt="NanoJev playing Snake on a base M1 through MLX, every move one forward pass, real recorded decisions" width="100%" />
</p>

# nanojev-mlx

**NanoJev without CUDA.** Runs the [NanoJev](https://github.com/TianyuCodings/NanoJev) decision model on a Mac: MLX on Apple Silicon, PyTorch CPU anywhere else.

Upstream cannot run here at all. Its inference entry point asserts a CUDA device and exits:

```
ValueError: 此原型推理入口需要可用CUDA设备；本命令未启用CPU或远程回退
            (this prototype inference entry requires CUDA; no CPU or remote fallback)
```

This repo is that fallback. Same weights, same prompt bytes, same answers, on the laptop you already own.

[![PyPI](https://img.shields.io/pypi/v/nanojev-mlx.svg)](https://pypi.org/project/nanojev-mlx/)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20weights-AnkitAI%2Fnanojev--mlx-blue)](https://huggingface.co/AnkitAI/nanojev-mlx)
[![Upstream](https://img.shields.io/badge/upstream-NanoJev-0a0a0a)](https://github.com/TianyuCodings/NanoJev)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Numbers, base M1 with 16 GB

Not an M3 Max. Not a Pro. The 8-core M1 with 16 GB that most Macs in the wild are. Measured on this machine, same requests, same weights, PyTorch built from upstream's own `DecisionModel`.

| request | candidate paths | PyTorch CPU | PyTorch MPS | **MLX fp16** | vs CPU | vs MPS |
|---|---:|---:|---:|---:|---:|---:|
| yes/no question | 1 | 194 ms | 82 ms | **36 ms** | 5.3x | 2.2x |
| 2-way choice | 2 | 247 ms | 116 ms | **64 ms** | 3.9x | 1.8x |
| 4-way choice | 4 | 394 ms | 246 ms | **119 ms** | 3.3x | 2.1x |
| 8-way choice | 8 | 513 ms | 328 ms | **140 ms** | 3.7x | 2.4x |
| choice + yes/no + score, one state | 8 | 567 ms | 386 ms | **169 ms** | 3.4x | 2.3x |
| model load | | 14.8 s | 15.8 s | **0.3 s** | | |

Median across all twelve fixtures: **3.5x faster than PyTorch CPU, 2.2x faster than PyTorch MPS**, and the model loads 50x faster. Reproduce with `tools/bench.py`; every timing sample is in [`benchmarks/`](benchmarks/).

**Parity:** the port selects the same answer as the PyTorch original on **15/15** fixture questions in both fp16 and fp32, checked against CPU and MPS references. Worst per-option probability delta is 1.2e-3 in fp16 and 8.6e-7 in fp32. Fixtures and reference outputs are committed under [`tests/fixtures/`](tests/fixtures/).

## Install

```bash
pip install nanojev-mlx
```

Apple Silicon, macOS 14+, Python 3.9+. The first load downloads ~1.1 GB of fp16 weights from the Hub; after that it is fully local.

```python
import nanojev_mlx

agent = nanojev_mlx.load("AnkitAI/nanojev-mlx")

result = agent.predict({"states": [{
    "id": "ticket-4411",
    "state": "Hi, we were billed twice for March. Refund the duplicate today or we cancel.",
    "questions": {
        "team":    {"type": "choice",  "instructions": "Which team should handle this?",
                    "criteria": {"billing": "invoices, refunds, payments",
                                 "support": "bugs and outages",
                                 "sales":   "pricing and contracts"}},
        "churn":   {"type": "boolean", "instructions": "The customer threatens to leave."},
        "urgency": {"type": "score",   "instructions": "How urgent is this?",
                    "criteria": ["can wait", "this week", "today"]},
    }}]})

answers = result["states"][0]["answers"]
answers["team"]["choice"]        # -> a label, with probabilities over all three
answers["churn"]["p_true"]       # -> a probability
answers["urgency"]["score"]      # -> expected level, 0..2
result["execution"]["model_ms"]  # -> the forward pass, measured
```

One forward pass answers every question. No tokens are generated, so there is nothing to parse and no label can be hallucinated: the model can only point at options you supplied.

## Watch it play

The GIF at the top is a real run, rendered at the speed the decisions were made. Reproduce it in your terminal:

```bash
nanojev-mlx play snake --model AnkitAI/nanojev-mlx --safety
```

`--safety` also asks the model, in the same forward pass, whether each move avoids a collision, and overrides its own pick when it says no. In the recorded run (seed 7) it survives 80 moves and eats 11 food with zero overrides needed.

## Serve it

```bash
nanojev-mlx serve AnkitAI/nanojev-mlx --port 8077
```

Two routes. `POST /predict` takes NanoJev's native `{"states": [...]}` payload. `POST /v1/systemone` takes a TypeSafe System One request (`state` + `questions`, with `noul`/`choice`/`score`), so any client written for Jev, Kev or Laya works unchanged:

```bash
curl -s localhost:8077/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size.",
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team?",
               "criteria": {"returns": "wrong or damaged items", "shipping": "delays"}},
    "urgent": {"type": "noul",   "instructions": "Needs a human today?"}}}'
```

Loopback only, no authentication. Keep it local.

## What this model is, honestly

The published NanoJev checkpoint is a **games model**: a Qwen3-0.6B backbone fine-tuned on Maze, Snake, ViZDoom Basic and Predict Position decisions. It answers arbitrary typed questions through the same interface, but that is not what it was trained on. Numbers measured here, on this port, same seeds every time:

| task | result on this machine |
|---|---|
| Snake 8x8, model alone, 10 seeds | median 8 food, best 16; always dies within 200 moves |
| Snake 8x8, model + its own safety answers | 11 food in 80 moves, survived (recorded run) |
| Maze, per-step agreement with a BFS oracle | 56% on 4x4, 49% on 6x6 |
| Maze, solved end to end | rare; a per-step rate near chance compounds |

Upstream's model card reports Maze 4/10 and Snake 8/8 on its matched test set, and a 30-food Snake showcase that it notes used "common code planners". The safety-question mode above is the all-model analogue of that showcase: no external planner, just more questions in the same pass.

For a general-purpose typed decision model, look at [Kev](https://github.com/jaredpalmer/kev) or [Laya](https://github.com/NandhaKishorM/laya). This port exists because NanoJev is the most-starred open replica and could not previously run on a Mac.

## A bug you will hit if you run upstream on transformers 4.x

The checkpoint's `backbone_config/config.json` was written by transformers 5.17 and stores the RoPE base as `rope_parameters.rope_theta = 1000000`. transformers 4.x does not read that key and **silently falls back to `rope_theta = 10000`**. Every hidden state changes, no error is raised, and the model quietly gets worse: on our fixtures the selected answer flipped on 4 of 15 questions.

`tools/reference_pytorch.py` detects the mismatch and forces the declared value, with a printed warning. If you build your own reference, do the same. The MLX port reads the declared value directly.

## How it works

Upstream's `DecisionModel`, reimplemented in ~150 lines of MLX ([`nanojev_mlx/model.py`](nanojev_mlx/model.py)):

```
for each candidate:  State: … \n Question type: … \n Candidate: … \n Decision: <eos>
                     └── Qwen3-0.6B ── hidden state at <eos>
                                        └── LayerNorm ── Linear(1024→1) ── base logit
choice questions only:
    candidates attend to each other through a 4-head set-attention block,
    conditioned on log(K), and add a residual correction to each logit
softmax over the offered candidates
```

Every candidate path of a state row starts with the same state tokens. The runtime computes those once and lets the K suffixes attend to a broadcast KV cache ([`pool_shared`](nanojev_mlx/model.py)). Same maths, less work: on a mid-game Snake state with the safety questions on, 599 ms becomes 374 ms. It switches on automatically above 96 shared tokens, where it starts to pay.

Backbone weights are fp16; the twelve head tensors stay fp32. The tokenizer is Hugging Face's Rust `tokenizers`, so PyTorch and Transformers are not runtime dependencies.

## Reproduce everything

```bash
git clone https://github.com/ankit-aglawe/nanojev-mlx && cd nanojev-mlx
pip install -e '.[dev,reference]'

# 1. convert the upstream checkpoint yourself (2.4 GB fp32 -> 1.1 GB fp16)
nanojev-mlx convert /path/to/C-Tianyu--NanoJev ~/.cache/nanojev-mlx/fp16

# 2. regenerate PyTorch references on CPU and MPS (needs an upstream checkout for its DecisionModel)
python tools/reference_pytorch.py --checkpoint /path/to/C-Tianyu--NanoJev \
    --source-scripts /path/to/NanoJev/scripts --cases tests/fixtures/cases.json \
    --out tests/fixtures/reference_cpu.json --device cpu

# 3. parity, latency, game capability
python tools/check_parity.py --model ~/.cache/nanojev-mlx/fp16 --reference tests/fixtures/reference_cpu.json
python tools/bench.py --model ~/.cache/nanojev-mlx/fp16 --cases tests/fixtures/cases.json
python tools/eval_games.py --model ~/.cache/nanojev-mlx/fp16

# 4. the GIF
python tools/render_gif.py --model ~/.cache/nanojev-mlx/fp16 --out assets/snake.gif --safety

pytest   # unit tests; the parity test runs when converted weights are present
```

## Limits

- Apple Silicon only for the MLX path. The PyTorch reference runner in `tools/` covers CPU and MPS.
- One request at a time in the server; no cross-caller batching.
- No training or fine-tuning; that lives upstream.
- The weights are upstream's, unchanged. Anything the model gets wrong, it also gets wrong on CUDA.

## Credits

Independent port, not an official NanoJev release. NanoJev is MIT, © 2026 OpenJev contributors; its request schema, prompt layout, Snake environment and model architecture are reproduced here under that licence (see [NOTICE](NOTICE)). Weights: [C-Tianyu/NanoJev](https://huggingface.co/C-Tianyu/NanoJev). Backbone: Qwen3-0.6B, Apache-2.0. Inspired by the interface of TypeSafe's Jev.

MIT.
