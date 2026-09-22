# nanojev-mlx — design

Date: 2026-09-22
Status: approved, pre-implementation

## Goal

Ship an MLX port of [NanoJev](https://github.com/TianyuCodings/NanoJev) that runs on a
base-model Apple Silicon Mac, and earn 100+ GitHub stars from the current TypeSafe Jev
ecosystem wave.

Stars are the objective. Everything below is chosen to serve that, not to serve
generality, research novelty, or long-term maintenance.

### Success criteria

| | target |
|---|---|
| GitHub stars | 100+ |
| Runs on | base M1 / 16 GB, no Pro, no Max |
| Numerical parity | selected answer matches PyTorch original on 100% of fixtures |
| Install | `pip install nanojev-mlx`, weights auto-download |
| Reproduction | one command produces the published latency table |

## Why NanoJev

Surveyed the five highest-starred repos in the category. Only NanoJev clears every bar.

| repo | stars | base model | MLX-portable | public weights |
|---|---|---|---|---|
| **NanoJev** | **1,074** | **Qwen3-0.6B** | **yes** | **yes** |
| jevlike | 1,008 | Qwen2.5-0.5B | yes | no |
| simple-jev | 429 | Gemma 4 26B / Qwen3.5 | no | n/a |
| decider | 99 | Qwen3.5 / 35B-A3B | no | yes |
| reflex | 79 | Qwen3.5 | no | n/a |

The deciding factor is the backbone. Everything below NanoJev is built on Qwen3.5, which
interleaves Gated DeltaNet layers. Those have no fast Metal kernels, which is why Kev sits
at 779 ms on an M5 and why its maintainer has not shipped MLX yet. Porting them means
writing recurrent kernels — unbounded work.

NanoJev's published config is `Qwen3ForCausalLM`, hidden size 1024, head dim 128, and
every entry in `layer_types` is `full_attention`. mlx-lm runs Qwen3 today.

Demand evidence: 2,543 downloads and 75 likes on `C-Tianyu/NanoJev`. No MLX port exists —
confirmed against the repo itself and against the Apple Silicon section of `awesome-jev`.

## Positioning

Every MLX entry in this category headlines a Max or Ultra number. `laya-mlx` leads with
7.4 ms on an M3 Max with 40 GPU cores and 128 GB.

We cannot win that race on an 8-core M1 and will not try. The claim instead is:

> **Runs on the Mac you already have.** Base M1, 16 GB, no Pro, no Max.

This is unclaimed, honest, and true because of the hardware constraint rather than in
spite of it. It also makes 0.6B the correct model size rather than a compromise.

The README leads with the M1 number and the demo GIF, not with a comparison to laya-mlx.

## Architecture

Four units, each independently testable.

### `convert.py` — weight conversion
Input: `C-Tianyu/NanoJev/best.safetensors` (2.4 GB, fp32).
Output: fp16 MLX-format weights plus a config sidecar.
Depends on: safetensors, mlx.core.
Streams tensor by tensor; must not hold the full fp32 model in memory at once, because
16 GB is the target machine.

### `model.py` — the runtime
Input: token ids, option-token positions.
Output: logits at those positions.
Depends on: mlx, mlx-lm's Qwen3 implementation.
The scoring mechanism reads logits off the standard LM head at the position where the
answer token would be generated, restricted to the single distinct token that labels each
candidate. Whether a
separate trained head also ships in the checkpoint is resolved in Milestone 1; if one
exists, it is reimplemented here and nowhere else.

### `decide.py` — the typed-decision API
Input: a state (string, dict or list) and a dict of typed questions.
Output: per-option probabilities, the selected answer, and a confidence value.
Depends on: `model.py`, the tokenizer.
Owns prompt construction, option-token placement, softmax, and calibration. This is the
only unit that knows what `choice` / `score` / `noul` mean.

### `serve.py` — System One endpoint
Exposes `POST /v1/systemone` matching TypeSafe's schema.
Depends on: `decide.py`, stdlib `http.server`.
Binds 127.0.0.1, no authentication, single request at a time. Documented as local-only.

Schema compatibility is deliberate leverage: it makes the port usable by every SDK in the
ecosystem and scoreable by Kev's `kev.benchmark --remote` harness without us writing an
evaluation framework.

## Data flow

```
state + questions
  -> decide.py builds one sequence per question, assigning each candidate a distinct
     label token and recording the position to read
  -> model.py runs one forward pass
  -> logits gathered at option-token positions
  -> softmax, then temperature calibration
  -> {answer, probabilities, confidence}
```

## Testing

- **Parity fixtures.** A frozen set of (state, questions) cases, run through the original
  PyTorch NanoJev once to produce reference outputs, committed as JSON. The MLX port must
  select the same answer on every case, with per-option probability deltas reported.
  `laya-mlx` published 63/63 across fp32 and fp16; that format is the credibility bar and
  we match it.
- **Conversion test.** Round-trip a small tensor set and assert dtype and shape.
- **API test.** One request per question type against a running server, asserting schema
  shape and that probabilities sum to 1.
- **Latency harness.** Reproduces the published table. Reports P50 and P95 over a fixed
  number of runs, with the machine identified.

No framework beyond pytest. No fixtures directory hierarchy. Four test files.

## Error handling

- Missing or corrupt weights: fail at load with the expected path and checksum.
- A question with fewer than two options, or an option list that does not tokenize to
  distinct option tokens: reject at request validation with the offending question id.
- Request larger than a fixed byte ceiling: reject. Cross-origin: reject.
- Non-finite values anywhere in the probability path: raise rather than return NaN.

## Non-goals

Training. Fine-tuning. The games data pipeline. vizdoom. The upstream web UI beyond one
demo. Batching across callers. Authentication. CUDA. Windows. Any model other than
NanoJev.

## Risks and kill criteria

| risk | response |
|---|---|
| M1 latency is embarrassing | **Kill criterion.** Measured at end of Milestone 1, before any MLX code. If the PyTorch baseline on this M1 is so slow that no honest framing works, stop and report. |
| Public checkpoint is the games model, not a general decision model | Resolved in Milestone 1. If games-only, the positioning shifts to the game demos, which is also the GIF source. Not a blocker. |
| Upstream never links us | Four independent discovery surfaces, below. |
| Someone ports it first | The window is days. Milestone 1 is half a day; we find out fast. |
| 34k lines of loose scripts, no package | Budgeted: Milestone 1 is archaeology, not code. |

## Distribution

Ordered by expected return, all cheap:

1. Issue or PR on NanoJev offering the port. Maintainers of repos in a hype cycle
   routinely add free ports to their README.
2. Converted weights published as their own Hugging Face model page. NanoJev has 2,543
   downloads, so that search term already carries traffic.
3. Submission to the `awesome-jev` lists. They auto-sync; the inbound links are free.
4. An Everyday AI guide on ankitaglawe.com pointing at the repo.

## Milestones

1. **Archaeology and baseline (half a day).** Confirm which checkpoint is public. Extract
   the inference path out of `scripts/` into one readable module. Run the PyTorch original
   on this M1 and record P50. Generate parity fixtures. **Kill gate here.**
2. **Conversion.** fp32 to fp16 MLX, streaming.
3. **Runtime and decision API.** `model.py` plus `decide.py`, passing parity fixtures.
4. **Server and tests.** `/v1/systemone`, four test files.
5. **README, latency table, demo GIF.**
6. **Distribution.** The four surfaces above.

## Cost

Zero. Conversion and inference run locally. Hugging Face hosting is free. No GPU rental.
