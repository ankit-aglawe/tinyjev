# Draft: issue for jaredpalmer/kev

**Title:** Kev-0.6B / Kev-4B (Qwen3 generation) on MLX and plain PyTorch via tinyjev — parity 14/14 against kev.model, transfer-v4 numbers inside

Hi Jared — thanks for the unusually careful repo; the frozen suites and `--remote` scoring made this easy to check.

I built a small runtime that serves the Qwen3-generation Kev checkpoints (the ones you've marked as no longer developed) on Apple Silicon through MLX, and on CPU/CUDA/MPS through a minimal PyTorch path with no peft/transformers-5 requirement: https://github.com/ankit-aglawe/tinyjev

What it does with your checkpoints:
- merges the rank-16 adapter into the base at convert time (alpha/r = 2, fp32, then fp16), attaches the pointer head from `head.pt`, keeps the head fp32
- runs each question as its own causal row continuing from the state, with the state's KV computed once and shared across questions (your `probs_with_prefix` design, on MLX)
- serves `/v1/systemone` unchanged, so the TypeSafe SDK and `kev.benchmark --remote` work against it

Verification:
- selected answer matches `kev.model` on 14/14 fixture questions on both backends; worst per-option probability delta 4.9e-3 (MLX fp16), 3.3e-3 (torch fp32); fixtures and the reference script are in the repo
- `kev.benchmark --remote` on `evals/v4/transfer-v4` development: **{TRANSFER_V4_ACC}** (your published kev-0.6b: 0.620)
- latency on a base M1 / 16 GB: **{M1_LATENCY}** for a three-question ticket through MLX

Converted weights: https://huggingface.co/AnkitAI/tinyjev-kev-0.6b (and kev-4b@qwen3 at https://huggingface.co/AnkitAI/tinyjev-kev-4b). Apache-2.0, attributed.

If you'd like a line under Serving Performance pointing Mac users at it for the Qwen3 models, happy to send the PR. Also happy to add an MLX path for the Qwen3.5 generation once fast DeltaNet kernels exist on MLX — I stopped at Qwen3 on purpose so as not to step on your planned backend.
