# Draft: issue for TianyuCodings/NanoJev

**Title:** transformers 4.x silently loads the wrong RoPE base (10000 instead of 1000000) — plus an MLX port so the checkpoint runs on Macs

Hi — two things, one bug and one contribution.

**Bug.** `backbone_config/config.json` in `C-Tianyu/NanoJev` was written by transformers 5.17 and stores the RoPE base under `rope_parameters.rope_theta = 1000000`. transformers 4.x (I hit it on 4.57) does not read that key and falls back to `rope_theta = 10000` with no warning. Every hidden state changes and the model quietly degrades: on 15 fixture questions the selected answer flipped on 4. Anyone loading the backbone through `AutoConfig` on 4.x gets a silently wrong model.

Repro: load the backbone config with transformers 4.57, print `hf.rotary_emb.inv_freq[1]` — it implies theta≈10000. Force `config.rope_theta = 1_000_000` and the hidden states match an independent implementation to 7e-5.

Suggested fix: also write a top-level `"rope_theta": 1000000` into `backbone_config/config.json` (5.x ignores it, 4.x reads it), or pin `transformers>=5` in the requirements.

**Port.** `predict_toy_decisions.DecisionPredictor` asserts CUDA, so the checkpoint cannot run on a Mac. I ported `DecisionModel` to MLX: https://github.com/ankit-aglawe/nanojev-mlx — same prompt bytes, same weights (converted to fp16, head kept fp32), 15/15 parity on selected answers against a CPU/MPS PyTorch reference built from your `DecisionModel`, worst probability delta 1.2e-3. On a base M1 it answers a yes/no in 36 ms and loads in 0.3 s. Weights: https://huggingface.co/AnkitAI/nanojev-mlx. MIT, attributed in NOTICE.

If useful, a one-line mention in the README under "run it on Apple Silicon" would help people find it. Happy to send a PR for the config fix too.
