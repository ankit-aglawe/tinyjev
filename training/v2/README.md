# TinyJev 4B v2: how it was trained

Everything needed to rebuild [AnkitAI/TinyJev-4B](https://huggingface.co/AnkitAI/TinyJev-4B) (v2) and to score it
through Ollama the way the model card does.

`ollama_format.py` is a byte-exact copy of how Ollama 0.35 turns a `/v1/systemone` request into one prompt per
question (`ollama/decision/systemone.go`). Every training row is rendered with it, so the model sees at training
time exactly the bytes Ollama sends at inference. `python ollama_format.py` runs its self-check.

1. **Data.**
   - Tev1's public training records: clone [togethercomputer/tev1](https://github.com/togethercomputer/tev1) (MIT),
     build its data, then `python convert_tev1.py <tev1>/data/new-v1 out/tev1` (37,840 train rows).
   - ContractNLI (CC BY 4.0): unzip [contract-nli.zip](https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip),
     then `python convert_cnli.py contract-nli out/cnli` (1,269 train rows, 122 dev, and the held-out eval requests).
2. **Train** on Modal, one A100-80GB, about two hours:
   `modal run modal_train_v2.py --run v2b --parts tev1,cnli --no-caps --rank 8 --dropout 0 --lr 5e-5 --min-lr 0 --warm-frac 0.03 --loss-kind full --max-tokens 16384`
3. **Export**: `modal run modal_train_v2.py::export --run v2b` merges the adapter and writes Q8_0 and Q4_K_M GGUFs
   (llama.cpp b11232 with `--no-mtp`, the converter Ollama 0.35.1 bundles).
4. **Package**: `python make_modelfile.py tinyjev-v2-Q8_0.gguf Modelfile && ollama create tinyjev -f Modelfile`
   (Ollama 0.35.1 or newer; the Modelfile's SYSTEM text must match training byte for byte).
5. **Score through Ollama**: `./eval_ollama.sh tinyjev` (JevBench public items and OpenDecision 500;
   `JEVBENCH=` points at a jevbench clone) and `python eval_cnli.py tinyjev --limit 60` (60 unseen NDAs).
