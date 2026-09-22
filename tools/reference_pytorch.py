"""Reference NanoJev runner in plain PyTorch, on CPU or MPS.

Upstream's DecisionPredictor asserts CUDA and refuses everything else, so it cannot
produce reference outputs on a Mac. This rebuilds the same graph from upstream's own
DecisionModel class and prepare_examples, with no device assertion, and is the oracle
the MLX port is checked against.
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def normalized_tokenizer_dir(tokenizer_dir, workdir):
    """Copy a tokenizer dir, repairing fields newer transformers refuses to load.

    NanoJev's checkpoint stores extra_special_tokens as a list; transformers expects a
    mapping and dies with "'list' object has no attribute 'keys'". Repair a copy so the
    checkpoint in the HF cache is never mutated.
    """
    import shutil
    src = Path(tokenizer_dir)
    dst = Path(workdir) / "tokenizer"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    cfg_path = dst / "tokenizer_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        changed = False
        extra = cfg.get("extra_special_tokens")
        if isinstance(extra, list):
            cfg["extra_special_tokens"] = {f"extra_{i}": t for i, t in enumerate(extra)}
            changed = True
        if cfg.get("tokenizer_class") in (None, "TokenizersBackend"):
            cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            cfg.pop("backend", None)
            cfg.pop("is_local", None)
            changed = True
        if changed:
            cfg_path.write_text(json.dumps(cfg, indent=2))
    return dst


def _load_upstream(src):
    """Import upstream's trainer + predictor modules straight from a NanoJev checkout."""
    src = Path(src).expanduser().resolve(strict=True)
    sys.path.insert(0, str(src))
    mods = {}
    for name in ("train_toy_decisions", "predict_toy_decisions"):
        spec = importlib.util.spec_from_file_location(f"nanojev_{name}", src / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["train_toy_decisions"], mods["predict_toy_decisions"]


class ReferenceRunner:
    def __init__(self, checkpoint_dir, source_scripts, device="cpu", max_length=None):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.torch = torch
        trainer, predictor = _load_upstream(source_scripts)
        self.prepare_examples = predictor.prepare_examples
        self.answer_from_probabilities = predictor.answer_from_probabilities

        root = Path(checkpoint_dir).expanduser().resolve(strict=True)
        run_config = json.loads((root / "config.json").read_text())
        set_head = run_config.get("set_head")
        if set_head not in {"none", "attention"}:
            raise ValueError(f"checkpoint config has no valid set_head: {set_head!r}")

        import tempfile
        self._workdir = tempfile.mkdtemp(prefix="nanojev-ref-")
        tok_dir = normalized_tokenizer_dir(root / "tokenizer", self._workdir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tok_dir), local_files_only=True, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        body_config = AutoConfig.from_pretrained(
            str(root / "backbone_config"), local_files_only=True, trust_remote_code=False)
        body_config.use_cache = False
        self._repair_rope(body_config, json.loads(
            (root / "backbone_config" / "config.json").read_text()))
        body = AutoModel.from_config(
            body_config, attn_implementation="sdpa", trust_remote_code=False).float()

        model = trainer.DecisionModel(body, set_head)
        model.load_state_dict(load_file(str(root / "best.safetensors"), device="cpu"), strict=True)
        model.to(device=torch.device(device), dtype=torch.float32).eval()

        self.model = model
        self.device = device
        self.limit = max_length or run_config.get("max_length", 512)
        self.set_head = set_head

    @staticmethod
    def _repair_rope(config, raw):
        """Force the rope base the checkpoint actually declares.

        NanoJev's backbone_config was written by transformers 5.x, which stores the base
        under `rope_parameters.rope_theta`. transformers 4.x does not read that key and
        silently falls back to rope_theta=10000, which changes every hidden state and
        quietly degrades the model. Without this the reference is wrong, not the port.
        """
        declared = (raw.get("rope_parameters") or {}).get("rope_theta", raw.get("rope_theta"))
        if declared is None:
            return
        current = getattr(config, "rope_theta", None)
        if current != declared:
            print(f"  [rope] transformers read rope_theta={current}, checkpoint declares "
                  f"{declared}; forcing the declared value", flush=True)
        config.rope_theta = declared
        if isinstance(getattr(config, "rope_parameters", None), dict):
            config.rope_parameters = {**config.rope_parameters, "rope_theta": declared}

    def logits(self, payload):
        """Return {example_id: {"candidate_ids", "type", "logits", "probabilities"}}."""
        torch = self.torch
        examples = self.prepare_examples(payload, self.tokenizer, self.limit)
        out = {}
        with torch.inference_mode():
            raw, _ = self.model(examples, self.tokenizer.pad_token_id)
        for example, values in zip(examples, raw):
            k = len(example["candidate_ids"])
            scores = values[:k].float()
            probs = scores.softmax(-1).cpu().tolist()
            out[example["id"]] = {
                "type": example["type"],
                "candidate_ids": example["candidate_ids"],
                "logits": scores.cpu().tolist(),
                "probabilities": probs,
                "answer": self.answer_from_probabilities(example, probs),
                "path_token_counts": [len(t) for t in example["leaf_tokens"]],
            }
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source-scripts", required=True)
    ap.add_argument("--cases", required=True, help="JSON file: list of System-One-ish payloads")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    t0 = time.perf_counter()
    runner = ReferenceRunner(args.checkpoint, args.source_scripts, device=args.device)
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f}s on {args.device}", flush=True)

    fixtures, timings = [], []
    for case in cases:
        # warm once so the recorded timings exclude lazy kernel compilation
        runner.logits(case["payload"])
        samples = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            result = runner.logits(case["payload"])
            samples.append((time.perf_counter() - t) * 1000.0)
        samples.sort()
        median = samples[len(samples) // 2]
        timings.append({"name": case["name"], "median_ms": round(median, 1),
                        "samples_ms": [round(s, 1) for s in samples]})
        fixtures.append({"name": case["name"], "payload": case["payload"], "expected": result})
        print(f"  {case['name']:<28} {median:8.1f} ms", flush=True)

    Path(args.out).write_text(json.dumps(
        {"device": args.device, "load_seconds": round(load_s, 2),
         "set_head": runner.set_head, "repeats": args.repeats,
         "timings": timings, "fixtures": fixtures}, indent=2))
    print(f"wrote {args.out}  ({len(fixtures)} fixtures)")


if __name__ == "__main__":
    main()
