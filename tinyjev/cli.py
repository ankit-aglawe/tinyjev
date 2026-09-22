"""tinyjev command line."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tinyjev", description="tiny Jev-style decision models, locally")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="convert an upstream checkpoint into the tinyjev layout")
    c.add_argument("family", choices=["nanojev", "kev"])
    c.add_argument("dest")
    c.add_argument("--source", help="nanojev: upstream checkpoint dir")
    c.add_argument("--adapter", help="kev: adapter dir (adapter_model.safetensors + head.pt)")
    c.add_argument("--base", help="kev: Qwen3 base model dir")
    c.add_argument("--name", default=None)
    c.add_argument("--dtype", default="float16", choices=["float16", "float32"])

    def model_args(p):
        p.add_argument("--backend", choices=["mlx", "torch"], default=None)
        p.add_argument("--device", default=None, help="torch backend: cpu, mps or cuda")

    s = sub.add_parser("serve", help="serve a checkpoint over HTTP (/predict, /v1/systemone)")
    s.add_argument("model"); s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8077); model_args(s)

    p = sub.add_parser("play", help="watch the model play snake or maze (nanojev)")
    p.add_argument("game", choices=["snake", "maze"]); p.add_argument("--model", default="nanojev")
    p.add_argument("--size", type=int); p.add_argument("--seed", type=int, default=17)
    p.add_argument("--steps", type=int, default=400); p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--safety", action="store_true"); p.add_argument("--legacy-format", action="store_true")
    model_args(p)

    a = sub.add_parser("ask", help="answer one request (System One or native shape) from a file or stdin")
    a.add_argument("model"); a.add_argument("request", nargs="?", default="-"); model_args(a)

    ls = sub.add_parser("models", help="list known checkpoints")

    args = parser.parse_args(argv)

    if args.command == "models":
        from .registry import HUB_REPO, MODELS
        for k, v in MODELS.items():
            print(f"{k:<10} {v['params']:<5} {v['family']:<8} {HUB_REPO}/{v['subfolder']:<12} <- {v['upstream']}")
        return 0
    if args.command == "convert":
        from .convert import convert
        kw = {"source": args.source, "adapter": args.adapter, "base": args.base, "name": args.name or args.family}
        convert(args.family, args.dest, dtype=args.dtype, **{k: v for k, v in kw.items() if v})
        return 0

    from . import load
    if args.command == "serve":
        from .serve import serve
        serve(load(args.model, backend=args.backend, device=args.device), host=args.host, port=args.port)
        return 0
    if args.command == "play":
        from .play import main as play_main
        argv2 = [args.game, "--model", args.model, "--seed", str(args.seed), "--steps", str(args.steps),
                 "--delay", str(args.delay)]
        if args.size: argv2 += ["--size", str(args.size)]
        if args.safety: argv2.append("--safety")
        if args.legacy_format: argv2.append("--legacy-format")
        if args.backend: argv2 += ["--backend", args.backend]
        return play_main(argv2)

    payload = json.load(sys.stdin) if args.request == "-" else json.load(open(args.request))
    agent = load(args.model, backend=args.backend, device=args.device)
    print(json.dumps(agent.predict(payload), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
