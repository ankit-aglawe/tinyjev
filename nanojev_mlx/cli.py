"""nanojev-mlx command line."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="nanojev-mlx", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="convert an upstream NanoJev checkpoint to MLX")
    c.add_argument("source"); c.add_argument("dest")
    c.add_argument("--dtype", default="float16", choices=["float16", "float32"])

    s = sub.add_parser("serve", help="serve a converted checkpoint over HTTP")
    s.add_argument("model"); s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8077)

    p = sub.add_parser("play", help="watch the model play snake or maze in the terminal")
    p.add_argument("game", choices=["snake", "maze"]); p.add_argument("--model", required=True)
    p.add_argument("--size", type=int); p.add_argument("--seed", type=int, default=17)
    p.add_argument("--steps", type=int, default=400); p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--safety", action="store_true")

    a = sub.add_parser("ask", help="answer one request from a JSON file or stdin")
    a.add_argument("model"); a.add_argument("request", nargs="?", default="-")

    args = parser.parse_args(argv)

    if args.command == "convert":
        from .convert import convert
        convert(args.source, args.dest, dtype=args.dtype)
        return 0

    if args.command == "play":
        from .play import main as play_main
        argv2 = [args.game, "--model", args.model, "--seed", str(args.seed),
                 "--steps", str(args.steps), "--delay", str(args.delay)]
        if args.size:
            argv2 += ["--size", str(args.size)]
        if args.safety:
            argv2.append("--safety")
        return play_main(argv2)

    from . import load
    if args.command == "serve":
        from .serve import serve
        serve(load(args.model), host=args.host, port=args.port)
        return 0

    payload = json.load(sys.stdin) if args.request == "-" else json.load(open(args.request))
    print(json.dumps(load(args.model).predict(payload), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
