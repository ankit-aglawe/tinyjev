"""Record a Snake run as a dashboard-style animated GIF at the real decision cadence.

Every frame shows the board, the model's probability over the offered moves, its
per-move safety answers, and the measured inference time. Nothing is staged: the
frames are the run.
"""
import argparse, platform, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image, ImageDraw, ImageFont
import tinyjev
from tinyjev.play import play_snake

S = 2  # supersampling factor
W, H = 960, 540
BG, PANEL, LINE = (11, 15, 20), (15, 20, 27), (38, 46, 56)
FG, DIM, MUTE = (226, 230, 236), (140, 150, 162), (78, 88, 100)
GREEN, GREEN_DK, MINT = (110, 226, 160), (31, 107, 74), (170, 240, 205)
ORANGE, BLUE, RED = (240, 176, 74), (120, 170, 240), (240, 110, 110)
FONTS = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/SFNSMono.ttf",
         "/Library/Fonts/Courier New.ttf"]


def font(px):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, px * S)
        except OSError:
            continue
    return ImageFont.load_default()


F_XS, F_S, F_M, F_L, F_XL = font(11), font(13), font(15), font(19), font(40)


def machine():
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True, check=True).stdout) // (1 << 30)
        return f"{chip} · {mem} GB"
    except Exception:
        return platform.machine()


MACHINE = machine()


def px(v):
    return int(v * S)


def text(d, xy, s, f=F_S, fill=FG):
    d.text((px(xy[0]), px(xy[1])), s, font=f, fill=fill)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_board(d, state, x0, y0, side):
    n = state["size"]
    cell = side / n
    d.rectangle([px(x0), px(y0), px(x0 + side), px(y0 + side)], fill=PANEL, outline=LINE, width=S)
    for r in range(n):
        for c in range(n):
            cx, cy = x0 + (c + 0.5) * cell, y0 + (r + 0.5) * cell
            d.ellipse([px(cx - 1), px(cy - 1), px(cx + 1), px(cy + 1)], fill=MUTE)
    body = state["body"]
    for i, (r, c) in enumerate(body):
        t = i / max(1, len(body) - 1)
        color = MINT if i == 0 else lerp(GREEN, GREEN_DK, t)
        pad = cell * 0.08
        d.rectangle([px(x0 + c * cell + pad), px(y0 + r * cell + pad),
                     px(x0 + (c + 1) * cell - pad), px(y0 + (r + 1) * cell - pad)], fill=color)
    fr, fc = state["food"]
    cx, cy = x0 + (fc + 0.5) * cell, y0 + (fr + 0.5) * cell
    rad = cell * 0.22
    d.ellipse([px(cx - rad), px(cy - rad), px(cx + rad), px(cy + rad)], fill=ORANGE)


def bar(d, x, y, w, h, frac, color, track=LINE):
    d.rectangle([px(x), px(y), px(x + w), px(y + h)], fill=track)
    if frac > 0:
        d.rectangle([px(x), px(y), px(x + w * min(1.0, frac)), px(y + h)], fill=color)


def render(info, best):
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)
    st, ans = info["state"], info["answers"]

    # window chrome
    d.rounded_rectangle([px(6), px(6), px(W - 6), px(H - 6)], radius=px(10), outline=LINE, width=S)
    for i, col in enumerate([RED, ORANGE, GREEN]):
        d.ellipse([px(18 + i * 14), px(16), px(26 + i * 14), px(24)], fill=col)
    text(d, (W / 2 - 118, 13), "tinyjev  /  real recorded decisions", F_XS, DIM)
    text(d, (24, 38), "NANOJEV  /  ON APPLE SILICON", F_M, DIM)
    text(d, (W - 190, 38), "RECORDED RUN · 1×", F_M, GREEN)
    d.line([px(24), px(62), px(W - 24), px(62)], fill=LINE, width=S)

    # board + big numbers
    text(d, (24, 76), "S N A K E", F_L, FG)
    text(d, (250, 80), f"{st['size']}x{st['size']}", F_S, DIM)
    draw_board(d, st, 24, 104, 300)
    for i, (label, val, col) in enumerate([("SCORE", st["score"], GREEN),
                                           ("LENGTH", len(st["body"]), FG),
                                           ("BEST", best, DIM)]):
        x = 24 + i * 108
        text(d, (x, 416), label, F_XS, DIM)
        text(d, (x, 432), f"{val:03d}", F_XL, col)

    # right panel: probabilities
    rx = 372
    text(d, (rx, 78), "NanoJev MLX", F_L, GREEN)
    text(d, (rx, 102), MACHINE + " · Local", F_S, DIM)
    text(d, (rx, 132), "NEXT MOVE", F_M, FG)
    text(d, (rx + 150, 132), "MODEL PROBABILITIES", F_M, DIM)
    probs = ans["action"]["probabilities"]
    y = 158
    for move in ("north", "east", "south", "west"):
        offered = move in probs
        p = probs.get(move, 0.0)
        chosen = move == info["chosen"]
        col = GREEN if chosen else (MUTE if not offered else DIM)
        text(d, (rx, y), ("› " if chosen else "  ") + move.upper(), F_S, col)
        if offered:
            bar(d, rx + 96, y + 3, 190, 12, p, GREEN if chosen else MUTE)
            text(d, (rx + 296, y), f"{p:.2f}", F_S, FG if chosen else DIM)
        else:
            text(d, (rx + 96, y), "not offered (reverse)", F_XS, MUTE)
        y += 22
    text(d, (rx, y + 8), "EXECUTING", F_M, DIM)
    text(d, (rx + 110, y + 8), info["chosen"].upper(), F_M, GREEN)
    if info["proposed"] != info["chosen"]:
        text(d, (rx + 190, y + 8), f"(model proposed {info['proposed'].upper()})", F_XS, ORANGE)
    y += 40

    # safety answers (only when asked)
    if info["safety"]:
        text(d, (rx, y), "SAFE NEXT STEP", F_M, DIM)
        text(d, (rx + 150, y), "P(no collision), same forward pass", F_XS, MUTE)
        y += 22
        for move in info["offered"]:
            ps = ans.get(f"safe_{move}", {}).get("p_true")
            if ps is None:
                continue
            col = GREEN if ps >= 0.5 else RED
            text(d, (rx, y), "  " + move.upper(), F_S, DIM)
            bar(d, rx + 96, y + 3, 190, 12, ps, col)
            text(d, (rx + 296, y), f"{ps:.2f}", F_S, DIM)
            y += 20
        text(d, (rx, y + 4), f"safety overrides  {info['overrides']:04d}", F_XS, ORANGE)
        y += 26

    # stats
    y = max(y + 6, 388)
    d.line([px(rx), px(y), px(W - 24), px(y)], fill=LINE, width=S)
    y += 8
    rows = [("INFERENCE", f"{info['elapsed_ms']:.1f} ms", FG),
            ("DECISIONS", f"{info['fps']:.1f} /s", FG),
            ("OUTPUT TOKENS", "0", FG),
            ("NETWORK", "OFFLINE", GREEN),
            ("ENGINE", "MLX · FP16 · shared prefix · Qwen3-0.6B", DIM)]
    for label, val, col in rows:
        text(d, (rx, y), label, F_XS, DIM)
        text(d, (rx + 130, y), val, F_XS, col)
        y += 16

    d.line([px(24), px(H - 40), px(W - 24), px(H - 40)], fill=LINE, width=S)
    text(d, (24, H - 30), "every move is one forward pass · no text generated · upstream refuses to run without CUDA",
         F_XS, DIM)
    text(d, (W - 110, H - 30), f"move {info['moves']:03d}", F_XS, DIM)
    return img.resize((W, H), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--safety", action="store_true")
    ap.add_argument("--preview", help="also save one mid-run frame as PNG")
    args = ap.parse_args()

    agent = tinyjev.load(args.model)
    frames, durations, stamps, best = [], [], [time.perf_counter()], [0]

    def on_frame(info):
        now = time.perf_counter()
        durations.append(max(60, int((now - stamps[-1]) * 1000)))
        stamps.append(now)
        best[0] = max(best[0], info["state"]["score"])
        frames.append(render(info, best[0]))

    result = play_snake(agent, size=args.size, seed=args.seed, max_steps=args.steps,
                        safety=args.safety, render=False, on_frame=on_frame)
    if not frames:
        raise SystemExit("no frames recorded")
    durations[-1] = 1800
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    if args.preview:
        frames[len(frames) // 2].save(args.preview)
    print(f"{len(frames)} frames -> {args.out} ({Path(args.out).stat().st_size // 1024} KB)")
    print(result)


if __name__ == "__main__":
    raise SystemExit(main())
