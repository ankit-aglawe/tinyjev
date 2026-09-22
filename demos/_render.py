"""Frame renderer for the demo GIFs: paper, ink, and the logo's orange.

Poppins throughout, to match the banner. Hairline rules instead of boxes, generous
margins, the gameplay panel dominant and the readout beside it kept small. Everything
drawn here comes from a real recorded run; nothing is mocked.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2                       # supersample, downscaled at the end
W, H = 840, 520

PAPER = (244, 241, 232)
PAPER_2 = (237, 233, 221)
INK = (45, 45, 45)          # the banner wordmark's charcoal
MUTED = (138, 133, 124)
RULE = (208, 202, 189)
ACCENT = (228, 100, 18)     # sampled from the ant in assets/ant.png
ACCENT_SOFT = (240, 190, 165)
MOSS = (92, 110, 84)

_FONTS = Path(__file__).resolve().parent.parent / "assets"


def _font(weight: str, px_: int):
    try:
        return ImageFont.truetype(str(_FONTS / f"Poppins-{weight}.ttf"), px_ * S)
    except OSError:
        return ImageFont.load_default()


F_TITLE = _font("SemiBold", 21)
F_SUB = _font("Regular", 10)
F_TINY = _font("Medium", 8)
F_LABEL = _font("Regular", 10)
F_BODY = _font("Regular", 11)
F_VALUE = _font("Medium", 11)
F_MODE = _font("SemiBold", 12)


def machine() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=True).stdout.strip()
        gb = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                text=True, check=True).stdout) // (1 << 30)
        return f"{chip} · {gb} GB"
    except Exception:
        return "local"


MACHINE = machine()


def px(v):
    return int(v * S)


class Frame:
    def __init__(self, title: str, subtitle: str):
        self.img = Image.new("RGB", (W * S, H * S), PAPER)
        self.d = ImageDraw.Draw(self.img)
        self.text((32, 22), title, F_TITLE, INK)
        self.text((32, 52), subtitle, F_SUB, MUTED)
        self.text((W - 32, 26), "tinyjev", F_VALUE, ACCENT, anchor="ra")
        self.text((W - 32, 44), MACHINE, F_TINY, MUTED, anchor="ra")
        self.rule(32, 72, W - 32)

    def text(self, xy, s, font=F_BODY, fill=INK, anchor=None):
        self.d.text((px(xy[0]), px(xy[1])), s, font=font, fill=fill, anchor=anchor)

    def rule(self, x0, y, x1, color=RULE):
        self.d.line([px(x0), px(y), px(x1), px(y)], fill=color, width=max(1, S // 2))

    def bar(self, x, y, w, h, frac, color=ACCENT, track=(226, 221, 209)):
        self.d.rounded_rectangle([px(x), px(y), px(x + w), px(y + h)], radius=px(h / 2), fill=track)
        if frac > 0.004:
            self.d.rounded_rectangle([px(x), px(y), px(x + w * min(1.0, frac)), px(y + h)],
                                     radius=px(h / 2), fill=color)

    def stat(self, x, y, label, value, color=INK):
        self.text((x, y), label.upper(), F_TINY, MUTED)
        self.text((x, y + 12), value, F_VALUE, color)

    def finish(self) -> Image.Image:
        self.rule(32, H - 30, W - 32)
        self.text((32, H - 24), "every frame is a real forward pass, recorded live",
                  F_TINY, MUTED)
        return self.img.resize((W, H), Image.LANCZOS)


def paste(frame: Frame, img: Image.Image, x, y, w, h, border=RULE):
    """Drop a raster panel (a game screen) into the frame at layout coords."""
    frame.img.paste(img.convert("RGB").resize((px(w), px(h)), Image.BOX), (px(x), px(y)))
    frame.d.rectangle([px(x), px(y), px(x + w), px(y + h)], outline=border, width=max(1, S // 2))


def options_ledger(f: Frame, x, y, width, options, probs, chosen,
                   title="THE MODEL'S CHOICE", label_w=104, row=22):
    f.text((x, y), title, F_TINY, MUTED)
    y += 16
    for name in options:
        p = probs.get(name)
        hit = name == chosen
        if hit:
            f.d.polygon([(px(x), px(y + 5)), (px(x + 6), px(y + 9)), (px(x), px(y + 13))],
                        fill=ACCENT)
        f.text((x + 13, y), name.replace("_", " "), F_VALUE if hit else F_BODY,
               INK if hit else MUTED)
        f.bar(x + label_w, y + 6, width - label_w - 40, 8, p or 0.0,
              ACCENT if hit else ACCENT_SOFT)
        f.text((x + width, y), f"{p:.2f}", F_VALUE if hit else F_BODY,
               INK if hit else MUTED, anchor="ra")
        y += row
    return y


UI_COLOURS = (PAPER, PAPER_2, INK, MUTED, RULE, ACCENT, ACCENT_SOFT, MOSS,
              (226, 221, 209), (255, 255, 255), (0, 0, 0))


def save_gif(frames, ms, path, hold_ms=1400, colors=64, keep=UI_COLOURS):
    """One shared palette so PIL can delta-encode; only the changed box is stored.

    The UI colours are pinned into the palette first. Left to median cut, Doom's browns
    outnumber them and the accent orange quantises away to mud."""
    if not frames:
        raise SystemExit("no frames")
    sample = frames[::max(1, len(frames) // 12)]
    montage = Image.new("RGB", (frames[0].width, frames[0].height * len(sample)))
    for i, fr in enumerate(sample):
        montage.paste(fr, (0, i * frames[0].height))

    pinned = list(dict.fromkeys(keep))[:colors - 2]
    scene = montage.quantize(colors=max(2, colors - len(pinned)), method=Image.MEDIANCUT)
    raw = scene.getpalette()[: 3 * max(2, colors - len(pinned))]
    entries = pinned + [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
    flat = [c for rgb in entries for c in rgb]
    flat += [0] * (768 - len(flat))
    master = Image.new("P", (1, 1))
    master.putpalette(flat)

    pal = [f.quantize(palette=master, dither=Image.NONE) for f in frames]
    durations = [ms] * len(pal)
    durations[-1] = hold_ms
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durations,
                loop=0, optimize=True)
    return Path(path).stat().st_size // 1024
