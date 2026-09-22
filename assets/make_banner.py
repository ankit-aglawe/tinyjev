"""Build the tinyjev header banner, matching the Parable/Sensible lockup.

    python assets/make_banner.py

1673x460, ink panel, origami mascot left, Poppins Bold wordmark right in the orange
sampled from the mascot itself. A solid panel reads on light and dark pages alike, so
there is one banner rather than a light/dark pair.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1673, 460
INK = (26, 26, 26, 255)
ANT_ORANGE = (228, 100, 18, 255)     # sampled from assets/ant.png
FONT = Path(__file__).with_name("Poppins-Bold.ttf")


def build(mascot_path, out, bg=INK, colour=ANT_ORANGE, size=196, word="TinyJev", flip=True):
    img = Image.new("RGBA", (W, H), bg)
    x = 40
    if mascot_path and Path(mascot_path).exists():
        m = Image.open(mascot_path).convert("RGBA")
        m = m.crop(m.split()[-1].getbbox())          # trim transparent margin
        if flip:
            m = m.transpose(Image.FLIP_LEFT_RIGHT)   # mascot should face the wordmark
        scale = (H - 48) / m.height
        m = m.resize((max(1, int(m.width * scale)), H - 48), Image.LANCZOS)
        img.paste(m, (x, 24), m)
        x += m.width + 56
    else:
        x = 230
    font = ImageFont.truetype(str(FONT), size)
    d = ImageDraw.Draw(img)
    # optical centring on the cap height, not the full em box
    box = d.textbbox((0, 0), word, font=font)
    d.text((x - box[0], (H - (box[3] - box[1])) // 2 - box[1]), word, font=font, fill=colour)
    img.convert("RGB").save(out)
    return img.size


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mascot", default="assets/ant.png")
    ap.add_argument("--size", type=int, default=196)
    ap.add_argument("--word", default="TinyJev")
    ap.add_argument("--no-flip", action="store_true", help="keep the mascot facing as drawn")
    a = ap.parse_args()
    out = Path(__file__).with_name("tinyjev_header.png")
    print("banner:", build(a.mascot, out, size=a.size, word=a.word, flip=not a.no_flip), out)
