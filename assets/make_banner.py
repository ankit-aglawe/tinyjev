"""Build the tinyjev header banners, matching the Parable/Sensible lockup.

    python assets/make_banner.py --mascot assets/ant.png

Transparent PNG, 1673x460, origami mascot left, Poppins Bold wordmark right.
Two variants: light (charcoal text, for light backgrounds) and dark (rust text).
Run with no --mascot to render the wordmark only, as a placeholder.
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1673, 460
CHARCOAL = (45, 45, 45, 255)
RUST = (196, 78, 32, 255)
FONT = Path(__file__).with_name("Poppins-Bold.ttf")


def build(mascot_path, out, colour, size=196, word="TinyJev", flip=True):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
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
    img.save(out)
    return img.size


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mascot", default="assets/ant.png")
    ap.add_argument("--size", type=int, default=196)
    ap.add_argument("--word", default="TinyJev")
    ap.add_argument("--no-flip", action="store_true", help="keep the mascot facing as drawn")
    a = ap.parse_args()
    here = Path(__file__).parent
    print("light:", build(a.mascot, here / "tinyjev_header.png", CHARCOAL, a.size, a.word, not a.no_flip))
    print("dark: ", build(a.mascot, here / "tinyjev_header_dark.png", RUST, a.size, a.word, not a.no_flip))
