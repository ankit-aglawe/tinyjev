"""Frames + race-time stamps from record_cdp.mjs -> an MP4 that plays at race speed, plus a GIF of one stretch.

    python3 make_video.py frames out/tinyjev-racer --gif-from 20 --gif-len 12     (run where record_cdp.mjs ran)
"""
import json, os, subprocess, sys

src, out, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
opt = lambda k, d: float(rest[rest.index(k) + 1]) if k in rest else d
gif_from, gif_len = opt("--gif-from", 0.0), opt("--gif-len", 12.0)
S = json.load(open(os.path.join(src, "stamps.json")))
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
lst = os.path.join(src, "concat.txt")
with open(lst, "w") as f:
    for a, b in zip(S, S[1:] + [None]):
        f.write(f"file '{os.path.abspath(a['f'])}'\nduration {(b['t'] if b else a['t'] + 1 / 15) - a['t']:.4f}\n")
    f.write(f"file '{os.path.abspath(S[-1]['f'])}'\n")          # concat demuxer needs the last file twice
run = lambda *a: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)
run("-f", "concat", "-safe", "0", "-i", lst, "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "20",
    "-movflags", "+faststart", out + ".mp4")
run("-ss", str(gif_from), "-t", str(gif_len), "-i", out + ".mp4", "-vf",
    "fps=15,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
    out + ".gif")
for ext in (".mp4", ".gif"):
    print(out + ext, f"{os.path.getsize(out + ext) / 1e6:.1f} MB")
print(f"race time {S[-1]['t']:.1f} s, {len(S)} frames, {S[-1]['d']} decisions")
