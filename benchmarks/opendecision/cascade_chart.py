"""One square image of the cascade curve, house style. assets/cascade.png"""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[2]
for w in ("Regular", "Medium", "SemiBold", "Bold"):
    fm.fontManager.addfont(str(ROOT / "assets" / f"Poppins-{w}.ttf"))
PAPER, INK, MUTED, ACCENT, ACCENT_TEXT, MOSS, RULE = "#f4f1e8", "#2d2d2d", "#605b52", "#e46412", "#b24808", "#40563a", "#c6bfb0"
plt.rcParams.update({"font.family": "Poppins", "figure.facecolor": PAPER, "axes.facecolor": PAPER, "text.color": INK})

rows = list(csv.DictReader(open("results/cascade.csv")))
S = json.load(open("results/summary.json"))
tj, op = S["tinyjev-0.6b"], S["claude-opus-5-5"]
xs = [float(r["sent_to_big"]) * 100 for r in rows]; ys = [float(r["cascade_accuracy"]) * 100 for r in rows]
big = float(rows[0]["big_only_accuracy"]) * 100
pick = {float(r["gate"]): r for r in rows}

fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=200)
fig.subplots_adjust(left=0.13, right=0.96, top=0.80, bottom=0.20)
ax.plot(xs, ys, color=ACCENT, linewidth=2.2, zorder=3)
ax.axhline(big, color=INK, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
ax.text(1, big + 0.25, f"Claude Opus 5.5 alone: {big:.1f}%", fontsize=7.5, color=INK, va="bottom")
ax.scatter([0], [tj["accuracy"] * 100], color=MOSS, s=26, zorder=4)
ax.text(2, tj["accuracy"] * 100 - 0.2, f"tinyjev alone: {tj['accuracy']*100:.1f}%", fontsize=7.5, color=MOSS, va="top")
for g, dy, txt in ((0.95, -2.6, "gate 0.95\n{sent:.0f}% of decisions go to Opus\nthe other {local:.0f}% stay on the laptop\nsame accuracy as Opus alone"),
                   (0.85, -3.6, "gate 0.85\n{sent:.0f}% to Opus, {local:.0f}% local\n{delta:+.1f} points")):
    r = pick[g]; x = float(r["sent_to_big"]) * 100; y = float(r["cascade_accuracy"]) * 100
    ax.scatter([x], [y], color=ACCENT, s=34, zorder=5, edgecolor=PAPER, linewidth=1)
    ax.annotate(txt.format(sent=x, local=100 - x, delta=float(r["delta_vs_big"]) * 100), (x, y), (x - 2, y + dy),
                fontsize=7, color=ACCENT_TEXT, ha="right", va="top",
                arrowprops=dict(arrowstyle="-", color=RULE, lw=0.8))
ax.set_xlim(-2, 102); ax.set_ylim(86.5, 100.5)
ax.set_xlabel("share of decisions sent to Opus 5.5", fontsize=8, color=MUTED, labelpad=6)
ax.set_ylabel("accuracy on all 500", fontsize=8, color=MUTED, labelpad=6)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=7, color=MUTED)
ax.set_yticks([88, 92, 96, 100]); ax.set_yticklabels([f"{v}%" for v in (88, 92, 96, 100)], fontsize=7, color=MUTED)
for s in ax.spines.values(): s.set_color(RULE)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=RULE, linewidth=0.5, alpha=0.7)
fig.text(0.06, 0.93, "Put a 596M model in front of Opus 5.5", fontsize=13.5, weight="semibold", color=INK)
fig.text(0.06, 0.885, "500 decisions, 25 domains it never trained on. The small model answers what it is\nsure of and hands the rest up. Both runs logged, every case, same inputs.",
         fontsize=7.6, color=MUTED, linespacing=1.4)
g95 = pick[0.95]; g85 = pick[0.85]
fig.text(0.06, 0.115, f"Opus 5.5 at $4 / $20 per MTok is roughly $2.7k–$5.7k per million decisions of this shape.\n"
                       f"Keeping {100-float(g95['sent_to_big'])*100:.0f}% of them local costs nothing in accuracy; keeping {100-float(g85['sent_to_big'])*100:.0f}% costs one point.\n"
                       f"The local half runs at {tj['mean_ms']:.0f} ms on a base M1, offline, $0.",
         fontsize=7.3, color=INK, linespacing=1.45)
fig.text(0.06, 0.045, "github.com/ankit-aglawe/tinyjev · benchmarks/opendecision", fontsize=7, color=ACCENT_TEXT)
fig.text(0.94, 0.045, "tinyjev", fontsize=8, color=ACCENT_TEXT, ha="right", weight="medium")
out = ROOT / "assets" / "cascade.png"
fig.savefig(out, facecolor=PAPER)
print(out, out.stat().st_size // 1024, "KB")
