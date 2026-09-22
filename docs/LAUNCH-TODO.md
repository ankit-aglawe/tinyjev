# Launch checklist — README as a marketing page

Decisions taken 2026-09-22 (user notes + calls made on their behalf).

## Visual identity
- [ ] Logo: same style as Parable's (look at Parable's logo/HF card first), but tiny — a **paper-origami ant**. SVG, mono + colour variants, dark/light lockups.
- [ ] Banner SVG: logo + "tinyjev" + one line ("tiny typed-decision models, on your laptop"). Used as the README hero and the HF card top.

## Three demos, one shape
Same-size containers (2-column on desktop, stack on phone — decide at build time by GIF aspect), each a **real recorded run** of tinyjev serving decisions, dashboard style like `tools/render_gif.py`:
- [ ] Snake — NanoJev playing its own game through our runtime (exists: `assets/snake.gif`, re-record on the unified format).
- [ ] Chess — Kev ships a chess playground (legal moves = choice options, a score question rates the position). Port to our server; record tinyjev-0.6b playing itself.
- [ ] Flight search, Manchester → Shanghai — browser-use/jev-ultrafast pointed at `localhost:8077`. Stop before booking. **Record only if a real run completes**; if the 0.6B can't drive it, say so on the page instead of faking it.

## Support
- [ ] Buy Me a Coffee: GitHub README gets the `<a href="https://www.buymeacoffee.com/AnkitAI"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" ...></a>` form (GitHub strips `<script>`); the HF card gets the same style as Parable's HF READMEs.

## Distribution
- [x] Decision: publish `tinyjev` on PyPI (name free) — needs a PyPI token in `~/.pypirc` or `TWINE_*` env, which the user provides; `pip install 'tinyjev[mlx]'` in the README assumes it.
- [x] Decision: "plugs into other tools" = the `/v1/systemone` endpoint; name the clients that already speak it (TypeSafe SDK, Kev's harness, Laya, jev-ultrafast, the MCP/n8n/Rails wrappers in awesome-jev).
- [ ] Push GitHub (`ankit-aglawe/tinyjev`) with the honest README first; marketing layer as the next commit; then launch.
- [ ] Launch: NanoJev issue (rope_theta bug + port), Kev issue (results + patch + link), awesome-jev submission, HF card final, Everyday AI guide.

## Content decisions
- Page order: banner → one line → install (2 lines) → the number table → three demos → "what we tried" (the negative results are the credibility) → runs-on-M1 table → how it works → reproduce → limits → credits → coffee.
- Tone: builder to builder, numbers with their suite and split, no hype words. The negative results stay above the fold of "how it works".
