# TinyJev drives Jev Grand Prix

[Jev Grand Prix](https://github.com/enoyola/jev-grand-prix) (enoyola, MIT) is an F1 game where a decision model
picks the racing line and the pedals several times a second and plain code steers. It was written for TypeSafe's
hosted Jev API. These files run it on any Ollama decision model instead, for example TinyJev 4B v2:

```bash
ollama pull parable/tinyjev                         # Ollama 0.35.1 or newer
git clone https://github.com/enoyola/jev-grand-prix && cd jev-grand-prix
git apply ../jev-grand-prix.patch && cp ../ollama_shim.py .
OLLAMA_MODEL=parable/tinyjev python3 server.py      # then open http://localhost:8765
```

The patch keeps the game's questions untouched. It adds `ollama_shim.py` as the client when `OLLAMA_MODEL` is set,
and three URL options: `?speed=0.05` slows the race clock for a model slower than the hosted API, `?driver=TinyJev`
names the car, and `?caption=...` burns one line of text into the frame.

On an M1 Mac mini, TinyJev 4B v2 takes about 5 s per decision (three questions, about 740 prompt tokens each), so
the clip in the main README was recorded at `speed=0.05` and replayed at race speed:

```bash
URL=http://localhost:8765 SPEED=0.05 LAPS=1 OUT=frames node record_cdp.mjs   # headless Chrome, frames stamped with race time
python3 make_video.py frames out/racer --gif-from 8 --gif-len 14            # MP4 at race speed + a GIF
```
