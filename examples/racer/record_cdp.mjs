// Record Jev Grand Prix with headless Chrome over CDP. The race clock runs at SPEED x real time, so a local model
// that answers in ~3 s gets about the same race time per decision as TypeSafe's hosted API (~0.27 s); each frame
// is stamped with race time so make_video.sh replays at race speed.
//   node record_cdp.mjs   (env: URL W H SPEED LAPS OUT STEP MAXMIN)
import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";

const env = process.env, W = +(env.W || 1080), H = +(env.H || 1080), SPEED = env.SPEED || "0.1";
const OUT = env.OUT || "frames", STEP = +(env.STEP || 1 / 15), MAXMIN = +(env.MAXMIN || 30), PORT = 9333;
const URL = `${env.URL || "http://localhost:8765"}/?film=1&driver=${env.DRIVER || "TinyJev"}&laps=${env.LAPS || 1}&auto=1&speed=${SPEED}` + (env.CAPTION ? `&caption=${encodeURIComponent(env.CAPTION)}` : "");
await mkdir(OUT, { recursive: true });
const chrome = spawn("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${OUT}/.chrome`, `--window-size=${W},${H}`,
   "--hide-scrollbars", "--mute-audio", "about:blank"], { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ws;
for (let i = 0; i < 50 && !ws; i++) {
  try {
    const tabs = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json();
    ws = new WebSocket(tabs.find(t => t.type === "page").webSocketDebuggerUrl);
  } catch { await sleep(200); }
}
await new Promise(r => ws.addEventListener("open", r, { once: true }));
let id = 0; const waiting = new Map();
ws.addEventListener("message", e => { const m = JSON.parse(e.data); if (waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); } });
const cdp = (method, params = {}) => new Promise((res, rej) => {
  const i = ++id; waiting.set(i, m => m.error ? rej(new Error(`${method}: ${m.error.message}`)) : res(m.result));
  ws.send(JSON.stringify({ id: i, method, params }));
});
const js = async expr => (await cdp("Runtime.evaluate", { expression: expr, returnByValue: true })).result.value;

try {
  await cdp("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await cdp("Page.navigate", { url: URL });
  for (let i = 0; i < 150 && (await js("window.game ? game.state : ''").catch(() => "")) !== "racing"; i++) await sleep(200);
  const stamps = []; let next = 0; const t0 = Date.now();
  while (Date.now() - t0 < MAXMIN * 60_000) {
    const s = await js("({t: game.t, state: game.state, d: game.decisions})");
    if (s.state !== "racing") { console.log("race", s); break; }
    if (s.t >= next) {
      const { data } = await cdp("Page.captureScreenshot", { format: "png" });
      const f = `${OUT}/${String(stamps.length).padStart(6, "0")}.png`;
      await writeFile(f, Buffer.from(data, "base64"));
      stamps.push({ f, t: s.t, d: s.d }); next = s.t + STEP;
      if (stamps.length % 150 === 0) console.log(`frame ${stamps.length} race ${s.t.toFixed(1)} s, ${s.d} decisions`);
    } else await sleep(30);
  }
  await writeFile(`${OUT}/stamps.json`, JSON.stringify(stamps));
  console.log(JSON.stringify({ frames: stamps.length, ...(await js(`({t: game.t, laps: cars.jev.laps, decisions: game.decisions,
    latency_ms: Math.round(game.latencyAvg), err: (document.querySelector("#error") || {}).textContent || ""})`)) }));
} finally { ws.close(); chrome.kill(); }
