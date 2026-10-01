#!/usr/bin/env bash
# Score decision models THROUGH OLLAMA's /v1/systemone (the path users run), same inputs for every model:
#   JevBench public items (231, JevBench's own typesafe adapter) and OpenDecision Original Choice 500 (our harness).
# Needs Ollama >= 0.35.1 (CAPABILITY decision). OLLAMA_URL defaults to a side server on :11435 so the installed app
# is untouched.
#   ./eval_ollama.sh tev1 qwen35-4b-zeroshot tinyjev-v2
set -euo pipefail
URL=${OLLAMA_URL:-http://127.0.0.1:11435}
HERE=$(cd "$(dirname "$0")" && pwd)
JB=${JEVBENCH:-$HERE/jevbench}               # a clone of github.com/fstandhartinger/jevbench
OD=$HERE/../../benchmarks/opendecision
PY=${PY:-/opt/homebrew/bin/python3}
for m in "$@"; do
  out=$HERE/runs/$m; mkdir -p "$out"
  for t in easy original hard; do
    [ -s "$out/jb-$t/results.jsonl" ] && continue
    (cd "$JB" && $PY -m jevbench.cli run --tasks datasets/public/$t.jsonl --adapter typesafe --endpoint "$URL" --key-env '' \
       --model "$m" --cost-basis local_open_weights_m1 --reserve-usd 0 --price-in-per-m 0 --price-out-per-m 0 \
       --results "$out/jb-$t/results.jsonl" --raw-dir "$out/jb-$t/raw" --ledger "$out/ledger.jsonl" --run-label "$m-$t" | tail -1) || true
    # jevbench stops a tier after 3 consecutive errors (e.g. prompts over the model's num_ctx); unattempted = wrong below
  done
  [ -s "$OD/results/ollama-$m.jsonl" ] || (cd "$OD" && $PY adapters/systemone_http_adapter.py --url "$URL/v1/systemone" \
       --name "ollama-$m" --model "$m" --note "via Ollama /v1/systemone" | tail -1)
  $PY - "$out" "$m" <<'EOF'
import json, sys
out, m = sys.argv[1], sys.argv[2]
tot = cor = 0; parts = {}
for t in ("easy", "original", "hard"):
    n = {"easy": 48, "original": 72, "hard": 111}[t]
    rows = [json.loads(l) for l in open(f"{out}/jb-{t}/results.jsonl")]
    c = sum(bool(r.get("correct")) for r in rows); parts[t] = f"{c}/{n}" + (f" ({n - len(rows)} unattempted)" if len(rows) < n else ""); tot += n; cor += c
print(f"{m}: JevBench public {cor}/{tot} = {cor/tot:.3f}  {parts}")
EOF
done
