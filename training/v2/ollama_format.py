"""Byte-exact copy of how Ollama 0.35 turns a /v1/systemone request into scoring prompts
(ollama/decision/systemone.go: Compile + compileField), for Qwen3.5 GGUFs whose native chat template
Ollama renders with thinking disabled.

Per question field Ollama builds ONE user message:
    json.Marshal({"context": <state>, "schema": [every field]}) + "\\n\\nRequested field: " + json.Marshal(name)
renders it after the Modelfile SYSTEM prompt with the model's chat template (think=false), and reads the
next-token logits of the choice codes "A", "B", ... Probabilities are a plain softmax over those codes.

Details that change bytes, all mirrored here:
- a string state is used as-is; an object/array state is the client's own JSON text, compacted (key order and
  escapes kept), so the same request encoded by Python (ensure_ascii) and by JS gives different prompts;
- Go's encoder escapes <, >, & and U+2028/2029, writes \\b and \\f as \\u0008 / \\u000c, and never escapes non-ASCII;
- noul is always [A=false "No", B=true "Yes"] unless criteria override the descriptions; score values are "0","1"...
"""
import json

_SHORT = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def go_str(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch in _SHORT:
            out.append(_SHORT[ch])
        elif ch < " " or ch in "<>&  ":
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def go_json(v) -> str:
    """encoding/json.Marshal for the value shapes Ollama marshals (dict in insertion order, list, str, bool)."""
    if isinstance(v, str): return go_str(v)
    if isinstance(v, bool): return "true" if v else "false"
    if v is None: return "null"
    if isinstance(v, (int, float)): return json.dumps(v)
    if isinstance(v, dict): return "{" + ",".join(go_str(k) + ":" + go_json(x) for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)): return "[" + ",".join(go_json(x) for x in v) + "]"
    raise TypeError(type(v))


def client_json(v, ensure_ascii: bool) -> str:
    """What a client sends for an object state: Python requests (ensure_ascii=True) or JS JSON.stringify (False)."""
    return json.dumps(v, ensure_ascii=ensure_ascii, separators=(",", ":"))


def content(raw) -> str:
    """decision.content: string as-is; object/array -> the client's compact JSON text."""
    if isinstance(raw, str): return raw
    if isinstance(raw, tuple) and raw[0] == "raw": return raw[1]            # pre-encoded client JSON
    return client_json(raw, ensure_ascii=True)


def compile_field(name: str, q: dict):
    """-> (field dict in Go's field order, list of choice codes)."""
    t = q["type"]
    ch = []
    add = lambda value, desc: ch.append({"code": chr(ord("A") + len(ch)), "value": value, "description": desc})
    if t == "noul":
        crit = q.get("criteria") or {}
        add(False, crit.get("false", "No")); add(True, crit.get("true", "Yes"))
    elif t == "choice":
        for k, d in q["criteria"].items():
            add(k, k if d is None else d)
    elif t == "score":
        for i, d in enumerate(q["criteria"]):
            add(str(i), d)
    else:
        raise ValueError(t)
    assert 2 <= len(ch) <= 26
    return {"name": name, "description": content(q["instructions"]), "choices": ch}, [c["code"] for c in ch]


def user_messages(state, questions: dict):
    """One (field_name, user_message, codes) per question, exactly as Ollama compiles them."""
    fields = [compile_field(n, q) for n, q in questions.items()]
    data = go_json({"context": content(state), "schema": [f for f, _ in fields]})
    return [(f["name"], data + "\n\nRequested field: " + go_str(f["name"]), codes) for f, codes in fields]


def render(tok, system: str, user: str) -> str:
    """Qwen3.5 chat template, think off: ends with '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


if __name__ == "__main__":
    # self-check against the escaping and layout rules above
    assert go_str('a<b>&"\\\n\t\x08\x0c é') == '"a\\u003cb\\u003e\\u0026\\"\\\\\\n\\t\\u0008\\u000c\\u2028é"'
    [(name, msg, codes)] = user_messages({"ticket": "I was charged twice & want a refund"},
                                         {"refund": {"type": "noul", "instructions": "Refund requested?"}})
    assert codes == ["A", "B"] and name == "refund"
    assert msg == ('{"context":"{\\"ticket\\":\\"I was charged twice \\u0026 want a refund\\"}","schema":[{"name":"refund",'
                   '"description":"Refund requested?","choices":[{"code":"A","value":false,"description":"No"},'
                   '{"code":"B","value":true,"description":"Yes"}]}]}\n\nRequested field: "refund"'), msg
    two = user_messages("state", {"a": {"type": "score", "instructions": "x", "criteria": ["lo", "hi"]},
                                  "b": {"type": "choice", "instructions": "y", "criteria": {"k": None, "j": "J"}}})
    assert [c for _, _, c in two] == [["A", "B"], ["A", "B"]] and '"value":"1"' in two[0][1] and '"description":"k"' in two[1][1]
    print("ollama_format self-check OK")
