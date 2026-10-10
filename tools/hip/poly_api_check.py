"""The server's API as a chat app or a coding agent (OpenCode, Bionic) uses it, against a running PolyStrata server.

    .venv\\Scripts\\python serve\\server.py --engine strata --config strata-<model>.json --port 8090
    .venv\\Scripts\\python tools\\hip\\poly_api_check.py --base http://127.0.0.1:8090 [--json out.json]

Each check sends what such a client sends (OpenAI chat completions, streamed or not, with tools; Anthropic messages)
and looks at what comes back: the models list, a system prompt that is obeyed, a streamed answer that ends with
[DONE] and its usage, a conversation of several turns that remembers what was said and reads only its new tokens
(usage.prompt_tokens_details.cached_tokens), a full tool cycle (call, result, answer), a long answer, a request
closed half way and the next one served, Anthropic's route streamed and not, and a session of many turns.
Thinking is off (chat_template_kwargs.enable_thinking false) except in the check named "thinking".
Exit 0 when every check passes.  Nothing here starts or changes the server.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8090"
KEY = ""
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def req(path: str, body: dict | None = None, stream: bool = False, timeout: float = 900.0):
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = f"Bearer {KEY}"
        h["x-api-key"] = KEY
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None, headers=h)
    resp = urllib.request.urlopen(r, timeout=timeout)
    if stream:
        return resp
    with resp:
        return json.loads(resp.read())


def chat(messages, **kw) -> dict:
    return req("/v1/chat/completions", {"model": "default", "messages": messages, "temperature": 0, **NOTHINK, **kw})


def sse(resp):
    """The data of each server-sent event, as a dict (or the string "[DONE]")."""
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:"):
                d = line[5:].strip()
                yield d if d == "[DONE]" else json.loads(d)


def stream_chat(messages, **kw):
    t = time.time()
    first, text, chunks, done, usage, finish, calls = None, "", 0, False, None, None, {}
    for d in sse(req("/v1/chat/completions", {"model": "default", "messages": messages, "temperature": 0,
                                              "stream": True, "stream_options": {"include_usage": True},
                                              **NOTHINK, **kw}, stream=True)):
        if d == "[DONE]":
            done = True
            continue
        if d.get("usage"):
            usage = d["usage"]
        for c in d.get("choices", []):
            delta = c.get("delta", {})
            if delta.get("content"):
                if first is None:
                    first = time.time() - t
                text += delta["content"]
                chunks += 1
            for tc in delta.get("tool_calls") or []:
                if first is None:
                    first = time.time() - t
                slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                slot["id"] = tc.get("id") or slot["id"]
                f = tc.get("function", {})
                slot["name"] += f.get("name") or ""
                slot["arguments"] += f.get("arguments") or ""
            if c.get("finish_reason"):
                finish = c["finish_reason"]
    return {"text": text, "chunks": chunks, "done": done, "usage": usage, "finish": finish, "ttft_s": first,
            "wall_s": time.time() - t, "tool_calls": [calls[k] for k in sorted(calls)]}


# ---------------------------------------------------------------------------------------------------------- checks
def c_models():
    m = req("/v1/models")
    ids = [x["id"] for x in m.get("data", [])]
    return bool(ids), {"models": ids}


def c_system():
    r = chat([{"role": "system", "content": "You are a pirate. Every answer starts with the word 'Arr'."},
              {"role": "user", "content": "What is 2+2?"}], max_tokens=60)
    t = r["choices"][0]["message"]["content"] or ""
    return t.strip().lower().startswith("arr"), {"answer": t[:200], "usage": r.get("usage")}


def c_stream():
    s = stream_chat([{"role": "user", "content": "Count from 1 to 20, separated by commas."}], max_tokens=120)
    ok = s["done"] and s["chunks"] > 5 and "20" in s["text"] and s["usage"] is not None
    return ok, {k: s[k] for k in ("chunks", "done", "finish", "ttft_s", "usage")} | {"text": s["text"][:120]}


def c_multiturn():
    msgs = [{"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "My name is Jorge and my favourite number is 417. Just say OK."}]
    out = []
    for q in ("What is my name?", "What is my favourite number plus one?"):
        r1 = stream_chat(msgs, max_tokens=60)
        msgs.append({"role": "assistant", "content": r1["text"]})
        msgs.append({"role": "user", "content": q})
        out.append({"q": msgs[-3]["content"][:40], "a": r1["text"][:80], "usage": r1["usage"], "ttft_s": r1["ttft_s"]})
    r = stream_chat(msgs, max_tokens=60)
    out.append({"q": msgs[-1]["content"], "a": r["text"][:80], "usage": r["usage"], "ttft_s": r["ttft_s"]})
    name_ok = "jorge" in out[2]["a"].lower() or "jorge" in out[1]["a"].lower()
    # the second and third turns repeat the conversation so far: most of it is read already
    cached = [(o["usage"] or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) for o in out]
    return name_ok and "418" in out[2]["a"] and cached[2] > 0, {"turns": out, "cached_tokens": cached}


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The current weather in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "the city"}},
                   "required": ["city"]}}}]


def c_tools(streamed: bool):
    msgs = [{"role": "user", "content": "What is the weather in Lisbon right now? Use the tool."}]
    if streamed:
        s = stream_chat(msgs, tools=TOOLS, max_tokens=300)
        calls, finish = s["tool_calls"], s["finish"]
    else:
        r = chat(msgs, tools=TOOLS, max_tokens=300)
        m = r["choices"][0]["message"]
        calls = [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                 for c in m.get("tool_calls") or []]
        finish = r["choices"][0]["finish_reason"]
    if not calls:
        return False, {"finish": finish, "calls": calls}
    c = calls[0]
    args = json.loads(c["arguments"] or "{}")
    msgs.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}]})
    msgs.append({"role": "tool", "tool_call_id": c["id"],
                 "content": json.dumps({"city": args.get("city"), "temp_c": 23, "sky": "sunny"})})
    s2 = stream_chat(msgs, tools=TOOLS, max_tokens=200)
    ok = c["name"] == "get_weather" and "lisbon" in str(args).lower() and finish == "tool_calls" and "23" in s2["text"]
    return ok, {"finish": finish, "call": c, "answer": s2["text"][:200], "usage": s2["usage"]}


def c_long():
    s = stream_chat([{"role": "user", "content": "Write a detailed essay of about 1500 words on the history of "
                                                 "Lisbon, with headings."}], max_tokens=2500)
    n = (s["usage"] or {}).get("completion_tokens", 0)
    return s["done"] and n >= 1000, {"completion_tokens": n, "finish": s["finish"], "chars": len(s["text"]),
                                     "tok_s": round(n / max(s["wall_s"] - (s["ttft_s"] or 0), 1e-3), 1)}


def c_cancel():
    resp = req("/v1/chat/completions", {"model": "default", "temperature": 0, "stream": True, "max_tokens": 2000,
                                        "messages": [{"role": "user", "content": "Write a very long story."}],
                                        **NOTHINK}, stream=True)
    got = 0
    for raw in resp:
        if raw.startswith(b"data:"):
            got += 1
            if got >= 10:
                break
    resp.close()   # the client goes away half way
    t = time.time()
    r = chat([{"role": "user", "content": "Say hello."}], max_tokens=20)
    wait = time.time() - t
    t = r["choices"][0]["message"]["content"] or ""
    return bool(t) and wait < 30, {"events_before_close": got, "next_request_s": round(wait, 2), "next": t[:60]}


def c_anthropic():
    body = {"model": "default", "max_tokens": 80, "system": "Answer in one short sentence.",
            "messages": [{"role": "user", "content": "What is the capital of France?"}], "temperature": 0,
            "thinking": {"type": "disabled"}}
    r = req("/v1/messages", body)
    text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
    kinds, st = [], ""
    for d in sse(req("/v1/messages", {**body, "stream": True}, stream=True)):
        if isinstance(d, dict):
            kinds.append(d.get("type"))
            if d.get("type") == "content_block_delta":
                st += d.get("delta", {}).get("text", "")
    ok = "paris" in text.lower() and "paris" in st.lower() and "message_stop" in kinds
    return ok, {"text": text[:100], "stream_text": st[:100], "events": sorted(set(kinds)), "usage": r.get("usage")}


def c_anthropic_tools():
    tools = [{"name": "get_weather", "description": "The current weather in a city.",
              "input_schema": TOOLS[0]["function"]["parameters"]}]
    msgs = [{"role": "user", "content": "What is the weather in Porto? Use the tool."}]
    r = req("/v1/messages", {"model": "default", "max_tokens": 300, "messages": msgs, "tools": tools,
                             "temperature": 0, "thinking": {"type": "disabled"}})
    use = [b for b in r.get("content", []) if b.get("type") == "tool_use"]
    if not use:
        return False, {"stop_reason": r.get("stop_reason"), "content": r.get("content")}
    msgs.append({"role": "assistant", "content": r["content"]})
    msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": use[0]["id"],
                                              "content": json.dumps({"temp_c": 19, "sky": "cloudy"})}]})
    r2 = req("/v1/messages", {"model": "default", "max_tokens": 200, "messages": msgs, "tools": tools,
                              "temperature": 0, "thinking": {"type": "disabled"}})
    text = "".join(b.get("text", "") for b in r2.get("content", []) if b.get("type") == "text")
    ok = r.get("stop_reason") == "tool_use" and use[0]["name"] == "get_weather" and "19" in text
    return ok, {"tool_use": use[0], "answer": text[:200]}


def c_thinking():
    r = req("/v1/chat/completions", {"model": "default", "temperature": 0, "max_tokens": 1500,
                                     "reasoning_effort": "low",
                                     "messages": [{"role": "user", "content": "Is 391 a prime number?"}]})
    m = r["choices"][0]["message"]
    ok = bool(m.get("reasoning_content")) and bool(m.get("content"))
    return ok, {"reasoning_chars": len(m.get("reasoning_content") or ""), "answer": (m.get("content") or "")[:160],
                "usage": r.get("usage")}


def c_session(turns: int):
    """A long session: each turn adds a paragraph to read and a question; each turn reads only what is new."""
    msgs = [{"role": "system", "content": "You are a careful assistant. Answer in at most two sentences."}]
    rows = []
    for i in range(turns):
        para = (f"Note {i + 1}: the warehouse in section {chr(65 + i % 26)} holds {100 + 37 * i} crates of "
                f"item number {9000 + i}. ") * 12
        msgs.append({"role": "user", "content": para + f"How many crates are in section {chr(65 + i % 26)}?"})
        s = stream_chat(msgs, max_tokens=60)
        msgs.append({"role": "assistant", "content": s["text"]})
        u = s["usage"] or {}
        rows.append({"turn": i + 1, "prompt_tokens": u.get("prompt_tokens"),
                     "cached": u.get("prompt_tokens_details", {}).get("cached_tokens"),
                     "ttft_s": round(s["ttft_s"] or 0, 3), "right": str(100 + 37 * i) in s["text"]})
    msgs.append({"role": "user", "content": "How many crates were in section A, at the start of our talk?"})
    s = stream_chat(msgs, max_tokens=60)
    recall = "100" in s["text"]
    # each turn reads again at most the turn the model answered (its template may write it otherwise as history)
    reuse = all((b["cached"] or 0) >= (a["prompt_tokens"] or 0) - 64 for a, b in zip(rows, rows[1:]))
    ok = recall and reuse and sum(r["right"] for r in rows) >= turns - 1
    return ok, {"turns": rows, "recall": s["text"][:120], "reuse_every_turn": reuse}


def listening() -> list[str]:
    """Where the port is listened on (Windows netstat): 127.0.0.1 only, unless an API key is set."""
    import subprocess
    port = BASE.rsplit(":", 1)[1].split("/")[0]
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
    except OSError:
        return []
    return sorted({l.split()[1] for l in out.splitlines() if "LISTENING" in l and l.split()[1].endswith(":" + port)})


def main() -> int:
    global BASE, KEY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--only", help="run only the checks whose name contains this (comma-separated)")
    ap.add_argument("--session-turns", type=int, default=12)
    ap.add_argument("--json")
    a = ap.parse_args()
    BASE, KEY = a.base.rstrip("/"), a.api_key
    checks = [("models", c_models), ("system", c_system), ("stream", c_stream), ("multiturn", c_multiturn),
              ("tools", lambda: c_tools(False)), ("tools-stream", lambda: c_tools(True)), ("long", c_long),
              ("cancel", c_cancel), ("anthropic", c_anthropic), ("anthropic-tools", c_anthropic_tools),
              ("thinking", c_thinking), ("session", lambda: c_session(a.session_turns))]
    if a.only:
        want = a.only.split(",")
        checks = [c for c in checks if any(w in c[0] for w in want)]
    res, good = {}, True
    where = listening()
    res["listening"] = where
    local = all(w.startswith(("127.0.0.1:", "[::1]:")) for w in where)
    print(f"listening on {where}: {'this PC only' if local else 'NOT only this PC'}", flush=True)
    good = good and (local or bool(KEY))
    for name, fn in checks:
        t = time.time()
        try:
            ok, info = fn()
        except Exception as e:   # noqa: BLE001 - a check that throws is a failed check
            ok, info = False, {"error": f"{type(e).__name__}: {e}"}
        info["seconds"] = round(time.time() - t, 2)
        res[name] = {"ok": ok, **info}
        good = good and ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:16s} {info['seconds']:7.2f} s  "
              f"{json.dumps(info, ensure_ascii=False)[:300]}", flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    print("api:", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    socket.setdefaulttimeout(900)
    sys.exit(main())
