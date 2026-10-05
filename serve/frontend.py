"""serve/frontend.py - plan v0.3 P8: the request/response layer between API clients and the engine.

Everything here is text: no model, no GPU. The engine consumes token ids and produces text deltas; this module

  * normalizes OpenAI Chat Completions and Anthropic Messages requests into the chat template's message form,
  * renders the model's own Jinja chat template (pack `tokenizer/chat_template.jinja`) exactly as Hugging Face
    does (checked against the pack's `chat_golden.json`, 10 cases incl. thinking and tools),
  * parses the streamed output incrementally into reasoning (`<think>...</think>`), content and tool calls in
    the template's XML form (`<tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call>`),
    never emitting a partial tag to the client.

Design note: plan v0.3 describes a single C++ process. The frontend is Python first because the template is Jinja
and the formats change often; the engine boundary is token ids in, text deltas out, which keeps a later C++ port
(or a pybind wrapper) a local change.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


# ------------------------------------------------------------------------------------------------ template
class TemplateRequestError(jinja2.exceptions.TemplateError, ValueError):
    """The template refused the request's messages (e.g. "No user query found in messages."): a ValueError, so the
    client gets a 400 with the template's message instead of a dropped connection (#365)."""


class ChatTemplate:
    """The model's chat template, rendered with the same Jinja settings as transformers' apply_chat_template."""

    def __init__(self, path: str | Path):
        def raise_exception(message):
            raise TemplateRequestError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        self.source = Path(path).read_text(encoding="utf-8")
        self.template = env.from_string(self.source)

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    **kwargs)


# ------------------------------------------------------------------------------------------------ requests
def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in
                   ("text", "input_text", None))


IMAGE_PARTS = ("image_url", "input_image", "image")

# Thinking levels.  The model's template knows low, medium and xhigh (its default; "high" means xhigh), and
# enable_thinking=false for none.  Clients spell these many ways; everything maps onto those four.
EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
          "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh", "maximum": "xhigh"}


def effort_kwargs(value) -> dict:
    """A reasoning effort as given by a client -> the template's kwargs.  Unknown values are a 400, not a crash."""
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    key = str(value).strip().lower()
    if key not in EFFORT:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium or high")
    level = EFFORT[key]
    return {"enable_thinking": False} if level is None else {"reasoning_effort": level}


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget (budget_tokens) -> a level: under 2K low, under 8K medium, else high."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "medium" if n < 8192 else "xhigh"}


def _has_image(content) -> bool:
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in content)


def _image_source(part: dict) -> str:
    """An image part's source as one string: a data: URL, an http(s) URL or a local file path.
    OpenAI: {"type": "image_url", "image_url": {"url": ...}} (or "image_url": "..."), Responses-style
    {"type": "input_image", "image_url": ...}; Anthropic: {"type": "image", "source": {"type": "base64",
    "media_type": ..., "data": ...}} or {"source": {"type": "url", "url": ...}}."""
    if part.get("type") == "image":
        src = part.get("source") or {}
        if src.get("type") == "base64":
            return f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
        return src.get("url") or src.get("path") or ""
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return url or ""


def _parts_of(content):
    """Message content for the template: a string when there is no image (unchanged behaviour), otherwise the
    template's list form - text items and image items, in order - whose image items carry their source."""
    if not _has_image(content):
        return _text_of(content)
    items = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in IMAGE_PARTS:
            items.append({"type": "image", "source": _image_source(part)})
        elif part.get("type") in ("text", "input_text", None) and "text" in part:
            items.append({"type": "text", "text": part.get("text", "")})
    return items


def images_of(messages: list[dict]) -> list[str]:
    """The image sources of the rendered conversation, in prompt order (the template renders one
    <|vision_start|><|image_pad|><|vision_end|> per image item, message by message)."""
    return [item["source"] for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") == "image"]


# #537: a literal <think> / </think> inside a message's text is plain text, not the model's reasoning markers.  The
# tokenizer matches those two strings as their special tokens everywhere (GGUF token type 4, as llama.cpp does), so
# a user quoting "</think>" used to hand the model a real end-of-reasoning token.  Before the template is rendered
# they are swapped for these private-use characters, and the server encodes the spans they mark as ordinary text.
THINK_TAGS = {"<think>": "\U000F0E01", "</think>": "\U000F0E02"}
THINK_MARKS = {v: k for k, v in THINK_TAGS.items()}


def _mark(text: str) -> str:
    for tag, mark in THINK_TAGS.items():
        text = text.replace(tag, mark)
    return text


def _mark_deep(v):
    if isinstance(v, str):
        return _mark(v)
    if isinstance(v, dict):
        return {k: _mark_deep(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_mark_deep(x) for x in v]
    return v


def _has_tag(v) -> bool:
    if isinstance(v, str):
        return "<think>" in v or "</think>" in v
    if isinstance(v, dict):
        return any(_has_tag(x) for x in v.values())
    if isinstance(v, list):
        return any(_has_tag(x) for x in v)
    return False


def mark_think_literals(messages: list[dict], tools: list[dict] | None):
    """#537: (messages, tools) with every literal <think> / </think> in their text swapped for THINK_TAGS' marks, and
    whether there was one (None: no change, the same objects back - a prompt without them renders as it always did).
    An assistant message whose content opens with a whole <think>...</think> block (clients that send the reasoning
    inline) keeps that one block as the model's markers, as before."""
    if not _has_tag(messages) and not _has_tag(tools):
        return messages, tools, False
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        for k, v in m.items():
            if k != "role":
                m[k] = _mark_deep(v)
        if m.get("role") == "assistant" and isinstance(content, str) and content.lstrip().startswith("<think>") \
                and "</think>" in content:
            i, j = content.index("<think>") + len("<think>"), content.index("</think>")
            m["content"] = content[:i] + _mark(content[i:j]) + "</think>" + _mark(content[j + len("</think>"):])
        out.append(m)
    return out, _mark_deep(tools), True


_THINK_MARK_RE = re.compile("|".join(THINK_MARKS))


def unmark_think_literals(prompt: str) -> tuple[str, list[tuple[int, int]]]:
    """The rendered prompt with THINK_TAGS' marks turned back into the tags' text, and the (start, end) spans of those
    tags in it: the server encodes them as ordinary text (the tokenizer's encode_plain_spans)."""
    out, spans, pos, n = [], [], 0, 0
    for m in _THINK_MARK_RE.finditer(prompt):
        out.append(prompt[pos:m.start()])
        n += m.start() - pos
        tag = THINK_MARKS[m.group(0)]
        out.append(tag)
        spans.append((n, n + len(tag)))
        n += len(tag)
        pos = m.end()
    out.append(prompt[pos:])
    return "".join(out), spans


def _late_system_to_user(messages: list[dict]) -> list[dict]:
    """The chat template takes a system message only at the start ("System message must be at the beginning").
    Clients also send them mid-conversation - Claude Code's hook context as {"role": "system"} after the first user
    turn, some OpenAI clients a late "developer" message (issue #56) - so those become user messages, in place:
    merging them into the first one would change the prompt's start and cost the conversation cache every turn."""
    return [dict(m, role="user") if m.get("role") == "system" and i > 0 else m for i, m in enumerate(messages)]


def _object_list(value, name: str) -> list[dict]:
    """#460: a request's "messages" (or a message's "tool_calls") as a list of objects.  Some clients send the array
    double-encoded, as a JSON string, which used to be iterated character by character and crashed on m.get: such a
    string is decoded.  Anything that is still not a list of objects is a ValueError, which the server answers with
    a 400 naming the field.  None (or no field) is an empty list, as a missing field always was."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(f"{name} must be a list of objects (a string was sent that is not JSON)") from None
    if not isinstance(value, list) or not all(isinstance(m, dict) for m in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def _tool_list(value, wrapper: str | None) -> list[dict]:
    """#592: a request's "tools" as a list of tool objects, each with a name - in the OpenAI shape
    {"type": "function", "function": {"name": ...}} (`wrapper` "function"; a bare {"name": ...} is still taken), or
    in the Anthropic shape {"name": ...} (`wrapper` None).  A value that is not (a string such as "auto", a list of
    names, an object without a name) is a ValueError - a 400 naming the field - where it used to take the request
    thread down with no reply at all.  No value (or an empty one) is no tools, as always."""
    if not value:
        return []
    shape = ('{"type": "function", "function": {"name": ..., "parameters": {...}}}' if wrapper else
             '{"name": ..., "input_schema": {...}}')
    try:
        tools = _object_list(value, "tools")
    except ValueError:
        raise ValueError(f"tools must be a list of tool objects, each {shape}") from None
    for i, t in enumerate(tools):
        fn = t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ValueError(f"tools[{i}] has no name: each tool must be {shape}")
    return tools


def _answers_in_call_order(messages: list[dict]) -> list[dict]:
    """Tool answers put in the order of the calls they answer.  A model reads them by position (Gemma's and Qwen's
    templates write no id), so answers sent in another order would be read as each other's.  A run of answers stays
    as it is unless each names a call of the assistant message before it, and each a different one."""
    out, i = [], 0
    while i < len(messages):
        out.append(messages[i])
        ids = [c.get("id") for c in messages[i].get("tool_calls") or []]
        i += 1
        end = i
        while end < len(messages) and messages[end].get("role") == "tool":
            end += 1
        run = messages[i:end]
        named = [t.get("tool_call_id") for t in run]
        if len(run) > 1 and all(ids) and len(set(named)) == len(named) and set(named) <= set(ids):
            run = sorted(run, key=lambda t: ids.index(t["tool_call_id"]))
        out += run
        i = end
    return out


def openai_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """OpenAI Chat Completions -> (template messages, template tools, template kwargs)."""
    messages = []
    for m in _object_list(req.get("messages"), "messages"):
        role = m.get("role")
        if role == "developer":
            role = "system"
        out = {"role": role, "content": _parts_of(m.get("content")) if role in ("user", "tool", "assistant") else _text_of(m.get("content"))}
        if m.get("reasoning_content"):
            out["reasoning_content"] = m["reasoning_content"]
        if role == "tool" and m.get("tool_call_id"):    # which call it answers: a template may name the tool by it
            out["tool_call_id"] = m["tool_call_id"]
        if m.get("tool_calls"):
            calls = []
            for c in _object_list(m["tool_calls"], "tool_calls"):
                fn = c.get("function", c)
                if not isinstance(fn, dict):
                    raise ValueError("tool_calls must be a list of objects (each with a \"function\" object)")
                args = fn.get("arguments")
                if isinstance(args, str):               # the template requires a mapping, not a JSON string
                    args = json.loads(args) if args.strip() else {}
                calls.append({"function": {"name": fn.get("name"), "arguments": args or {}},
                              **({"id": c["id"]} if c.get("id") else {})})
            out["tool_calls"] = calls
        messages.append(out)
    tools = [t.get("function", t) if t.get("type") == "function" else t
             for t in _tool_list(req.get("tools"), "function")] or None
    kwargs = {}
    # OpenAI Chat Completions: "reasoning_effort"; Responses style: "reasoning": {"effort": ...}
    reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
    kwargs.update(effort_kwargs(req.get("reasoning_effort") or reasoning.get("effort")))
    # the vLLM / llama.cpp convention: {"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}}
    for k, v in (req.get("chat_template_kwargs") or {}).items():
        if k == "enable_thinking" and not v:
            kwargs = {"enable_thinking": False}
        elif k == "reasoning_effort" and "enable_thinking" not in kwargs:
            kwargs.update(effort_kwargs(v))
    return _answers_in_call_order(_late_system_to_user(messages)), tools, kwargs


def anthropic_to_messages(req: dict, think_unasked: bool = True) -> tuple[list[dict], list[dict] | None, dict]:
    """Anthropic Messages -> (template messages, template tools, template kwargs).  `think_unasked`: a request
    without "thinking", an effort or a budget gets the template's default (it thinks), as through 0.1.31; False
    renders it without thinking (#278, the config's "anthropic_thinking": "on_request")."""
    messages = []
    system = req.get("system")
    if system:
        messages.append({"role": "system", "content": _text_of(system)})
    for m in _object_list(req.get("messages"), "messages"):
        content = m.get("content")
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        if m.get("role") == "user" and _has_image(content) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            messages.append({"role": "user", "content": _parts_of(content)})
            continue
        text, reasoning, calls = [], [], []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                text.append(block.get("text", ""))
            elif kind == "thinking":
                reasoning.append(block.get("thinking", ""))
            elif kind == "tool_use":
                calls.append({"function": {"name": block.get("name"), "arguments": block.get("input") or {}},
                              **({"id": block["id"]} if block.get("id") else {})})
            elif kind == "tool_result":
                messages.append({"role": "tool", "content": _text_of(block.get("content")),
                                 **({"tool_call_id": block["tool_use_id"]} if block.get("tool_use_id") else {})})
        if text or calls or reasoning:
            out = {"role": m["role"], "content": "".join(text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
    tools = [{"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}
             for t in _tool_list(req.get("tools"), None)] or None
    kwargs = {}
    # Anthropic: "thinking": {"type": "disabled"} or {"type": "enabled", "budget_tokens": N};
    # "output_config": {"effort": "low" | "medium" | "high"}
    thinking = req.get("thinking")
    effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        kwargs["enable_thinking"] = False
    elif effort:
        kwargs.update(effort_kwargs(effort))
    elif isinstance(thinking, dict) and thinking.get("budget_tokens"):
        kwargs.update(budget_effort(thinking["budget_tokens"]))
    elif thinking is None and not req.get("reasoning_budget_tokens") and not think_unasked:
        # Opt-in (the config's "anthropic_thinking": "on_request"; the default thinks as 0.1.31 did, since a
        # client that never asks would otherwise lose the thinking on every turn).  Anthropic's thinking is
        # opt-in there. Claude Code's helper calls (a session title, a topic check) ask for none
        # and allow a few dozen tokens, which the model otherwise spent thinking and answered with no text at all.
        # A config's reasoning_effort still applies: Service.with_shared sets output_config before this runs.  A
        # request that gives its own reasoning_budget_tokens (#123) asks for thinking, so it thinks as before.
        kwargs["enable_thinking"] = False
    return _answers_in_call_order(_late_system_to_user(messages)), tools, kwargs


# ------------------------------------------------------------------------------------------------ output parser
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])


@dataclass
class Event:
    kind: str                     # "reasoning" | "content" | "tool_call"
    text: str = ""
    call: ToolCall | None = None


THINK_END = "</think>"
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"


PARAM_END = "</parameter>"
FUNC_END = "</function>"


def param_end(text: str, final: bool = False) -> int:
    """Where a parameter value in `text` ends: the first `</parameter>` followed (after whitespace) by the next
    `<parameter=` or `</function>` - the same text inside a value (a file that documents the call format, #210) is
    part of the value.  -1: none yet; -2: a candidate whose follower has not arrived (streaming; `final` accepts it)."""
    at = text.find(PARAM_END)
    while at >= 0:
        after = text[at + len(PARAM_END):].lstrip()
        if after.startswith(("<parameter=", FUNC_END)):
            return at
        if not after or "<parameter=".startswith(after) or FUNC_END.startswith(after):
            return at if final else -2
        at = text.find(PARAM_END, at + 1)
    return -1


def call_end(text: str) -> int:
    """Where a tool call's body ends (#210): the `</tool_call>` after the call's own `</function>`, found by walking
    its parameters with param_end, so a value may contain either tag.  -1: not complete yet.  A body that is not in
    the call format ends at the first `</tool_call>`, as before."""
    s = text.lstrip()
    pos = len(text) - len(s)
    if not s.startswith("<function="):
        return text.find(CALL_END) if not "<function=".startswith(s) else -1
    gt = text.find(">", pos)
    if gt < 0:
        return -1
    pos = gt + 1
    while True:
        rest = text[pos:]
        s = rest.lstrip()
        pos += len(rest) - len(s)
        if s.startswith("<parameter="):
            gt = text.find(">", pos)
            if gt < 0:
                return -1
            end = param_end(text[gt + 1:])
            if end < 0:
                return -1
            pos = gt + 1 + end + len(PARAM_END)
        elif s.startswith(FUNC_END):
            rest = text[pos + len(FUNC_END):]
            s = rest.lstrip()
            if s.startswith(CALL_END):
                return pos + len(FUNC_END) + len(rest) - len(s)
            return -1 if CALL_END.startswith(s) else text.find(CALL_END, pos)
        elif not s or "<parameter=".startswith(s) or FUNC_END.startswith(s):
            return -1
        else:
            return text.find(CALL_END, pos)


def parse_tool_call(body: str, schema: dict | None = None) -> ToolCall:
    """`<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>...</function>` -> ToolCall. Values are JSON-decoded
    when the tool's schema says the parameter is not a string (or, without a schema, when they parse as JSON
    objects/arrays/numbers/booleans)."""
    body = body.strip()
    if not body.startswith("<function=") or ">" not in body:
        raise ValueError("malformed tool call: " + body[:80])
    name = body[len("<function="):body.index(">")]
    rest = body[body.index(">") + 1:]
    props = ((schema or {}).get("parameters") or {}).get("properties") or {}
    args = {}
    while "<parameter=" in rest:
        rest = rest[rest.index("<parameter=") + len("<parameter="):]
        pname = rest[:rest.index(">")]
        rest = rest[rest.index(">") + 1:]
        end = param_end(rest, final=True)
        value = rest[:end] if end >= 0 else rest
        rest = rest[end + len(PARAM_END):] if end >= 0 else ""
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        declared = (props.get(pname) or {}).get("type")
        if declared == "string":
            args[pname] = value
        else:
            try:
                args[pname] = json.loads(value)
            except ValueError:
                args[pname] = value
    return ToolCall(name=name, arguments=args)


ARG_KEY, ARG_KEY_END = "<arg_key>", "</arg_key>"
ARG_VALUE, ARG_VALUE_END = "<arg_value>", "</arg_value>"


def glm_call_name(text: str) -> str | None:
    """A GLM call's name, or None while it is still arriving."""
    at = [i for i in (text.find(ARG_KEY), text.find(CALL_END)) if i >= 0]
    return text[:min(at)].strip() if at else None


def glm_call_end(text: str) -> int:
    """Where a GLM tool call's body ends: the first `</tool_call>` right after the name or an `</arg_value>`, so an
    argument value may contain the tag.  -1: not complete yet."""
    at = text.find(CALL_END)
    while at >= 0:
        before = text[:at].rstrip()
        if before.endswith(ARG_VALUE_END) or ARG_KEY not in before:
            return at
        at = text.find(CALL_END, at + 1)
    return -1


def parse_glm_tool_call(body: str, schema: dict | None = None) -> ToolCall:
    """`NAME<arg_key>K</arg_key><arg_value>V</arg_value>...` -> ToolCall (GLM's chat template writes calls this
    way).  Values are typed as in parse_tool_call."""
    name, _, rest = body.partition(ARG_KEY)
    props = ((schema or {}).get("parameters") or {}).get("properties") or {}
    args = {}
    for part in rest.split(ARG_KEY) if rest else []:
        key, _, tail = part.partition(ARG_KEY_END)
        value = tail.partition(ARG_VALUE)[2]
        end = value.rfind(ARG_VALUE_END)        # the last one: a value may quote the tag
        value = value[:end] if end >= 0 else value
        if (props.get(key.strip()) or {}).get("type") == "string":
            args[key.strip()] = value
        else:
            try:
                args[key.strip()] = json.loads(value)
            except ValueError:
                args[key.strip()] = value
    return ToolCall(name=name.strip(), arguments=args)


# Gemma 4 writes a call as <|tool_call>call:NAME{key:value,...}<tool_call|>.  A value is a string between two
# <|"|> tokens, a number, true, false, null, a {key:value} mapping or a [value] list (its chat template's
# format_argument; llama.cpp's common/parsers/gemma4.cpp reads the same).
GEMMA_CALL, GEMMA_CALL_END, GEMMA_QUOTE = "<|tool_call>", "<tool_call|>", '<|"|>'


def gemma_call_name(text: str) -> str | None:
    """A Gemma call's name, or None while it is still arriving."""
    at = [i for i in (text.find("{"), text.find(GEMMA_CALL_END)) if i >= 0]
    return text[:min(at)].strip().removeprefix("call:") if at else None


def gemma_call_end(text: str) -> int:
    """Where a Gemma tool call's body ends: the first `<tool_call|>` that is not inside a string.  -1: not yet."""
    pos = 0
    while True:
        quote, end = text.find(GEMMA_QUOTE, pos), text.find(GEMMA_CALL_END, pos)
        if end >= 0 and (quote < 0 or end < quote):
            return end
        close = text.find(GEMMA_QUOTE, quote + len(GEMMA_QUOTE)) if quote >= 0 else -1
        if close < 0:
            return -1
        pos = close + len(GEMMA_QUOTE)


def _gemma_value(s: str, i: int):
    """The value of Gemma's call notation that starts at s[i] -> (value, the index after it).  Text that ends
    early (a call the output stopped in) gives what was there."""
    while i < len(s) and s[i].isspace():
        i += 1
    if s.startswith(GEMMA_QUOTE, i):
        i += len(GEMMA_QUOTE)
        end = s.find(GEMMA_QUOTE, i)
        return (s[i:], len(s)) if end < 0 else (s[i:end], end + len(GEMMA_QUOTE))
    if s.startswith(("{", "["), i):
        mapping, close = s[i] == "{", "}" if s[i] == "{" else "]"
        out, i = ({} if mapping else []), i + 1
        while True:
            while i < len(s) and (s[i].isspace() or s[i] == ","):
                i += 1
            if i >= len(s) or s[i] == close:
                return out, i + 1
            if mapping:
                colon = s.find(":", i)
                if colon < 0:
                    return out, len(s)
                key, i = s[i:colon].strip().removeprefix(GEMMA_QUOTE).removesuffix(GEMMA_QUOTE), colon + 1
                out[key], i = _gemma_value(s, i)
            else:
                value, i = _gemma_value(s, i)
                out.append(value)
    end = i
    while end < len(s) and s[end] not in ",}]":
        end += 1
    word = s[i:end].strip()
    try:
        return json.loads(word), end
    except ValueError:
        return word, end


def parse_gemma_tool_call(body: str, schema: dict | None = None) -> ToolCall:
    """`call:NAME{key:value,...}` -> ToolCall.  A parameter the tool's schema declares a string stays one when the
    model wrote it bare."""
    brace = body.find("{")
    args = _gemma_value(body, brace)[0] if brace >= 0 else {}
    props = ((schema or {}).get("parameters") or {}).get("properties") or {}
    for key, value in args.items():
        if (props.get(key) or {}).get("type") == "string" and not isinstance(value, str):
            args[key] = json.dumps(value, ensure_ascii=False)
    return ToolCall(name=gemma_call_name(body + GEMMA_CALL_END), arguments=args)


def gemma_model_turn(prompt: str, thinking: bool) -> str:
    """Gemma's template opens no turn for the model after an answer that came with text, tool calls and their
    results: the prompt ends on that closed turn and, with thinking on, on a thought opened outside any turn.  This
    ends it as the template ends every other prompt.  (llama.cpp adds the turn too, in the case without thinking.)"""
    thought = "<|channel>thought\n"
    if prompt.endswith("<turn|>\n" + thought):
        prompt = prompt[:-len(thought)]
    if prompt.endswith("<turn|>\n"):
        prompt += "<|turn>model\n" + ("" if thinking else thought + "<channel|>")
    return prompt


class OutputParser:
    """Incremental parser of the model's text. Feed deltas; get events. A tag split across deltas is held back
    until it is complete, so clients never see `<tool_` or `</thi`."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 call_format: str = "function", opened: bool = False):
        # GLM's <arg_key>/<arg_value> calls and Gemma's call:NAME{...} instead of <function=...>: such a call is
        # announced when its name is complete and its arguments follow whole.  (its name, its end, its parser)
        self.whole = {"glm": (glm_call_name, glm_call_end, parse_glm_tool_call),
                      "gemma": (gemma_call_name, gemma_call_end, parse_gemma_tool_call)}.get(call_format)
        self.call_start, self.call_stop = (GEMMA_CALL, GEMMA_CALL_END) if call_format == "gemma" \
            else (CALL_START, CALL_END)
        # Gemma opens its reasoning itself, as a channel, and ends it with a token of its own; the others' reasoning
        # is opened by the prompt and ends with </think>.  `opened`: this prompt ended on Gemma's opening (its
        # template writes it after a tool's answer), so the reply starts inside the reasoning
        self.think_open, self.think_end = ("<|channel>thought\n", "<channel|>") if call_format == "gemma" \
            else ("", THINK_END)
        # the reply may still start with think_open.  Asked not to think, Gemma can still open with a thought
        # (an empty one, after a tool's answer): it is read either way
        self.opening = bool(self.think_open) and not opened
        self.state = "reasoning" if thinking and (opened or not self.think_open) else "content"
        self.buf = ""
        self.lead = False
        self.schemas = {t.get("name"): t for t in tools or []}
        # stream_tools: a tool call is also reported while it is being written - "tool_start" (its name and id) as
        # soon as the name is known, then "tool_args" pieces of its JSON arguments (string parameters character by
        # character; other types whole, once complete) - before the final "tool_call".  Without it, a client sees
        # nothing until the call is complete, which for a large file write can be many minutes.
        self.stream_tools = stream_tools
        self._reset_scan()

    def _reset_scan(self):
        self.sp = 0                  # how much of self.buf (the call body) the scanner has consumed
        self.ss = "name"             # name -> between -> str|raw -> ... -> done
        self.scall = None            # the ToolCall being streamed (its id is reused by the final event)
        self.sfirst = True
        self.sval_started = False
        self.sdeclared = {}

    def _scan(self) -> list[Event]:
        """Advance the streaming view of the call body in self.buf (see stream_tools)."""
        out = []

        def args(s):
            if s:
                out.append(Event("tool_args", s, call=self.scall))
        while True:
            rest = self.buf[self.sp:]
            if self.ss == "name":
                a = rest.find("<function=")
                b = rest.find(">", a + 10) if a >= 0 else -1
                if b < 0:
                    return out
                name = rest[a + 10:b]
                self.scall = ToolCall(name=name, arguments={})
                props = ((self.schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
                self.sdeclared = {k: (v or {}).get("type") for k, v in props.items()}
                out.append(Event("tool_start", call=self.scall))
                args("{")
                self.sp += b + 1
                self.ss = "between"
            elif self.ss == "between":
                stripped = rest.lstrip()
                self.sp += len(rest) - len(stripped)
                if stripped.startswith("<parameter="):
                    b = stripped.find(">")
                    if b < 0:
                        return out
                    pname = stripped[11:b]
                    args(("" if self.sfirst else ",") + json.dumps(pname) + ":")
                    self.sfirst = False
                    self.sp += b + 1
                    if self.sdeclared.get(pname) == "string":
                        args('"')
                        self.ss, self.sval_started = "str", False
                    else:
                        self.ss = "raw"
                elif stripped.startswith("</function>"):
                    args("}")
                    self.sp += len("</function>")
                    self.ss = "done"
                else:
                    return out          # a tag still arriving (or trailing text): wait
            elif self.ss == "str":
                if not self.sval_started:
                    if not rest:
                        return out
                    if rest[0] == "\n":          # the value's leading newline is not part of it
                        self.sp += 1
                        rest = rest[1:]
                    self.sval_started = True
                end = param_end(rest)
                if end >= 0:
                    value = rest[:end]
                    if value.endswith("\n"):
                        value = value[:-1]
                    args(json.dumps(value)[1:-1] + '"')
                    self.sp += end + len(PARAM_END)
                    self.ss = "between"
                    continue
                # an undecided </parameter> (-2) is held from its start, like a tag still arriving
                safe = rest.find(PARAM_END) if end == -2 else len(rest) - self._hold(rest, (PARAM_END,))
                if safe > 0 and rest[safe - 1] == "\n":   # may be the trailing newline before </parameter>
                    safe -= 1
                if safe > 0:
                    args(json.dumps(rest[:safe])[1:-1])
                    self.sp += safe
                return out
            elif self.ss == "raw":
                end = param_end(rest)
                if end < 0:
                    return out
                value = rest[:end]
                if value.startswith("\n"):
                    value = value[1:]
                if value.endswith("\n"):
                    value = value[:-1]
                try:
                    v = json.loads(value)
                except ValueError:
                    v = value
                args(json.dumps(v, ensure_ascii=False))
                self.sp += end + len(PARAM_END)
                self.ss = "between"
            else:
                return out

    def _close_scan(self) -> list[Event]:
        """A streamed call that ended without a clean </function>: close its JSON so clients can still parse it."""
        out = []
        if self.scall is None or self.ss == "done":
            return out
        tail = ""
        if self.ss == "str":
            tail += '"'
        elif self.ss == "raw":
            tail += json.dumps(self.buf[self.sp:].strip("\n"))
        tail += "}"
        out.append(Event("tool_args", tail, call=self.scall))
        self.ss = "done"
        return out

    def _hold(self, text: str, tags: tuple[str, ...]) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
        best = 0
        for tag in tags:
            for n in range(1, len(tag)):
                if text.endswith(tag[:n]):
                    best = max(best, n)
        return best

    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        out: list[Event] = []
        while True:
            if self.state == "reasoning":
                i = self.buf.find(self.think_end)
                if i < 0:
                    keep = self._hold(self.buf, (self.think_end,))
                    if len(self.buf) > keep:
                        out.append(Event("reasoning", self.buf[:len(self.buf) - keep]))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                if i:
                    out.append(Event("reasoning", self.buf[:i]))
                self.buf = self.buf[i + len(self.think_end):]
                self.state, self.lead = "content", True
            elif self.state == "content":
                if self.opening:                                # the model's own opening of its reasoning
                    if self.buf.startswith(self.think_open):
                        self.buf, self.state, self.opening = self.buf[len(self.think_open):], "reasoning", False
                        continue
                    if self.think_open.startswith(self.buf):
                        return out
                    self.opening = False
                if self.lead:                                   # newlines right after </think> or a call
                    stripped = self.buf.lstrip("\n")
                    if not stripped:
                        self.buf = ""
                        return out
                    self.buf, self.lead = stripped, False
                i = self.buf.find(self.call_start)
                if i < 0:
                    # Hold a partial tag AND the newlines before it: if a tool call follows, they are dropped,
                    # so emitting them early would make streamed and whole outputs differ.
                    j = len(self.buf) - self._hold(self.buf, (self.call_start,))
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self.buf[:j]))
                        self.buf = self.buf[j:]
                    return out
                if i and self.buf[:i].strip():
                    out.append(Event("content", self.buf[:i].rstrip("\n")))
                self.buf = self.buf[i + len(self.call_start):]
                self.state = "call"
            elif self.whole:
                # the call is announced when its name is complete; its arguments follow whole when it ends
                name_of, end_of, parse = self.whole
                if self.stream_tools and self.scall is None:
                    name = name_of(self.buf)
                    if name is not None:
                        self.scall = ToolCall(name=name, arguments={})
                        out.append(Event("tool_start", call=self.scall))
                i = end_of(self.buf)
                if i < 0:
                    return out
                body = self.buf[:i]
                self.buf = self.buf[i + len(self.call_stop):]
                call = parse(body, self.schemas.get(name_of(body + self.call_stop)))
                if self.scall is not None:
                    call.id = self.scall.id
                    out.append(Event("tool_args", json.dumps(call.arguments, ensure_ascii=False), call=self.scall))
                out.append(Event("tool_call", call=call))
                self._reset_scan()
                self.state, self.lead = "content", True
            else:
                i = call_end(self.buf)
                if self.stream_tools:
                    if i >= 0:
                        whole, self.buf = self.buf, self.buf[:i]     # scan only the body
                        out += self._scan()
                        out += self._close_scan()
                        self.buf = whole
                    else:
                        out += self._scan()
                if i < 0:
                    return out
                body = self.buf[:i]
                self.buf = self.buf[i + len(CALL_END):]
                name = body.strip()[len("<function="):].split(">", 1)[0]
                call = parse_tool_call(body, self.schemas.get(name))
                if self.scall is not None:
                    call.id = self.scall.id
                out.append(Event("tool_call", call=call))
                self._reset_scan()
                self.state, self.lead = "content", True

    def finish(self) -> list[Event]:
        """End of generation: flush whatever is held (an unterminated tool call is returned as content; one that was
        already announced stays unfinished: its JSON is not closed and no "tool_call" follows it, #211)."""
        out = []
        if self.state == "call" and self.stream_tools and self.scall is not None:
            out += self._scan()                 # the output ended inside a call that was already announced
            if self.ss == "done":               # only its </tool_call> is missing: the call itself is whole
                out.append(Event("tool_call", call=self.scall))
            self.buf = ""
            self._reset_scan()
            return out
        if self.buf:
            kind = {"reasoning": "reasoning", "content": "content"}.get(self.state, "content")
            text = self.buf if self.state != "call" else self.call_start + self.buf
            out.append(Event(kind, text))
            self.buf = ""
        return out
