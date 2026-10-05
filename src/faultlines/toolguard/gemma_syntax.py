"""Parse and render Gemma 4's native tool-call syntax.

Gemma 4 emits tool calls as special-token-delimited text, e.g.

    <|tool_call>call:read_file{path:<|"|>src/app.py<|"|>,start_line:10}<tool_call|>

Keys are bare identifiers, strings are wrapped in the <|"|> token, numbers and
booleans are bare, and objects / arrays use {} and [].
Source: https://ai.google.dev/gemma/docs/capabilities/function-calling
(verify against the chat template of the exact checkpoint you run).

This module never guesses content. It reports *what* is wrong so that
`repair.py` can decide what is safe to fix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CALL_OPEN = "<|tool_call>"
CALL_CLOSE = "<tool_call|>"
STR = '<|"|>'

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")


class GemmaSyntaxError(ValueError):
    """Raised when a call body cannot be parsed. `code` is a stable reason id."""

    def __init__(self, code: str, detail: str, pos: int = -1):
        super().__init__(f"{code}: {detail} (at {pos})")
        self.code = code
        self.detail = detail
        self.pos = pos


@dataclass
class RawCall:
    """One `<|tool_call>...` region found in model output."""

    body: str               # text between the open tag and the close tag (or end of text)
    start: int              # offset of the open tag in the original text
    closed: bool            # False when the close tag is missing (usually MAX_TOKENS truncation)


@dataclass
class ParsedBody:
    name: str
    args: dict[str, Any]
    warnings: list[str] = field(default_factory=list)  # lenient-parse notes, e.g. BARE_STRING


def extract_calls(text: str) -> list[RawCall]:
    """Find every tool-call region in `text`, including an unclosed trailing one."""
    calls: list[RawCall] = []
    i = 0
    while True:
        start = text.find(CALL_OPEN, i)
        if start < 0:
            break
        body_start = start + len(CALL_OPEN)
        # A string literal may legally contain the close tag's characters only if the
        # model emitted it inside <|"|>...<|"|>; scan string-aware.
        end = _find_close(text, body_start)
        if end < 0:
            calls.append(RawCall(body=text[body_start:], start=start, closed=False))
            break
        calls.append(RawCall(body=text[body_start:end], start=start, closed=True))
        i = end + len(CALL_CLOSE)
    return calls


def _find_close(text: str, pos: int) -> int:
    """String-aware search for the close tag.

    Falls back to a plain search when string delimiters are unbalanced, so a call
    that is closed but has a broken string is reported as MALFORMED, not TRUNCATED.
    """
    end = _find_close_string_aware(text, pos)
    if end >= 0:
        return end
    plain = text.find(CALL_CLOSE, pos)
    next_open = text.find(CALL_OPEN, pos)
    if plain >= 0 and (next_open < 0 or plain < next_open):
        return plain
    return -1


def _find_close_string_aware(text: str, pos: int) -> int:
    in_str = False
    i = pos
    while i < len(text):
        if text.startswith(STR, i):
            in_str = not in_str
            i += len(STR)
            continue
        if not in_str and text.startswith(CALL_CLOSE, i):
            return i
        if not in_str and text.startswith(CALL_OPEN, i):
            # A new call opened before this one closed: treat this one as unclosed.
            return -1
        i += 1
    return -1


def parse_body(body: str, allow_trailing: bool = False) -> ParsedBody:
    """Parse `call:NAME{...}` into a name and an argument dict (strict Gemma syntax).

    With `allow_trailing`, text after the closing brace is ignored and noted as a
    TRAILING_TEXT warning (useful when the close tag is missing and the model kept
    talking after a complete call).
    """
    s = body.strip()
    if not s.startswith("call:"):
        raise GemmaSyntaxError("MISSING_CALL_PREFIX", "body does not start with 'call:'", 0)
    s = s[len("call:"):]
    m = _IDENT.match(s)
    if not m:
        raise GemmaSyntaxError("BAD_TOOL_NAME", "no tool name after 'call:'", len("call:"))
    name = m.group(0)
    rest = s[m.end():].strip()
    if rest == "":
        return ParsedBody(name=name, args={})  # zero-arg call without braces
    p = _Parser(rest)
    args = p.parse_object()
    p.skip_ws()
    if p.i != len(p.t):
        if not allow_trailing:
            raise GemmaSyntaxError("TRAILING_TEXT", repr(p.t[p.i:p.i + 40]), p.i)
        p.warnings.append("TRAILING_TEXT")
    if not isinstance(args, dict):
        raise GemmaSyntaxError("ARGS_NOT_OBJECT", "arguments must be an object", 0)
    return ParsedBody(name=name, args=args, warnings=p.warnings)


class _Parser:
    def __init__(self, text: str):
        self.t = text
        self.i = 0
        self.warnings: list[str] = []

    def skip_ws(self) -> None:
        while self.i < len(self.t) and self.t[self.i] in " \t\r\n":
            self.i += 1

    def peek(self) -> str:
        return self.t[self.i] if self.i < len(self.t) else ""

    def expect(self, ch: str) -> None:
        self.skip_ws()
        if self.peek() != ch:
            code = "UNEXPECTED_END" if self.i >= len(self.t) else "UNEXPECTED_CHAR"
            raise GemmaSyntaxError(code, f"expected {ch!r}, got {self.peek()!r}", self.i)
        self.i += 1

    def parse_value(self) -> Any:
        self.skip_ws()
        if self.t.startswith(STR, self.i):
            return self.parse_string()
        c = self.peek()
        if c == "{":
            return self.parse_object()
        if c == "[":
            return self.parse_array()
        if c == "":
            raise GemmaSyntaxError("UNEXPECTED_END", "value expected", self.i)
        return self.parse_bare()

    def parse_string(self) -> str:
        self.i += len(STR)
        j = self.t.find(STR, self.i)
        if j < 0:
            raise GemmaSyntaxError("UNTERMINATED_STRING", "missing closing <|\"|>", self.i)
        val = self.t[self.i:j]
        self.i = j + len(STR)
        return val

    def parse_bare(self) -> Any:
        j = self.i
        while j < len(self.t) and self.t[j] not in ",}]":
            if self.t.startswith(STR, j):
                break
            j += 1
        tok = self.t[self.i:j].strip()
        self.i = j
        if tok == "":
            raise GemmaSyntaxError("EMPTY_VALUE", "empty value", self.i)
        low = tok.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low in ("null", "none"):
            return None
        try:
            return int(tok)
        except ValueError:
            pass
        try:
            return float(tok)
        except ValueError:
            pass
        # Lenient: an unquoted string. Record it; repair.py treats it as a warning.
        self.warnings.append(f"BARE_STRING:{tok[:40]}")
        return tok

    def parse_key(self) -> str:
        self.skip_ws()
        if self.t.startswith(STR, self.i):
            return self.parse_string()
        c = self.peek()
        if c in "\"'":
            j = self.t.find(c, self.i + 1)
            if j < 0:
                raise GemmaSyntaxError("UNTERMINATED_KEY", "key quote not closed", self.i)
            key = self.t[self.i + 1:j]
            self.i = j + 1
            self.warnings.append("QUOTED_KEY")
            return key
        m = _IDENT.match(self.t, self.i)
        if not m:
            code = "UNEXPECTED_END" if self.i >= len(self.t) else "BAD_KEY"
            raise GemmaSyntaxError(code, f"bad key near {self.t[self.i:self.i + 20]!r}", self.i)
        self.i = m.end()
        return m.group(0)

    def parse_object(self) -> dict[str, Any]:
        self.expect("{")
        out: dict[str, Any] = {}
        self.skip_ws()
        if self.peek() == "}":
            self.i += 1
            return out
        while True:
            key = self.parse_key()
            self.expect(":")
            if key in out:
                self.warnings.append(f"DUPLICATE_KEY:{key}")
            out[key] = self.parse_value()
            self.skip_ws()
            c = self.peek()
            if c == ",":
                self.i += 1
                self.skip_ws()
                if self.peek() == "}":  # trailing comma
                    self.warnings.append("TRAILING_COMMA")
                    self.i += 1
                    return out
                continue
            if c == "}":
                self.i += 1
                return out
            code = "UNEXPECTED_END" if c == "" else "UNEXPECTED_CHAR"
            raise GemmaSyntaxError(code, f"expected ',' or '}}', got {c!r}", self.i)

    def parse_array(self) -> list[Any]:
        self.expect("[")
        out: list[Any] = []
        self.skip_ws()
        if self.peek() == "]":
            self.i += 1
            return out
        while True:
            out.append(self.parse_value())
            self.skip_ws()
            c = self.peek()
            if c == ",":
                self.i += 1
                continue
            if c == "]":
                self.i += 1
                return out
            code = "UNEXPECTED_END" if c == "" else "UNEXPECTED_CHAR"
            raise GemmaSyntaxError(code, f"expected ',' or ']', got {c!r}", self.i)


def render_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return f"{STR}{v}{STR}"
    if isinstance(v, list):
        return "[" + ",".join(render_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{render_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"cannot render {type(v)}")


def render_call(name: str, args: dict[str, Any]) -> str:
    """Render a call in canonical Gemma 4 syntax (inverse of extract_calls + parse_body)."""
    return f"{CALL_OPEN}call:{name}{render_value(args)}{CALL_CLOSE}"
