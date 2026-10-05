"""3-tier replacement used by edit_file, as documented for adk-eval-core's apply_replacement.

    exact     character-for-character match (after \r\n -> \n)
    flexible  line-by-line match ignoring leading/trailing whitespace; new text is
              re-indented to the matched block's indentation
    regex     tokens split around code delimiters ( ) : [ ] { } > = < joined by \s*
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class EditError(Exception):
    pass


@dataclass
class EditResult:
    content: str
    occurrences: int
    strategy: str


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _exact(content: str, old: str, new: str, allow_multiple: bool) -> EditResult | None:
    n = content.count(old)
    if n == 0:
        return None
    if n > 1 and not allow_multiple:
        raise EditError(f"old_string matches {n} locations; make it unique or set allow_multiple=true")
    return EditResult(content.replace(old, new) if allow_multiple else content.replace(old, new, 1), n, "exact")


def _flexible(content: str, old: str, new: str, allow_multiple: bool) -> EditResult | None:
    lines = content.split("\n")
    old_lines = old.strip("\n").split("\n")
    target = [l.strip() for l in old_lines]
    if not any(target):
        return None
    k = len(target)
    hits = [i for i in range(len(lines) - k + 1) if [l.strip() for l in lines[i:i + k]] == target]
    if not hits:
        return None
    if len(hits) > 1 and not allow_multiple:
        raise EditError(f"old_string matches {len(hits)} locations (whitespace-insensitive); make it unique")
    new_lines = new.strip("\n").split("\n")
    old_base = _indent(next(l for l in old_lines if l.strip()))
    for i in reversed(hits):
        base = _indent(next(l for l in lines[i:i + k] if l.strip()))
        reindented = []
        for l in new_lines:
            if not l.strip():
                reindented.append("")
            elif l.startswith(old_base):
                reindented.append(base + l[len(old_base):])
            else:
                reindented.append(base + l.lstrip())
        lines[i:i + k] = reindented
    return EditResult("\n".join(lines), len(hits), "flexible")


_DELIMS = r"([()\[\]{}:<>=,])"


def _regex(content: str, old: str, new: str, allow_multiple: bool) -> EditResult | None:
    tokens = [t for t in re.split(r"\s+|" + _DELIMS, old) if t]
    if not tokens:
        return None
    pattern = r"\s*".join(re.escape(t) for t in tokens)
    matches = list(re.finditer(pattern, content))
    if not matches:
        return None
    if len(matches) > 1 and not allow_multiple:
        raise EditError(f"old_string matches {len(matches)} locations (token match); make it unique")
    out = content
    for m in reversed(matches if allow_multiple else matches[:1]):
        out = out[: m.start()] + new + out[m.end():]
    return EditResult(out, len(matches), "regex")


def apply_replacement(content: str, old: str, new: str, allow_multiple: bool = False) -> EditResult:
    if old == "":
        raise EditError("old_string must not be empty")
    content = content.replace("\r\n", "\n")
    old = old.replace("\r\n", "\n")
    new = new.replace("\r\n", "\n")
    for strategy in (_exact, _flexible, _regex):
        res = strategy(content, old, new, allow_multiple)
        if res is not None:
            return res
    # Over-escaped arguments ("a\\nb" for a two-line string): decode and retry, exact + unique only.
    from ..toolguard.repair import is_over_escaped, unescape_string
    if is_over_escaped(old):
        old_u = unescape_string(old)
        new_u = unescape_string(new) if is_over_escaped(new) else new
        if content.count(old_u) == 1:
            res = _exact(content, old_u, new_u, False)
            if res is not None:
                res.strategy = "unescape"
                return res
    raise EditError("old_string not found in file")
