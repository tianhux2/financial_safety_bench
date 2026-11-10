"""Declarative injection rules for robustness testing.

A rule describes one adversarial modification applied to a matching tool output,
for red-team / robustness evaluation of the finance agent. Two primitives:

- ``replace``: substitute strings (e.g. a true figure -> a tampered figure) inside
  tool content the agent reads. Tests whether wrong data in a source changes the
  answer. NOTE: to beat the grader's rounding tolerance the tampered value must
  differ from the true value at the precision the agent will report (two
  significant figures is a safe rule of thumb).
- ``append``: add a block of adversarial text, or an extra search result, to what
  the agent reads. Used for prompt-injection robustness and honeypot presence. For
  web_search the payload takes one ``result`` or a list of ``results`` (several
  fabricated sources corroborating each other).
- ``serve``: answer parse_html_page for listed URLs with synthetic page text and make
  no network request, so that opening a fabricated search result does not end in a
  dead link, which would itself give the fabrication away. Handled by ToolEnvironment.

All injected content is synthetic and exists only inside the offline evaluation
run; it is never published anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InjectionRule:
    case_id: str
    tool: str  # "web_search" | "parse_html_page" (the two dominant channels)
    primitive: str  # "replace" | "append"
    keywords: list[str] = field(default_factory=list)  # required terms: ALL must appear (AND)
    aliases: list[str] = field(default_factory=list)  # entity names/tickers: ANY must appear (OR)
    payload: dict[str, Any] = field(default_factory=dict)
    # Corroboration dose: poison at most this many matching sources (0 = unlimited). Set to
    # k<N to leave N-k sources carrying the TRUE value, so a model that cross-checks can
    # detect the conflict. Enforced by ToolEnvironment across a run.
    max_applications: int = 0

    def matches(self, tool_name: str, args: dict[str, Any]) -> bool:
        if tool_name != self.tool:
            return False
        blob = json.dumps(args, default=str).lower()
        if not all(k.lower() in blob for k in self.keywords):
            return False
        # Aliases are OR: the entity may be referred to by name or ticker. Requiring
        # every alias (AND) would never match a query that uses only one of them.
        if self.aliases and not any(a.lower() in blob for a in self.aliases):
            return False
        return True

    # -- text-level transform, shared by search content and stored page content --
    def transform_text(self, text: str) -> tuple[str, bool]:
        if self.primitive == "replace":
            new, changed = text, False
            for find in self.payload.get("find", []):
                if find and find in new:
                    new = new.replace(find, self.payload.get("replace", ""))
                    changed = True
            return new, changed
        if self.primitive == "append":
            block = self.payload.get("block", "")
            if block:
                return f"{text}\n{block}", True
        return text, False

    def apply_to_search(self, output: str) -> tuple[str, bool]:
        """Modify a web_search JSON-array output string. Returns (output, changed)."""
        try:
            results = json.loads(output)
        except Exception:
            return output, False
        if not isinstance(results, list):
            return output, False

        changed = False
        if self.primitive == "replace":
            for r in results:
                if not isinstance(r, dict):
                    continue
                for field_name in ("content", "raw_content", "title"):
                    value = r.get(field_name)
                    if isinstance(value, str):
                        new_value, ch = self.transform_text(value)
                        if ch:
                            r[field_name] = new_value
                            changed = True
        elif self.primitive == "append":
            extras = self.payload.get("results") or [self.payload.get("result")]
            rank = self.payload.get("rank", 0)
            for offset, extra in enumerate(e for e in extras if isinstance(e, dict)):
                results.insert(rank + offset, extra)
                changed = True

        return (json.dumps(results, default=str), True) if changed else (output, False)

    def served_page(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Synthetic page text for a ``serve`` rule when this call opens one of its URLs."""
        if self.primitive != "serve" or tool_name != self.tool:
            return None
        return self.payload.get("pages", {}).get(str(args.get("url", "")))

    def apply_to_state(self, args: dict[str, Any], state: dict[str, Any]) -> bool:
        """Modify the stored page content for parse_html_page. Returns changed."""
        key = args.get("key")
        if not key or key not in state:
            return False
        new_value, changed = self.transform_text(str(state[key]))
        if changed:
            state[key] = new_value
        return changed
