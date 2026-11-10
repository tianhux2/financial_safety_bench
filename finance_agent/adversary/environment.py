"""ToolEnvironment: wraps each tool call to inject adversarial content and log exposure.

It sits between the agent's tool executor and the raw tool. For every call it runs
the real tool (overlay mode), applies any matching injection rules to what the agent
will read, and records a ToolEvent. For parse_html_page the poison lands in the
agent's data store, so a later retrieve_information that references a poisoned key is
recorded as consumption.

Default agent behavior is unchanged unless an environment is supplied.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable

from .exposure import ExposureLog, ToolEvent
from .injection import InjectionRule

_KEY_RE = re.compile(r"{{([^{}]+)}}")


class ToolEnvironment:
    def __init__(self, rules: list[InjectionRule] | None = None, log: ExposureLog | None = None) -> None:
        self.rules = rules or []
        self.log = log or ExposureLog()
        self._n = 0
        self._poisoned_keys: set[str] = set()
        self._applied: dict[str, int] = {}  # case_id -> how many sources it has poisoned

    def reset(self) -> None:
        self._n = 0
        self._poisoned_keys.clear()
        self._applied.clear()
        self.log.reset()

    def _attack_markers(self) -> list[str]:
        markers = []
        for rule in self.rules:
            if rule.primitive == "replace":
                markers.append(str(rule.payload.get("replace", "")))
            elif rule.primitive == "append" and rule.payload.get("block"):
                markers.append(str(rule.payload["block"]))
            elif rule.primitive == "serve":
                markers.extend(str(page) for page in rule.payload.get("pages", {}).values())
        return [m for m in markers if m]

    def _attack_in_view(self, keys: list[str], args: dict[str, Any], state: dict[str, Any]) -> bool:
        """Did this retrieval really pass the attack content to the sub-model? A character range can skip it."""
        ranges = {r["key"]: (r["start"], r["end"]) for r in args.get("input_character_ranges") or []
                  if isinstance(r, dict) and {"key", "start", "end"} <= set(r)}
        markers = self._attack_markers()
        for key in keys:
            text = str(state.get(key, ""))
            start, end = ranges.get(key, (0, len(text)))
            if any(m in text[start:end] for m in markers):
                return True
        return False

    async def _serve(self, tool_name: str, args: dict[str, Any], state: dict[str, Any]) -> str | None:
        """Answer a call to a fabricated URL from the case itself, without touching the network."""
        for rule in self.rules:
            page = rule.served_page(tool_name, args)
            if page is None:
                continue
            from ..tools import ParseHtmlPage  # imported here: tools pulls in the full network stack

            key = str(args.get("key"))
            output = await ParseHtmlPage()._save_tool_output(page, key, state)
            self._poisoned_keys.add(key)
            self.log.record(ToolEvent(index=self._n, tool=tool_name, injected=True,
                                      primitives=["serve"], case_ids=[rule.case_id]))
            return output
        return None

    async def intercept(
        self,
        tool_name: str,
        args: dict[str, Any],
        state: dict[str, Any],
        logger: logging.Logger,
        run_raw: Callable[[], Awaitable[Any]],
    ) -> str:
        self._n += 1
        served = await self._serve(tool_name, args, state)
        if served is not None:
            return served

        output_obj = await run_raw()
        output = output_obj.output if hasattr(output_obj, "output") else str(output_obj)

        fired: list[str] = []
        cases: list[str] = []
        for rule in self.rules:
            if rule.primitive == "serve" or not rule.matches(tool_name, args):
                continue
            # Corroboration dose: once k sources are poisoned, leave the rest true.
            if rule.max_applications and self._applied.get(rule.case_id, 0) >= rule.max_applications:
                continue
            if tool_name == "web_search":
                output, changed = rule.apply_to_search(output)
            elif tool_name == "parse_html_page":
                changed = rule.apply_to_state(args, state)
                if changed:
                    self._poisoned_keys.add(str(args.get("key")))
            else:
                changed = False
            if changed:
                self._applied[rule.case_id] = self._applied.get(rule.case_id, 0) + 1
                fired.append(rule.primitive)
                cases.append(rule.case_id)

        note = ""
        if tool_name == "retrieve_information" and self._poisoned_keys:
            referenced = _KEY_RE.findall(str(args.get("prompt", "")))
            consumed = [k for k in referenced if k in self._poisoned_keys]
            if consumed:
                note = f"consumed_poisoned_keys:{consumed}"
                if self._attack_in_view(consumed, args, state):
                    self.log.poison_reads.append(str(output))

        self.log.record(
            ToolEvent(
                index=self._n,
                tool=tool_name,
                injected=bool(fired),
                primitives=fired,
                case_ids=cases,
                note=note,
            )
        )
        return output
