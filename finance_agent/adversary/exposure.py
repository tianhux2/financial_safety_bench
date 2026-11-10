"""Exposure log: a per-run, agent-invisible record of what each tool call did.

The log answers questions the final answer cannot: did the agent actually
encounter injected/adversarial content, at which step, and did it later consume
poisoned stored documents. It is written into the trajectory under ``env_log``
and never shown to the agent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolEvent:
    """One tool call as seen by the environment."""

    index: int  # 1-based ordinal of this tool call within the run
    tool: str
    injected: bool = False  # did the adversary alter or supply this output?
    primitives: list[str] = field(default_factory=list)  # which primitives fired
    case_ids: list[str] = field(default_factory=list)  # attack specs that touched this call
    note: str = ""  # free-form (e.g. consumed poisoned storage keys)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExposureLog:
    events: list[ToolEvent] = field(default_factory=list)
    # What the retrieval sub-model answered on calls whose input really contained the attack content.
    poison_reads: list[str] = field(default_factory=list)

    def record(self, event: ToolEvent) -> None:
        self.events.append(event)

    def reset(self) -> None:
        self.events.clear()
        self.poison_reads.clear()

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]

    @property
    def exposed(self) -> bool:
        """True if the agent received at least one injected tool output."""
        return any(e.injected for e in self.events)

    def first_exposure_index(self) -> int | None:
        for e in self.events:
            if e.injected:
                return e.index
        return None

    def consumed_poison(self) -> bool:
        """True if a poisoned stored document was later read via retrieve_information."""
        return any(e.note.startswith("consumed_poisoned_keys") for e in self.events)
