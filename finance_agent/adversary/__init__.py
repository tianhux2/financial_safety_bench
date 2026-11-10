"""Adversarial information environment and behavior checks for finance-agent safety evaluation.

- ToolEnvironment / InjectionRule: inject adversarial content into tool outputs and log exposure.
- ExposureLog: agent-invisible record of what each tool call did.
- scan_trajectory: Tier-1 passive behavior checks over a finished trajectory.
"""

from .checks import scan_trajectory
from .environment import ToolEnvironment
from .exposure import ExposureLog, ToolEvent
from .injection import InjectionRule

__all__ = [
    "ToolEnvironment",
    "InjectionRule",
    "ExposureLog",
    "ToolEvent",
    "scan_trajectory",
]
