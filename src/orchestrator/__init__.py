"""Orchestrator skeleton — registry, workspace, dispatcher, loop.

The orchestrator is the strange-loop research subject (CLAUDE.md principle 3).
See ARCHITECTURE.md "Architecture overview".
"""

from src.orchestrator.dispatcher import Dispatcher
from src.orchestrator.loop import Orchestrator, Session
from src.orchestrator.registry import SchemaRegistry
from src.orchestrator.workspace import Workspace

__all__ = [
    "Dispatcher",
    "Orchestrator",
    "SchemaRegistry",
    "Session",
    "Workspace",
]
