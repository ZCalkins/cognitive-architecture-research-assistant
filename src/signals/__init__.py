"""Engagement signal pipeline — polymorphic substrate for claim (A).

See ARCHITECTURE.md "Phase 1 deliverable 6" and CLAUDE.md claim (A).
"""

from src.signals.handlers import (
    DefaultLoggingHandler,
    PaperTagHandler,
    SignalHandler,
    SignalRegistry,
)
from src.signals.markdown import (
    TriageMarkdownParser,
    TriageMarkdownWatcher,
    TriageMarkdownWriter,
)
from src.signals.types import SignalEvent, SignalKind

__all__ = [
    "DefaultLoggingHandler",
    "PaperTagHandler",
    "SignalEvent",
    "SignalHandler",
    "SignalKind",
    "SignalRegistry",
    "TriageMarkdownParser",
    "TriageMarkdownWatcher",
    "TriageMarkdownWriter",
]
