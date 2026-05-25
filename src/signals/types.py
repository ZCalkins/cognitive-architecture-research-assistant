"""Engagement signal types — the substrate for the (A) personalization claim.

Strange-loop role
-----------------
Signal events are the substrate for claim (A): the system's structural
personalization is learned from the user's engagement. The pipeline is
polymorphic from v0 — new signal kinds are added by extending ``SignalKind``
and registering new ``SignalHandler``s, NOT by restructuring the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
from typing import Any
from uuid import UUID, uuid4


class SignalKind(StrEnum):
    """Kinds of engagement signal. Extend here to add a new modality."""

    PENDING = auto()
    READ = auto()
    SKIM = auto()
    DISCARD = auto()
    FLAG = auto()
    NOTE = auto()
    OPENED = auto()
    TIME_ON_FILE = auto()


@dataclass
class SignalEvent:
    """A single engagement signal.

    Note: required fields precede default fields (Python dataclass ordering),
    so ``event_id`` carries a default factory rather than leading the field
    list. Construct with keyword arguments.
    """

    paper_id: str
    kind: SignalKind
    timestamp: datetime
    event_id: UUID = field(default_factory=uuid4)
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
