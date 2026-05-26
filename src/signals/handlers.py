"""Signal handlers and the dispatch registry — polymorphic for (A) scaling.

Strange-loop role
-----------------
New signal kinds are absorbed by registering new handlers, never by editing the
dispatch core. This polymorphism is the design choice that lets the (A)
personalization substrate scale to new engagement modalities (code commits,
calendar moves) in later versions without restructuring the pipeline.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Protocol

from src.signals.types import SignalEvent, SignalKind

if TYPE_CHECKING:
    from src.kb.connector import KBConnector
    from src.persistence.store import SQLiteStore

logger = logging.getLogger(__name__)


class SignalHandler(Protocol):
    """Handles a subset of signal kinds. Side effects only."""

    kinds: tuple[SignalKind, ...]

    def handle(self, event: SignalEvent, store: SQLiteStore, kb: KBConnector) -> None: ...


class SignalRegistry:
    """Persists every dispatched event, then fans out to registered handlers."""

    def __init__(self, store: SQLiteStore, kb: KBConnector) -> None:
        self.store = store
        self.kb = kb
        self._handlers: dict[SignalKind, list[SignalHandler]] = defaultdict(list)

    def register(self, handler: SignalHandler) -> None:
        for kind in handler.kinds:
            self._handlers[kind].append(handler)

    def dispatch(self, event: SignalEvent) -> None:
        self.store.record_engagement_signal(event)
        for handler in self._handlers.get(event.kind, []):
            handler.handle(event, self.store, self.kb)

    def dispatch_batch(self, events: list[SignalEvent]) -> None:
        for event in events:
            self.dispatch(event)


class DefaultLoggingHandler:
    """Logs every signal kind to stderr at INFO level."""

    kinds = tuple(SignalKind)

    def handle(self, event: SignalEvent, store: SQLiteStore, kb: KBConnector) -> None:
        logger.info(
            "signal kind=%s paper=%s payload=%s",
            event.kind.value,
            event.paper_id,
            event.payload,
        )


class PaperTagHandler:
    """Tags the KB paper with the engagement kind for read/skim/discard/flag."""

    kinds = (SignalKind.READ, SignalKind.SKIM, SignalKind.DISCARD, SignalKind.FLAG)

    def handle(self, event: SignalEvent, store: SQLiteStore, kb: KBConnector) -> None:
        kb.tag_paper(
            event.paper_id,
            event.kind.value,
            provenance={
                "signal_event_id": str(event.event_id),
                "ts": event.timestamp.isoformat(),
            },
        )
