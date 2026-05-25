"""Tests for the engagement signal pipeline.

Serves claim (A): engagement signals are the personalization substrate. The
pipeline is polymorphic — handlers, not the dispatch core, carry signal-specific
logic. Hermetic: in-memory store + FakeKBConnector.
"""

from datetime import datetime

import torch

from src.kb import FakeKBConnector, Paper
from src.persistence import SQLiteStore
from src.signals import (
    DefaultLoggingHandler,
    PaperTagHandler,
    SignalEvent,
    SignalKind,
    SignalRegistry,
)


def _event(kind, paper_id="p1"):
    return SignalEvent(paper_id=paper_id, kind=kind, timestamp=datetime.now())


class _RecordingHandler:
    kinds = (SignalKind.READ,)

    def __init__(self):
        self.seen = []

    def handle(self, event, store, kb):
        self.seen.append(event)


def test_signal_kind_enum_values():
    """Serves (A): signal kinds carry stable string values."""
    assert SignalKind.READ.value == "read"
    assert SignalKind.PENDING.value == "pending"
    assert SignalKind.TIME_ON_FILE.value == "time_on_file"


def test_signal_event_dataclass():
    """Serves (A): event carries id, payload, and source defaults."""
    event = _event(SignalKind.READ)
    assert event.paper_id == "p1"
    assert event.source == "unknown"
    assert event.payload == {}
    assert event.event_id is not None


def test_signal_registry_dispatches_to_correct_handler():
    """Serves (A): a handler only fires for its declared kinds."""
    store = SQLiteStore(":memory:")
    registry = SignalRegistry(store, FakeKBConnector())
    handler = _RecordingHandler()
    registry.register(handler)
    registry.dispatch(_event(SignalKind.READ))
    registry.dispatch(_event(SignalKind.SKIM))
    assert len(handler.seen) == 1
    store.close()


def test_signal_registry_persists_event_to_store():
    """Serves (A): every dispatched event is persisted."""
    store = SQLiteStore(":memory:")
    registry = SignalRegistry(store, FakeKBConnector())
    registry.dispatch(_event(SignalKind.FLAG, "paperX"))
    rows = store.engagement_signals_for_paper("paperX")
    assert len(rows) == 1
    assert rows[0]["kind"] == "flag"
    store.close()


def test_paper_tag_handler_tags_kb_on_read():
    """Serves (A): a read signal tags the KB paper."""
    store = SQLiteStore(":memory:")
    kb = FakeKBConnector()
    kb.add_paper(Paper("paperY", "T", [], "abs"), torch.ones(2))
    registry = SignalRegistry(store, kb)
    registry.register(PaperTagHandler())
    registry.dispatch(_event(SignalKind.READ, "paperY"))
    assert any(ref.paper_id == "paperY" for ref in kb.papers_with_tag("read"))
    store.close()


def test_default_logging_handler_handles_all_kinds():
    """Serves (A): the default handler accepts every signal kind."""
    handler = DefaultLoggingHandler()
    assert set(handler.kinds) == set(SignalKind)
    store = SQLiteStore(":memory:")
    kb = FakeKBConnector()
    for kind in SignalKind:
        handler.handle(_event(kind), store, kb)
    store.close()
