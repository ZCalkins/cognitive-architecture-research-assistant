"""Tests for the triage markdown writer, parser, and watcher.

Serves (A): the markdown surface is how engagement signals are collected. The
writer/parser are dumb infrastructure (the (C) firewall: no interpretation
beyond explicit fields). The watcher's event handler is invoked directly rather
than spinning a real Observer thread.
"""

from datetime import datetime

from src.kb import FakeKBConnector, Paper
from src.persistence import SQLiteStore
from src.signals import SignalRegistry
from src.signals.markdown import (
    TriageMarkdownParser,
    TriageMarkdownWatcher,
    TriageMarkdownWriter,
)


def _paper(paper_id, title="Title", abstract="Abstract text."):
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=["Alice", "Bob"],
        abstract=abstract,
        arxiv_id=paper_id,
        categories=["cs.AI"],
        submitted_date=datetime(2024, 1, 1),
    )


def test_writer_produces_parseable_output(tmp_path):
    """Serves (A): writer output round-trips through the parser."""
    writer = TriageMarkdownWriter(tmp_path)
    scores = {"a": {"novelty_vs_kb": 1.0}, "b": {"novelty_vs_kb": 2.0}}
    path = writer.write_daily([_paper("a"), _paper("b")], scores, date=datetime(2024, 1, 1))
    parsed = TriageMarkdownParser().parse_file(path)
    assert set(parsed.keys()) == {"a", "b"}
    assert parsed["a"]["status"] == "pending"
    assert "novelty_vs_kb" in parsed["a"]["specialists"]


def test_writer_preserves_user_set_fields_on_rewrite(tmp_path):
    """Serves (A): regeneration preserves user-set status and notes."""
    writer = TriageMarkdownWriter(tmp_path)
    papers = [_paper("a")]
    scores = {"a": {"novelty_vs_kb": 1.0}}
    path = writer.write_daily(papers, scores, date=datetime(2024, 1, 1))
    edited = path.read_text().replace("- status: pending", "- status: read")
    edited = edited.replace("**Notes:**\n", "**Notes:**\n\nGreat paper.\n")
    path.write_text(edited)
    writer.write_daily(papers, scores, date=datetime(2024, 1, 1))
    parsed = TriageMarkdownParser().parse_file(path)
    assert parsed["a"]["status"] == "read"
    assert "Great paper." in parsed["a"]["notes"]


def test_parser_extracts_paper_blocks(tmp_path):
    """Serves (A): the parser extracts per-paper prelude fields."""
    writer = TriageMarkdownWriter(tmp_path)
    path = writer.write_daily([_paper("a"), _paper("b")], {"a": {}, "b": {}}, date=datetime(2024, 1, 1))
    parsed = TriageMarkdownParser().parse_file(path)
    assert parsed["a"]["paper_id"] == "a"
    assert parsed["a"]["authors"] == "Alice; Bob"


def test_parser_extracts_notes_block():
    """Serves (A): the notes block is extracted verbatim (stripped)."""
    text = (
        "## [P-001] T\n\n"
        "- paper_id: a\n"
        "- status: read\n"
        "- specialists:\n"
        "  - novelty_vs_kb: 1.0\n\n"
        "**Abstract.** Some abstract.\n\n"
        "**Notes:**\n\n"
        "My note here.\n"
    )
    parsed = TriageMarkdownParser().parse_text(text)
    assert parsed["a"]["notes"] == "My note here."


def test_parser_tolerates_indentation_variation():
    """Serves (A): the parser classifies by key, tolerating indentation."""
    text = (
        "## [P-001] T\n\n"
        "  - paper_id: a\n"
        "    - status: skim\n"
        "- specialists:\n"
        "      - novelty_vs_kb: 0.5\n\n"
        "**Notes:**\n"
    )
    parsed = TriageMarkdownParser().parse_text(text)
    assert parsed["a"]["paper_id"] == "a"
    assert parsed["a"]["status"] == "skim"
    assert parsed["a"]["specialists"]["novelty_vs_kb"] == 0.5


def test_watcher_emits_signal_on_status_change(tmp_path):
    """Serves (A): a status edit becomes an engagement signal."""
    store = SQLiteStore(":memory:")
    registry = SignalRegistry(store, FakeKBConnector())
    parser = TriageMarkdownParser()
    writer = TriageMarkdownWriter(tmp_path)
    path = writer.write_daily([_paper("a")], {"a": {}}, date=datetime(2024, 1, 1))

    watcher = TriageMarkdownWatcher(tmp_path, registry, parser)
    watcher._snapshot[path] = parser.parse_file(path)
    path.write_text(path.read_text().replace("- status: pending", "- status: read"))
    watcher._on_modified(path)

    reads = [row for row in store.recent_engagement_signals() if row["kind"] == "read"]
    assert len(reads) == 1
    assert reads[0]["paper_id"] == "a"
    store.close()
