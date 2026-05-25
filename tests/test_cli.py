"""Tests for the strange-loop CLI.

Hermetic: heavy services are substituted via the ``make_*`` factories. Serves
(A) — the CLI is the operator surface through which engagement flows — and (D)
falsifiability (the end-to-end build runs with fakes, no network/Ollama).
"""

from datetime import datetime
from unittest.mock import MagicMock

import torch
import yaml
from click.testing import CliRunner

import src.cli as cli_module
from src.cli import cli
from src.kb import FakeKBConnector, Paper
from src.persistence import SQLiteStore
from src.signals import SignalEvent, SignalKind
from src.signals.markdown import TriageMarkdownWriter


def _write_config(tmp_path):
    config = {
        "embedding": {"model_name": "fake", "dimension": 8, "cache_dir": str(tmp_path / "emb")},
        "kb": {
            "papers_db_path": str(tmp_path / "papers.db"),
            "embeddings_index_path": str(tmp_path / "p.faiss"),
            "embeddings_metadata_path": str(tmp_path / "m.jsonl"),
        },
        "ingestion": {"arxiv_categories": ["cs.AI"], "arxiv_max_results_per_run": 5},
        "triage": {
            "output_directory": str(tmp_path / "triage"),
            "filename_format": "%Y-%m-%d.md",
            "papers_per_day_target": 30,
        },
        "persistence": {"db_path": str(tmp_path / "strange_loop.db")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_cli_version():
    """Serves (D): the CLI reports its version."""
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.0.1" in result.output


def test_cli_config_show_renders_yaml(tmp_path):
    """Serves (D): config-show renders the loaded config."""
    result = CliRunner().invoke(
        cli, ["--config-path", str(_write_config(tmp_path)), "config-show"]
    )
    assert result.exit_code == 0
    assert "embedding" in result.output


def test_cli_kb_stats_renders_counts(tmp_path, monkeypatch):
    """Serves (A): kb stats reports the corpus size."""
    fake_kb = FakeKBConnector()
    fake_kb.add_paper(Paper("a", "A", [], "x"), torch.ones(8))
    fake_kb.add_paper(Paper("b", "B", [], "x"), torch.ones(8))
    monkeypatch.setattr(cli_module, "make_embedding_service", lambda config: MagicMock())
    monkeypatch.setattr(cli_module, "make_kb", lambda config, emb: fake_kb)
    result = CliRunner().invoke(
        cli, ["--config-path", str(_write_config(tmp_path)), "kb", "stats"]
    )
    assert result.exit_code == 0
    assert "papers: 2" in result.output


def test_cli_triage_status_updates_markdown_and_emits_signal(tmp_path, monkeypatch):
    """Serves (A): a status command updates the file and emits a signal."""
    config_path = _write_config(tmp_path)
    writer = TriageMarkdownWriter(tmp_path / "triage")
    path = writer.write_daily([Paper("a", "A", [], "x")], {"a": {}}, date=datetime.now())
    store = SQLiteStore(":memory:")
    monkeypatch.setattr(cli_module, "make_store", lambda config: store)
    monkeypatch.setattr(cli_module, "make_embedding_service", lambda config: MagicMock())
    monkeypatch.setattr(cli_module, "make_kb", lambda config, emb: FakeKBConnector())
    result = CliRunner().invoke(
        cli, ["--config-path", str(config_path), "triage", "status", "a", "read"]
    )
    assert result.exit_code == 0, result.output
    assert "- status: read" in path.read_text()
    reads = [row for row in store.recent_engagement_signals() if row["kind"] == "read"]
    assert len(reads) == 1


def test_cli_signals_recent_lists_events(tmp_path, monkeypatch):
    """Serves (A): signals recent lists persisted events."""
    store = SQLiteStore(":memory:")
    store.record_engagement_signal(
        SignalEvent(paper_id="px", kind=SignalKind.FLAG, timestamp=datetime.now())
    )
    monkeypatch.setattr(cli_module, "make_store", lambda config: store)
    result = CliRunner().invoke(
        cli, ["--config-path", str(_write_config(tmp_path)), "signals", "recent"]
    )
    assert result.exit_code == 0
    assert "flag" in result.output
    assert "px" in result.output


def test_cli_triage_build_runs_end_to_end_with_stubs(tmp_path, monkeypatch):
    """Serves the Phase 1 substrate: ingest -> triage runs end-to-end with stubs."""
    config_path = _write_config(tmp_path)
    papers = [
        Paper(
            f"p{i}", f"Title {i}", ["Auth"], f"Abstract {i}",
            arxiv_id=f"p{i}", categories=["cs.AI"], submitted_date=datetime(2024, 1, 1),
        )
        for i in range(3)
    ]
    fake_source = MagicMock()
    fake_source.source_name = "arxiv"
    fake_source.fetch_recent.return_value = papers
    fake_emb = MagicMock()
    fake_emb.cached_embed_paper.side_effect = lambda paper: torch.ones(8)
    fake_emb.embed_paper.side_effect = lambda paper: torch.ones(8)
    monkeypatch.setattr(cli_module, "make_embedding_service", lambda config: fake_emb)
    monkeypatch.setattr(cli_module, "make_kb", lambda config, emb: FakeKBConnector())
    monkeypatch.setattr(cli_module, "make_source", lambda config: fake_source)
    monkeypatch.setattr(cli_module, "make_store", lambda config: SQLiteStore(":memory:"))

    result = CliRunner().invoke(
        cli, ["--config-path", str(config_path), "triage", "build"]
    )
    assert result.exit_code == 0, result.output
    triage_file = tmp_path / "triage" / f"{datetime.now().strftime('%Y-%m-%d.md')}"
    assert triage_file.exists()
    content = triage_file.read_text()
    assert "Title 0" in content
    assert "novelty_vs_kb" in content
    assert content.count("## [P-") == 3
