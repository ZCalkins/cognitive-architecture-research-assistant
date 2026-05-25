"""strange-loop CLI (Session 3).

Operator surface for the v0 paper-triage loop: ingest papers, build the daily
triage, and inspect engagement signals. The orchestrator remains the research
subject; the CLI is the operator's handle on it.

Service construction goes through module-level ``make_*`` factories so the
hermetic test suite can substitute fakes without touching disk, network, or
Ollama.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import click
import yaml
from dateutil import parser as date_parser

from src.ingestion import ArxivSource, IngestionPipeline
from src.kb import EmbeddingService, LocalKBConnector
from src.orchestrator import Dispatcher, Orchestrator, Workspace
from src.persistence import SQLiteStore
from src.signals import (
    DefaultLoggingHandler,
    PaperTagHandler,
    SignalEvent,
    SignalKind,
    SignalRegistry,
    TriageMarkdownParser,
    TriageMarkdownWatcher,
    TriageMarkdownWriter,
)
from src.specialists import build_v0_specialist_registry

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "conf" / "config.yaml"
_VALID_STATUSES = ("pending", "read", "skim", "discard", "flag")


# --- service factories (monkeypatched in tests) ---------------------------


def make_embedding_service(config: dict) -> EmbeddingService:
    cfg = config["embedding"]
    return EmbeddingService(
        model_name=cfg["model_name"], cache_dir=cfg.get("cache_dir"), device="cpu"
    )


def make_kb(config: dict, embedding_service: EmbeddingService) -> LocalKBConnector:
    cfg = config["kb"]
    return LocalKBConnector(
        cfg["papers_db_path"],
        cfg["embeddings_index_path"],
        cfg["embeddings_metadata_path"],
        embedding_service,
        embedding_dim=config["embedding"]["dimension"],
    )


def make_source(config: dict) -> ArxivSource:
    cfg = config["ingestion"]
    return ArxivSource(
        categories=cfg["arxiv_categories"],
        max_results_per_run=cfg["arxiv_max_results_per_run"],
    )


def make_store(config: dict) -> SQLiteStore:
    return SQLiteStore(config["persistence"]["db_path"])


# --- helpers --------------------------------------------------------------


def _load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _expand(path_value: str | Path) -> Path:
    return Path(os.path.expanduser(str(path_value)))


def _triage_path(config: dict, date: datetime | None = None) -> Path:
    date = date or datetime.now()
    return _expand(config["triage"]["output_directory"]) / date.strftime(
        config["triage"]["filename_format"]
    )


def _last_ingestion_path(config: dict) -> Path:
    return Path(config["persistence"]["db_path"]).parent / ".last_ingestion"


def _resolve_since(since: str | None, config: dict) -> datetime:
    if since:
        return date_parser.parse(since)
    marker = _last_ingestion_path(config)
    if marker.exists():
        try:
            return datetime.fromtimestamp(float(marker.read_text().strip()))
        except (ValueError, OSError):
            pass
    return datetime.now() - timedelta(days=7)


def _update_last_ingestion(config: dict) -> None:
    marker = _last_ingestion_path(config)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(time.time()))


def _score(_output, alpha) -> float:
    return round(float(alpha.mean()), 4)


def _set_status_in_text(text: str, paper_id: str, status: str) -> str:
    parts = re.split(r"(?=^## \[P-)", text, flags=re.MULTILINE)
    paper_line = re.compile(rf"^-\s*paper_id:\s*{re.escape(paper_id)}\s*$", re.MULTILINE)
    out = []
    for part in parts:
        if paper_line.search(part):
            part = re.sub(
                r"^(-\s*status:\s*).*$", rf"\g<1>{status}", part, count=1, flags=re.MULTILINE
            )
        out.append(part)
    return "".join(out)


def _signal_registry(config: dict) -> SignalRegistry:
    registry = SignalRegistry(make_store(config), make_kb(config, make_embedding_service(config)))
    registry.register(PaperTagHandler())
    registry.register(DefaultLoggingHandler())
    return registry


# --- CLI ------------------------------------------------------------------


@click.group()
@click.version_option(version="0.0.1", prog_name="strange-loop")
@click.option("--config-path", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
@click.pass_context
def cli(ctx: click.Context, config_path: str) -> None:
    """strange-loop research companion CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["config"] = _load_config(config_path)


@cli.command("config-show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Print the loaded configuration as YAML."""
    click.echo(yaml.safe_dump(ctx.obj["config"], sort_keys=False))


@cli.group("kb")
def kb_group() -> None:
    """Knowledge-base commands."""


@kb_group.command("stats")
@click.pass_context
def kb_stats(ctx: click.Context) -> None:
    """Show KB paper counts."""
    kb = make_kb(ctx.obj["config"], make_embedding_service(ctx.obj["config"]))
    click.echo(f"papers: {kb.count()}")


@cli.group("triage")
def triage_group() -> None:
    """Daily triage commands."""


@triage_group.command("build")
@click.option("--since", default=None, help="Ingest papers submitted since this date.")
@click.option("--max-papers", default=None, type=int, help="Cap papers triaged.")
@click.pass_context
def triage_build(ctx: click.Context, since: str | None, max_papers: int | None) -> None:
    """Ingest new papers and write today's triage markdown."""
    config = ctx.obj["config"]
    embedding_service = make_embedding_service(config)
    kb = make_kb(config, embedding_service)
    store = make_store(config)
    pipeline = IngestionPipeline([make_source(config)], kb, embedding_service)

    report = pipeline.run(since=_resolve_since(since, config))

    latent_dim = config["embedding"]["dimension"]
    registry = build_v0_specialist_registry(latent_dim=latent_dim)
    workspace = Workspace(registry)
    dispatcher = Dispatcher(registry, workspace)
    # The orchestrator owns the action-selection surface; it is the research subject.
    orchestrator = Orchestrator(registry, workspace, dispatcher, store)

    target = max_papers or config["triage"].get("papers_per_day_target", 30)
    papers = kb.all_papers()
    papers = papers[-target:] if len(papers) > target else papers

    scores: dict[str, dict[str, float]] = {}
    for paper in papers:
        context = embedding_service.cached_embed_paper(paper)
        # Dispatch ALL specialists so the triage shows every specialist's score.
        results = orchestrator.dispatcher.dispatch(context, top_k=len(registry))
        scores[paper.paper_id] = {
            schema.meta.referent: _score(output, alpha)
            for schema, output, alpha in results
        }

    writer = TriageMarkdownWriter(
        _expand(config["triage"]["output_directory"]), config["triage"]["filename_format"]
    )
    path = writer.write_daily(papers, scores)
    _update_last_ingestion(config)
    click.echo(
        f"Ingested {report.papers_added} new ({report.papers_seen} seen); "
        f"triaged {len(papers)}; output: {path}"
    )


@triage_group.command("today")
@click.pass_context
def triage_today(ctx: click.Context) -> None:
    """Print today's triage path, building it if absent."""
    path = _triage_path(ctx.obj["config"])
    if not path.exists():
        ctx.invoke(triage_build)
    click.echo(str(path))


@triage_group.command("status")
@click.argument("paper_id")
@click.argument("status")
@click.pass_context
def triage_status(ctx: click.Context, paper_id: str, status: str) -> None:
    """Set a paper's status in today's triage file and emit the signal."""
    if status not in _VALID_STATUSES:
        raise click.BadParameter(f"status must be one of {list(_VALID_STATUSES)}")
    config = ctx.obj["config"]
    path = _triage_path(config)
    if not path.exists():
        raise click.ClickException(f"No triage file for today at {path}")
    path.write_text(
        _set_status_in_text(path.read_text(encoding="utf-8"), paper_id, status),
        encoding="utf-8",
    )
    kind = SignalKind(status)
    _signal_registry(config).dispatch(
        SignalEvent(paper_id=paper_id, kind=kind, timestamp=datetime.now(), source="cli")
    )
    click.echo(f"Set {paper_id} status to {status}")


@triage_group.command("watch")
@click.pass_context
def triage_watch(ctx: click.Context) -> None:
    """Watch the triage directory and emit signals on edits (foreground)."""
    config = ctx.obj["config"]
    watcher = TriageMarkdownWatcher(
        _expand(config["triage"]["output_directory"]),
        _signal_registry(config),
        TriageMarkdownParser(),
    )
    watcher.start()
    click.echo("Watching for triage edits. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()


@cli.group("signals")
def signals_group() -> None:
    """Engagement-signal inspection commands."""


@signals_group.command("recent")
@click.option("--limit", default=20, type=int)
@click.pass_context
def signals_recent(ctx: click.Context, limit: int) -> None:
    """List the most recent engagement signals."""
    store = make_store(ctx.obj["config"])
    for row in store.recent_engagement_signals(limit=limit):
        click.echo(f"{row['ts']:.0f} {row['kind']} {row['paper_id']} {row['payload_json']}")


@signals_group.command("for-paper")
@click.argument("paper_id")
@click.pass_context
def signals_for_paper(ctx: click.Context, paper_id: str) -> None:
    """List engagement signals for a paper, oldest first."""
    store = make_store(ctx.obj["config"])
    for row in store.engagement_signals_for_paper(paper_id):
        click.echo(f"{row['ts']:.0f} {row['kind']} {row['payload_json']}")


def main() -> None:
    cli()
