"""Triage markdown surface — write, parse, and watch the daily triage file.

The daily triage is a markdown file the user annotates in any editor. The
writer emits a stable, parseable prelude per paper; the parser reads it back;
the watcher turns file edits into engagement signals.

Strange-loop role
-----------------
This is dumb infrastructure. The writer and parser do not *interpret* anything
beyond the explicit prelude fields and the notes block — interpretation lives
in the signal handlers (the (C) firewall: the file is a substrate, not a
decider). Time-on-file is a coarse, best-effort proxy; the pipeline is robust
to its imprecision.
"""

from __future__ import annotations

import contextlib
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.signals.types import SignalEvent, SignalKind

if TYPE_CHECKING:
    from src.kb.types import Paper
    from src.signals.handlers import SignalRegistry

SPECIALIST_ORDER = (
    "novelty_vs_kb",
    "methodological_rigor",
    "theoretical_claim_evaluator",
    "relevance_to_projects",
    "citation_graph_position",
    "author_history",
)

_PRELUDE_KEYS = {
    "paper_id",
    "arxiv",
    "status",
    "overall_score",
    "authors",
    "categories",
    "submitted",
    "opened_at",
}

_BLOCK_SPLIT = re.compile(r"(?=^## \[P-)", re.MULTILINE)
_FIELD_RE = re.compile(r"^\s*-\s*([A-Za-z_]+):\s*(.*)$")
_HEADER_RE = re.compile(r"^##\s*\[P-\d+\]\s*(.*)$")
_STATUS_SIGNAL = {
    "read": SignalKind.READ,
    "skim": SignalKind.SKIM,
    "discard": SignalKind.DISCARD,
    "flag": SignalKind.FLAG,
}


class TriageMarkdownParser:
    """Parses a triage markdown file into ``{paper_id: field_dict}``."""

    def parse_file(self, path: Path | str) -> dict[str, dict]:
        return self.parse_text(Path(path).read_text(encoding="utf-8"))

    def parse_text(self, text: str) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for part in _BLOCK_SPLIT.split(text):
            if not part.lstrip().startswith("## [P-"):
                continue
            block = self._parse_block(part)
            if "paper_id" in block:
                result[block["paper_id"]] = block
        return result

    def _parse_block(self, part: str) -> dict:
        lines = part.splitlines()
        data: dict = {"specialists": {}}
        header = _HEADER_RE.match(lines[0]) if lines else None
        if header:
            data["title"] = header.group(1).strip()

        in_specialists = False
        in_notes = False
        notes_lines: list[str] = []
        for line in lines[1:]:
            if in_notes:
                notes_lines.append(line)
                continue
            stripped = line.strip()
            if stripped.startswith("**Notes:**"):
                in_notes = True
                continue
            if stripped.startswith("**Abstract.**"):
                data["abstract"] = stripped[len("**Abstract.**") :].strip()
                continue
            match = _FIELD_RE.match(line)
            if not match:
                continue
            key, value = match.group(1), match.group(2).strip()
            if key == "specialists":
                in_specialists = True
            elif key in _PRELUDE_KEYS:
                in_specialists = False
                data[key] = value
            elif in_specialists:
                with contextlib.suppress(ValueError):
                    data["specialists"][key] = float(value)
        data["notes"] = "\n".join(notes_lines).strip()
        return data


class TriageMarkdownWriter:
    """Writes the daily triage file, preserving user-set fields on rewrite."""

    def __init__(
        self, output_directory: Path | str, filename_format: str = "%Y-%m-%d.md"
    ) -> None:
        self.output_directory = Path(output_directory)
        self.filename_format = filename_format
        self._parser = TriageMarkdownParser()

    def write_daily(
        self,
        papers: list[Paper],
        scores: dict[str, dict[str, float]],
        date: datetime | None = None,
    ) -> Path:
        date = date or datetime.now()
        path = self.output_directory / date.strftime(self.filename_format)
        path.parent.mkdir(parents=True, exist_ok=True)

        preserved: dict[str, dict] = {}
        prior_order: list[str] = []
        if path.exists():
            preserved = self._parser.parse_file(path)
            prior_order = list(preserved.keys())

        papers_by_id = {paper.paper_id: paper for paper in papers}
        overall = {
            paper_id: (sum(s.values()) / len(s) if s else 0.0)
            for paper_id, s in scores.items()
        }
        new_ids = sorted(
            (pid for pid in papers_by_id if pid not in preserved),
            key=lambda pid: overall.get(pid, 0.0),
            reverse=True,
        )
        ordered_ids = [pid for pid in prior_order if pid in papers_by_id] + new_ids

        blocks = [
            self._render_block(
                sequence=index,
                paper=papers_by_id[paper_id],
                specialist_scores=scores.get(paper_id, {}),
                overall=overall.get(paper_id, 0.0),
                preserved=preserved.get(paper_id, {}),
            )
            for index, paper_id in enumerate(ordered_ids, start=1)
        ]
        path.write_text("\n".join(blocks), encoding="utf-8")
        return path

    @staticmethod
    def _render_block(
        sequence: int,
        paper: Paper,
        specialist_scores: dict[str, float],
        overall: float,
        preserved: dict,
    ) -> str:
        status = preserved.get("status", "pending")
        opened_at = preserved.get("opened_at", "null")
        notes = preserved.get("notes", "")
        submitted = (
            paper.submitted_date.date().isoformat() if paper.submitted_date else "null"
        )
        lines = [
            f"## [P-{sequence:03d}] {paper.title}",
            "",
            f"- paper_id: {paper.paper_id}",
            f"- arxiv: {paper.arxiv_id if paper.arxiv_id else 'null'}",
            f"- status: {status}",
            f"- overall_score: {overall:.4f}",
            "- specialists:",
        ]
        lines += [
            f"  - {name}: {specialist_scores.get(name, 0.0):.4f}"
            for name in SPECIALIST_ORDER
        ]
        lines += [
            f"- authors: {'; '.join(paper.authors)}",
            f"- categories: {', '.join(paper.categories)}",
            f"- submitted: {submitted}",
            f"- opened_at: {opened_at}",
            "",
            f"**Abstract.** {paper.abstract}",
            "",
            "**Notes:**",
            "",
        ]
        if notes:
            lines += [notes, ""]
        return "\n".join(lines)


class TriageMarkdownWatcher:
    """Turns triage-file edits into engagement signals (best-effort)."""

    def __init__(
        self,
        triage_directory: Path | str,
        signal_registry: SignalRegistry,
        parser: TriageMarkdownParser,
    ) -> None:
        self.triage_directory = Path(triage_directory)
        self.signal_registry = signal_registry
        self.parser = parser
        self._snapshot: dict[Path, dict[str, dict]] = {}
        self._first_open_seen: set[Path] = set()
        self._last_seen: dict[Path, datetime] = {}
        self._observer = None

    def start(self) -> None:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event) -> None:
                if not event.is_directory:
                    watcher._on_modified(Path(event.src_path))

            def on_created(self, event) -> None:
                if not event.is_directory:
                    watcher._on_modified(Path(event.src_path))

        for existing in self.triage_directory.glob("*.md"):
            self._snapshot[existing] = self.parser.parse_file(existing)
        self._observer = Observer()
        self._observer.schedule(_Handler(), str(self.triage_directory), recursive=False)
        self._observer.start()

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    def _on_modified(self, path: Path) -> None:
        path = Path(path)
        if path.suffix != ".md":
            return
        now = datetime.now()

        if path not in self._first_open_seen:
            self._first_open_seen.add(path)
            self.signal_registry.dispatch(
                SignalEvent(
                    paper_id="",
                    kind=SignalKind.OPENED,
                    timestamp=now,
                    payload={"path": str(path)},
                    source="watcher",
                )
            )

        try:
            current = self.parser.parse_file(path)
        except OSError:
            return
        prior = self._snapshot.get(path, {})

        for paper_id, fields in current.items():
            prior_fields = prior.get(paper_id, {})
            new_status = fields.get("status", "pending")
            if new_status != prior_fields.get("status", "pending") and new_status in _STATUS_SIGNAL:
                self.signal_registry.dispatch(
                    SignalEvent(
                        paper_id=paper_id,
                        kind=_STATUS_SIGNAL[new_status],
                        timestamp=now,
                        payload={"status": new_status},
                        source="watcher",
                    )
                )
            new_notes = fields.get("notes", "")
            if new_notes and new_notes != prior_fields.get("notes", ""):
                self.signal_registry.dispatch(
                    SignalEvent(
                        paper_id=paper_id,
                        kind=SignalKind.NOTE,
                        timestamp=now,
                        payload={"note": new_notes},
                        source="watcher",
                    )
                )

        last_seen = self._last_seen.get(path)
        if last_seen is not None:
            self.signal_registry.dispatch(
                SignalEvent(
                    paper_id="",
                    kind=SignalKind.TIME_ON_FILE,
                    timestamp=now,
                    payload={"seconds": (now - last_seen).total_seconds()},
                    source="watcher",
                )
            )
        self._last_seen[path] = now
        self._snapshot[path] = current
