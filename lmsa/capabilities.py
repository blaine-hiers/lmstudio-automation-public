"""Everything `tasks/` can do, derived from live metadata -- never a
hand-maintained list (#67). Backs the dashboard's `/capabilities` page and
the static `output/capabilities.html` report.

Task facts come from the same sources the launcher and Setup checklist
already read:
  - `tasks.dashboard.discover_tasks` (ast via `lmsa.task_meta`, never an
    import): group, cadence, model, curated/derived options, write flags,
    whether it's new;
  - the whole docstring (`lmsa.task_meta.TaskMeta.full_doc`), not just its
    first line;
  - each task's live SETUP status from `dash.setup.status_all()`, the same
    probe round the Setup page runs (`tool-<name>` items, one per task that
    declares a `SETUP` block);
  - its newest row in the `runs` table (`dash.last_runs()` / `report_url`).

Briefings (the #52 briefing engine, built in parallel) are listed from
`config/briefings/*.toml` if that directory exists at all -- this module
never imports `lmsa.briefing`, so it works whether or not #52 has landed.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

MAX_BRIEFING_TEXT = 400
MAX_BRIEFING_SOURCES_SHOWN = 40


@dataclass(frozen=True)
class ArgCap:
    flag: str
    kind: str  # positional | int | bool | text
    label: str
    write: bool
    required: bool


@dataclass(frozen=True)
class TaskCap:
    name: str
    group: str
    tone: str
    summary: str  # docstring, first line
    doc: str  # docstring, in full
    cadence: str | None
    model: str | None
    args: tuple[ArgCap, ...]
    write_flags: tuple[str, ...]
    setup: dict | None  # the module's own SETUP literal, or None if it declares none
    setup_state: str  # "ok" | "missing" | "warning" | "none"
    setup_reason: str
    last_run: dict | None  # {"started", "finished", "status", "report_url"}
    new: bool
    warnings: tuple[str, ...]

    @property
    def writes(self) -> bool:
        return bool(self.write_flags) or any(a.write for a in self.args)


@dataclass(frozen=True)
class Briefing:
    name: str
    title: str
    description: str
    sources: int
    warning: str = ""


@dataclass(frozen=True)
class NotReady:
    id: str
    title: str
    reason: str


@dataclass(frozen=True)
class Capabilities:
    tasks: tuple[TaskCap, ...]
    not_ready: tuple[NotReady, ...]
    briefings: tuple[Briefing, ...]


def _clean_text(value: object, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def briefings(root: Path) -> tuple[Briefing, ...]:
    """One entry per `config/briefings/*.toml` (title, description, source
    count), tolerant of unknown keys -- or nothing at all if the briefing
    engine (#52) hasn't created the directory yet."""
    briefings_dir = Path(root) / "config" / "briefings"
    if not briefings_dir.is_dir():
        return ()
    out: list[Briefing] = []
    for path in sorted(briefings_dir.glob("*.toml")):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            out.append(Briefing(path.stem, path.stem, "", 0, warning=f"can't read ({type(exc).__name__})"))
            continue
        if not isinstance(data, dict):
            out.append(Briefing(path.stem, path.stem, "", 0, warning="not a table"))
            continue
        # The briefing engine's shape: [briefing] title/description, and
        # [[sources.<kind>]] entry lists under a `sources` table.
        head = data.get("briefing") if isinstance(data.get("briefing"), dict) else {}
        title = _clean_text(head.get("title"), 120) or path.stem
        description = _clean_text(head.get("description"), MAX_BRIEFING_TEXT)
        sources = data.get("sources") if isinstance(data.get("sources"), dict) else {}
        n_sources = sum(len(v) for v in sources.values() if isinstance(v, list))
        out.append(Briefing(path.stem, title, description, min(n_sources, 10_000)))
    return tuple(out)


def _setup_status(status_by_id: dict[str, dict], t) -> tuple[str, str]:
    if t.setup is None and not t.setup_warnings:
        return "none", "declares no SETUP needs"
    row = status_by_id.get(f"tool-{t.name}")
    if row is None:
        return "warning", "not probed"
    return row["state"], row["reason"]


def build(dash) -> Capabilities:
    """Everything the capabilities page shows, read fresh each call: an
    ast-parse per module (cached by mtime/size in `lmsa.task_meta`) plus the
    same `Setup.status_all()` probe round the Setup page already pays for."""
    from lmsa.task_meta import read_meta

    tasks = dash.catalog()
    last = dash.last_runs()
    status = dash.setup.status_all()["items"]
    status_by_id = {row["id"]: row for row in status}
    titles = {it.id: it.title for it in dash.setup.items()}

    caps = []
    for t in tasks:
        meta = read_meta(dash.tasks_dir / f"{t.name}.py")
        setup_state, setup_reason = _setup_status(status_by_id, t)
        lr = last.get(t.name)
        last_run = None
        if lr is not None:
            last_run = {**lr, "report_url": dash.report_url(lr.get("report"))}
        caps.append(TaskCap(
            name=t.name,
            group=t.group,
            tone=t.tone,
            summary=t.doc,
            doc=meta.full_doc or t.doc,
            cadence=t.cadence,
            model=t.model,
            args=tuple(ArgCap(f.flag, f.kind, f.label, f.write, f.required) for f in t.fields),
            write_flags=t.write_flags,
            setup=t.setup,
            setup_state=setup_state,
            setup_reason=setup_reason,
            last_run=last_run,
            new=t.new,
            warnings=t.warnings,
        ))

    not_ready = tuple(
        NotReady(row["id"], titles.get(row["id"], row["id"]), row["reason"])
        for row in status if row["state"] != "ok"
    )
    return Capabilities(tuple(caps), not_ready, briefings(dash.root))
