# CLAUDE.md

Local-LLM automations over LM Studio. Python 3.11+, venv at `.venv`.

## Commands

- Install: `.venv\Scripts\python -m pip install -e ".[dev]"`
- Test: `.venv\Scripts\python -m pytest`
- Run a task: `.venv\Scripts\python -m tasks.<name>`

## Conventions

- **One job = one module in `tasks/`** with a `main()` and an `if __name__ == "__main__"` guard. Tasks call `lm.ensure_server()` first — they run unattended.
- **All model calls go through `lmsa.LMStudio`.** Don't construct an `OpenAI` client in a task. Use `extract()` with a JSON schema whenever the result feeds code; never regex-parse free text.
- **All HTML goes through `lmsa.report`.** Never hand-roll styles in a task. A missing component is added to `report.py` + `afterhours.css` and shown in `tasks/style_preview.py`.
- **Untrusted text (email bodies, scraped pages, model output) is escaped.** `Section.raw()` is for trusted markup only.
- **Reads before writes.** A task that sends, deletes, labels, or pays anything defaults to dry-run and writes a report of what it *would* do.
- Model defaults live in `lmsa/config.py`; override via env vars listed in `.env.example`. Secrets come from Windows Credential Manager (`python -m lmsa.secrets set <name>`) or a gitignored local `.env`, never the repo.
- `output/` and `data/` are gitignored — generated artifacts and local caches go there. Reports land in `output/<YYYY-MM-DD>/<type>/<file>` (`lmsa/layout.py`; `Report.write` does it for you, and `python -m lmsa.layout` sorts a flat folder). Links between reports use `layout.link()`.
