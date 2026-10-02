"""Config-driven briefing engine (#52): `config/briefings/<name>.toml` ->
collect -> pre-filter -> embed -> cluster -> rank -> extract -> report.

- `lmsa.briefing.config`  -- `load(name)`, `names()`, `BriefingConfig`, `ConfigError`
- `lmsa.briefing.sources` -- `Doc`, `SourceContext`, `register`, `register_extension`
- `lmsa.briefing.engine`  -- `run_briefing`, `build_report`, `StoryResult`, table/vector ids
- `lmsa.briefing.quotes`  -- `verify_quote`
- `lmsa.briefing.adapters` -- drop-in adapter modules (imported by `load_plugins()`)
"""

from .config import BriefingConfig, ConfigError
from .sources import Doc, SourceContext, register, register_extension

__all__ = ["BriefingConfig", "ConfigError", "Doc", "SourceContext", "register", "register_extension"]
