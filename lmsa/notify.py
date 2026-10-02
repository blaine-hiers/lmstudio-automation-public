"""Alerts: a Windows toast and a Slack webhook post, fanned out by `notify()`.

Both channels fail soft — a broken or unconfigured channel logs a warning
and returns False rather than raising, so sending an alert never breaks the
task that's trying to send it.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import urllib.request

from .secrets import resolve

log = logging.getLogger(__name__)

LEVELS = {"info", "urgent"}

# A well-known AUMID (Windows PowerShell's own) so CreateToastNotifier works
# without registering a real app identity.
_TOAST_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


def _ps_string(value: str) -> str:
    """A PowerShell single-quoted string literal for `value`.

    Single-quoted strings in PowerShell do no interpolation and no escaping
    beyond doubling an embedded quote, so this is injection-safe.
    """
    return "'" + value.replace("'", "''") + "'"


def toast(title: str, body: str, url: str | None = None) -> bool:
    """Show a Windows toast via the built-in WinRT notification APIs.

    No extra package: PowerShell loads `Windows.UI.Notifications` itself.
    Fails soft — off Windows, or if PowerShell/WinRT can't show it, this
    logs a warning and returns False instead of raising.
    """
    if sys.platform != "win32":
        log.warning("toast skipped: not running on Windows")
        return False

    lines = [
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications,"
        " ContentType = WindowsRuntime] > $null",
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] > $null",
        "$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02)",
        "$texts = $template.GetElementsByTagName('text')",
        f"$texts.Item(0).AppendChild($template.CreateTextNode({_ps_string(title)})) > $null",
        f"$texts.Item(1).AppendChild($template.CreateTextNode({_ps_string(body)})) > $null",
    ]
    if url:
        lines.append(f"$template.SelectSingleNode('/toast').SetAttribute('launch', {_ps_string(url)})")
    lines += [
        "$toast = New-Object Windows.UI.Notifications.ToastNotification $template",
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        f"{_ps_string(_TOAST_APP_ID)}).Show($toast)",
    ]
    script = "\n".join(lines)

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        log.warning("toast failed: could not run powershell", exc_info=True)
        return False

    if result.returncode != 0:
        log.warning("toast failed: %s", result.stderr.decode(errors="replace").strip())
        return False
    return True


def slack(text: str) -> bool:
    """POST `text` to the Slack incoming webhook: `LMSA_SLACK_WEBHOOK` if
    set, else the keyring entry `python -m lmsa.secrets set slack` stored.

    Skips silently (returns False) when no webhook is configured.
    """
    try:
        webhook = resolve("slack")
    except Exception as exc:  # keyring backend missing/broken: fail soft, like every channel
        log.warning("slack notify skipped: can't read the keyring (%s)", type(exc).__name__)
        return False
    if not webhook:
        return False

    payload = json.dumps({"text": text}).encode("utf-8")
    try:
        req = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        # The webhook URL is a credential, and ValueError("unknown url type:
        # '<url>'"), http.client.InvalidURL and friends quote it: log the
        # exception type only, never its text or traceback.
        log.warning("slack notify failed (%s)", type(exc).__name__)
        return False


def notify(title: str, body: str, url: str | None = None, level: str = "info") -> None:
    """Fan out an alert to whatever channels are configured (toast, Slack).

    `level` is "info" or "urgent" — urgent alerts get a Slack prefix so they
    stand out among routine ones. Each channel fails soft on its own; a
    channel that isn't configured or doesn't work is simply skipped.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")

    toast(title, body, url)

    prefix = ":rotating_light: " if level == "urgent" else ""
    text = f"{prefix}*{title}*\n{body}"
    if url:
        text += f"\n{url}"
    slack(text)
