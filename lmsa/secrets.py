"""Secret storage for lmsa.

This repo is vault-agnostic: it knows nothing about Doppler, Bitwarden, or
any other password manager. A secret is either the matching env var (from a
gitignored local `.env`) or an entry in the OS keyring (Windows Credential
Manager). Wiring a real vault into either of those is the environment's job.

    python -m lmsa.secrets set imap      # reads stdin if piped, else prompts
    python -m lmsa.secrets check imap    # prints present/missing, never the value
    python -m lmsa.secrets set slack     # the Slack incoming-webhook URL for alerts

At runtime, `resolve(name)` is the only way callers should read a secret:
env var first, then the keyring. The value is never written to disk, logged,
or printed.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

SERVICE = "lmsa"

# secret name -> env var that overrides the keyring entry
_ENV_VARS = {
    "imap": "LMSA_IMAP_PASSWORD",
    "slack": "LMSA_SLACK_WEBHOOK",  # the incoming-webhook URL is a credential too
    "firms": "LMSA_FIRMS_MAP_KEY",  # NASA FIRMS API key (tasks.disasters, #59); optional
}


def resolve(name: str) -> str | None:
    """Env var first (local `.env`), then the OS keyring. `None` if neither has it."""
    env_var = _ENV_VARS[name]
    value = os.getenv(env_var)
    if value:
        return value
    import keyring

    return keyring.get_password(SERVICE, name)


def _read_value() -> str:
    if sys.stdin.isatty():
        return getpass.getpass("value: ")
    return sys.stdin.readline().rstrip("\n")


def _set(name: str) -> None:
    value = _read_value()
    if not value:
        print("no value given", file=sys.stderr)
        raise SystemExit(1)
    import keyring

    keyring.set_password(SERVICE, name, value)
    print(f"{name}: stored")


def _check(name: str) -> None:
    present = resolve(name) is not None
    print(f"{name}: {'present' if present else 'missing'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m lmsa.secrets")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("set", "check"):
        p = sub.add_parser(action)
        p.add_argument("name", choices=sorted(_ENV_VARS))
    args = parser.parse_args(argv)
    if args.action == "set":
        _set(args.name)
    else:
        _check(args.name)


if __name__ == "__main__":
    main()
