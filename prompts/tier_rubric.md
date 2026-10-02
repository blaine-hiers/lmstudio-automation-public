# Tier rubric

Used by `tasks/pretriage.py` for a free first pass over an issue, before a
paid model triages it. Classify the issue and return only the requested
JSON fields.

## Type

- `bug` — something that used to work, or should work per spec, doesn't.
- `feature` — a wholly new capability.
- `improvement` — an enhancement to something that already works.

## Tier

Copied verbatim from Nightshift's `CLAUDE.md` (the model-tier table); pick
the *least* powerful tier that could plausibly do the work — default down,
not up.

| Tier | Use for |
|---|---|
| `haiku` | Mechanical, well-specified tickets: doc/comment drift, rename, add a missing guard, one-line regex or flag fix, status/label bookkeeping. |
| `sonnet` | Ordinary bugfixes: reproduce, trace a few files, patch, run tests. This is the normal default for a well-written ticket. |
| `opus` | Only when needed: unclear root cause, cross-cutting refactor, concurrency/data-loss/security reasoning, or after a lower tier has failed. |
| `fable` | The hardest long-horizon agentic work, or opus already failed. Most capable model, but 2x opus cost ($10/$50 per MTok vs $5/$25) and single requests can run many minutes, so the drain gives it a longer clock but only one attempt. |

(`codex` is not a tier this pass ever assigns — it's an opt-in human choice,
never derived from the ticket.)

## Workable

`workable: true` means an agent could start right now with zero questions —
the same bar Nightshift's `status/todo` uses. Say `false`, and list what's
missing, when the issue lacks any of:

- A clear, reproducible problem statement (for a bug) or a concrete
  acceptance criterion (for a feature/improvement).
- Enough detail to act on without the original conversation — no undefined
  design decision left for the implementer to guess at.

## Output

- `missing`: short strings naming each gap (empty list if `workable` is
  `true`).
- `reason`: one or two sentences explaining the `type`/`tier`/`workable`
  call.
