# AI handoff protocol

This file is the portable workflow for any AI tool working in this repository. The Codex `experiment-handoff` skill adds more detailed guidance when available; this document is sufficient for tools that cannot load Codex skills.

## Keep the information stream usable

- `.trae/HANDOFF.md` is the single read-first pointer for current work. Keep it short and current: goal, verified state, important decisions, links to evidence, and the next concrete action.
- `documents/NOTES.md` is chronological, durable history. Append dated questions, decisions, failed attempts, and results; do not erase superseded history.
- `todo/TODO.md` is the user-owned task list. Follow `documents/RULES.md`; agents must not create or remove TODO items unless the user explicitly asks.
- `documents/DESIGN.md` and `documents/STRUCTURE.md` remain the source of truth for project purpose and code layout. Update them when the corresponding facts change.

## While working

At the start, read `.trae/HANDOFF.md`, then follow the full read-first sequence in `documents/RULES.md`. Verify current state from files, git status, processes/sessions, logs, checkpoints, and reports as relevant; an old note or chat message is not proof that a run is still active or complete.

For multi-step or long-running work, update the handoff as milestones change and before context switches. Label each run `planned`, `running`, `completed`, `failed`, or `blocked`. A `running` claim needs a live process/session check; a `completed` claim needs a successful exit and expected artifact check. Record timestamp and timezone for transient status.

For each experiment, preserve enough to resume: command/config, revision or changed files, data and checkpoint identity, seed, device, output/log/session paths, progress, last verified checkpoint, metrics, and caveats. Link detailed plans and reports instead of duplicating them. Separate measured results from interpretation and estimates.

Before handing off, state one concrete next action, its prerequisites and success signal, and any user decision or authorization boundary. If no task is active, say so and point to the user-owned TODO list and unfinished design milestones without converting them into new tasks.
