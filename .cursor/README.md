# Cursor setup for `retargeting_from_scratch`

`AGENTS.md` (repo root) is the canonical, self-contained guide for this project
and is meant to be read by **every** agent or human. `CLAUDE.md` is a symlink to
it, so Claude Code reads the same content. The files in this directory are
**Cursor-specific conveniences** that mirror `AGENTS.md`; if they ever disagree,
`AGENTS.md` wins.

Read order for agents working in Cursor:

1. `AGENTS.md`
2. `.cursor/rules/retarget-architecture.mdc`
3. `.cursor/rules/schema-authoring-and-targets.mdc`
4. `.cursor/rules/demo-pipeline.mdc`
5. `.cursor/rules/demo-tracks-resampling.mdc`
6. `.cursor/rules/sync-alignment.mdc`
7. `.cursor/rules/testing.mdc`
8. `.cursor/rules/python-style.mdc`
9. `.cursor/rules/running-commands.mdc`

The rules reflect the current architecture: a typed-first, **enum-free** model in
which `TypedDict` schemas plus dual-purpose frozen dataclasses are both the
authoring objects and the bound runtime query surface, with string-based targets
(`MarkerTarget` / `PatchTarget` / `SegmentTarget` / `SegmentKey`) as stable
runtime keys. Demonstrations use `Demonstration` / `DemonstrationView`,
alignment-aware `resample_to`, discrete contact resampling, conservative mocap
resampling, and graph-based sync.

`.cursor/commands/*.md` are runnable snippets and `.cursor/hooks/*.sh` are local
helper scripts. Hooks are not installed by Git unless you wire them into
`.git/hooks/` yourself (see `.cursor/hooks/README.md`).
