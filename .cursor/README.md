# Cursor setup for `retargeting_from_scratch`

This directory contains repo-specific Cursor rules, command snippets, and local hook scripts for agents working on the retargeting toolkit.

Read order for agents:

1. `AGENTS.md`
2. `.cursor/rules/retarget-architecture.mdc`
3. `.cursor/rules/schema-authoring-and-targets.mdc`
4. `.cursor/rules/demo-pipeline.mdc`
5. `.cursor/rules/demo-tracks-resampling.mdc`
6. `.cursor/rules/sync-alignment.mdc`
7. `.cursor/rules/testing.mdc`
8. `.cursor/rules/python-style.mdc`
9. `.cursor/rules/running-commands.mdc`

The rules reflect the current demo architecture: typed track IDs, `Demonstration`/`DemonstrationView`, alignment-aware `resample_to`, discrete contact resampling, conservative mocap resampling, and graph-based sync.

Hook scripts are provided as runnable local helpers. They are not automatically installed by Git unless you wire them into `.git/hooks/` yourself.

Current architecture direction: TypedDict schema authoring compiles into normalized runtime specs/views/targets. See `.cursor/rules/schema-authoring-and-targets.mdc` and `agent_plans/typed_schema_authoring_refactor.md`.
