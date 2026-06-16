# Implement typed schema authoring refactor

Use this command guide when working on the TypedDict schema authoring change.

## Read first

```bash
sed -n '1,220p' AGENTS.md
sed -n '1,240p' .cursor/rules/schema-authoring-and-targets.mdc
sed -n '1,220p' .cursor/rules/retarget-architecture.mdc
sed -n '1,180p' .cursor/rules/python-style.mdc
```

## Search current spec/target code

```bash
rg -n "class .*Spec|SegmentKey|MarkerHandle|PatchHandle|PatchTarget|SceneSpec|SubjectSpec|SegmentSpec" src/retarget examples tests
```

## Suggested narrow tests

After adding the package primitives/build step, add or update tests around core schema behavior and run something like:

```bash
pytest -q tests/test_core_schema_authoring.py tests/test_core_scene_specs.py tests/test_core_targets.py
```

If those files do not exist yet, create focused tests rather than stuffing assertions into unrelated demo tests like a gremlin with a suitcase.

## Full verification

```bash
python3 -m compileall -q src/retarget
python3 -m compileall -q examples
pytest -q
```

If the project uses uv in the current checkout, prefer:

```bash
uv run pytest -q
```
