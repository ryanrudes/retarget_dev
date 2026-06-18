# Run full repository checks

Use this when finishing a task. A change is done only when all three pass.

```bash
python3 -m compileall -q src/retarget examples
uv run pytest -q
uv run mypy
```

If any command fails, fix the first failure before moving on. The later failures may just be noise wearing a costume.
