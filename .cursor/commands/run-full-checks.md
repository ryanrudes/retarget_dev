# Run full repository checks

Use this when finishing a task.

```bash
python -m compileall -q src/retarget
python -m compileall -q examples
pytest -q
```

If any command fails, fix the first failure before moving on. The later failures may just be noise wearing a costume.
