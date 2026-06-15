# Inspect public API surface

Use this after adding new public helpers or modules.

```bash
python - <<'PY'
import retarget.demo.alignment as alignment
import retarget.demo.demo as demo
import retarget.demo.resampling as resampling
import retarget.demo.sync as sync

for module in [alignment, demo, resampling, sync]:
    print(f"\n{module.__name__}")
    for name in sorted(n for n in dir(module) if not n.startswith('_')):
        print(" ", name)
PY
```

This is not a substitute for tests. It is a quick sanity check so imports do not rot quietly like leftovers in a lab fridge.
