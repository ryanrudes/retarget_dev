# Run demo/resampling/sync tests

Use this after touching `src/retarget/demo/**`.

```bash
pytest -q \
  tests/test_demo_resampling.py \
  tests/test_demo_contact.py \
  tests/test_demo_mocap.py \
  tests/test_demo.py \
  tests/test_demo_container.py \
  tests/test_demo_sync.py
```
