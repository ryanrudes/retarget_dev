# Run a narrow test

Replace `<path>` with the relevant test path or node id.

```bash
pytest -q <path>
```

Examples:

```bash
pytest -q tests/test_demo_sync.py::test_estimate_sync_to_reference_returns_root_alignments
pytest -q tests/test_demo_mocap.py
```
