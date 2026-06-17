# Sync and resample demonstrations

The common workflow is:

1. Build a `Demonstration` from time-indexed tracks.
2. Define a `SyncPlan` describing which tracks should be aligned.
3. Estimate sync and materialize a slice on the reference timeline.

```python
from retarget.demo.sync import (
    SyncEdge,
    SyncPlan,
    estimate_sync_and_resample_to_reference,
)

plan = SyncPlan(
    reference="mocap",
    edges=(
        SyncEdge(
            source="contact",
            reference="mocap",
            source_signal=contact_signal,
            reference_signal=mocap_signal,
            max_lag_seconds=0.5,
        ),
    ),
)

aligned = estimate_sync_and_resample_to_reference(
    demo,
    plan,
    start=0.0,
    stop=10.0,
)