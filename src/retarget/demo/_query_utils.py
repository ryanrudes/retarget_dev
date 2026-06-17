"""Internal helpers for demonstration-layer time-series queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np


def resolve_indices(length: int, indices: tuple[int, ...] | None) -> tuple[int, ...]:
    if indices is not None:
        return indices
    return tuple(range(length))


def slice_timestamps(timestamps: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return timestamps[list(indices)]


def normalize_entity_input[E](
    entity: E | Sequence[E],
    entity_type: type[E],
) -> tuple[tuple[E, ...], bool]:
    """Return (entities, is_many) for a single entity or a sequence of entities."""
    if isinstance(entity, entity_type):
        return (entity,), False
    if isinstance(entity, str):
        raise TypeError(
            f"Expected {entity_type.__name__} or a sequence of "
            f"{entity_type.__name__}; got raw string {entity!r}"
        )
    entities = tuple(cast(Sequence[E], entity))
    for item in entities:
        if not isinstance(item, entity_type):
            raise TypeError(
                f"Expected all items to be {entity_type.__name__}; "
                f"got {type(item).__name__}"
            )
    return entities, True


def stack_entity_arrays[T](
    entities: Sequence[T],
    arrays: Sequence[np.ndarray],
    *,
    return_dict: bool,
) -> np.ndarray | Mapping[T, np.ndarray]:
    if return_dict:
        return dict(zip(entities, arrays, strict=True))
    if not arrays:
        return np.empty((0, 0))
    first = arrays[0]
    if first.shape[0] == 0:
        trailing = first.shape[1:] if first.ndim > 1 else ()
        return np.empty((0, len(arrays), *trailing), dtype=first.dtype)
    return np.stack(arrays, axis=1)
