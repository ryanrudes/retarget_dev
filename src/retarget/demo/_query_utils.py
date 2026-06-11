"""Internal helpers for demonstration-layer time-series queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from retarget.core.enums import MarkerId, NameId, PatchId


def resolve_indices(length: int, indices: tuple[int, ...] | None) -> tuple[int, ...]:
    if indices is not None:
        return indices
    return tuple(range(length))


def slice_timestamps(timestamps: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return timestamps[list(indices)]


def finite_difference_velocity(
    positions: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Return velocity with shape matching ``positions`` using ``np.gradient``."""
    if len(timestamps) == 0:
        return np.empty_like(positions)
    if len(timestamps) == 1:
        shape = positions.shape
        return np.zeros(shape, dtype=positions.dtype)
    return np.gradient(positions, timestamps, axis=0)


def speed_from_velocity(velocity: np.ndarray) -> np.ndarray:
    return np.linalg.norm(velocity, axis=-1)


def coerce_marker_id[M: MarkerId](marker: MarkerId, marker_type: type[M]) -> M:
    if isinstance(marker, marker_type):
        return marker
    raise TypeError(
        f"Expected marker of type {marker_type.__name__}, got {type(marker).__name__}"
    )


def coerce_patch_id[P: PatchId](patch: PatchId, patch_type: type[P]) -> P:
    if isinstance(patch, patch_type):
        return patch
    raise TypeError(
        f"Expected patch of type {patch_type.__name__}, got {type(patch).__name__}"
    )


def normalize_entity_input[E](
    entity: E | Sequence[E],
    entity_type: type[E],
) -> tuple[tuple[E, ...], bool]:
    if isinstance(entity, entity_type):
        return (entity,), False
    if isinstance(entity, str):
        raise TypeError(
            f"Expected {entity_type.__name__} or a sequence of "
            f"{entity_type.__name__}; got raw string {entity!r}"
        )
    if isinstance(entity, NameId):
        raise TypeError(
            f"Expected {entity_type.__name__}; got {type(entity).__name__}"
        )
    entities = tuple(entity)
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
