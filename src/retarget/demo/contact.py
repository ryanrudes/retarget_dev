"""Contact track skeleton for derived contact state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from retarget.core.targets import PatchTarget
from retarget.demo._query_utils import normalize_entity_input, slice_timestamps, stack_entity_arrays


def _validate_strictly_increasing_timestamps(timestamps: np.ndarray) -> None:
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")


@dataclass(frozen=True, slots=True)
class ContactTrack:
    """Time-varying contact state keyed by scene-level patch targets."""

    timestamps: np.ndarray
    contacts: Mapping[PatchTarget[Any], np.ndarray]
    confidences: Mapping[PatchTarget[Any], np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        object.__setattr__(self, "timestamps", timestamps)
        if timestamps.ndim != 1:
            raise ValueError("timestamps must be a 1D array")
        _validate_strictly_increasing_timestamps(timestamps)
        num_timesteps = len(timestamps)
        frozen_contacts: dict[PatchTarget[Any], np.ndarray] = {}
        for target, values in self.contacts.items():
            array = np.asarray(values)
            if array.shape != (num_timesteps,):
                raise ValueError(
                    f"Contact array for {target} has shape {array.shape}, "
                    f"expected ({num_timesteps},)"
                )
            if array.dtype != np.bool_:
                raise TypeError("contact arrays must have bool dtype")
            frozen_contacts[target] = array
        unknown_confidence_targets = set(self.confidences) - set(self.contacts)
        if unknown_confidence_targets:
            raise ValueError(
                "confidences contains targets not present in contacts: "
                f"{unknown_confidence_targets}"
            )
        frozen_confidences: dict[PatchTarget[Any], np.ndarray] = {}
        for target, values in self.confidences.items():
            array = np.asarray(values)
            if array.shape != (num_timesteps,):
                raise ValueError(
                    f"Confidence array for {target} has shape {array.shape}, "
                    f"expected ({num_timesteps},)"
                )
            if not np.issubdtype(array.dtype, np.floating):
                raise TypeError("contact confidences must be floating-point arrays")
            if np.any(array < 0) or np.any(array > 1):
                raise ValueError("contact confidences must be in [0, 1]")
            frozen_confidences[target] = array
        object.__setattr__(self, "contacts", MappingProxyType(frozen_contacts))
        object.__setattr__(self, "confidences", MappingProxyType(frozen_confidences))

    def slice_time(self, start: float, stop: float) -> ContactTrackView:
        mask = (self.timestamps >= start) & (self.timestamps < stop)
        indices = tuple(int(index) for index in np.nonzero(mask)[0])
        return ContactTrackView(source=self, indices=indices)


@dataclass(frozen=True, slots=True)
class ContactTrackView:
    """Sliced view into a :class:`ContactTrack`."""

    source: ContactTrack
    indices: tuple[int, ...]

    @property
    def timestamps(self) -> np.ndarray:
        return slice_timestamps(self.source.timestamps, self.indices)

    def state(
        self,
        target: PatchTarget[Any] | Sequence[PatchTarget[Any]],
        *,
        return_dict: bool = False,
    ) -> np.ndarray | Mapping[PatchTarget[Any], np.ndarray]:
        return self._query(self.source.contacts, target, return_dict=return_dict)

    def confidence(
        self,
        target: PatchTarget[Any] | Sequence[PatchTarget[Any]],
        *,
        return_dict: bool = False,
    ) -> np.ndarray | Mapping[PatchTarget[Any], np.ndarray]:
        return self._query(self.source.confidences, target, return_dict=return_dict)

    def _query(
        self,
        mapping: Mapping[PatchTarget[Any], np.ndarray],
        target: PatchTarget[Any] | Sequence[PatchTarget[Any]],
        *,
        return_dict: bool,
    ) -> np.ndarray | Mapping[PatchTarget[Any], np.ndarray]:
        targets, is_many = normalize_entity_input(target, PatchTarget)
        if not self.indices:
            arrays = [np.empty((0,), dtype=mapping[t].dtype) for t in targets]
        else:
            arrays = [mapping[t][list(self.indices)] for t in targets]
        if not is_many and not return_dict:
            return arrays[0]
        return stack_entity_arrays(targets, arrays, return_dict=return_dict)
