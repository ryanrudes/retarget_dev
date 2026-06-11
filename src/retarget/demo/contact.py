"""Contact tracks for discrete patch-level contact state.

A :class:`ContactTrack` stores boolean contact state over time for scene-level
:class:`~retarget.core.targets.PatchTarget` keys. Optional confidence arrays
can be stored alongside the boolean contact arrays using the same target keys.

A :class:`ContactTrackView` is a cheap indexed view over a source
``ContactTrack``. It does not copy data until a query or resampling operation
asks for the visible arrays.

The private ``_ContactQueryMixin`` keeps the public query surface shared
between full tracks and views. Concrete classes only define which contact and
confidence arrays are visible; the mixin implements ``state(...)``,
``confidence(...)``, and discrete ``resample_to(...)`` once.

Contact resampling is intentionally discrete. It delegates timestamp/index
policy to :mod:`retarget.demo.resampling` and returns a new ``ContactTrack``
on the requested timestamps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from retarget.core.targets import PatchTarget
from retarget.demo._query_utils import (
    normalize_entity_input,
    slice_timestamps,
    stack_entity_arrays,
)
from retarget.demo.resampling import (
    ResampleMethod,
    resample_indices,
    validate_resample_timestamps,
    validate_strictly_increasing_timestamps,
)
from retarget.demo.tracks import Track, TrackView


class _ContactQueryMixin:
    """Shared query/resampling surface for contact tracks and views."""

    @property
    def timestamps(self) -> np.ndarray:
        raise NotImplementedError

    def _visible_contacts(self) -> Mapping[PatchTarget[Any], np.ndarray]:
        raise NotImplementedError

    def _visible_confidences(self) -> Mapping[PatchTarget[Any], np.ndarray]:
        raise NotImplementedError

    def state(
        self,
        target: PatchTarget[Any] | Sequence[PatchTarget[Any]],
        *,
        return_dict: bool = False,
    ) -> np.ndarray | Mapping[PatchTarget[Any], np.ndarray]:
        return _query_contact_mapping(
            self._visible_contacts(),
            target,
            return_dict=return_dict,
        )

    def confidence(
        self,
        target: PatchTarget[Any] | Sequence[PatchTarget[Any]],
        *,
        return_dict: bool = False,
    ) -> np.ndarray | Mapping[PatchTarget[Any], np.ndarray]:
        return _query_contact_mapping(
            self._visible_confidences(),
            target,
            return_dict=return_dict,
        )

    def resample_to(
        self,
        timestamps: np.ndarray,
        *,
        method: ResampleMethod | str = ResampleMethod.NEAREST,
    ) -> ContactTrack:
        """Return this contact track/view resampled onto requested timestamps."""
        target_timestamps = validate_resample_timestamps(timestamps)
        indices = resample_indices(
            source_timestamps=self.timestamps,
            target_timestamps=target_timestamps,
            method=method,
        )
        return _resampled_contact_track(
            timestamps=target_timestamps,
            contacts=self._visible_contacts(),
            confidences=self._visible_confidences(),
            indices=indices,
        )


@dataclass(frozen=True, slots=True)
class ContactTrack(_ContactQueryMixin, Track):
    """Time-varying contact state keyed by scene-level patch targets."""

    contacts: Mapping[PatchTarget[Any], np.ndarray]
    timestamps: np.ndarray
    confidences: Mapping[PatchTarget[Any], np.ndarray] = field(default_factory=dict)
    nominal_hz_override: float | None = None

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        object.__setattr__(self, "timestamps", timestamps)
        if timestamps.ndim != 1:
            raise ValueError("timestamps must be a 1D array")
        validate_strictly_increasing_timestamps(timestamps)
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

    @classmethod
    def _view_type(cls) -> type[ContactTrackView]:
        return ContactTrackView

    def _visible_contacts(self) -> Mapping[PatchTarget[Any], np.ndarray]:
        return self.contacts

    def _visible_confidences(self) -> Mapping[PatchTarget[Any], np.ndarray]:
        return self.confidences


@dataclass(frozen=True, slots=True)
class ContactTrackView(_ContactQueryMixin, TrackView[ContactTrack]):
    """Sliced view into a :class:`ContactTrack`."""

    @property
    def timestamps(self) -> np.ndarray:
        return slice_timestamps(self.source.timestamps, self.indices)

    def _visible_contacts(self) -> Mapping[PatchTarget[Any], np.ndarray]:
        return _slice_contact_mapping(self.source.contacts, self.indices)

    def _visible_confidences(self) -> Mapping[PatchTarget[Any], np.ndarray]:
        return _slice_contact_mapping(self.source.confidences, self.indices)


def _slice_contact_mapping(
    mapping: Mapping[PatchTarget[Any], np.ndarray],
    indices: tuple[int, ...],
) -> Mapping[PatchTarget[Any], np.ndarray]:
    index_list = list(indices)
    return {target: values[index_list] for target, values in mapping.items()}


def _resampled_contact_track(
    *,
    timestamps: np.ndarray,
    contacts: Mapping[PatchTarget[Any], np.ndarray],
    confidences: Mapping[PatchTarget[Any], np.ndarray],
    indices: np.ndarray,
) -> ContactTrack:
    return ContactTrack(
        timestamps=timestamps,
        contacts={target: values[indices] for target, values in contacts.items()},
        confidences={
            target: values[indices] for target, values in confidences.items()
        },
    )


def _query_contact_mapping(
    mapping: Mapping[PatchTarget[Any], np.ndarray],
    target: PatchTarget[Any] | Sequence[PatchTarget[Any]],
    *,
    return_dict: bool,
) -> np.ndarray | Mapping[PatchTarget[Any], np.ndarray]:
    targets, is_many = normalize_entity_input(target, PatchTarget)
    arrays = [mapping[t] for t in targets]
    if not is_many and not return_dict:
        return arrays[0]
    return stack_entity_arrays(targets, arrays, return_dict=return_dict)
