"""Per-patch, per-named-support contact state (the multi-contact substrate).

A :class:`SupportStateTrack` stores, for each :class:`PatchTarget`, a boolean
contact array per *named* support (several may be True at once) plus the matching
scores, and -- optionally -- a per-patch ``invalid`` mask flagging frames whose
pose can't be trusted (dropped tracking). The categorical "air / ground /
skateboard / unknown" label is a read-time *view* produced by a resolver:

* :func:`priority` -- the first support (in declared order) in contact wins;
* :func:`highest_score` -- the strongest-scoring support in contact wins;
* :func:`multi_label` -- keep the full set of simultaneous contacts.

By default :meth:`support_state` layers ``priority().where(invalid, unknown_label)``
so flagged frames read as ``"unknown"`` automatically. Pass an explicit
``resolve`` to take full control.

This mirrors :class:`~retarget.demo.contact.ContactTrack` (discrete resampling,
a lazy slice view) but generalizes it from one boolean to many named supports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from retarget.core.support_resolve import (
    ResolveFn,
    SupportContacts,
    SupportResolver,
    SupportScores,
    as_resolver,
    highest_score,
    multi_label,
    priority,
)
from retarget.core.targets import PatchTarget
from retarget.core.types import FloatArray1D, LabelArray, TimeBool
from retarget.demo.resampling import (
    ResampleMethod,
    resample_indices,
    validate_resample_timestamps,
    validate_strictly_increasing_timestamps,
)
from retarget.demo.tracks import Track, TrackView

if TYPE_CHECKING:
    from retarget.core.schema.patch import Patch

# A patch query accepts either a stable PatchTarget or the bound Patch you already
# hold (so reading off the returned track needs no re-fetch and no keys).
type PatchLike = PatchTarget | "Patch[Any]"


def _target_of(target: PatchLike) -> PatchTarget:
    return target if isinstance(target, PatchTarget) else target.target

__all__ = [
    "SupportStateTrack",
    "SupportStateTrackView",
    "SupportContacts",
    "SupportScores",
    "SupportResolver",
    "ResolveFn",
    "priority",
    "highest_score",
    "multi_label",
]

type NestedContacts = Mapping[PatchTarget, SupportContacts]
type NestedScores = Mapping[PatchTarget, SupportScores]
type InvalidMasks = Mapping[PatchTarget, TimeBool]


def _validate_support_arrays(
    contacts: NestedContacts,
    scores: NestedScores,
    num_timesteps: int,
) -> tuple[NestedContacts, NestedScores]:
    frozen_contacts: dict[PatchTarget, SupportContacts] = {}
    for target, per_support in contacts.items():
        inner: dict[str, TimeBool] = {}
        for name, values in per_support.items():
            array = np.asarray(values)
            if array.shape != (num_timesteps,):
                raise ValueError(f"contact array for {target}/{name!r} must have shape ({num_timesteps},)")
            if array.dtype != np.bool_:
                raise TypeError("support contact arrays must have bool dtype")
            inner[name] = cast(TimeBool, array)
        frozen_contacts[target] = MappingProxyType(inner)
    frozen_scores: dict[PatchTarget, SupportScores] = {}
    for target, per_support_scores in scores.items():
        if target not in frozen_contacts:
            raise ValueError(f"scores contains target not present in contacts: {target}")
        inner_scores: dict[str, FloatArray1D] = {}
        for name, score_values in per_support_scores.items():
            if name not in frozen_contacts[target]:
                raise ValueError(f"scores for {target} contains support {name!r} not present in contacts")
            score_array = np.asarray(score_values)
            if score_array.shape != (num_timesteps,):
                raise ValueError(f"score array for {target}/{name!r} must have shape ({num_timesteps},)")
            if not np.issubdtype(score_array.dtype, np.floating):
                raise TypeError("support scores must be floating-point arrays")
            if np.any(score_array < 0) or np.any(score_array > 1):
                raise ValueError("support scores must be in [0, 1]")
            inner_scores[name] = cast(FloatArray1D, score_array)
        frozen_scores[target] = MappingProxyType(inner_scores)
    return frozen_contacts, frozen_scores


def _validate_invalid(invalid: InvalidMasks, num_timesteps: int) -> InvalidMasks:
    frozen: dict[PatchTarget, TimeBool] = {}
    for target, values in invalid.items():
        array = np.asarray(values)
        if array.shape != (num_timesteps,):
            raise ValueError(f"invalid mask for {target} must have shape ({num_timesteps},)")
        if array.dtype != np.bool_:
            raise TypeError("invalid masks must have bool dtype")
        frozen[target] = cast(TimeBool, array)
    return MappingProxyType(frozen)


def _slice_nested(mapping: NestedContacts | NestedScores, indices: Sequence[int]) -> dict[PatchTarget, dict[str, np.ndarray]]:
    index_list = list(indices)
    return {
        target: {name: values[index_list] for name, values in per_support.items()}
        for target, per_support in mapping.items()
    }


def _slice_flat(mapping: InvalidMasks, indices: Sequence[int]) -> dict[PatchTarget, np.ndarray]:
    index_list = list(indices)
    return {target: values[index_list] for target, values in mapping.items()}


class _SupportStateQueryMixin:
    """Shared query/resampling surface for support-state tracks and views."""

    @property
    def timestamps(self) -> np.ndarray:
        raise NotImplementedError

    def _visible_contacts(self) -> NestedContacts:
        raise NotImplementedError

    def _visible_scores(self) -> NestedScores:
        raise NotImplementedError

    def _visible_invalid(self) -> InvalidMasks:
        raise NotImplementedError

    def support_contacts(self, target: PatchLike) -> dict[str, TimeBool]:
        """Per-support boolean contact arrays for ``target`` (the multi-contact truth)."""
        key = _target_of(target)
        visible = self._visible_contacts()
        return dict(visible[key]) if key in visible else {}

    def support_scores(self, target: PatchLike) -> dict[str, FloatArray1D]:
        """Per-support contact scores for ``target``."""
        key = _target_of(target)
        visible = self._visible_scores()
        return dict(visible[key]) if key in visible else {}

    def support_names(self, target: PatchLike) -> tuple[str, ...]:
        """Names of the supports detected for ``target``, in declared order."""
        key = _target_of(target)
        visible = self._visible_contacts()
        return tuple(visible[key]) if key in visible else ()

    def support_state(
        self,
        target: PatchLike,
        *,
        resolve: SupportResolver | ResolveFn | None = None,
        none: str | None = None,
        unknown: str | None = None,
    ) -> LabelArray:
        """Resolve ``target``'s per-support contacts to a per-frame categorical label.

        ``target`` is the ``Patch`` you already hold (or its ``PatchTarget``), so
        reading off the track classify returned needs no attach and no re-fetch.
        Labels are yours: values are the support names from ``classify`` (``None``
        for no contact, unless you pass a ``none`` label). Pass ``unknown="..."``
        to surface detector-flagged untrustworthy frames under that label; omit it
        and those frames read as ``none``. ``resolve`` (e.g. :func:`highest_score`,
        :func:`multi_label`, a ``priority().where(...)`` chain) overrides the
        default :func:`priority` reduction.
        """
        key = _target_of(target)
        contacts = self.support_contacts(key)
        scores = self.support_scores(key)
        if not contacts:
            return cast(LabelArray, np.full(len(self.timestamps), none, dtype=object))
        resolver = priority(none_label=none) if resolve is None else as_resolver(resolve)
        labels = np.asarray(resolver(contacts, scores), dtype=object)
        if unknown is not None:
            invalid = self._visible_invalid().get(key)
            if invalid is not None:
                labels = labels.copy()
                labels[np.asarray(invalid, dtype=np.bool_)] = unknown
        return cast(LabelArray, labels)

    def resample_to(
        self,
        timestamps: np.ndarray,
        *,
        output_timestamps: np.ndarray | None = None,
        method: ResampleMethod | str = ResampleMethod.NEAREST,
    ) -> SupportStateTrack:
        """Discretely resample this support-state track to ``timestamps``."""
        sample_timestamps = validate_resample_timestamps(timestamps)
        if output_timestamps is None:
            result_timestamps = sample_timestamps
        else:
            result_timestamps = validate_resample_timestamps(output_timestamps)
            if len(result_timestamps) != len(sample_timestamps):
                raise ValueError("output_timestamps must have the same length as timestamps")
        indices = resample_indices(
            source_timestamps=self.timestamps,
            target_timestamps=sample_timestamps,
            method=method,
        )
        index_list = [int(i) for i in indices]
        return SupportStateTrack(
            contacts=_slice_nested(self._visible_contacts(), index_list),
            scores=_slice_nested(self._visible_scores(), index_list),
            timestamps=result_timestamps,
            invalid=_slice_flat(self._visible_invalid(), index_list),
        )


@dataclass(frozen=True, slots=True)
class SupportStateTrack(_SupportStateQueryMixin, Track):
    """Per-patch, per-named-support contact state (multi-contact) with scores.

    ``invalid`` holds an optional per-patch ``(T,)`` bool mask of untrustworthy
    frames; surface them by passing ``unknown="..."`` to :meth:`support_state`.
    """

    contacts: NestedContacts
    scores: NestedScores
    timestamps: np.ndarray
    invalid: InvalidMasks = field(default_factory=dict)
    nominal_hz_override: float | None = None

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        object.__setattr__(self, "timestamps", timestamps)
        if timestamps.ndim != 1:
            raise ValueError("timestamps must be a 1D array")
        validate_strictly_increasing_timestamps(timestamps)
        frozen_contacts, frozen_scores = _validate_support_arrays(
            self.contacts, self.scores, len(timestamps)
        )
        object.__setattr__(self, "contacts", MappingProxyType(dict(frozen_contacts)))
        object.__setattr__(self, "scores", MappingProxyType(dict(frozen_scores)))
        object.__setattr__(self, "invalid", _validate_invalid(self.invalid, len(timestamps)))

    @classmethod
    def _view_type(cls) -> type[SupportStateTrackView]:
        return SupportStateTrackView

    def _visible_contacts(self) -> NestedContacts:
        return self.contacts

    def _visible_scores(self) -> NestedScores:
        return self.scores

    def _visible_invalid(self) -> InvalidMasks:
        return self.invalid

    def select_indices(self, indices: Sequence[int]) -> SupportStateTrack:
        """Return a materialized track at the given native-time indices."""
        index_list = [int(i) for i in indices]
        return SupportStateTrack(
            contacts=_slice_nested(self.contacts, index_list),
            scores=_slice_nested(self.scores, index_list),
            timestamps=self.timestamps[index_list],
            invalid=_slice_flat(self.invalid, index_list),
            nominal_hz_override=self.nominal_hz_override,
        )


@dataclass(frozen=True, slots=True)
class SupportStateTrackView(_SupportStateQueryMixin, TrackView[SupportStateTrack]):
    """Sliced view into a :class:`SupportStateTrack`."""

    @property
    def timestamps(self) -> np.ndarray:
        return self.source.timestamps[list(self.indices)]

    def _visible_contacts(self) -> NestedContacts:
        return _slice_nested(self.source.contacts, list(self.indices))

    def _visible_scores(self) -> NestedScores:
        return _slice_nested(self.source.scores, list(self.indices))

    def _visible_invalid(self) -> InvalidMasks:
        return _slice_flat(self.source.invalid, list(self.indices))
