"""Resolvers that reduce per-named-support contact booleans to a label.

A patch may be in contact with several named supports at once. A
:class:`SupportResolver` collapses, per frame, the set of supports in contact to
a single categorical label (or, for :func:`multi_label`, keeps the whole set):

* :func:`most_confident` -- the support with the highest contact confidence wins
  (the default; integrates all signals, not declared order);
* :func:`priority` -- the first support (in declared order) in contact wins;
* :func:`highest_score` -- alias of :func:`most_confident` (scores are confidences);
* :func:`multi_label` -- keep the full ``frozenset`` of simultaneous contacts.

Resolvers compose fluently: ``priority().where(invalid, "unknown")`` returns a
resolver that emits ``"unknown"`` at the flagged frames and otherwise defers to
``priority()``. These are pure NumPy helpers with no scene/demo/contacts
dependency, so the schema layer (:meth:`retarget.core.schema.patch.Patch.support_state`)
and the contacts layer (:class:`retarget.contacts.state.SupportStateTrack`) share
one implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from retarget.core.types import FloatArray1D, LabelArray, TimeBool

type SupportContacts = Mapping[str, TimeBool]
"""Per-support boolean contact arrays, each shape ``(T,)``, keyed by support name."""

type SupportScores = Mapping[str, FloatArray1D]
"""Per-support contact scores in ``[0, 1]``, each shape ``(T,)``, keyed by support name."""

type ResolveFn = Callable[[SupportContacts, SupportScores], LabelArray]
"""The bare callable a :class:`SupportResolver` wraps: ``(contacts, scores) -> (T,) labels``."""


def num_timesteps(contacts: SupportContacts) -> int:
    """Number of frames implied by a per-support contact mapping (``0`` if empty)."""
    for array in contacts.values():
        return len(array)
    return 0


@dataclass(frozen=True, slots=True)
class SupportResolver:
    """A composable rule mapping per-support contacts to per-frame labels.

    Call it as ``resolver(contacts, scores) -> (T,) LabelArray``. Use
    :meth:`where` to layer an override (e.g. mark dropped-tracking frames
    ``"unknown"``) on top of any resolver.
    """

    fn: ResolveFn

    def __call__(self, contacts: SupportContacts, scores: SupportScores) -> LabelArray:
        return self.fn(contacts, scores)

    def where(self, invalid: TimeBool, label: str = "unknown") -> SupportResolver:
        """Return a resolver that emits ``label`` wherever ``invalid`` is True.

        Args:
            invalid: Per-frame mask, shape ``(T,)``; True frames get ``label``.
            label: Categorical label written at the flagged frames.

        Returns:
            A new :class:`SupportResolver` deferring to ``self`` elsewhere.
        """
        inner = self.fn
        invalid_mask = np.asarray(invalid, dtype=np.bool_)

        def resolve(contacts: SupportContacts, scores: SupportScores) -> LabelArray:
            labels = np.asarray(inner(contacts, scores), dtype=object).copy()
            if invalid_mask.shape != labels.shape:
                raise ValueError(
                    f"invalid mask shape {invalid_mask.shape} does not match label shape {labels.shape}"
                )
            labels[invalid_mask] = label
            return labels

        return SupportResolver(resolve)


def as_resolver(resolver: SupportResolver | ResolveFn) -> SupportResolver:
    """Coerce a bare ``(contacts, scores) -> labels`` callable into a :class:`SupportResolver`."""
    return resolver if isinstance(resolver, SupportResolver) else SupportResolver(resolver)


def priority(order: Sequence[str] | None = None, none_label: str | None = None) -> SupportResolver:
    """Resolver: the first support in ``order`` (or declared order) in contact wins.

    ``none_label`` (default ``None``) is the value where nothing is in contact.
    """

    def resolve(contacts: SupportContacts, scores: SupportScores) -> LabelArray:
        names = list(order) if order is not None else list(contacts)
        labels: LabelArray = np.full(num_timesteps(contacts), none_label, dtype=object)
        for name in reversed(names):
            if name in contacts:
                labels[np.asarray(contacts[name], dtype=np.bool_)] = name
        return labels

    return SupportResolver(resolve)


def highest_score(none_label: str | None = None) -> SupportResolver:
    """Resolver: among supports in contact, the highest contact score wins each frame."""

    def resolve(contacts: SupportContacts, scores: SupportScores) -> LabelArray:
        names = list(contacts)
        count = num_timesteps(contacts)
        if not names:
            return np.full(count, none_label, dtype=object)
        in_contact = np.column_stack([np.asarray(contacts[name], dtype=np.bool_) for name in names])
        score_matrix = np.column_stack(
            [np.asarray(scores.get(name, np.zeros(count)), dtype=np.float64) for name in names]
        )
        masked = np.where(in_contact, score_matrix, -np.inf)
        best = np.asarray(names, dtype=object)[np.argmax(masked, axis=1)]
        return cast(LabelArray, np.where(in_contact.any(axis=1), best, none_label))

    return SupportResolver(resolve)


def most_confident(none_label: str | None = None) -> SupportResolver:
    """Resolver: among supports in contact, the highest contact confidence wins each frame.

    This is the default for ``support_state(...)``. Per-support scores are calibrated
    contact confidences (integrating clearance and motion), so the winner is the support
    the data most supports -- not the first declared. Alias of :func:`highest_score`.
    """
    return highest_score(none_label)


def multi_label() -> SupportResolver:
    """Resolver: keep the full ``frozenset`` of supports in contact each frame."""

    def resolve(contacts: SupportContacts, scores: SupportScores) -> LabelArray:
        names = list(contacts)
        count = num_timesteps(contacts)
        columns = {name: np.asarray(contacts[name], dtype=np.bool_) for name in names}
        out: LabelArray = np.empty(count, dtype=object)
        for i in range(count):
            out[i] = frozenset(name for name in names if columns[name][i])
        return out

    return SupportResolver(resolve)
