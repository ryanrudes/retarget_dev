"""A small registry mapping a SMPL-family ``Variant`` to its model class.

Mirrors ``urdf.robots.registry``: the built-in variants are a closed, pre-known :class:`Variant`
``StrEnum`` (so they autocomplete and type-narrow), while the backing map stays string-keyed so extra
variants can still be registered at runtime, and a one-shot :func:`load` resolves a variant plus an
``.npz`` model file straight to a constructed model. The model files themselves are never shipped
(licensing); callers bring their own path, exactly as ``urdf`` needs a ``.urdf``.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from os import PathLike
from typing import Literal, overload

from smpl.io.load import BodyModelData, load_npz
from smpl.models.smpl import SmplModel
from smpl.models.smplx import SmplxModel


class Variant(StrEnum):
    """Identifiers for the built-in, pre-known family of SMPL-family body-model variants.

    The SMPL family is a closed, published set, so the built-ins are an enum (mirroring ``urdf``'s
    ``RobotId``). Because :class:`enum.StrEnum` members *are* ``str``, a ``Variant`` is accepted
    anywhere a variant string is, and the string-keyed registry still admits custom runtime variants
    via :func:`register_model`.
    """

    SMPL = "smpl"
    """The original SMPL body model (24 joints)."""

    SMPLX = "smplx"
    """SMPL-X: SMPL plus articulated hands and a face (55 joints)."""


type AnyModel = SmplModel | SmplxModel
"""Any registered SMPL-family model instance."""

type ModelFactory = Callable[[BodyModelData], AnyModel]
"""A callable that builds a model from loaded body-model arrays (a model class is one)."""

_MODELS: dict[str, ModelFactory] = {
    Variant.SMPL: SmplModel,
    Variant.SMPLX: SmplxModel,
}
_BUILT_IN_VARIANTS = frozenset(str(variant) for variant in Variant)


@overload
def get_model_class(variant: Literal[Variant.SMPL]) -> type[SmplModel]: ...


@overload
def get_model_class(variant: Literal[Variant.SMPLX]) -> type[SmplxModel]: ...


@overload
def get_model_class(variant: str) -> ModelFactory: ...


def get_model_class(variant: str) -> ModelFactory:
    """Return the model factory (the model class) registered for ``variant``.

    Pass a :class:`Variant` member (e.g. ``Variant.SMPLX``) to narrow the return type to the
    concrete model class; a plain string also resolves but returns the broad factory type.
    """
    try:
        return _MODELS[variant]
    except KeyError as exc:
        raise KeyError(f"No model registered for variant {variant!r}.") from exc


@overload
def load(variant: Literal[Variant.SMPL], npz_path: str | PathLike[str]) -> SmplModel: ...


@overload
def load(variant: Literal[Variant.SMPLX], npz_path: str | PathLike[str]) -> SmplxModel: ...


@overload
def load(variant: str, npz_path: str | PathLike[str]) -> AnyModel: ...


def load(variant: str, npz_path: str | PathLike[str]) -> AnyModel:
    """Load an ``.npz`` body-model file and wrap it in the model class for ``variant``.

    Args:
        variant: A registered variant -- a :class:`Variant` member (e.g. ``Variant.SMPLX``, which
            narrows the return type) or a registered variant string.
        npz_path: Path to the SMPL-family ``.npz`` model file (you supply your own).

    Returns:
        The constructed model for ``variant``.
    """
    return get_model_class(variant)(load_npz(npz_path))


def register_model(variant: str, factory: ModelFactory, *, replace: bool = False) -> None:
    """Register a model factory (e.g. a model class) under ``variant`` for runtime lookup.

    The built-in family is the closed :class:`Variant` enum; this adds *extra* variants at runtime
    (e.g. a torch-backed factory, or a research variant) under any non-empty string.

    Args:
        variant: The non-empty variant name to register.
        factory: A callable mapping :class:`~smpl.io.load.BodyModelData` to a model instance.
        replace: If ``False`` (default), raise when ``variant`` is already registered.
    """
    if not variant:
        raise ValueError("Model variant names cannot be empty.")
    if variant in _MODELS and not replace:
        raise ValueError(f"A model is already registered for variant {variant!r}.")
    _MODELS[variant] = factory


def unregister_model(variant: str, *, allow_builtin: bool = False) -> ModelFactory:
    """Remove and return the model factory registered under ``variant``.

    Built-in :class:`Variant` members are protected unless ``allow_builtin`` is set.
    """
    if variant in _BUILT_IN_VARIANTS and not allow_builtin:
        raise ValueError(f"Cannot unregister built-in variant {variant!r}.")
    try:
        return _MODELS.pop(variant)
    except KeyError as exc:
        raise KeyError(f"No model registered for variant {variant!r}.") from exc


def registered_variants() -> frozenset[str]:
    """Return the names of all registered variants (built-in :class:`Variant` members + custom)."""
    return frozenset(_MODELS)
