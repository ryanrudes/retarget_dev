from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from typing import Literal, overload

from smpl.io.load import BodyModelData, load_npz
from smpl.models.smpl import SmplModel
from smpl.models.smplx import SmplxModel

"""A small runtime registry mapping a variant name to its SMPL-family model class.

Mirrors the spirit of ``urdf.robots.registry``: built-in variants are registered up front, extra
variants can be registered at runtime, and a one-shot :func:`load` resolves a variant name plus an
``.npz`` model file straight to a constructed model. The model files themselves are never shipped
(licensing); callers bring their own path, exactly as ``urdf`` needs a ``.urdf``.
"""

type AnyModel = SmplModel | SmplxModel
"""Any registered SMPL-family model instance."""

type ModelFactory = Callable[[BodyModelData], AnyModel]
"""A callable that builds a model from loaded body-model arrays (a model class is one)."""

_MODELS: dict[str, ModelFactory] = {
    "smpl": SmplModel,
    "smplx": SmplxModel,
}
_BUILT_IN_VARIANTS = frozenset(_MODELS)


@overload
def get_model_class(variant: Literal["smpl"]) -> type[SmplModel]: ...


@overload
def get_model_class(variant: Literal["smplx"]) -> type[SmplxModel]: ...


@overload
def get_model_class(variant: str) -> ModelFactory: ...


def get_model_class(variant: str) -> ModelFactory:
    """Return the model factory (the model class) registered for ``variant``."""
    try:
        return _MODELS[variant]
    except KeyError as exc:
        raise KeyError(f"No model registered for variant {variant!r}.") from exc


@overload
def load(variant: Literal["smpl"], npz_path: str | PathLike[str]) -> SmplModel: ...


@overload
def load(variant: Literal["smplx"], npz_path: str | PathLike[str]) -> SmplxModel: ...


@overload
def load(variant: str, npz_path: str | PathLike[str]) -> AnyModel: ...


def load(variant: str, npz_path: str | PathLike[str]) -> AnyModel:
    """Load an ``.npz`` body-model file and wrap it in the model class for ``variant``.

    Args:
        variant: A registered variant name, e.g. ``"smpl"`` or ``"smplx"``.
        npz_path: Path to the SMPL-family ``.npz`` model file (you supply your own).

    Returns:
        The constructed model for ``variant``.
    """
    return get_model_class(variant)(load_npz(npz_path))


def register_model(variant: str, factory: ModelFactory, *, replace: bool = False) -> None:
    """Register a model factory (e.g. a model class) under ``variant`` for runtime lookup.

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

    Built-in variants (``"smpl"``, ``"smplx"``) are protected unless ``allow_builtin`` is set.
    """
    if variant in _BUILT_IN_VARIANTS and not allow_builtin:
        raise ValueError(f"Cannot unregister built-in variant {variant!r}.")
    try:
        return _MODELS.pop(variant)
    except KeyError as exc:
        raise KeyError(f"No model registered for variant {variant!r}.") from exc


def registered_variants() -> frozenset[str]:
    """Return the names of all registered variants."""
    return frozenset(_MODELS)
