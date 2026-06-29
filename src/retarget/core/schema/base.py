"""Structural root + reflection kernel for authored schema containers.

The five schema bases (``Markers``/``Patches``/``Segments``/``Subjects``/``Tracks``) are empty
frozen+slotted dataclasses subclassing :class:`_Schema`; concrete user schemas subclass those with
typed attribute fields, and attribute access (``markers.heel``) projects the field type -- the typed
deep chain, no codegen. These helpers reflect over the dataclass fields so the binding layer and the
runtime walkers stay generic over any concrete schema.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class _Schema:
    """Structural root of every authored schema container.

    Being a dataclass itself, its bound supplies ``__dataclass_fields__``, so ``fields(x: SchemaT)``
    type-checks with no ignore for any ``SchemaT: _Schema``.
    """

    # --- transitional bridge ---------------------------------------------------------------------
    # Keeps un-migrated ``schema["x"]`` / iteration / ``.items()`` working during the staged
    # migration from subscript to attribute access. Removed once every access site is
    # attribute-based -- after which a stray subscript is a hard mypy + runtime error.
    def __getitem__(self, name: str) -> Any:
        return _schema_get(self, name)

    def __iter__(self) -> Iterator[str]:
        return iter(_schema_fields(self))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in _schema_fields(self)

    def items(self) -> Iterator[tuple[str, Any]]:
        return _schema_items(self)

    def values(self) -> Iterator[Any]:
        return _schema_values(self)

    def keys(self) -> tuple[str, ...]:
        return _schema_fields(self)


def _schema_items(schema: _Schema) -> Iterator[tuple[str, Any]]:
    """The schema's ``(field_name, value)`` pairs, in declaration order."""
    for field in fields(schema):
        yield field.name, getattr(schema, field.name)


def _schema_values(schema: _Schema) -> Iterator[Any]:
    """The schema's field values, in declaration order."""
    for field in fields(schema):
        yield getattr(schema, field.name)


def _schema_fields(schema: _Schema) -> tuple[str, ...]:
    """The schema's field names, in declaration order."""
    return tuple(field.name for field in fields(schema))


def _schema_get(schema: _Schema, name: str) -> Any:
    """Dynamic string lookup of one field value (for IO / name-driven walkers); raises ``KeyError``."""
    if name not in _schema_fields(schema):
        raise KeyError(name)
    return getattr(schema, name)


def _rebuild_schema[SchemaT: _Schema](schema: SchemaT, bind: Callable[[str, Any], Any]) -> SchemaT:
    """Rebuild a schema dataclass of the same concrete type, mapping each field through ``bind``.

    The generic equivalent of the old ``{name: bound}`` dict construction: ``type(schema)`` is the
    concrete subclass, so the rebuilt schema keeps its precise ``SchemaT`` type for the deep chain.
    """
    return type(schema)(**{name: bind(name, value) for name, value in _schema_items(schema)})
