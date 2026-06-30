"""Unit tests for the schema reflection kernel (``core/schema/base.py``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from retarget.core.schema.base import (
    _rebuild_schema,
    _schema_fields,
    _schema_get,
    _schema_items,
    _schema_values,
    _Schema,
)


@dataclass(frozen=True, slots=True)
class _Toy(_Schema):
    a: int
    b: int


@dataclass(frozen=True, slots=True)
class _Empty(_Schema):
    pass


def test_schema_items_values_fields_in_declaration_order() -> None:
    toy = _Toy(a=1, b=2)
    assert list(_schema_items(toy)) == [("a", 1), ("b", 2)]
    assert list(_schema_values(toy)) == [1, 2]
    assert _schema_fields(toy) == ("a", "b")
    assert _schema_fields(_Empty()) == ()


def test_rebuild_schema_preserves_concrete_type_and_passes_name_value() -> None:
    seen: list[tuple[str, Any]] = []

    def record(name: str, value: Any) -> Any:
        seen.append((name, value))
        return value * 10

    rebuilt = _rebuild_schema(_Toy(a=1, b=2), record)
    assert isinstance(rebuilt, _Toy)
    assert rebuilt == _Toy(a=10, b=20)
    assert seen == [("a", 1), ("b", 2)]
    # an empty schema rebuilds to an equal empty instance (no fields visited)
    assert _rebuild_schema(_Empty(), record) == _Empty()


def test_schema_get_returns_field_or_raises_keyerror() -> None:
    toy = _Toy(a=1, b=2)
    assert _schema_get(toy, "a") == 1
    with pytest.raises(KeyError):
        _schema_get(toy, "missing")


def test_schema_is_not_subscriptable_after_bridge_removal() -> None:
    # Stage 3 removed the transitional bridge: a stray schema subscript is now a hard error,
    # which is exactly what locks the attribute-symbol surface.
    toy = _Toy(a=1, b=2)
    with pytest.raises(TypeError):
        _ = toy["a"]  # type: ignore[index]  # deliberately exercising the removed bridge
