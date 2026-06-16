# Axes

## `CoordinateAxis`
`CoordinateAxis` is an integer enumeration of the standard cartesian axes X (0), Y (1), and Z (2). `CoordinateAxis` implements the + and - prefix unitary operators, each of which return a `SignedAxis`.

## `SemanticAxis`
`SemanticAxis` is a string enumeration of symbolic axes `RIGHT`, `FORWARD`, and `UP`. It supports the + and - prefix unitary operators, as well as multiplication, which returns a `SemanticAxisTranslation`.

## `SemanticAxisTranslation`
`SemanticAxisTranslation` is a representation of a translation by some distance along a semantic axis. For example "3 units along the right axis". It supports multiplication and prefix unitary negation. It also provides a `resolve(segment: AxisResolvable) -> Vec3` function, which takes a protocol `AxisResolvable` (anything that can resolve semantic axes into local-frame vectors) and does just that using its own axis. Some examples of `AxisResolvable` types are `MarkerTranslation`, `BodyFrameTranslation`, and `SemanticAxisTranslation`.

## `SignedAxis`
A `SignedAxis` is a dataclass represented by a `CoordinateAxis` and an integer sign. It can represent concepts such as -X, +Y, etc.

`SignedAxis` supports the + and - prefix unitary operators, which overwrite the sign. It also provides `flip()`, which inverts the sign. Finally, the `vector()` function outputs the 3D unit vector representation.

## `AxisConvention`
An `AxisConvention` is a mapping from semantic axes to the coordinate axes. For instance, to define a Z-up convention:

```python
Z_UP_AXES = AxisConvention({
    SemanticAxis.RIGHT:   -CoordinateAxis.Y,
    SemanticAxis.FORWARD: +CoordinateAxis.X,
    SemanticAxis.UP:      +CoordinateAxis.Z,
})
```

## Built-Ins

The built-in axis conventions are `Z_UP_AXES` and `Y_UP_AXES`. Additionally, there are `MUJOCO_AXES` and `ISAAC_AXES`, which are both just aliases for `Z_UP_AXES`.