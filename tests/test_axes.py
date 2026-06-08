def test_axis_convention() -> None:
    from retarget.core import Z_UP_AXES, SemanticAxis

    assert Z_UP_AXES.vector(SemanticAxis.UP).shape == (3,)