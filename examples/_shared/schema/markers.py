from dataclasses import dataclass

from retarget.core import Markers, Marker


@dataclass(frozen=True, slots=True)
class LeftShoeMarkers(Markers):
    # Each marker is declared once, here, as a field default (its Vicon ``mocap_name``). Authoring code
    # then constructs ``LeftShoeMarkers()`` and refers to the symbols as ``m.heel`` -- no second listing.
    heel: Marker = Marker("heel")
    toe: Marker = Marker("toe")

    heel_inner_1: Marker = Marker("heel_inner_1")
    heel_inner_2: Marker = Marker("heel_inner_2")
    heel_outer_1: Marker = Marker("heel_outer_1")
    heel_outer_2: Marker = Marker("heel_outer_2")

    toe_inner: Marker = Marker("toe_inner")
    toe_outer: Marker = Marker("toe_outer")

    toe_grid_1: Marker = Marker("toe_grid_1")
    toe_grid_2: Marker = Marker("toe_grid_2")
    toe_grid_3: Marker = Marker("toe_grid_3")
    toe_grid_4: Marker = Marker("toe_grid_4")

    sole_inner: Marker = Marker("sole_inner")
    sole_outer: Marker = Marker("sole_outer")

    plane_rear: Marker = Marker("plane_rear")
    plane_inner: Marker = Marker("plane_inner")
    plane_outer: Marker = Marker("plane_outer")


@dataclass(frozen=True, slots=True)
class BalanceBoardMarkers(Markers):
    # mocap_name case differs from the field name, so each default is explicit.
    surface1: Marker = Marker("Surface1")
    surface2: Marker = Marker("Surface2")
    surface3: Marker = Marker("Surface3")
    surface4: Marker = Marker("Surface4")
    edge1: Marker = Marker("Edge1")
    edge2: Marker = Marker("Edge2")
