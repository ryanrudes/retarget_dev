from retarget.core import Markers, Marker


class LeftShoeMarkers(Markers):
    heel: Marker
    toe: Marker

    heel_inner_1: Marker
    heel_inner_2: Marker
    heel_outer_1: Marker
    heel_outer_2: Marker

    toe_inner: Marker
    toe_outer: Marker

    toe_grid_1: Marker
    toe_grid_2: Marker
    toe_grid_3: Marker
    toe_grid_4: Marker

    sole_inner: Marker
    sole_outer: Marker

    plane_rear: Marker
    plane_inner: Marker
    plane_outer: Marker


class SkateboardMarkers(Markers):
    front_center_board: Marker
    front_left_board: Marker
    rear_left_board: Marker
    rear_right_board: Marker

    front_left_pole_left: Marker
    front_left_pole_right: Marker
    front_right_pole_left: Marker
    front_right_pole_right: Marker
    rear_left_pole_left: Marker
    rear_left_pole_right: Marker
    rear_right_pole_left: Marker
    rear_right_pole_right: Marker

    front_left_wheel: Marker
    front_right_wheel: Marker
    rear_left_wheel: Marker
    rear_right_wheel: Marker
