from retarget.core import Patches, Patch, RectangularRegion


class LeftShoePatches(Patches):
    sole: Patch[RectangularRegion]


class RightShoePatches(Patches):
    pass


class BalanceBoardPatches(Patches):
    surface: Patch[RectangularRegion]