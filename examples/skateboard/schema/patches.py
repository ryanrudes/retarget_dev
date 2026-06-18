from retarget.core import Patches, Patch, RectangularRegion


class LeftShoePatches(Patches):
    sole: Patch[RectangularRegion]


class SkateboardPatches(Patches):
    surface: Patch[RectangularRegion]
