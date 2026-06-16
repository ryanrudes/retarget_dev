# Enums

This library uses enumerations as type-safe alternatives to strings or integers for all vocabularies, such as marker names, segment names, subject names, patch names, etc. It is not always possible to resolve types perfectly when using enum members are arguments, so some methods accept either some structure or the enum symbol which identifies it.

## `NameId`
All vocabularies are some descendent of `NameId`, which is a `StrEnum`. `NameId` assigns an `index` to each item in the order in which they are defined, and provides an alias attribute `label` for the string member value. It also provides helpers `size()`, `members()`, and `labels()`. Classes which inherit from `NameId` are mostly for the sake of readable naming and intuitive typing. Those classes are the following:
- `SubjectId`: Vocabulary for the subjects which make up a scene.
- `SegmentId`: Vocabulary for the segments which make up a subject.
- `MarkerId`: Vocabulary for the markers which make up a segment.
- `PatchId`: Vocabulary for the contact patches which make up a segment.
- `TrackId`: Base class for user-defined demonstration track identifiers.

## Other Enums

There are other `StrEnum` classes defined, which do not inherit from `NameId`.
- `MarkerRole`: `TRACKING`, `CALIBRATION`, and `TRACKING_AND_CALIBRATION`.
- `RotationFormat`: `MATRIX`, `QUATERNION_XYZW`, `QUATERNION_WXYZ`, and `ROTVEC`.
- `PoseFormat`: `RIGID_TRANSFORM`, `MATRIX_4X4`, `TRANSLATION_QUATERNION_XYZW`, `TRANSLATION_ROTATION_MATRIX`.
- `QuaternionOrder`: `WXYZ` and `XYZW`.