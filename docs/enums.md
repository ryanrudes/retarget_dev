# Enums

Scene identity (subjects, segments, markers, patches, tracks) is expressed with
authored **string** names, not identifier enums. There is no `NameId`,
`SubjectId`, `SegmentId`, `MarkerId`, `PatchId`, or `TrackId`. See
`handles_and_targets.md` for the identity model.

The library keeps a small set of **value** enums (all `StrEnum`):

- `MarkerRole`: `TRACKING`, `CALIBRATION`, `TRACKING_AND_CALIBRATION`.
- `RotationFormat`: `MATRIX`, `QUATERNION_XYZW`, `QUATERNION_WXYZ`, `ROTVEC`.
- `PoseFormat`: `RIGID_TRANSFORM`, `MATRIX_4X4`, `TRANSLATION_QUATERNION_XYZW`,
  `TRANSLATION_ROTATION_MATRIX`.
- `QuaternionOrder`: `WXYZ`, `XYZW`.
- `ResampleMethod` (in `retarget.demo.resampling`): `NEAREST`, `PREVIOUS`.
