# smpl

A clean, typed, standalone library for loading SMPL-family body models and
running forward kinematics. Vendored alongside `urdf`; `retarget` consumes it
through the `BodyModel` protocol.

Model files (`.npz`/`.pkl`) are registration-gated at the MPI sites and are
**not** redistributed here — bring your own.
