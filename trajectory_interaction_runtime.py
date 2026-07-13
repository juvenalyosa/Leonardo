"""Legacy import shim for :mod:`leonardo.trajectory_interaction_runtime`."""
from leonardo import trajectory_interaction_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
