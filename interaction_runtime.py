"""Legacy import shim for :mod:`leonardo.interaction_runtime`."""
from leonardo import interaction_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
