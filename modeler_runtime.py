"""Legacy import shim for :mod:`leonardo.modeler_runtime`."""
from leonardo import modeler_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
