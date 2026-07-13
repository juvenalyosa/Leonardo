"""Legacy import shim for :mod:`leonardo.leadopt_runtime`."""
from leonardo import leadopt_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
