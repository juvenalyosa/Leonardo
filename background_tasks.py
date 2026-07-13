"""Legacy import shim for :mod:`leonardo.background_tasks`."""
from leonardo import background_tasks as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
