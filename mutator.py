"""Legacy import shim for :mod:`leonardo.mutator`."""
from leonardo import mutator as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
