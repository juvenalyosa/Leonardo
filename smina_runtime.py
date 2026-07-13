"""Legacy import shim for :mod:`leonardo.smina_runtime`."""

from leonardo import smina_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
