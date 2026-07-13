"""Legacy import shim for :mod:`leonardo.bioseq_runtime`."""
from leonardo import bioseq_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
