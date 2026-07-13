"""Legacy import shim for :mod:`leonardo.qmmm_mopac_runtime`."""
from leonardo import qmmm_mopac_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
