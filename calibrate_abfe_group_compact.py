"""Legacy import shim for :mod:`leonardo.calibrate_abfe_group_compact`."""
from leonardo import calibrate_abfe_group_compact as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
