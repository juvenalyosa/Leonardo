"""Legacy import shim for :mod:`leonardo.train_abfe_steering_policy`."""
from leonardo import train_abfe_steering_policy as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
