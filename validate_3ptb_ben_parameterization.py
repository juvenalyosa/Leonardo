"""Legacy import shim for :mod:`leonardo.validate_3ptb_ben_parameterization`."""
from leonardo import validate_3ptb_ben_parameterization as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
