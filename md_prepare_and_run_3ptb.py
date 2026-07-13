"""Legacy import shim for :mod:`leonardo.md_prepare_and_run_3ptb`."""
from leonardo import md_prepare_and_run_3ptb as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
