"""Legacy import shim for :mod:`leonardo.compare_abfe_vs_docking`."""
from leonardo import compare_abfe_vs_docking as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
