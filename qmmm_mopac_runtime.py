"""Legacy import shim for :mod:`leonardo.qmmm_mopac_runtime`."""
import sys
from leonardo import qmmm_mopac_runtime as _impl
sys.modules[__name__] = _impl
