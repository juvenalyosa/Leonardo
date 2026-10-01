"""Legacy import shim for :mod:`leonardo.pka_runtime`."""
import sys
from leonardo import pka_runtime as _impl
sys.modules[__name__] = _impl
