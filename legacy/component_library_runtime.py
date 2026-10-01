"""Legacy import shim for :mod:`leonardo.component_library_runtime`."""
import sys
from leonardo import component_library_runtime as _impl
sys.modules[__name__] = _impl
