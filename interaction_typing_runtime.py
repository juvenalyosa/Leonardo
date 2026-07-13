"""Legacy import shim for :mod:`leonardo.interaction_typing_runtime`."""
import sys
from leonardo import interaction_typing_runtime as _impl
sys.modules[__name__] = _impl
