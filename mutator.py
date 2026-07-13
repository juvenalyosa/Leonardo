"""Legacy import shim for :mod:`leonardo.mutator`."""
import sys
from leonardo import mutator as _impl
sys.modules[__name__] = _impl
