"""Legacy import shim for :mod:`leonardo.config`."""
import sys
from leonardo import config as _impl
sys.modules[__name__] = _impl
