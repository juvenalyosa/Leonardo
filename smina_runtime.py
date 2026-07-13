"""Legacy import shim for :mod:`leonardo.smina_runtime`."""

import sys
from leonardo import smina_runtime as _impl
sys.modules[__name__] = _impl
