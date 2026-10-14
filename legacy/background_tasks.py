"""Legacy import shim for :mod:`leonardo.background_tasks`."""
import sys

from leonardo import background_tasks as _impl

sys.modules[__name__] = _impl
