"""Legacy import shim for :mod:`leonardo.ui_layout_runtime`."""
import sys

from leonardo import ui_layout_runtime as _impl

sys.modules[__name__] = _impl
