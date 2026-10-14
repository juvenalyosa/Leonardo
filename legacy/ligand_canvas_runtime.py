"""Legacy import shim for :mod:`leonardo.ligand_canvas_runtime`."""
import sys

from leonardo import ligand_canvas_runtime as _impl

sys.modules[__name__] = _impl
