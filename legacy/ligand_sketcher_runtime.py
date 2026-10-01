"""Legacy import shim for :mod:`leonardo.ligand_sketcher_runtime`."""
import sys
from leonardo import ligand_sketcher_runtime as _impl
sys.modules[__name__] = _impl
