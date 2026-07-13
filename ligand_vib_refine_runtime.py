"""Legacy import shim for :mod:`leonardo.ligand_vib_refine_runtime`."""
import sys
from leonardo import ligand_vib_refine_runtime as _impl
sys.modules[__name__] = _impl
