"""Legacy import shim for :mod:`leonardo.ligand_vib_refine_runtime`."""
from leonardo import ligand_vib_refine_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
