"""Legacy import shim for :mod:`leonardo.run_ligand_vib_refine`."""
from leonardo import run_ligand_vib_refine as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
