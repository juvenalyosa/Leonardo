"""Legacy import shim for :mod:`leonardo.run_ligand_vib_refine`."""
import sys

from leonardo import run_ligand_vib_refine as _impl

sys.modules[__name__] = _impl
if __name__ == "__main__":
    raise SystemExit(_impl.main())
