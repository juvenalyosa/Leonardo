"""Legacy import shim for :mod:`leonardo.compare_abfe_vs_docking`."""
import sys

from leonardo import compare_abfe_vs_docking as _impl

sys.modules[__name__] = _impl
if __name__ == "__main__":
    raise SystemExit(_impl.main())
