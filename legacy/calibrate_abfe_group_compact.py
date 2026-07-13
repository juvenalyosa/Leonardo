"""Legacy import shim for :mod:`leonardo.calibrate_abfe_group_compact`."""
import sys
from leonardo import calibrate_abfe_group_compact as _impl
sys.modules[__name__] = _impl
if __name__ == "__main__":
    raise SystemExit(_impl.main())
