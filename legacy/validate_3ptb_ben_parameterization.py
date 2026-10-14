"""Legacy import shim for :mod:`leonardo.validate_3ptb_ben_parameterization`."""
import sys

from leonardo import validate_3ptb_ben_parameterization as _impl

sys.modules[__name__] = _impl
if __name__ == "__main__":
    raise SystemExit(_impl.main())
