"""Legacy import shim for :mod:`leonardo.train_abfe_steering_policy`."""
import sys

from leonardo import train_abfe_steering_policy as _impl

sys.modules[__name__] = _impl
if __name__ == "__main__":
    raise SystemExit(_impl.main())
