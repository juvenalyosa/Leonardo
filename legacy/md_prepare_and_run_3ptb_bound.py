"""Legacy import shim for :mod:`leonardo.md_prepare_and_run_3ptb_bound`."""
import sys

from leonardo import md_prepare_and_run_3ptb_bound as _impl

sys.modules[__name__] = _impl
if __name__ == "__main__":
    raise SystemExit(_impl.main())
