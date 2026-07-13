"""Legacy import shim for :mod:`leonardo.report_scientific_readiness`."""
import sys
from leonardo import report_scientific_readiness as _impl
sys.modules[__name__] = _impl
