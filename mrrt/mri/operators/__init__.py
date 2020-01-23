"""System matrix objects (operators) for Cartesian and Non-Cartesian MRI.

"""
from numpy.testing import Tester

# Cartesian MRI Operator (includes multi-coil and fieldmap processing)
from ._MRI_Cartesian import MRI_Cartesian  # noqa

# NonUniform FFT operator
try:
    # Basic NUFFT Operator
    from ._NUFFT import NUFFT_Operator  # noqa

    # MRI Operator (includes multi-coil and fieldmap processing)
    from ._MRI import MRI_Operator  # noqa
except (ImportError, NameError):
    print(
        "NUFFT_Operator & MRI_OPerator not available (requires mrrt.nufft "
        "and mrrt.operators packages)"
    )

test = Tester().test
