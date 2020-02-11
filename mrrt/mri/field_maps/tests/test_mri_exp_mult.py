import itertools

import numpy as np
import pytest

from mrrt.mri import mri_exp_mult
from mrrt.utils import config

all_xp = [np]
if config.have_cupy:
    import cupy

    all_xp += [cupy]

__all__ = ["test_mri_exp_mult"]


@pytest.mark.parametrize(
    "xp, dtype", itertools.product(all_xp, [np.float32, np.float64])
)
def test_mri_exp_mult(xp, dtype):
    """
    test_mri_exp_mult
    """
    L = 1000
    N = 3000
    M = 200
    rstate = xp.random.RandomState(0)
    A = rstate.randn(N, L).astype(dtype)
    A = A + 1j * rstate.randn(N, L).astype(dtype)
    ur = rstate.randn(N).astype(dtype)
    ui = rstate.randn(N).astype(dtype)
    vr = rstate.randn(M).astype(dtype)
    vi = rstate.randn(M).astype(dtype)
    u = ur.astype(dtype) + 1j * ui.astype(dtype)
    v = vr.astype(dtype) + 1j * vi.astype(dtype)

    # Test with ur, v
    y = mri_exp_mult(A, ur, v)
    assert y.dtype == np.promote_types(dtype, np.complex64)

    # Repeat with u, vr
    y = mri_exp_mult(A, u, vr)
    assert y.dtype == np.promote_types(dtype, np.complex64)

    return
