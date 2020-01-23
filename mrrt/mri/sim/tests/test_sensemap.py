from itertools import product
import numpy as np
import pytest

from mrrt.mri import sensemap_sim

from mrrt.utils import config

all_xp = [np]
if config.have_cupy:
    import cupy

    all_xp += [cupy]


@pytest.mark.parametrize(
    "xp, dtype", product(all_xp, [np.complex64, np.complex128])
)
def test_sensemap_sim_2d(xp, dtype):
    shape = (64, 64)
    spacings = tuple(240 / s for s in shape)
    ncoil = 8
    smap = sensemap_sim(shape=shape, spacings=spacings, ncoil=8, dtype=dtype)
    assert smap.shape == shape + (ncoil,)
    assert smap.dtype == dtype
