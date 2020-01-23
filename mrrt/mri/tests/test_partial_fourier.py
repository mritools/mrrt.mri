from itertools import product

import numpy as np
import pytest

from mrrt.mri.sim import mri_object_2d, mri_object_3d
from mrrt.mri import mri_partial_fourier_nd
from mrrt.utils import config, ifftnc, ImageGeometry

all_xp = [np]
if config.have_cupy:
    import cupy

    all_xp += [cupy]


# TODO: add test case with non-zero phase


@pytest.mark.parametrize(
    "xp, dtype", product(all_xp, [np.complex64, np.complex128])
)
def test_partial_fourier_2d(xp, dtype):
    shape = (256, 192)
    ig = ImageGeometry(shape, distances=(1, 1), offsets="dsp")
    obj = mri_object_2d(ig.fov)
    coords = ig.fgrid()

    # fully-sampled k-space
    kspace_full = xp.asarray(obj.kspace(*coords), dtype=dtype)

    # partial-Fourier k-space
    pf_fractions = (0.6, 0.7)
    nkeep = [int(f * s) for f, s in zip(pf_fractions, shape)]
    pf_mask = xp.zeros(shape, dtype=xp.bool)
    pf_mask[: nkeep[0], : nkeep[1]] = 1
    kspace_pf = kspace_full[: nkeep[0], : nkeep[1]]

    # direct reconstruction using zero-filled k-space
    direct_recon = ifftnc(kspace_full * pf_mask)

    # partial Fourier reconstruction
    pf_recon = mri_partial_fourier_nd(kspace_pf, pf_mask)
    # dtype is preserved
    assert pf_recon.dtype == dtype

    # ground truth image
    # x_true = xp.asarray(obj.image(*ig.grid()))

    # recon from fully sampled k-space
    x_full = xp.asarray(ifftnc(kspace_full))

    # Error of partial-Fourier recon should be much less than for zero-filling
    mse_pf = xp.mean(xp.abs(x_full - pf_recon) ** 2)
    mse_direct = xp.mean(xp.abs(x_full - direct_recon) ** 2)
    assert mse_pf < 0.2 * mse_direct


@pytest.mark.parametrize(
    "xp, dtype, pf_fractions",
    product(
        all_xp,
        [np.complex64, np.complex128],
        [(1, 0.6, 1), (0.6, 0.7, 1), (0.65, 0.65, 0.65)],
    ),
)
def test_partial_fourier_3d(xp, dtype, pf_fractions):
    shape = (128, 128, 64)
    ig = ImageGeometry(shape, distances=(1,) * len(shape), offsets="dsp")
    obj = mri_object_3d(ig.fov)
    coords = ig.fgrid()

    # fully-sampled k-space
    kspace_full = xp.asarray(obj.kspace(*coords), dtype=dtype)

    # partial-Fourier k-space
    nkeep = [int(f * s) for f, s in zip(pf_fractions, shape)]
    pf_mask = xp.zeros(shape, dtype=xp.bool)
    pf_mask[: nkeep[0], : nkeep[1]] = 1
    kspace_pf = kspace_full[: nkeep[0], : nkeep[1]]

    # direct reconstruction using zero-filled k-space
    direct_recon = ifftnc(kspace_full * pf_mask)

    # partial Fourier reconstruction
    pf_recon = mri_partial_fourier_nd(kspace_pf, pf_mask)
    # dtype is preserved
    assert pf_recon.dtype == dtype

    # ground truth image
    # x_true = xp.asarray(obj.image(*ig.grid()))

    # recon from fully sampled k-space
    x_full = xp.asarray(ifftnc(kspace_full))

    # Error of partial-Fourier recon should be much less than for zero-filling
    mse_pf = xp.mean(xp.abs(x_full - pf_recon) ** 2)
    mse_direct = xp.mean(xp.abs(x_full - direct_recon) ** 2)
    assert mse_pf < 0.25 * mse_direct
