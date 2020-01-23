import numpy as np
import pytest

from mrrt.mri.coils import calculate_prewhitening, prewhiten

from mrrt.utils import config

all_xp = [np]
if config.have_cupy:
    import cupy

    all_xp += [cupy]


@pytest.mark.parametrize("xp", all_xp)
def test_calculate_prewhitening(xp):
    nsamples = 2048
    ncoils = 16
    rstate = xp.random.RandomState(16)
    noise = rstate.randn(nsamples, ncoils)
    w, r = calculate_prewhitening(noise, coil_axis=-1, return_full=True)
    assert w.shape == (ncoils, ncoils)
    assert r.shape == (ncoils, ncoils)

    # r should be close to the identity matrix in this case
    eye = xp.eye(ncoils)
    norm_ratio = xp.linalg.norm(r - eye) / xp.linalg.norm(eye)
    assert norm_ratio < 0.1

    # for white noise, w should be nearly diagonal
    assert xp.linalg.norm(xp.diag(w)) / xp.linalg.norm(w) > 0.95

    w = calculate_prewhitening(noise, coil_axis=-1, return_full=False)
    assert w.shape == (ncoils, ncoils)

    # can use arbitrary noise shape with user-specified coil axis
    noise2 = noise.reshape(nsamples // 4, 4, ncoils)
    new_coil_axis = 1
    noise2 = noise2.swapaxes(-1, new_coil_axis)
    w2 = calculate_prewhitening(
        noise2, coil_axis=new_coil_axis, return_full=False
    )
    xp.testing.assert_array_almost_equal(w, w2)


@pytest.mark.parametrize("xp", all_xp)
def test_prewhiten(xp):
    nsamples = 2048
    ncoils = 16
    rstate = xp.random.RandomState(16)
    noise_data = rstate.randn(nsamples, ncoils)

    # some fake data
    data = xp.arange(64 * 64 * ncoils).reshape((64, 64, ncoils))
    data = data + 1j * data[::-1, :, :]

    dataw, w, r = prewhiten(data, noise_data, coil_axis=-1, coil_axis_noi=-1)
    assert w.shape == (ncoils, ncoils)
    assert r.shape == (ncoils, ncoils)
    assert dataw.shape == data.shape
    assert dataw.dtype == data.dtype

    # r should be close to the identity matrix in this case
    eye = xp.eye(ncoils)
    norm_ratio = xp.linalg.norm(r - eye) / xp.linalg.norm(eye)
    assert norm_ratio < 0.1

    # for white noise, w should be nearly diagonal
    assert xp.linalg.norm(xp.diag(w)) / xp.linalg.norm(w) > 0.95

    # transposed data with coils as first axis
    data_trans = data.transpose()
    dataw, w, r = prewhiten(
        data_trans, noise_data, coil_axis=0, coil_axis_noi=-1
    )
    assert w.shape == (ncoils, ncoils)
    assert r.shape == (ncoils, ncoils)
    assert dataw.shape == data_trans.shape
    assert dataw.dtype == data.dtype

    # noise also transposed
    dataw2, w2, r2 = prewhiten(
        data_trans, noise_data.T, coil_axis=0, coil_axis_noi=0
    )
    # result should be the same as previously
    xp.testing.assert_array_almost_equal(dataw, dataw2)
