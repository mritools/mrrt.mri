import numpy as np
from numpy.testing import assert_raises, assert_array_almost_equal

from mrrt.mri.operators._MRI import PixelBasis


def _kspace_cartesian(shape, fov):
    if len(fov) != len(shape):
        raise ValueError("shape mismatch")

    omegas = [(np.arange(s) / s - 0.5) * 2 * np.pi for s in shape]
    omegas_grid = np.meshgrid(*omegas, indexing="ij")
    omega = np.stack([o.ravel() for o in omegas_grid], axis=-1)

    # convert to physical units
    kspace = omega.copy()
    for d in range(kspace.shape[-1]):
        dx = fov[d] / shape[d]
        kspace[:, d] /= 2 * np.pi * dx

    return kspace, omega


def test_PixelBasis(show_figure=False):
    fov = np.asarray([240, 240])
    Nd = np.asarray([64, 64])
    dx = fov / Nd
    kspace, omega = _kspace_cartesian(shape=Nd, fov=fov)

    # try both 'dirac' and 'rect'
    b = PixelBasis(kspace, "dirac", fov=fov, Nd=Nd)
    b2 = PixelBasis(kspace, "rect", fov=fov, Nd=Nd)
    b3 = PixelBasis(kspace, "sinc", fov=fov, Nd=Nd)
    assert b.transform is None
    assert b2.transform is not None
    assert b3.transform is not None

    if show_figure:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(131), plt.plot(kspace[:, 0], kspace[:, 1], "k."),
        plt.xlabel("kx"), plt.ylabel("ky"), plt.axis("image")
        plt.subplot(132), plt.imshow(
            b3.transform.reshape(Nd) / b3.transform.max(), vmin=0, vmax=1
        )
        plt.title("sinc")
        plt.subplot(133), plt.imshow(
            b2.transform.reshape(Nd) / b2.transform.max(), vmin=0, vmax=1
        )
        plt.title("rect")

    # uncrecognized basis
    assert_raises(ValueError, PixelBasis, kspace, "rect2", fov=fov, Nd=Nd)

    # test alternate input argument forms
    b3_v2 = PixelBasis(kspace, "sinc", fov=fov, dx=dx)
    b3_v3 = PixelBasis(kspace, "sinc", fov=fov, dx=dx[0])
    assert_array_almost_equal(b3.transform, b3_v2.transform)
    assert_array_almost_equal(b3.transform, b3_v3.transform)
