import itertools

import time
import numpy as np
import scipy.ndimage as ndi
import pytest

from mrrt.utils import ImageGeometry, ellipse_im
from mrrt.mri import mri_exp_approx

__all__ = ["test_mri_exp_approx"]


def _test_mri_exp_approx1(
    segments=4,
    nx=64,
    tmax=25e-3,
    dt=5e-6,
    autocorr=False,
    use_rmap=True,
    atype="hist,time,unif",
    nhist=None,
    ctest=True,
    verbose=False,
    tol=None,
):
    if verbose:
        from matplotlib import pyplot as plt
        from pyvolplot import subplot_stack

    ti = np.arange(0, tmax, dt)

    if True:
        # Generate a synthetic fieldmap
        fmap = np.zeros((64, 64))
        fmap[6:27, 9:20] = 90
        fmap[36:57, 9:20] = 120
        fmap[5:26, 29:60] = 30
        fmap[37:58, 29:60] = 60
        if nx != 64:
            fmap = ndi.zoom(fmap, nx / 64, order=0)
        kernel_size = int(np.round(5 * nx / 64))
        smoothing_kernel = np.ones((kernel_size, kernel_size)) / (
            kernel_size ** 2
        )
        ndi.convolve(fmap, smoothing_kernel, output=fmap)
        fmap = fmap + 10
        if verbose:
            plt.figure()
            plt.imshow(fmap, interpolation="nearest", cmap="gray")
            plt.title("Field Map")

    if use_rmap:
        # generate a T2 relaxation map
        rmap = (
            np.asarray(
                [[0, 0, 18, 23, 0, 20 * 64 / nx], [6, 0, 8, 8, 0, 3 * 64 / nx]]
            )
            * nx
            / 64
        )
        ig = ImageGeometry(shape=(nx, nx), fov=(nx, nx))
        rmap, params = 1 * ellipse_im(ig, rmap, oversample=3)
        if verbose:
            plt.figure()
            plt.imshow(rmap, cmap="gray", interpolation="nearest")
            plt.title("Relax Map"),
    else:
        rmap = 0

    zmap = rmap + (2j * np.pi) * fmap

    if not nhist:
        if not np.any(rmap > 0):
            nhist = [40]
        else:
            nhist = [40, 10]

    # autocorr_arg = ['autocorr', True] # test autocorrelation version

    if True:  # convert to single precision
        ti = np.asarray(ti, dtype="float32")
        zmap = np.asarray(zmap, dtype="complex64")  # single precision complex

    if isinstance(segments, int):
        pass
    elif isinstance(segments, (list, tuple)) and len(segments) == 2:
        pass
    else:
        raise ValueError("Invalid choice for segments")

    kwargs = {"autocorr": autocorr, "ctest": ctest, "verbose": verbose}
    tstart = time.time()
    if tol is None:
        B, C, hk, zk = mri_exp_approx(
            ti, zmap, segments, approx_type=(atype, nhist), **kwargs
        )
    else:
        B, C, hk, zk = mri_exp_approx(
            ti, zmap, [segments, tol], approx_type=(atype, nhist), **kwargs
        )
    print("\tduration=%g s" % (time.time() - tstart))

    if ctest:
        Eh = np.exp(-ti[:, np.newaxis] * zk.ravel()[np.newaxis, :])
    else:
        Eh = np.exp(-ti[:, np.newaxis] * zmap.ravel()[np.newaxis, :])
    Ep = np.dot(B, C)  # matrix product
    err = np.abs(Eh - Ep)
    mse = np.mean(np.power(err, 2), axis=0)
    if ctest:
        wrms = np.sqrt(np.dot(mse, hk) / np.sum(hk))
    else:
        wrms = -1

    if verbose:
        subplot_stack(1000 * ti, B, title="Basis Components", colors=["k", "m"])

    nf = np.floor(nhist[0] / 4)
    if len(nhist) == 2:
        ik = np.array([0, nf, 2 * nf, 3 * nf, nhist[1] - 1]) + 2 * nf * nhist[1]
        ik = ik.tolist()
    elif len(nhist) == 1:
        ik = [0, nf, 2 * nf, 3 * nf, nhist[0] - 1]
    ik = np.asarray(ik, dtype=int)

    mse_mean = mse.mean()
    max_err = err.max()

    if verbose:
        fig = subplot_stack(1000 * ti, Eh[:, ik], colors=["g", "k"])
        fig = subplot_stack(
            1000 * ti,
            Ep[:, ik],
            colors=["b--", "r--"],
            fig=fig,
            title="True and Approx",
        )
        fig = subplot_stack(
            1000 * ti,
            err[:, ik],
            colors=["b--", "r--"],
            title="True and Approx",
        )
        print(
            "\tfor L=%d, wrms=%g, mse = %g, max_err=%g"
            % (B.shape[1], wrms, mse_mean, max_err)
        )
    return wrms, mse_mean, max_err


@pytest.mark.parametrize(
    "use_rmap, alg",
    itertools.product(
        [False, True],
        [
            "hist,svd",
            "hist,time,unif",
            "time,unif",
            "hist,fs,unif",
            "hist,fs,prctile",
            "hist,fs,kmeans",
        ],
    ),
)
@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_mri_exp_approx(use_rmap, alg, verbose=False):
    if alg == ["hist,fs,kmeans"]:
        pytest.importorskip("sklearn")

    tmax = 25e-3  # overall time duration (s)
    wrms, mse_mean, max_err = _test_mri_exp_approx1(
        segments=8,  # number of segments
        nx=64,
        tmax=tmax,
        dt=1e-3,
        autocorr=False,
        use_rmap=use_rmap,
        atype=alg,
        nhist=None,
        ctest=True,
        verbose=verbose,
    )

    if alg == "hist,fs,prctile" or (use_rmap and alg == "hist,fs,kmeans"):
        # may want to just remove these options, as they perform relatively
        # poorly
        assert mse_mean < 0.01
    else:
        assert mse_mean < 1e-5
