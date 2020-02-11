"""Approximation of complex exponentials for off-resonance corrected MRI.

Notes
-----
Details of various fieldmap approximations are given in [1], [2].

References
----------
.. [1] J.A. Fessler, S. Lee, V.T. Olafsson, H.R. Shi, D.C. Noll. Toeplitz-based
    iterative image reconstruction for MRI with correction for magnetic field
    inhomogeneity. IEEE Trans. Signal Process. 2005; 53:3393–3402.

.. [2] B. P. Sutton, D. C. Noll, and J. A. Fessler. Fast, iterative image
    reconstruction for MRI in the presence of field inhomogeneities.
    IEEE Trans. Med. Imag. 2003; 22(2):178–188.

"""
import warnings

import numpy as np

from ._mri_exp_mult import mri_exp_mult
from mrrt.utils import (
    get_array_module,
    hist_equal,
    outer_sum,
    pinv_tol,
    profile,
)


__all__ = ["mri_exp_approx"]


@profile
def mri_exp_approx(
    ti=None,
    zmap=None,
    segments=6,
    approx_type="default",
    ctest=False,
    autocorr=False,
    verbose=False,
    tol=None,
    xp=None,
):
    """Approximation to exponentials for iterative MR image reconstruction.

    Generalizes "time segmentation" and "frequency segmentation" methods.
    This is used by the MRI_Operator object for field-corrected MR
    reconstruction.

    Parameters
    ----------
    ti : ndarray
        Sample times (in seconds).
    zmap : The field map to approximate.
        The real component corresponds to T2(*) decay while the imaginary
        component corresponds to off-resonance (in rad/s).
    segments : int, optional
        The number of segments to use for the approximation.
    approx_type : str, optional
        The type of field map approximation to apply. The default is
        histogram-based time segmentation.
        TODO: describe the options.

    Additional Parameters
    ---------------------
    ctest : bool, optional
        If True, a reduced size output with shape (L, nhist) instead of (L, N)
        is returned. This can be used for testing.
    autocorr : bool, optional
        If True, autocorrelate the fieldmap histogram (for the Toeplitz case).
    tol : float, "fro", or None, optional
        The tolerance used internally when computing the pseudo-inverse.
    xp : {numpy, cupy}
        The array module to use.



    Returns
    -------
    TODO

    Notes
    -----
    Details of various fieldmap approximations are given in [1], [2].

    References
    ----------
    .. [1] J.A. Fessler, S. Lee, V.T. Olafsson, H.R. Shi, D.C. Noll.
        Toeplitz-based iterative image reconstruction for MRI with correction
        for magnetic field inhomogeneity. IEEE Trans. Signal Process. 2005;
        53:3393–3402.
        :DOI:10.1109/TSP.2005.853152

    .. [2] B. P. Sutton, D. C. Noll, and J. A. Fessler. Fast, iterative image
        reconstruction for MRI in the presence of field inhomogeneities.
        IEEE Trans. Med. Imag. 2003; 22(2):178–188.
        :DOI:10.1109/TMI.2002.808360

    """
    """
    % in
    %    ti    [M,1]    sample times
    %    zmap    [N,1]    rate map: relax_map + 2i*pi*field_map
    %            *_map and ti should have reciprocal units.
    %            usually relax_map is 0, so zmap is purely imaginary!
    %    segments        number of components: 1 <= L << N
    %            Or, use {Linit, rmsmax}: give initial L to try,
    %            then increase L until RMS error <= rmsmax < 1.
    %            This is implemented only for 'hist,time,unif' type.
    %
    % options
    %    'ctest'    1|0    return C as [L,nhist] instead of full [L, N], for testing
    %    'autocorr'    1|0    autocorrelate the fmap histogram (for Toeplitz case)
    %    'tol'    tol    tolerance for pinv(), see pinv_tol() below.
    %            use {'fro'} for tol = 1e-8 * norm(X,'fro') (for large L)
    %    'approx_type'     what type of approximation (see choices below)
    %
    % out
    %    B    [M, L]    basis functions
    %    C    [L, N]    coefficients, such that B * C \approx exp(-ti * zmap.')
    %    hk,zk    [Ko, Kr]    histogram values and 'frequencies', if used
    %
    % type of approximation choices:
    %
    %    {'hist,time,unif', nhist}
    %        with uniform time samples and LS coefficients (using histogram).
    %        This approach works almost as well as the SVD method
    %        unless L is chosen so small that the error is large.
    %    'time,unif'
    %        Time segmentation with uniform time samples (LS coef).
    %        No histogram, so it is very slow.  not recommended.
    %
    %    {'hist,svd', nhist}
    %        Nearly optimal approach that uses an SVD and a zmap histogram.
    %        nhist is the # of histogram bins; about 40 is recommended.
    %
    %    {'hist,fs,unif', nhist}
    %        Not recommended since it works poorly except for uniform distn.
    %    {'hist,fs,prctile', nhist}
    %        Not recommended since it works quite poorly.
    %    {'hist,fs,lbg', nhist}
    %        Frequency segmentation methods (exponential bases)
    %        with different choices for the nominal frequency components.
    %        in all cases the coefficients are chosen by LS (Man, MRM, 1997).
    %
    %    for relaxation cases, nhist should be [Ko=#omap Kr=#rmap]
    %
    % Copyright 2004-7-1, Jeff Fessler, The University of Michigan
    """

    # defaults
    xp, on_gpu = get_array_module(zmap, xp)
    # make sure ti is on the same device as zmap
    ti = xp.asarray(ti)

    if verbose:
        from matplotlib import pylab as plt

    if approx_type == "default":
        if xp.any(zmap.real.ravel()):
            approx_type = ("hist,time,unif", [40, 10])
        else:
            approx_type = ("hist,time,unif", [40])

    atype = approx_type[0]

    segments = np.atleast_1d(segments)
    if len(segments) == 2:
        rmsmax = segments[1]
        segments = segments[0]
    elif len(segments) == 1:
        segments = segments[0]
        rmsmax = 1 / xp.finfo(float).eps
    else:
        raise ValueError(
            "segments must be an integer or a tuple containing an initial "
            "guess and an rms error tolerance"
        )
    zmap = zmap.ravel().reshape(-1, 1)  # [N,1]
    if ti.ndim > 1:
        warnings.warn("expected 1d ti")
    ti = ti.ravel()

    rmap = zmap.real
    fmap = zmap.imag / (2 * xp.pi)

    # histogram the field map
    warned = False
    hk = None
    zk = None
    if ("hist," in atype) or (ctest):
        nhist = approx_type[1]
        if len(nhist) == 1:  # fmap only

            if xp.any(rmap.ravel()):
                raise ValueError("rmap requires len(nhist) == 2")

            hk, zc = xp.histogram(zmap.imag, bins=nhist[0])
            if zc.dtype != zmap.imag.dtype:
                # use single precision if fieldmap was single precision
                zc = zc.astype(zmap.imag.dtype)

            # convert bin edges to bin centers to match matlab behavior!
            zc = zc[:-1] + xp.diff(zc) / 2.0

            # Note: skimage.exposure.histogram returns the bin centers instead
            # scipy.ndimage.histogram doesn't return bin edges
            # skimage.exposure.histogram(zmap, nhist[0]) -> seems to recenter
            # values

            zk = 0 + 1j * zc.ravel()
            if verbose:
                plt.figure(),
                bar_width = xp.mean(xp.diff(zc) / (2 * xp.pi)) * 0.66
                plt.bar(xp.asarray(zk.imag) / (2 * xp.pi), hk, width=bar_width)

            if autocorr:
                hk = xp.correlate(hk, hk, mode="full")
                zk = xp.arange(-(nhist[0] - 1), (nhist[0])) * (zk[1] - zk[0])
                if verbose:  # Todo
                    plt.figure(),
                    bar_width = xp.mean(xp.diff(zk.imag) / (2 * xp.pi)) * 0.8
                    plt.bar(
                        xp.asarray(zk.imag) / (2 * xp.pi), hk, width=bar_width
                    )
        else:
            hk, zc = hist_equal(
                xp.concatenate((zmap.imag, zmap.real), axis=1), nhist
            )
            if zc[0].dtype != zmap.imag.dtype:
                zc[0] = zc[0].astype(zmap.imag.dtype)
            if zc[1].dtype != zmap.real.dtype:
                zc[1] = zc[1].astype(zmap.real.dtype)
            zk = outer_sum(1j * zc[0], zc[1]).T  # [K1, K2]

            if autocorr:  # matlab version of this case by Valur Olafsson
                # TODO: haven't tested this case much
                import scipy.signal

                hk = scipy.signal.convolve2d(
                    hk, hk[::-1, ...], mode="full", boundary="fill", fillvalue=0
                )
                zc[0] = xp.arange(-(nhist[0] - 1), (nhist[0])) * (
                    zc[0][1] - zc[0][0]
                )
                zc[1] = xp.linspace(
                    2 * xp.min(zc[1]), 2 * xp.max(zc[1]), 2 * nhist[1] - 1
                )
                zk = outer_sum(1j * zc[0], zc[1]).T

        hk = hk.ravel()

        # GRL  Add this to conserve memory for cases where Eh is not needed
        if atype == "hist,svd" or verbose or (rmsmax < 1):
            Eh = xp.exp(
                xp.dot(-ti[:, np.newaxis], zk.ravel()[np.newaxis, :])
            )  # [N,K]

    #
    # SVD approach (optimal, to within histogram approximation)
    #
    if atype == "hist,svd":
        import scipy.linalg

        try:
            if xp is np:
                from scipy.sparse import spdiags
            else:
                from cupyx.scipy.sparse import spdiags
            # weighted signals
            Ew = xp.asarray(
                xp.asmatrix(Eh)
                * spdiags(xp.sqrt(hk), 0, len(hk), len(hk), format="csr")
            )
        except ImportError:
            Ew = xp.dot(Eh, xp.diag(xp.sqrt(hk)))
        if xp is np:
            U, s, V = scipy.linalg.svd(Ew, full_matrices=False)
            B = U[:, :segments]  # [M, L] keep desired components
        else:
            U, s, V = scipy.linalg.svd(Ew.get(), full_matrices=False)
            B = U[:, :segments]  # [M, L] keep desired components
            B = xp.asarray(B)

    #
    # time segmentation approaches (recommended)
    #
    elif (atype == "time,unif") or (atype == "hist,time,unif"):

        rms = xp.inf
        while (rms > rmsmax) and (segments < 40):
            # time sample locations [0 ... end]
            if segments == 1:
                tl = ti.mean()
            else:
                p = [100 * q / (segments - 1) for q in range(segments)]
                tl = xp.asarray(xp.percentile(ti, q=p))
                if tl.dtype != zmap.imag.dtype:
                    tl = tl.astype(zmap.imag.dtype)

            if (rmsmax < 1) or ctest or (atype == "hist,time,unif"):
                # Ch is shape (L, K)
                # Ch = xp.exp(xp.dot(-tl[:, np.newaxis], zk[np.newaxis, :]))

                Ch = xp.exp(
                    xp.dot(-tl[:, np.newaxis], zk.ravel()[np.newaxis, :])
                )

            if (segments > 9) and (tol is None) and (not warned):
                warnings.warn("warning: For large L, trying tol='fro'")
                warned = True
                if isinstance(tol, list):
                    tol[0] = "fro"
                else:
                    tol = "fro"

            if atype == "time,unif":
                # [L, N] - classic TS
                C = xp.exp(xp.dot(-tl[:, np.newaxis], zmap.reshape(1, -1)))
                X = C.T  # [N,L]
                X = pinv_tol(X, tol).T  # [N,L]
                B = mri_exp_mult(
                    X.conj(),
                    zmap.ravel(),
                    # zmap.astype(xp.promote_types(zmap.dtype, xp.float64)),
                    ti,
                ).T
            elif atype == "hist,time,unif":  # Fessler section 5.4.3.5
                if xp is np:
                    from scipy.sparse import spdiags
                else:
                    from cupyx.scipy.sparse import spdiags
                W = spdiags(xp.sqrt(hk), 0, len(hk), len(hk), format="csr")
                if W.dtype != zmap.imag.dtype:
                    # use single precision if fieldmap was single precision
                    W = W.astype(zmap.imag.dtype)
                if xp is np:
                    P = pinv_tol(W * Ch.T, tol) * W  # [L,K], weighted pinv
                else:
                    # TODO: fix pinv_tol on the GPU.
                    #       for now run it on the CPU and then copy result back to the GPU
                    import cupy

                    P = cupy.asarray(
                        pinv_tol(W.get() * Ch.T.get(), tol) * W.get()
                    )  # [L,K], weighted pinv
                # print("zk.dtype = {}".format(zk.dtype))
                # print("P.dtype = {}".format(P.dtype))

                if False and verbose:
                    # rP = numpyutils.matrixrank(P);
                    rP = xp.linalg.matrix_rank(P)  # crashes?

                    if rP != segments:
                        warnings.warn("rank=%d < L=%d" % (rP, segments))
                B = mri_exp_mult(P.conj().T, zk.ravel(), ti).T

            if rmsmax < 1:
                segments += 1
                Ep = xp.dot(B, Ch)
                atmp = xp.abs(Eh - Ep)
                atmp *= atmp
                mse = xp.mean(atmp, axis=0)
                rms = xp.dot(xp.sqrt(mse), hk) / xp.sum(hk)  # hk-weighted rms
                print("rms={} at segments={}".format(rms, segments))
                if verbose:  # GRL added
                    from pyvolplot import subplot_stack

                    iworst = xp.argmax(mse)
                    ik = [
                        0,
                        np.floor(nhist[0] / 4),
                        2 * np.floor(nhist[0] / 4),
                        3 * np.floor(nhist[0] / 4),
                        nhist[0] - 1,
                        np.prod(nhist) - 1,
                        iworst,
                    ]
                    fig = subplot_stack(1000 * ti, Eh[:, ik], colors=["g", "k"])
                    fig = subplot_stack(
                        1000 * ti,
                        Ep[:, ik],
                        colors=["b--", "r--"],
                        fig=fig,
                        title="True and Approx, segments={}".format(segments),
                    )

            else:
                break  # check

        if segments == 40:
            raise ValueError("max segments reached!?")

        if ctest:
            C = Ch
        elif not atype == "time,unif":
            C = xp.dot(-tl[:, np.newaxis], zmap.reshape(1, -1))  # [L, N]
            xp.exp(C, out=C)

        return B, C, hk, zk

    #
    # freq. segmentation approaches
    #
    elif atype[:8] == "hist,fs,":

        if atype == "hist,fs,unif":  # uniform spacing
            fl = xp.linspace(xp.min(fmap), xp.max(fmap), segments + 2)
            fl = fl[1:-1]
            rl = 0  # lazy: 'uniform' in 2D seems too arbitrary
            # this may stink if rmap is nonzero.

        elif atype == "hist,fs,prctile":  # histogram percentiles
            p = [100 * q / (segments + 1) for q in range(1, segments + 1)]
            fl = xp.array(xp.percentile(fmap, q=p))
            rl = 0  # lazy again

        elif atype == "hist,fs,kmeans":
            # requires scikit-learn package
            try:
                from sklearn.cluster import KMeans
            except ImportError as e:
                print("hist,fs,kmeans requires sklearn")
                raise (e)
            k_means = KMeans(n_clusters=int(segments), n_init=10, n_jobs=-1)
            if xp is not np:
                fmap_np = fmap.get()
                rmap_np = rmap.get()
            else:
                fmap_np = fmap
                rmap_np = rmap

            if xp.any(rmap):
                k_means.fit(
                    xp.stack((rmap_np.ravel(), fmap_np.ravel()), axis=-1)
                )
                centers = k_means.cluster_centers_
                si = xp.argsort(centers[:, 1])
                fl = xp.sort(centers[:, 1][si])
                rl = xp.sort(centers[:, 0][si])
            else:
                k_means.fit(fmap.reshape((-1, 1)))
                centers = k_means.cluster_centers_.squeeze()
                fl = xp.sort(xp.asarray(centers))
                rl = 0

        elif atype == "hist,fs,lbg":  # LBG quantization of histogram
            raise ValueError(
                "Haven't implemented the Linde–Buzo–Gray (LBG) algorithm in "
                "python.  Try hist,fs,kmeans instead"
            )
        else:
            raise ValueError("fs type {} unknown".format(atype))

        zl = rl + (2j * xp.pi) * fl
        B = xp.exp(
            xp.dot(-ti[:, np.newaxis], zl.ravel()[np.newaxis, :])
        )  # (M, L)

    else:
        raise ValueError("type {} unknown".format(atype))

    # given the basis, now compute the LS coefficients,
    # i.e., (B'*B) \ B' * E
    Bpp = pinv_tol(B, tol).T.conj()  # [M, L]

    if ctest:
        # [L, K] coefficients
        C = mri_exp_mult(Bpp, ti, zk.ravel())
    else:
        # [L, N] coefficients
        C = mri_exp_mult(Bpp, ti, zmap.ravel())

    # make sure outputs are all arrays
    B = xp.asarray(B)
    C = xp.asarray(C)
    if hk is not None:
        hk = xp.asarray(hk)
        zk = xp.asarray(zk)
    return B, C, hk, zk


if False:
    from mrrt.mri.sim import generate_fieldmap

    zmap = generate_fieldmap((64, 64), 80)
