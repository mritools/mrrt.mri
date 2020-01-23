import numpy as np

from mrrt.utils import (
    embed,
    get_array_module,
    masker,
    pinv_tol,
)


__all__ = ["mri_decay_approx"]


def mri_decay_approx(
    nshots,
    cmap,
    background_mask=None,
    segments=6,
    approx_type=("hist,svd", [40]),
    verbose=False,
    tol="fro",
    xp=None,
):

    xp, on_gpu = get_array_module(cmap)
    si = xp.arange(nshots).reshape(-1, 1)
    atype = approx_type[0]
    segments = xp.atleast_1d(segments)
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

    if background_mask is not None:
        # values within the mask only
        meanval = cmap[~background_mask].mean()
        # cmap2 = cmap[background_mask]
        cmap = masker(cmap, background_mask, order="F")

        # append average value from background region
        cmap = xp.concatenate((cmap, [meanval]))

    if verbose:
        import matplotlib.pyplot as plt

    if "hist," in atype:
        nhist = approx_type[1]
        hk, zc = xp.histogram(cmap, bins=nhist[0])
        zc = xp.asarray(zc, dtype=cmap.dtype)
        # convert bin edges to bin centers to match matlab behavior!
        zc = zc[:-1] + xp.diff(zc) / 2.0

        # Note: skimage.exposure.histogram returns the bin centers instead
        # scipy.ndimage.histogram doesn't return bin edges
        # skimage.exposure.histogram(zmap, nhist[0]) -> seems to recenter
        # values?

        zk = zc.reshape((-1, 1))  # [K,1]

        hk = hk.ravel()

        # GRL  Add this to conserve memory for cases where Eh is not needed
        if atype == "hist,svd" or verbose or (rmsmax < 1):

            # Eh = # xp.exp(xp.dot(-ti, zk.reshape(1, -1)))  # [N,K]
            Eh = zk.reshape(1, -1) ** si
            if verbose:
                plt.figure()
                plt.plot(Eh)

    #
    # SVD approach (optimal, to within histogram approximation)
    #

    print("atype={}".format(atype))
    if atype == "hist,svd":
        try:
            # weighted signals
            if xp is np:
                from scipy.sparse import spdiags
            else:
                from cupyx.scipy.sparse import spdiags
            Ew = xp.asarray(
                xp.asmatrix(Eh)
                * spdiags(xp.sqrt(hk), 0, len(hk), len(hk), format="csr")
            )
        except ImportError:
            Ew = xp.dot(Eh, xp.diag(xp.sqrt(hk)))
        if xp is np:
            import scipy.linalg

            U, s, V = scipy.linalg.svd(Ew, full_matrices=False)
        else:
            raise NotImplementedError("TODO")
        B = U[:, :segments]  # [M, L] keep desired components

    Bpp = pinv_tol(B, tol).T.conj()  # [M, L]
    Bpp = xp.conj(Bpp).T

    segment_size = 500
    nsegments = int(xp.ceil(cmap.size / segment_size))
    # to avoid running out of memory do:
    cmap = cmap.ravel(order="F")
    all_C = []
    for seg in range(nsegments):
        print("seg {} of {}".format(seg + 1, nsegments))
        sl = slice(seg * segment_size, (seg + 1) * segment_size)
        tmp = cmap[sl].reshape((1, -1), order="F") ** si
        all_C.append(xp.dot(Bpp, tmp))
    C = xp.concatenate(all_C, axis=1)
    if background_mask is not None:
        C_background = C[:, -1]
        C = C[:, :-1]
        C = C.T
        C = embed(C, background_mask, order="F")
        # bg_l = xp.stack((background_mask, )*segments, axis=-1)
        for l in range(segments):
            tmp = xp.ascontiguousarray(C[..., l])
            tmp[background_mask] = C_background[l]
            C[..., l] = tmp
        C = C.reshape((-1, C.shape[-1]), order="F")
        C = C.T
        # plt.figure(); plt.plot(xp.dot(B, C_background[:, xp.newaxis]))
    else:
        pass
    # make sure outputs are all numpy arrays
    B = xp.asarray(B)
    C = xp.asarray(C)
    if hk is not None:
        hk = xp.asarray(hk)
        zk = xp.asarray(zk)

    if False:
        if xp is np:
            from skimage.measure import compare_nrmse

            Ep = xp.dot(B, C)
            Eh = cmap.reshape(1, -1) ** si
            nrmse = compare_nrmse(Eh, Ep)
            print("nrmse of approximation = {}".format(nrmse))
    return B, C, hk, zk

