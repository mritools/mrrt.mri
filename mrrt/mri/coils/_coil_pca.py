"""Coil compression, pre-whitening and sensitivy map estimation utilities.

Much of this code is modified from idmrmrd-python-tools
https://github.com/ismrmrd/ismrmrd-python-tools

Much code that was 2D only has been updated to work with 3D (or nD) data.
The user can specify which axis of the array corresponds to the coils.

The function calculate_csm_inati_iter is an adaptation of a coil sensitivity
map estimation function by Souheil Inati that is included in Gadgetron.

"""

import warnings
import numpy as np
from mrrt.utils import reale, get_array_module


__all__ = ["apply_pca_weights", "coil_pca", "coil_pca_noncart"]


def _pca_inner(caldata, neig, percentile=None, verbose=False, xp=None):
    # caldata must be shape (ncal, ncoils)
    xp, on_gpu = get_array_module(caldata, xp)
    if neig is None:
        neig = caldata.shape[-1]
    elif neig > caldata.shape[-1]:
        raise ValueError(
            "number of eigencoils cannot exceed the number of original coils"
        )

    cov_mat = xp.dot(xp.conj(caldata.T), caldata)
    # cov_mat is Hermitian symmetric so use the faster eigh instead of eig
    if xp is np:
        w, v = np.linalg.eigh(cov_mat)
    else:
        # can only run eigh on the CPU at the moment
        w, v = np.linalg.eigh(cov_mat.get())
        w = xp.asarray(w)
        v = xp.asarray(v)

    w = reale(w)
    si = xp.argsort(w)[::-1]  # largest to smallest
    if verbose:
        from matplotlib import pyplot as plt

        plt.figure()
        plt.subplot(121)
        if on_gpu:
            plt.plot((w[si] / xp.sum(w[si])).get(), "k.-")
        else:
            plt.plot(w[si] / xp.sum(w[si]), "k.-")
        plt.ylabel("normalized eigenvalue")
        plt.xlabel("channel (sorted by descending eigenvalue)")
        plt.subplot(122)
        x = np.arange(1, caldata.shape[-1] + 1)
        if on_gpu:
            plt.plot(x, xp.cumsum(w[si] / xp.sum(w[si])).get(), "k.-")
        else:
            plt.plot(x, xp.cumsum(w[si] / xp.sum(w[si])), "k.-")
        plt.plot([x[0], x[-1]], [0.95, 0.95], "k--")
        plt.plot([x[0], x[-1]], [0.98, 0.98], "k:")
        plt.plot([neig, neig], [0, 1], "r:")
        plt.ylabel("normalized eigenvalue cumsum")
        plt.ylabel("# of channels")
        plt.axis("tight")

    v = v[:, si]  # eigenvectors sorted
    if percentile is None:
        return v
    else:
        if percentile < 90:
            warnings.warn("percentile < 90%: was this intentional?")
        neig = xp.where(xp.cumsum(w[si] / xp.sum(w[si])) >= percentile / 100)[
            0
        ][0]
        return v, neig


def apply_pca_weights(data, pca_matrix, neig, coil_axis=-1, xp=None):
    xp, on_gpu = get_array_module(data, xp)
    if coil_axis != -1:
        # coil axis must come last
        data = xp.swapaxes(data, -1, coil_axis)
    s = data.shape[:-1] + (neig,)
    data = xp.dot(data, pca_matrix[:, :neig])
    data = data.reshape(s, order="F")
    if coil_axis != -1:
        # swap coil axis back to original location
        data = xp.swapaxes(data, -1, coil_axis)
    return data


def coil_pca_noncart(
    data,
    kabs=None,
    rkeep=1,
    coil_axis=-1,
    neig=12,
    verbose=False,
    pca_matrix_only=False,
    xp=None,
):
    xp, on_gpu = get_array_module(data, xp)
    if coil_axis != -1:
        # coil axis must come last
        data = xp.swapaxes(data, -1, coil_axis)

    ncoils = data.shape[-1]
    if ncoils < neig:
        raise ValueError("too few coils")

    if rkeep < 1:
        if data.ndim != 2:
            raise ValueError("expected a 2D array of data [nsamples, ncoils]")

        if kabs is None:
            raise ValueError("rkeep option requires kspace magnitude vector")

        cal_keep = xp.where(kabs <= rkeep)
        caldata = data[cal_keep[0], :]
        s = None
    else:
        s = data.shape
        caldata = data.reshape((-1, ncoils), order="F")
    pca_mtx = _pca_inner(caldata, neig=neig, verbose=verbose)
    if pca_matrix_only:
        return pca_mtx

    data = apply_pca_weights(data, pca_matrix=pca_mtx, neig=neig)
    if s is not None:
        s = list(s)
        s[-1] = neig
        data = data.reshape(s, order="F")

    if coil_axis != -1:
        # swap coil axis back to original location
        data = xp.swapaxes(data, -1, coil_axis)
    return data


def coil_pca(
    data,
    ncal_x=64,
    ncal_y=64,
    ncal_z=64,
    coil_axis=-1,
    neig=None,
    percentile=100,
    verbose=True,
    pca_matrix_only=False,
    xp=None,
):
    xp, on_gpu = get_array_module(data, xp)
    ncoils = data.shape[coil_axis]
    #    if neig >= ncoils:
    #        return data

    if coil_axis != -1:
        # coil axis must come last
        data = xp.swapaxes(data, -1, coil_axis)

    ncal_y = min(data.shape[0], ncal_x)
    ncal_x = min(data.shape[1], ncal_y)
    ncal_z = min(data.shape[2], ncal_z)

    if data.ndim == 4:
        nx, ny, nz = data.shape[:-1]
    else:
        nx, ny = data.shape[:-1]
    nx_2 = nx // 2
    ny_2 = ny // 2
    if data.ndim == 4:
        nz_2 = nz // 2

    if ncal_x == -1:
        slice_x = slice(None)
    else:
        slice_x = slice(nx_2 - ncal_x // 2, nx_2 + ncal_x // 2)
    if ncal_y == -1:
        slice_y = slice(None)
    else:
        slice_y = slice(ny_2 - ncal_y // 2, ny_2 + ncal_y // 2)

    if data.ndim == 4:
        if ncal_z == -1:
            slice_z = slice(None)
        else:
            slice_z = slice(nz_2 - ncal_z // 2, nz_2 + ncal_z // 2)
        cal_slices = [slice_x, slice_y, slice_z, slice(None)]
    else:
        cal_slices = [slice_x, slice_y, slice(None)]

    caldata = data[cal_slices].reshape((-1, ncoils), order="F")
    data = data.reshape((-1, ncoils), order="F")
    if False:
        caldata_demean = caldata - caldata.mean(-1)[:, xp.newaxis]
        U, s, pca_mtx = xp.linalg.svd(caldata_demean, full_matrices=False)
    else:
        pca_mtx, neig = _pca_inner(
            caldata, neig, percentile=percentile, verbose=verbose
        )
        if pca_matrix_only:
            return pca_mtx, neig
        data = apply_pca_weights(data, pca_mtx, neig)
    if coil_axis != -1:
        # swap coil axis back to original location
        data = xp.swapaxes(data, -1, coil_axis)
    return data
