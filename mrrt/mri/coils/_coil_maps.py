"""Coil compression, pre-whitening and sensitivy map estimation utilities.

Much of this code is modified from idmrmrd-python-tools
https://github.com/ismrmrd/ismrmrd-python-tools

Much code that was 2D only has been updated to work with 3D (or nD) data.
The user can specify which axis of the array corresponds to the coils.

The function calculate_csm_inati_iter is an adaptation of a coil sensitivity
map estimation function by Souheil Inati that is included in Gadgetron.

"""

import numpy as np
from scipy import ndimage as ndi
from mrrt.utils._cupy import get_array_module


__all__ = ["apply_csm", "calculate_csm_inati"]


def smooth(img, box=5, xp=None):
    """Smooths coil images

    :param img: Input complex images, ``[y, x] or [z, y, x]``
    :param box: Smoothing block size (default ``5``)

    :returns simg: Smoothed complex image ``[y,x] or [z,y,x]``
    """
    xp, on_gpu = get_array_module(img, xp)
    if xp is np:
        t_real = np.zeros(img.shape, dtype=img.real.dtype)
        t_imag = np.zeros(img.shape, dtype=img.imag.dtype)

        ndi.filters.uniform_filter(img.real, size=box, output=t_real)
        ndi.filters.uniform_filter(img.imag, size=box, output=t_imag)
        simg = t_real + 1j * t_imag
    else:
        try:
            from cupyimg import convolve_separable
        except ImportError:
            # TODO: allow fallback to cupyx.scipy.ndimage when not found
            raise ImportError(
                "GPU-based smoothing requires the cupyimg package."
            )

        if np.isscalar(box):
            box = [box] * img.ndim
        boxfilt = [np.ones(wsize, dtype=img.real.dtype) for wsize in box]
        boxfilt = [w / w.sum() for w in boxfilt]
        boxfilt = [xp.asarray(f) for f in boxfilt]
        simg = convolve_separable(img, boxfilt)
    return simg


def calculate_csm_global(data, roi_mask=None, xp=None):

    xp, on_gpu = get_array_module(data, xp)
    if not ((data.ndim == 3) or (data.ndim == 4)):
        raise ValueError("Data dimension error: data must be 3D or 4D")

    nc = data.shape[-1]

    if roi_mask is not None:
        beta = []
        for ch in range(nc):
            beta.append(data[..., ch][roi_mask].sum())
        beta = xp.asarray(beta)
    else:
        beta = xp.sum(data, axis=tuple(range(data.ndim - 1)))

    beta /= xp.linalg.norm(beta)
    cs = beta.reshape((1,) * (data.ndim - 1) + (nc,)) * xp.ones(
        data.shape, data.dtype
    )

    comim = xp.squeeze((xp.conj(cs) * data).sum(axis=-1))

    rho = xp.sqrt(xp.mean(xp.abs(comim) ** 2, axis=tuple(range(data.ndim - 1))))

    return cs, rho, comim


def calculate_csm_inati(
    data, smoothing=5, niter=5, verbose=True, roi_mask=None, xp=None
):
    """ Fast, iterative coil map estimation for 2D or 3D acquisitions.

    Parameters
    ----------
    im : (..., coil) ndarray
        Input images, (x, y, coil) or (x, y, z, coil).
    smoothing : int or array-like, optional
        Smoothing block size(s) for the spatial axes.
    niter : int, optional
        Maximal number of iterations to run.
    thresh : float, optional
        Threshold on the relative coil map change required for early
        termination of iterations.  If ``thresh=0``, the threshold check
        will be skipped and all `niter` iterations will be performed.
    verbose : bool, optional
        If true, progress information will be printed out at each iteration.

    Returns
    -------
    coil_map : (..., coil) array
        Relative coil sensitivity maps, (x, y, coil) or (x, y, z, coil).
    rho : array
        TODO
    coil_combined : array
        The coil combined image volume, (x, y) or (x, y, z).

    Notes
    -----
    The implementation corresponds to the algorithm described in [1]_ and is a
    port of Gadgetron's `coil_map_3d_Inati_Iter` routine.

    For non-isotropic voxels it may be desirable to use non-uniform smoothing
    kernel sizes, so a length 3 array of smoothings is also supported.

    References
    ----------
    .. [1] S Inati, MS Hansen, P Kellman.  A Fast Optimal Method for Coil
        Sensitivity Estimation and Adaptive Coil Combination for Complex
        Images.  In: ISMRM proceedings; Milan, Italy; 2014; p. 4407.
    """
    xp, on_gpu = get_array_module(data, xp)
    data = xp.ascontiguousarray(data)

    if not ((data.ndim == 3) or (data.ndim == 4)):
        raise ValueError("Data dimension error: data must be 3D or 4D")

    if not (xp.all(xp.asarray(smoothing) > 0)):
        raise ValueError("Box size error: box must be a positive integer")

    if xp.isscalar(smoothing):
        smoothing = (smoothing,) * (data.ndim - 1) + (1,)
    else:
        # add singleton on channel dimension
        smoothing = np.concatenate((np.asarray(smoothing), (1,)), axis=0)
        if smoothing.size != data.ndim:
            raise ValueError(
                "smoothing kernel must match the number of " "image dimensions"
            )

    # initialize
    (cs_global, rho, comim) = calculate_csm_global(data, roi_mask=roi_mask)
    comim = comim

    eps = xp.finfo(data.dtype).eps * xp.abs(data).mean()
    for iter in range(niter):
        if verbose:
            print(
                "Inati coilmap estimation: iteration {} of {}".format(
                    iter + 1, niter
                )
            )

        # comim is D*v, (i.e. u*s)
        # rho is s
        # (u^H*s)*D, i.e. s^2 * v^H
        cs = smooth(
            xp.squeeze(xp.conj(comim[..., xp.newaxis]) * data), smoothing
        )

        # combine s*s*v, using the global combiner
        comim_glob = apply_csm(cs, cs_global, coil_axis=-1)
        # and remove the phase
        cs *= xp.exp(-1j * xp.angle(comim_glob))[..., xp.newaxis]

        # normalize s*s*v, i.e. compute s*s
        csnorm = xp.abs(cs)
        csnorm *= csnorm
        csnorm = csnorm.sum(-1, keepdims=True)
        csnorm = xp.sqrt(csnorm)
        csnorm += eps
        cs /= csnorm

        # D*v = u*s
        comim = apply_csm(data, cs, coil_axis=-1)

    # compute s
    rho = xp.squeeze(xp.sqrt(csnorm))

    return cs, rho, comim


def apply_csm(img, csm, coil_axis=-1, xp=None):
    """Apply coil sensitivity maps to combine images

    :param img: Input images, ``[coil, y, x]``, or ``[coil, z, y, x]``
    :param csm: Coil sensitivity maps, ``[coil, y, x]``, or ``[coil, z, y, x]``

    :returns comim: Combined image, ``[y, x]`` or ``[z, y, x]``
    """
    xp, on_gpu = get_array_module(img, xp)
    img = xp.asarray(img)
    csm = xp.asarray(csm)
    if img.shape != csm.shape:
        raise ValueError(
            "Images and coil sensitivities must have matching shape"
        )
    comim = xp.squeeze(xp.sum(xp.conj(csm) * img, axis=coil_axis))

    return comim
