"""Noise pre-whitening of parallel MRI data.

This code is adapted from idmrmrd-python-tools
https://github.com/ismrmrd/ismrmrd-python-tools

"""

import numpy as np
from mrrt.utils import get_array_module


__all__ = ["apply_prewhitening", "calculate_prewhitening", "prewhiten"]


def calculate_prewhitening(
    noise, coil_axis=-1, scale_factor=1.0, return_full=False, xp=None
):
    """Calculates the noise prewhitening matrix

    Parameters
    ----------
    noise : ndarray
        Input noise data (array or matrix)
    coil_axis : int
        Must correspond to the axis in ``noise`` that corresponds to coils.
    scale_factor : float
        Applied on the noise covariance matrix. Used to adjust for effective
        noise bandwith and difference in sampling rate between noise
        calibration and actual measurement:
        scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio
    return_full : bool
        If True, also return the noise correlation matrix.

    Returns
    -------
    W : ndarray
        Prewhitening matrix (upper triangular) of shape (coil, coil).
        ``data_pw = numpy.dot(data, w)`` where `data` is an (nsamples, ncoils)
        array gives prewhitened data.

    R : ndarray
        noise correlation matrix, shape = (coil, coil).

    References
    ----------
    .. [1] Pruessman KP, Weiger M, Bornert P and Boesiger P.  Advances in
    Sensitivity Encoding with Arbitrary k-Space Trajectories.
    Magn. Reson. Med. 46:638-651
    """
    xp, on_gpu = get_array_module(noise, xp)
    noise = xp.asarray(noise)
    coil_axis = coil_axis % noise.ndim
    ncoils = noise.shape[coil_axis]
    if coil_axis != noise.ndim - 1:
        # coil axis must come last
        noise = xp.swapaxes(noise, -1, coil_axis)
    noise = noise.reshape((noise.size // ncoils, ncoils), order="F")
    M = float(noise.shape[0])
    R = (1 / (M - 1)) * xp.dot(noise.T, xp.conj(noise))
    W = xp.linalg.inv(xp.linalg.cholesky(R))
    W = W.T * xp.sqrt(2) * xp.sqrt(scale_factor)
    if return_full:
        return W, R
    else:
        return W


def apply_prewhitening(data, W, order="F", coil_axis=-1, xp=None):
    """Apply the noise prewhitening matrix.

    Parameters
    ----------
    noise : ndarray
        The data to prewhiten.
    W : ndarray
        Input noise prewhitening matrix. This can be computed from noise-only
        data via ``calculate_prewhitening``.
    coil_axis : int
        The axis in ``data`` containing coils.

    Returns
    -------
    w_data : ndarray
        Prewhitened data.

    References
    ----------
    .. [1] Pruessman KP, Weiger M, Bornert P and Boesiger P.  Advances in
    Sensitivity Encoding with Arbitrary k-Space Trajectories.
    Magn. Reson. Med. 46:638-651
    (2001).
    """
    xp, on_gpu = get_array_module(data, xp)
    data = xp.asanyarray(data)
    W = xp.asanyarray(W)
    coil_axis = coil_axis % data.ndim
    ncoils = data.shape[coil_axis]
    if coil_axis != data.ndim - 1:
        # coil axis must come last
        data = xp.swapaxes(data, -1, coil_axis)
    s = data.shape
    data = data.reshape((-1, ncoils), order="F")
    data = xp.dot(data, W).reshape(s, order="F")
    if coil_axis != -1:
        # restore coil axis back to original position
        data = xp.swapaxes(data, -1, coil_axis)
    return data


def prewhiten(data, noise_cal_data, coil_axis=-1, coil_axis_noi=None, xp=None):
    """Noise prewhitening of multichannel MRI data.

    Parameters
    ----------
    data : ndarray
        The data to prewhiten.
    noise_cal_data : ndarray
        Noise calibration data.
    coil_axis : int, optional
        The axis in ``data`` corresponding to coils. By default, the last axis
        is assumed.
    coil_axis_noi : int, optional
        The axis in ``noise_cal_data`` corresponding to coils. By default, the
        last axis is assumed.
    xp : {np, cupy}
        The array module to use.

    Returns
    -------
    data_prewhite
        The prewhitened data.

    W : xp.ndarray
        The noise prewhitening matrix.

    R : xp.ndarray
        The noise correlatin matrix corresponding to ``noise_cal_data``.
    """
    xp, on_gpu = get_array_module(data, xp)
    if coil_axis_noi is None:
        coil_axis_noi = np.argmin(noise_cal_data.shape)

    if not xp.iscomplexobj(data):
        raise ValueError("data must have a complex dtype")

    if data.shape[coil_axis] != noise_cal_data.shape[coil_axis_noi]:
        raise ValueError(
            "data and noise calibration data must have the "
            "same number of channels."
        )

    W, R = calculate_prewhitening(
        noise_cal_data, coil_axis=coil_axis_noi, return_full=True
    )
    data_prewhite = apply_prewhitening(data, W, coil_axis=coil_axis)
    return data_prewhite, W, R
