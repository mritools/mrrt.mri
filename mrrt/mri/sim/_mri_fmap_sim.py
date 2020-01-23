import numpy as np

__all__ = ["generate_fieldmap"]


def generate_fieldmap(shape, fmap_peak=80):
    """Return a 2d or 3d Gaussian bump for use as a simulated filedmap.

    Parameters
    ----------
    shape : tuple of int
        The shape of the fieldmap.
    fmap_peak : float, optional
        The maximum magnitude of the fieldmap, in Hz.

    Returns
    -------
    z_map : ndarray
        The field map in complex form as expected by the MRI NonCartesian
        operator. The imaginary component corresponds to the off-resonance map
        in radians. (The real component corresponds to relaxation rather than
        off-resonance and will be zero).

    Notes
    -----
    ``z_map = r_map + 1j * f_map`` where ``r_map`` is a T2* relaxation map and
    ``f_map`` is a fieldmap in radians.

    """
    ndim = len(shape)
    if ndim < 2 or ndim > 3:
        raise ValueError("Only 2D and 3D cases supported")
    # make up a fake gaussian fieldmap...
    a = 2.5
    k1 = np.arange(-(shape[0] - 1) / 2, (shape[0] - 1) / 2 + 1)
    w1 = np.exp(-0.5 * (a * k1 / (shape[0] / 2)) ** 2)
    k2 = np.arange(-(shape[1] - 1) / 2, (shape[1] - 1) / 2 + 1)
    w2 = np.exp(-0.5 * (a * k2 / (shape[1] / 2)) ** 2)

    if ndim == 3:
        k3 = np.arange(-(shape[2] - 1) / 2, (shape[2] - 1) / 2 + 1)
        w3 = np.exp(-0.5 * (a * k3 / (shape[2] / 2)) ** 2)
        w1 = w1[:, None, None]
        w2 = w2[None, :, None]
        w3 = w3[None, None, :]
        z_map = 0 + 1j * 2 * np.pi * fmap_peak * (w1 * w2 * w3)
    elif ndim == 2:
        w1 = w1[:, None]
        w2 = w2[None, :]
        z_map = 0 + 1j * 2 * np.pi * fmap_peak * (w1 * w2)

    return z_map
