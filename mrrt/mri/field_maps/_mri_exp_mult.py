from __future__ import division, print_function, absolute_import
import numpy as np

from mrrt.utils import get_array_module  # TODO: move

__all__ = ["mri_exp_mult"]


def mri_exp_mult(A, u, v, xp=None, debug=False):
    """
    Y = A.H * exp(-u * v.T)      [L x M]
    """
    xp, on_gpu = get_array_module(A, xp)

    if debug:
        print("A.shape = {}".format(A.shape))
        print("u.shape = {}".format(u.shape))
        print("v.shape = {}".format(v.shape))

    if A.ndim == 1:
        A = A[:, np.newaxis]
    elif A.ndim != 2:
        raise ValueError("A must be 2d")
    n, segs = A.shape

    if u.ndim > 1 or v.ndim > 1:
        raise ValueError("u, v must be 1d")

    u = u[:, np.newaxis]
    v = v[np.newaxis, :]
    m = v.size

    if n != u.size:
        raise ValueError(
            "Inconsistent Dimensions: n={}, u.shape[1]={}".format(n, u.shape[1])
        )

    if debug:
        print("mri_exp_mult:  n={}, m={}, segs={}".format(n, m, segs))

    if v.size < 4e6:
        tmp = -xp.dot(u, v)
        xp.exp(tmp, out=tmp)
        return xp.dot(xp.conj(A.T), tmp)
    else:
        # break into chunks to reduce memory required
        nchunks = int(xp.ceil(v.size / 1e6))
        nper = int(xp.ceil(v.size / nchunks))
        arrays = []
        for ci in range(nchunks):
            print("computing chunk {} of {}".format(ci + 1, nchunks))
            if ci == nchunks - 1:
                sl = (slice(None), slice(ci * nper, v.size))
            else:
                sl = (slice(None), slice(ci * nper, (ci + 1) * nper))
            tmp = -xp.dot(u, v[sl])
            tmp = xp.exp(tmp, out=tmp)
            arrays.append(xp.dot(xp.conj(A.T), tmp))
        return xp.concatenate(arrays, axis=1)
