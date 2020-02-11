"""Functions related to Non-Cartesian MRI sampling density compensation.

See also the separate ``mrrt.sdc3`` module for a more well-tested algorithm.
"""
import warnings

import numpy as np


from mrrt.nufft._nufft import nufft_forward, nufft_adj

# TODO: add docstring and literature reference


def mri_dcf_pipe(nufft_op, niter=8, thresh=0.02, verbose=False):
    from mrrt.mri.operators import MRI_Operator, NUFFT_Operator

    if isinstance(nufft_op, MRI_Operator):
        nufft_op = nufft_op.Gnufft
    elif not isinstance(nufft_op, NUFFT_Operator):
        raise ValueError("Gnufft_op must be an MRI_Operator or NUFFT_Operator")
    xp = nufft_op.xp
    wi = xp.ones(nufft_op.shape[0], dtype=nufft_op._cplx_dtype)
    itr = 0
    goal = np.inf
    max_diff = xp.max(xp.abs(goal - 1))
    while max_diff > thresh:
        itr += 1
        goal = nufft_adj(nufft_op, xk=wi, grid_only=True)
        goal = nufft_forward(nufft_op, x=goal, grid_only=True)
        max_diff = xp.max(xp.abs(goal - 1))
        if verbose:
            print("iteration {}, max_err = {}".format(itr, max_diff))
        goal.shape = wi.shape
        wi = wi / goal.real
        if itr > niter:
            warnings.warn(
                "thresh not reached prior to {} iterations".format(niter)
            )
            break
    return wi.real
