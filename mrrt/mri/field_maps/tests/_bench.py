import time
import numpy as np

# from pyir.utils import max_percent_diff
from mrrt.mri import mri_exp_mult
from mrrt.utils import config

if config.have_cupy:
    import cupy


def bench_mri_exp_mult(verbose=True):
    """
    test_mri_exp_mult
    """
    L = 1000
    N = 3000
    M = 200
    rstate = np.random.RandomState(0)
    for dtype in [np.float32, np.float64]:
        if dtype == np.float32:
            atol = rtol = 1e-3
        else:
            atol = rtol = 1e-12
        A = rstate.randn(N, L).astype(dtype)
        A = A + 1j * rstate.randn(N, L).astype(dtype)
        ur = rstate.randn(N).astype(dtype)
        ui = rstate.randn(N).astype(dtype)
        vr = rstate.randn(M).astype(dtype)
        vi = rstate.randn(M).astype(dtype)
        u = ur.astype(dtype) + 1j * ui.astype(dtype)
        v = vr.astype(dtype) + 1j * vi.astype(dtype)

        # Test with ur, v
        tstart = time.time()
        d1 = mri_exp_mult(A, ur, v)
        t_python = time.time() - tstart

        if verbose:
            print("duration (python) = {} s".format(t_python))

        if config.have_cupy:

            tstart = time.time()
            Ag = cupy.asarray(A)
            urg = cupy.asarray(ur)
            vg = cupy.asarray(v)
            d2 = mri_exp_mult(Ag, urg, vg).get()
            t_gpu = time.time() - tstart
            cupy.testing.assert_allclose(d1, d2, atol=atol, rtol=rtol)

            tstart = time.time()
            d3 = mri_exp_mult(Ag, urg, vg)
            t_gpu_single2 = time.time() - tstart
            cupy.testing.assert_allclose(d1, d3.get(), atol=atol, rtol=rtol)
            if verbose:
                print("duration (gpu) = {} s".format(t_gpu))
                print(
                    "duration (gpu, no transfer) = {} s".format(t_gpu_single2)
                )

        # Repeat with u, vr
        d1 = mri_exp_mult(A, u, vr)

        if config.have_cupy:
            vrg = cupy.asarray(vr)
            ug = cupy.asarray(u)
            d2 = mri_exp_mult(Ag, ug, vrg)
            cupy.testing.assert_allclose(d1, d2, atol=atol, rtol=rtol)
    return
