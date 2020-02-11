"""Tests related to Non-Cartesian MRI reconstruction."""
from itertools import product
import time

import numpy as np
from numpy.testing import assert_, assert_equal
import pytest

from mrrt.operators.LinOp import DiagonalOperatorMulti
from mrrt.mri.operators.tests._generate_testdata import generate_sim_data
from mrrt.nufft import dtft, dtft_adj
from mrrt.utils import embed, have_cupy, profile

import os

OMIT_CPU = int(os.environ.get("OMIT_CPU", False))
OMIT_GPU = int(os.environ.get("OMIT_GPU", False))

all_xp = [np]
cpu_cases = ["CPU,Tab0", "CPU,Tab", "CPU,Sp"] if not OMIT_CPU else []
all_cases = cpu_cases
if have_cupy:
    import cupy

    if cupy.cuda.runtime.getDeviceCount() > 0 and not OMIT_GPU:
        gpu_cases = ["GPU,Tab0", "GPU,Tab", "GPU,Sp"]
        all_cases += gpu_cases
        all_xp += [cupy]

# To ignore PendingDeprecationWarning related to scipy.sparse we use:
#     @pytest.mark.filterwarnings("ignore:the matrix subclass is not")
#
# This class of warnings could also be ignored via the command line, e.g.:
#     pytest -W ignore::PendingDeprecationWarning test_mri_reconstruction.py


@profile
def _test_mri_multi(
    ndim=3,
    N0=8,
    grid_os_factor=1.5,
    J0=4,
    Ld=4096,
    n_coils=1,
    fieldmap_segments=None,
    precisions=["single", "double"],
    phasings=["real", "complex"],
    recon_cases=["CPU,Tab0", "CPU,Tab", "CPU,Sp"],
    rtol=1e-3,
    compare_to_exact=False,
    show_figures=False,
    nufft_kwargs={},
    navg_time=1,
    n_creation=1,
    return_errors=False,
    gpu_memflags=None,
    verbose=False,
    return_operator=False,
    spectral_offsets=None,
):
    """Run a batch of NUFFT tests."""
    all_err_forward = np.zeros(
        (len(recon_cases), len(precisions), len(phasings))
    )
    all_err_adj = np.zeros((len(recon_cases), len(precisions), len(phasings)))
    alltimes = {}
    if not np.isscalar(navg_time):
        navg_time_cpu, navg_time_gpu = navg_time
    else:
        navg_time_cpu = navg_time_gpu = navg_time
    for i, recon_case in enumerate(recon_cases):
        if "CPU" in recon_case:
            navg_time = navg_time_cpu
        else:
            navg_time = navg_time_gpu

        for j, precision in enumerate(precisions):
            for k, phasing in enumerate(phasings):
                if verbose:
                    print(
                        "phasing={}, precision={}, type={}".format(
                            phasing, precision, recon_case
                        )
                    )

                if "Tab" in recon_case:
                    # may want to create twice when benchmarking GPU case
                    # because the custom kernels are compiled the first time
                    ncr_max = n_creation
                else:
                    ncr_max = 1
                # on_gpu = ('GPU' in recon_case)
                for ncr in range(ncr_max):
                    (
                        Gn,
                        wi_full,
                        xTrue,
                        ig,
                        data_true,
                        times,
                    ) = generate_sim_data(
                        recon_case=recon_case,
                        ndim=ndim,
                        N0=N0,
                        J0=J0,
                        grid_os_factor=grid_os_factor,
                        fieldmap_segments=fieldmap_segments,
                        Ld=Ld,
                        n_coils=n_coils,
                        precision=precision,
                        phasing=phasing,
                        nufft_kwargs=nufft_kwargs,
                        MRI_object_kwargs=dict(gpu_memflags=gpu_memflags),
                        spectral_offsets=spectral_offsets,
                    )

                xp = Gn.xp

                # time the forward operator
                sim_data = Gn * xTrue  # dry run
                tstart = time.time()
                for nt in range(navg_time):
                    sim_data = Gn * xTrue
                    sim_data += 0.0
                sim_data = xp.squeeze(sim_data)  # TODO: should be 1D already?
                # print("type(xTrue) = {}".format(type(xTrue)))
                # print("type(sim_data) = {}".format(type(sim_data)))
                t_for = (time.time() - tstart) / navg_time
                times["MRI: forward"] = t_for

                # time the norm operator
                Gn.norm(xTrue)  # dry run
                tstart = time.time()
                for nt in range(navg_time):
                    Gn.norm(xTrue)
                t_norm = (time.time() - tstart) / navg_time

                times["MRI: norm"] = t_norm
                if precision == "single":
                    dtype_real = np.float32
                    dtype_cplx = np.complex64
                else:
                    dtype_real = np.float64
                    dtype_cplx = np.complex128

                if "Tab" in recon_case:
                    if phasing == "complex":
                        assert_equal(Gn.Gnufft.h[0].dtype, dtype_cplx)
                    else:
                        assert_equal(Gn.Gnufft.h[0].dtype, dtype_real)
                else:
                    if phasing == "complex":
                        assert_equal(Gn.Gnufft.p.dtype, dtype_cplx)
                    else:
                        assert_equal(Gn.Gnufft.p.dtype, dtype_real)
                assert_equal(sim_data.dtype, dtype_cplx)

                if compare_to_exact:
                    # compare_to_exact only currently for single-coil,
                    # no fieldmap case
                    if spectral_offsets is not None:
                        raise NotImplementedError(
                            "compare_to_exact doesn't currently support "
                            "spectral offsets"
                        )
                    nshift_exact = tuple(s / 2 for s in Gn.Nd)
                    sim_data2 = dtft(
                        xTrue, Gn.omega, shape=Gn.Nd, n_shift=nshift_exact
                    )

                    sd2_norm = xp.linalg.norm(sim_data2)
                    rel_err = xp.linalg.norm(sim_data - sim_data2) / sd2_norm
                    if "GPU" in recon_case:
                        if hasattr(rel_err, "get"):
                            rel_err = rel_err.get()
                    all_err_forward[i, j, k] = rel_err
                    print(
                        "{},{},{}: forward error = {}".format(
                            recon_case, precision, phasing, rel_err
                        )
                    )
                    rel_err_mag = (
                        xp.linalg.norm(np.abs(sim_data) - np.abs(sim_data2))
                        / sd2_norm
                    )
                    print(
                        f"{recon_case},{precision},{phasing}: "
                        f"forward mag diff error = {rel_err_mag}"
                    )
                    assert rel_err < rtol

                # TODO: update DiagonalOperatorMulti to auto-set loc_in,
                #       loc_out appropriately
                if xp is np:
                    diag_args = dict(loc_in="cpu", loc_out="cpu")
                else:
                    diag_args = dict(loc_in="gpu", loc_out="gpu")
                diag_op = DiagonalOperatorMulti(wi_full, **diag_args)
                if n_coils == 1:
                    data_dcf = diag_op * data_true
                else:
                    data_dcf = diag_op * sim_data

                # time the adjoint operation
                im_est = Gn.H * data_dcf  # dry run
                tstart = time.time()
                for nt in range(navg_time):
                    im_est = Gn.H * data_dcf
                t_adj = (time.time() - tstart) / navg_time
                times["MRI: adjoint"] = t_adj

                if hasattr(Gn, "mask") and Gn.mask is not None:
                    im_est = embed(im_est, Gn.mask)
                else:
                    if spectral_offsets is None:
                        im_est = im_est.reshape(Gn.Nd, order=Gn.order)
                    else:
                        im_est = im_est.reshape(
                            tuple(Gn.Nd) + (len(spectral_offsets),),
                            order=Gn.order,
                        )

                if compare_to_exact:
                    im_est_exact = dtft_adj(
                        data_dcf, Gn.omega, shape=Gn.Nd, n_shift=nshift_exact
                    )
                    ex_norm = xp.linalg.norm(im_est_exact)
                    rel_err = xp.linalg.norm(im_est - im_est_exact) / ex_norm
                    all_err_adj[i, j, k] = rel_err
                    if verbose:
                        print(
                            "{},{},{}: adjoint error = {}".format(
                                recon_case, precision, phasing, rel_err
                            )
                        )
                    rel_err_mag = (
                        xp.linalg.norm(np.abs(im_est) - np.abs(im_est_exact))
                        / ex_norm
                    )
                    if verbose:
                        print(
                            "{},{},{}: adjoint mag diff error = {}".format(
                                recon_case, precision, phasing, rel_err
                            )
                        )
                    assert_(rel_err < rtol)

                title = ", ".join([recon_case, precision, phasing])
                if show_figures:
                    from matplotlib import pyplot as plt
                    from pyvolplot import volshow

                    if compare_to_exact:
                        volshow(
                            [
                                im_est_exact,
                                im_est,
                                im_est_exact - im_est,
                                xp.abs(im_est_exact) - xp.abs(im_est),
                            ]
                        )
                    else:
                        volshow(im_est)
                        plt.title(title)
                alltimes[title] = times

    if return_operator:
        if return_errors:
            return Gn, alltimes, all_err_forward, all_err_adj
        return Gn, alltimes

    if return_errors:
        return alltimes, all_err_forward, all_err_adj
    return alltimes


@pytest.mark.parametrize(
    "recon_case, precision, phasing",
    product(all_cases, ["single", "double"], ["complex", "real"]),
)
@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_mri_2d_nocoils_nofieldmap_nocompare(
    recon_case, precision, phasing, show_figures=False, verbose=False
):

    _test_mri_multi(
        ndim=2,
        N0=16,
        grid_os_factor=1.5,
        J0=6,
        Ld=4096,
        n_coils=1,
        fieldmap_segments=None,
        precisions=[precision],
        phasings=[phasing],
        recon_cases=[recon_case],
        show_figures=show_figures,
        verbose=verbose,
        compare_to_exact=False,
        nufft_kwargs={},
        rtol=1e-4,
    )


# @dec.slow
@pytest.mark.parametrize(
    "recon_case, precision, phasing",
    product(all_cases, ["single", "double"], ["complex", "real"]),
)
@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_mri_2d_nocoils_nofieldmap(
    recon_case, precision, phasing, show_figures=False, verbose=False
):

    _test_mri_multi(
        ndim=2,
        N0=16,
        grid_os_factor=1.5,
        J0=6,
        Ld=4096,
        n_coils=1,
        fieldmap_segments=None,
        precisions=[precision],
        phasings=[phasing],
        recon_cases=[recon_case],
        show_figures=show_figures,
        verbose=verbose,
        compare_to_exact=True,
        nufft_kwargs={},
        rtol=1e-3,
    )


# @dec.slow
@pytest.mark.parametrize(
    "recon_case, precision, phasing",
    product(all_cases, ["single", "double"], ["complex", "real"]),
)
@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_mri_2d_nocoils_nofieldmap_kernels(
    recon_case, precision, phasing, show_figures=False, verbose=False
):
    N = 32
    grid_os_factor = 2
    J = 6
    t, ef, ea = _test_mri_multi(
        ndim=2,
        N0=N,
        grid_os_factor=grid_os_factor,
        J0=J,
        Ld=512,
        n_coils=1,
        fieldmap_segments=None,
        precisions=[precision],
        phasings=[phasing],
        recon_cases=[recon_case],
        show_figures=show_figures,
        verbose=verbose,
        compare_to_exact=True,
        nufft_kwargs={},
        rtol=1e-2,
        return_errors=True,
    )


# @dec.slow
@pytest.mark.parametrize(
    "recon_case, precision, phasing",
    product(all_cases, ["single", "double"], ["complex", "real"]),
)
@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_mri_2d_multicoils_nofieldmap(
    recon_case, precision, phasing, show_figures=False, verbose=False
):
    _test_mri_multi(
        ndim=2,
        N0=16,
        grid_os_factor=1.5,
        J0=6,
        Ld=4096,
        n_coils=4,
        fieldmap_segments=None,
        precisions=[precision],
        phasings=[phasing],
        recon_cases=[recon_case],
        compare_to_exact=False,
        rtol=1e-3,
        show_figures=show_figures,
        verbose=verbose,
    )


# @dec.slow
@pytest.mark.parametrize(
    "recon_case, precision, phasing",
    product(all_cases, ["single", "double"], ["complex", "real"]),
)
@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_mri_2d_multicoils_fieldmap(
    recon_case, precision, phasing, show_figures=False, verbose=False
):
    _test_mri_multi(
        ndim=2,
        N0=16,
        grid_os_factor=1.5,
        J0=6,
        Ld=4096,
        n_coils=4,
        fieldmap_segments=6,
        precisions=[precision],
        phasings=[phasing],
        recon_cases=[recon_case],
        compare_to_exact=False,
        show_figures=show_figures,
        verbose=verbose,
    )


# @dec.slow
@pytest.mark.parametrize(
    "recon_case, precision, phasing",
    product(all_cases, ["single", "double"], ["complex", "real"]),
)
@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_mri_3d_nocoils_nofieldmap(
    recon_case, precision, phasing, show_figures=False, verbose=False
):
    if "Tab0" in recon_case:
        rtol = 1e-2
    else:
        rtol = 1e-3
    _test_mri_multi(
        ndim=3,
        N0=12,
        grid_os_factor=1.5,
        J0=4,
        Ld=4096,
        n_coils=1,
        fieldmap_segments=None,
        precisions=[precision],
        phasings=[phasing],
        recon_cases=[recon_case],
        compare_to_exact=True,
        show_figures=show_figures,
        rtol=rtol,
        verbose=verbose,
    )


# @dec.slow
@pytest.mark.parametrize(
    "recon_case, precision, phasing",
    product(all_cases, ["single", "double"], ["complex", "real"]),
)
@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_mri_3d_multicoils_nofieldmap(
    recon_case, precision, phasing, show_figures=False, verbose=False
):
    _test_mri_multi(
        ndim=3,
        N0=12,
        grid_os_factor=1.5,
        J0=4,
        Ld=4096,
        n_coils=4,
        fieldmap_segments=None,
        precisions=[precision],
        phasings=[phasing],
        recon_cases=[recon_case],
        compare_to_exact=False,
        show_figures=show_figures,
        verbose=verbose,
    )


# @dec.slow
@pytest.mark.parametrize(
    "recon_case, precision, phasing",
    product(all_cases, ["single", "double"], ["complex", "real"]),
)
@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_mri_3d_multicoils_fieldmap(
    recon_case, precision, phasing, show_figures=False, verbose=False
):
    _test_mri_multi(
        ndim=3,
        N0=12,
        grid_os_factor=1.5,
        J0=4,
        Ld=4096,
        n_coils=4,
        fieldmap_segments=6,
        precisions=[precision],
        phasings=[phasing],
        recon_cases=[recon_case],
        compare_to_exact=False,
        show_figures=show_figures,
        verbose=verbose,
    )
