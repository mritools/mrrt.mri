import os
from os.path import join as pjoin
import time

import numpy as np

from mrrt.utils import ImageGeometry
import mrrt.mri
from mrrt.mri.sim import (
    mri_object_1d,
    mri_object_2d,
    mri_object_3d,
    mri_object_2d_multispectral,
    generate_fieldmap,
    sensemap_sim,  # TODO: rename this as generate_coilmap
)
from mrrt.mri.operators import MRI_Operator

data_dir = pjoin(os.path.dirname(mrrt.mri.__file__), "data")
test_data = np.load(pjoin(data_dir, "multiecho_radial_traj.npz"))


def generate_radial_kspace_from_angles(
    azimuths, elevations, n=128, fov=256, os_factor=2
):
    azimuths = np.atleast_1d(azimuths)
    elevations = np.atleast_1d(elevations)
    n = np.atleast_1d(n)
    fov = np.atleast_1d(fov)
    if len(n) == 1:
        n = np.concatenate((n, n, n))
    if len(fov) == 1:
        fov = np.concatenate((fov, fov, fov))

    if len(n) != 3 or len(fov) != 3:
        raise ValueError("need 3D coordinates")

    # [-n/2, n/2)
    kr = np.arange(-n.max() / 2, n.max() / 2, 1 / os_factor)  # +(.5/os_factor)

    # convert to physical units
    kr /= np.abs(kr).max()
    kr *= np.max(n / fov) / 2

    num_shots = len(azimuths)
    kspace = np.zeros((len(kr), num_shots, 3), dtype=np.float32)
    for n in range(num_shots):
        az = azimuths[n]
        el = elevations[n]
        ca = np.cos(az)
        sa = np.sin(az)
        ce = np.cos(el)
        se = np.sin(el)
        kspace[:, n, 0] = kr * ca * ce
        kspace[:, n, 1] = kr * sa * ce
        kspace[:, n, 2] = kr * se
    kspace = kspace.reshape((-1, 3), order="F")
    return kspace


def genkspace_radial_VPR5(all_rot, weights, res, r=None):
    n_shots = all_rot.shape[2]

    weights = test_data["weights"]
    ksp = test_data["ksp"]
    all_rot = test_data["all_rot"]

    res = res.max()
    if r is None:
        if res <= 200:
            r = 2
        # measured trajectory was for res~240 and is oversampled along read so
        # we may want to reduce the density of the sampling at lower
        # resolutions
        elif res <= 128:
            r = 3
        elif res <= 64:
            r = 6
        elif res <= 32:
            r = 12

    kx1 = ksp[0::r, 0] / 10
    ky1 = ksp[0::r, 1] / 10
    kz1 = ksp[0::r, 2] / 10
    weights = weights[0::r]
    weights = np.tile(weights, (1, n_shots))
    nr = kx1.size
    allkx = np.zeros((nr, 1, n_shots), order="F")
    allky = np.zeros((nr, 1, n_shots), order="F")
    allkz = np.zeros((nr, 1, n_shots), order="F")

    for shotIDX in range(n_shots):
        R = all_rot[:, :, shotIDX]
        allkx[:, 0, shotIDX] = kx1 * R[0, 0] + ky1 * R[0, 1] + kz1 * R[0, 2]
        allky[:, 0, shotIDX] = kx1 * R[1, 0] + ky1 * R[1, 1] + kz1 * R[1, 2]
        allkz[:, 0, shotIDX] = kx1 * R[2, 0] + ky1 * R[2, 1] + kz1 * R[2, 2]

    kspace = np.concatenate(
        (
            allkx.ravel(order="F")[:, np.newaxis],
            allky.ravel(order="F")[:, np.newaxis],
            allkz.ravel(order="F")[:, np.newaxis],
        ),
        axis=1,
    )
    return (kspace, weights, nr)


def generate_ideal_object(
    ndim,
    shape,
    fov=240,
    precision="single",
    return_x_true=False,
    spectral_offsets=None,
):

    if precision == "single":
        obj_dtype = np.complex64
    elif precision == "double":
        obj_dtype = np.complex128
    else:
        raise ValueError("precision must be single or double")

    if ndim == 3:
        if spectral_offsets is not None:
            raise NotImplementedError("TODO")
        ig = ImageGeometry(
            shape=shape, offsets="dsp", fov=fov  # [-n/2:n/2-1] for mri
        )  # 24 cm transaxial fov

        mri_obj = mri_object_3d(ig.fov, units="mm", dtype=obj_dtype)
    elif ndim == 2:
        ig = ImageGeometry(shape=shape, offsets="dsp", fov=fov)
        if spectral_offsets is None:
            # object model
            mri_obj = mri_object_2d(ig.fov, units="mm", dtype=obj_dtype)
        else:
            # mri_obj will be a tuple where the second object represents
            # the spectral offsets for each location in the first
            # (i.e. there are no partial-volume voxels at the moment)
            mri_obj = mri_object_2d_multispectral(
                ig.fov,
                spectral_offsets=spectral_offsets,
                units="mm",
                dtype=obj_dtype,
            )
    elif ndim == 1:
        if spectral_offsets is not None:
            raise NotImplementedError("TODO")
        ig = ImageGeometry(shape=shape, ny=1, offsets="dsp", fov=fov)

        # object model
        mri_obj = mri_object_1d(ig.fov, units="mm", dtype=obj_dtype)
    if return_x_true:
        # samples of continuous-space
        x_true = mri_obj.image(*ig.grid())
        # MUST BE COMPLEX FOR MRI_Operator COMPATIBILITY
        x_true = np.asarray(x_true, dtype=obj_dtype)
        return mri_obj, ig, x_true
    return mri_obj, ig


def generate_ideal_kspace(ndim, mri_obj, kspace, precision="single", t=None):
    if ndim == 1:
        data_true = mri_obj.kspace(kspace[:, 0], t=t)
    elif ndim == 2:
        data_true = mri_obj.kspace(kspace[:, 0], kspace[:, 1], t=t)
    else:
        data_true = mri_obj.kspace(
            kspace[:, 0], kspace[:, 1], kspace[:, 2], t=t
        )
    # samples of continuous-space
    if precision == "double":
        data_true = np.asarray(data_true, dtype=np.complex128)
    elif precision == "single":
        data_true = np.asarray(data_true, dtype=np.complex64)
    else:
        raise ValueError("precision must be single or double")
    return data_true


def generate_MRI_object(
    N,
    kspace,
    fovs,
    smap3d,
    n_shots,
    mask=None,
    recon_case="CPU,Tab",
    J=4,
    grid_os_factor=1.5,
    phasing="real",
    Ld=512,
    precision="single",
    fieldmap_segments=None,
    allti=None,
    zmap=None,
    extra_args={},
    nufft_kwargs={},
    debug=False,
    verbose=False,
    spectral_offsets=None,
    fieldmap_during_init=False,
):
    acq = {}
    acq["traj"] = "radial"
    # stack-of-stars
    acq["dens"] = []
    acq["basis"] = "dirac"

    default_nufft_kwargs = dict(n_shift=N / 2)
    for k, v in default_nufft_kwargs.items():
        if k not in nufft_kwargs:
            nufft_kwargs[k] = v

    ndim = kspace.shape[-1]
    if np.isscalar(N):
        N = (N,) * ndim

    # setup NUFFT arguments.  will modify table_based or not below
    nufft_kwargs["Jd"] = J
    nufft_kwargs["Ld"] = Ld
    nufft_kwargs["phasing"] = phasing
    nufft_kwargs["n_shift"] = N / 2

    if mask is None:
        mask = np.ones(N, dtype=np.bool)
    # mask = tuple(N)

    tstart = time.time()
    if recon_case == "GPU,Sp":
        extra_args["loc_in"] = "gpu"
        extra_args["loc_out"] = "gpu"
        nufft_kwargs["mode"] = "sparse"
    elif recon_case == "GPU,Tab0":
        extra_args["loc_in"] = "gpu"
        extra_args["loc_out"] = "gpu"
        nufft_kwargs["mode"] = "table0"
    elif recon_case == "GPU,Tab":
        extra_args["loc_in"] = "gpu"
        extra_args["loc_out"] = "gpu"
        nufft_kwargs["mode"] = "table1"
    elif recon_case == "CPU,Sp":
        extra_args["loc_in"] = "cpu"
        extra_args["loc_out"] = "cpu"
        nufft_kwargs["mode"] = "sparse"
    elif recon_case == "CPU,Tab":
        extra_args["loc_in"] = "cpu"
        extra_args["loc_out"] = "cpu"
        nufft_kwargs["mode"] = "table1"
    elif recon_case == "CPU,Tab0":
        extra_args["loc_in"] = "cpu"
        extra_args["loc_out"] = "cpu"
        nufft_kwargs["mode"] = "table0"
    else:
        raise ValueError("Invalid recon type: {}".format(recon_case))
    if "n_shift" in nufft_kwargs:
        extra_args["n_shift"] = nufft_kwargs["n_shift"]

    if debug:
        print("default_nufft_kwargs={}".format(default_nufft_kwargs))
        print("nufft_kwargs={}".format(nufft_kwargs))
        print("extra_args={}".format(extra_args))
        print("N={}".format(N))
        # if mask is not None:
        #     print("mask.shape={}".format(mask.shape))
        print("kspace.shape={}".format(kspace.shape))
        if smap3d is not None:
            print("smap3d.shape={}".format(smap3d.shape))
        print("fov={}".format(fovs))

    if spectral_offsets is not None:
        spectral_args = dict(spectral_offsets=spectral_offsets, ti=allti)
    else:
        spectral_args = {}

    on_gpu = "GPU" in recon_case
    if on_gpu:
        import cupy

        kspace = cupy.asarray(kspace)
        if smap3d is not None:
            smap3d = cupy.asarray(smap3d)
        if zmap is not None:
            zmap = cupy.asarray(zmap)
            allti = cupy.asarray(allti)
        # loc_in = loc_out = 'gpu'
    # else:
    # loc_in = loc_out = 'cpu'
    if fieldmap_during_init and fieldmap_segments is not None:
        # configure fieldmap at the time of object initialization
        fieldmap_kwargs = dict(zmap=zmap, fieldmap_segments=fieldmap_segments)
        if not spectral_offsets:
            nti = allti.shape[0] // n_shots
            fieldmap_kwargs["ti"] = allti[:nti]
            fieldmap_kwargs["n_shots"] = n_shots
    else:
        fieldmap_kwargs = {}

    Gn = MRI_Operator(
        shape=N,
        kspace=kspace,
        grid_os_factor=grid_os_factor,
        mask=mask,
        fov=fovs,
        exact=False,
        pixel_basis=acq["basis"],
        coil_sensitivities=smap3d,
        nufft_kwargs=nufft_kwargs,
        on_gpu=on_gpu,
        precision=precision,
        # loc_in=loc_in,
        # loc_out=loc_out,
        **spectral_args,
        **fieldmap_kwargs,
        **extra_args,
    )
    # TODO: can zmap be specified at creation time rather than later via new_zmap?
    # ti=ti, zmap=zmap, fieldmap_segments=6, table_based=True)

    tgen = time.time() - tstart
    if verbose:
        print("tgen = {}".format(tgen))

    if not fieldmap_during_init:
        if fieldmap_segments is not None:
            nti = allti.shape[0] // n_shots
            tstart = time.time()
            Gn.new_zmap(
                ti=allti[:nti],
                zmap=zmap,
                fieldmap_segments=fieldmap_segments,
                n_shots=n_shots,
            )
            tgen_fmap = time.time() - tstart
        else:
            tgen_fmap = 0
    else:
        tgen_fmap = 0

    return Gn, tgen, tgen_fmap


def generate_sim_data(
    recon_case="CPU,Tab",
    mri_obj=None,
    ig=None,
    kspace=None,
    wi_full=None,
    smap3d=None,
    Gn=None,
    ndim=3,
    N0=8,
    J0=3,
    fov=240,
    grid_os_factor=1.5,
    fieldmap_segments=None,
    n_coils=8,
    precision="single",
    phasing="complex",
    Ld=512,
    nufft_kwargs={},
    MRI_object_kwargs={},
    debug=False,
    verbose=False,
    n_shots=None,
    spectral_offsets=None,
):

    nufft_kwargs = nufft_kwargs.copy()

    # CPU,Tab  CPU,Tab0   CPU,Sp
    if ig is not None:
        res = np.atleast_1d(ig.shape)
    else:
        N0 = np.atleast_1d(N0)
        if len(N0) == 1:
            # can set to 32, 64, 128, 192, etc...
            res = np.asarray([N0[0]] * ndim)
        elif len(N0) == ndim:
            res = N0
        else:
            raise ValueError(
                "Invalid Resolution.  Must be an integer or "
                "length ndim array"
            )

    J0 = np.atleast_1d(J0)
    if len(J0) == 1:
        J = [J0] * len(res)  # kernel size along each dimension
    elif len(J0) == ndim:
        J = J0
    else:
        raise ValueError(
            "Invalid J0.  Must be an integer or length ndim " "array"
        )

    if precision == "double":
        real_dtype = np.float64
    elif precision == "single":
        real_dtype = np.float32
    else:
        raise ValueError("precision must be single or double")

    if mri_obj is None:
        mri_obj, ig = generate_ideal_object(
            ndim=ndim,
            shape=res,
            fov=fov,
            precision=precision,
            spectral_offsets=spectral_offsets,
        )
    else:
        if ig is None:
            raise ValueError(
                "if mri_obj is provided, a corresponding "
                "ImageGeometry object must also be provided."
            )

        # make sure ordering of the offsets matches the genrated object
        spectral_offsets = mri_obj.unique_offsets
    x_true = mri_obj.image(*ig.grid())

    # MUST BE COMPLEX FOR MRI_Operator COMPATIBILITY
    x_true = np.asarray(
        x_true, dtype=np.promote_types(real_dtype, np.complex64)
    )

    N = np.asarray(ig.shape)
    if ndim == 1:
        N = N[:1]

    if kspace is None:
        if ndim == 3:
            weights = test_data["weights"]
            all_rot = test_data["all_rot"]

            # There are 4096 shots stored in the file above, just use a subset
            # of them.  Each shot is 5 full 3D projections
            if n_shots is None:
                n_shots = int(np.round(400 * res.max() / 256))
            all_rot = all_rot[:, :, :n_shots]
            (kspace, weights, nr) = genkspace_radial_VPR5(all_rot, weights, res)

            kabs = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=1))
            kmax = kabs.max()
            # rescale kspace extent to match resolution
            kspace = kspace / kmax * ig.shape[0] / np.asarray(ig.fov) / 2
        elif ndim == 2:
            os_factor = 2
            kr = np.arange(
                -max(N) / 2, max(N) / 2, 1 / os_factor
            )  # +(.5/os_factor)
            kr /= np.abs(kr).max()
            kr *= np.max(N / np.asarray(ig.fov)) / 2
            nr = len(kr)
            if n_shots is None:
                n_shots = na = np.ceil(nr * np.pi)
            else:
                na = n_shots
            ang = np.arange(na) / na * np.pi
            n_shots = len(ang)
            kx = np.zeros((nr, n_shots))
            ky = np.zeros((nr, n_shots))
            for shot in range(n_shots):
                kx[:, shot] = np.cos(ang[shot]) * kr
                ky[:, shot] = -np.sin(ang[shot]) * kr
            kspace = np.concatenate(
                (
                    kx.reshape((-1, 1), order="F"),
                    ky.reshape((-1, 1), order="F"),
                ),
                axis=1,
            )
        elif ndim == 1:
            os_factor = 2
            kr = np.arange(
                -max(N) / 2, max(N) / 2, 1 / os_factor
            )  # +(.5/os_factor)
            kr /= np.abs(kr).max()
            kr *= np.max(N / np.asarray(ig.fovs())) / 2
            nr = len(kr)
            n_shots = 1
            kspace = kr.reshape((-1, 1))
    else:
        nr = kspace.shape[0]
        n_shots = 1

    show_kspace = False
    if show_kspace:
        from matplotlib import pyplot as plt

        plt.figure()
        plt.plot(kspace[:1000, 0])
        plt.hold("on")
        plt.plot(kspace[:1000, 1])
        if ndim > 2:
            plt.plot(kspace[:1000, 2])

    if fieldmap_segments is not None or spectral_offsets is not None:
        # make up a fake time vector
        ti = np.linspace(0, 12e-3, nr)
        allti = np.tile(ti, kspace.shape[0] // len(ti))
    else:
        allti = None

    data_true = generate_ideal_kspace(ndim, mri_obj, kspace, t=allti)

    if precision == "single":
        kspace = np.float32(kspace)
        data_complex_dtype = np.complex64
    else:
        kspace = np.float64(kspace)
        data_complex_dtype = np.complex128

    if smap3d is None:
        if n_coils == 1 or n_coils is None:
            smap3d = None
        else:
            smap2d = sensemap_sim(
                shape=ig.shape[:2],
                spacings=ig.distances[:2],
                ncoil=n_coils,
                rcoil=140,
            )

            # generate a rough intensity correction map to account for
            # non-uniform coil sensitivities
            i_corr = 1 / np.sqrt(np.sum(np.abs(smap2d) ** 2, axis=2))
            i_corr = i_corr ** 1.5

            # for simplicity, set sensitivities to constant along z
            if False:
                smap3d = np.zeros(
                    (ig.mask.sum(), n_coils), dtype=data_complex_dtype
                )
                for coilIDX in range(n_coils):
                    smap3d[:, coilIDX] = np.tile(
                        smap2d[:, :, coilIDX][:, :, np.newaxis],
                        [1, 1, ig.shape[2]],
                    )[ig.mask]

            else:
                if ndim == 3:
                    smap3d = np.zeros(
                        (list(N) + [n_coils]),
                        dtype=data_complex_dtype,
                        order="F",
                    )
                    for coilIDX in range(n_coils):
                        smap3d[:, :, :, coilIDX] = np.tile(
                            smap2d[:, :, coilIDX][:, :, np.newaxis],
                            [1, 1, ig.shape[2]],
                        )
                elif ndim == 2:
                    smap3d = smap2d
                smap3d = smap3d.reshape((-1, n_coils), order="F")
                smap3d = smap3d[ig.mask.ravel(order="F"), :]
            if ndim == 3:
                i_corr = np.tile(i_corr[:, :, np.newaxis], (1, 1, ig.shape[2]))

    if fieldmap_segments is not None:
        zmap = generate_fieldmap(N, fmap_peak=80)
    else:
        zmap = None

    on_gpu = "GPU" in recon_case
    if on_gpu:
        import cupy

        xp = cupy
        x_true = cupy.asarray(x_true)
        kspace = cupy.asarray(kspace)
        if zmap is not None:
            zmap = cupy.asarray(zmap)
            kspace = cupy.asarray(kspace)
        if smap3d is not None:
            smap3d = cupy.asarray(smap3d)
    else:
        xp = np

    if Gn is None:
        # if ig.mask is not None:
        #     mask = np.squeeze(ig.mask)
        # else:
        #     mask = None
        Gn, tgen, tgen_fmap = generate_MRI_object(
            N=N,
            kspace=kspace,
            fovs=ig.fov,
            smap3d=smap3d,
            n_shots=n_shots,
            mask=np.squeeze(
                ig.mask
            ),  # TODO: remove need to squeeze ig.mask in 1D case
            recon_case=recon_case,
            J=J,
            grid_os_factor=grid_os_factor,
            phasing=phasing,
            Ld=Ld,
            precision=precision,
            fieldmap_segments=fieldmap_segments,
            allti=allti,
            zmap=zmap,
            extra_args=MRI_object_kwargs,
            nufft_kwargs=nufft_kwargs,
            debug=debug,
            verbose=verbose,
            spectral_offsets=spectral_offsets,
        )
        timings = {}
        timings["MRI: object creation"] = tgen
        if fieldmap_segments is not None:
            timings["MRI: fieldmap init"] = tgen_fmap

    kabs = xp.sqrt(xp.sum(kspace * kspace, axis=1))

    try:
        if wi_full is None:
            if ndim == 3:
                # wi_full=kabs.^2;
                wi_full = weights.flatten(order="F")
            else:
                wi_full = kabs
            if on_gpu:
                wi_full = cupy.asarray(wi_full)

            wi_full = xp.tile(wi_full[:, xp.newaxis], (n_coils, 1))
    except UnboundLocalError:
        # weights or kabs not available (e.g. using user-provided kspace)
        wi_full = None

    return Gn, wi_full, x_true, ig, data_true, timings
