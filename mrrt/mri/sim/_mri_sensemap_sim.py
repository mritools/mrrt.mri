import numpy as np
from scipy.special import ellipk, ellipe

from mrrt.utils import get_array_module  # TODO: move

__all__ = ["sensemap_sim"]


def _mri_smap1(x, y, z, a, xp=None):
    """circular coil in "x-y plane" of radius a

    note that coil x-y plane is not same as object x-y plane!
    """
    xp, on_gpu = get_array_module(x)
    x = x / a
    y = y / a
    z = z / a
    r = xp.sqrt(x * x + y * y)
    r[r == 0] = 1e-7  # avoid divide by zero

    zsq = z * z
    rp1sq = (r + 1) ** 2

    M = 4 * r / (rp1sq + zsq)
    # elliptic integral of the first and second kind
    if xp != np:
        # CuPy doesn't currently have ellipk or ellipe so have to transfer to
        # the CPU to evaluate the elliptic integrals.
        M = M.get()
    (K, E) = ellipk(M), ellipe(M)
    if xp != np:
        K, E = map(xp.asarray, (K, E))

    # B_z in eqn (18) in grivich:00:tmf
    tmp = (rp1sq + zsq) ** (-0.5)
    tmp2 = (1 - r) ** 2 + zsq
    rsq = r * r
    smap_z = 2 * tmp * (K + (1 - rsq - zsq) / tmp2 * E)
    smap_z /= a

    # B_r in eqn (17) in grivich:00:tmf
    smap_r = 2 * z / r * tmp * ((1 + rsq + zsq) / tmp2 * E - K)

    bad = xp.abs(r) < 1e-6
    smap_r[bad] = 3 * xp.pi * z[bad] / ((1 + z[bad] ** 2) ** 2.5) * r[bad]
    smap_r /= a

    if xp.any(xp.isnan(smap_r)) or xp.any(xp.isnan(smap_z)):
        raise Exception("Nan found in smap")

    phi = xp.arctan2(y, x)
    smap_x = smap_r * xp.cos(phi)
    smap_y = smap_r * xp.sin(phi)

    return (smap_x, smap_y, smap_z)


def sensemap_sim(
    shape=(64, 64),
    spacings=(3, 3),
    ncoil=8,
    rcoil=100,
    orbit=360,
    orbit_start=None,
    coil_distance=1.5,
    nring=1,
    dz_coil=None,
    scale="default",
    dtype=np.complex128,
    xp=np,
):
    """Simulate sensitivity maps for sensitivity-encoded MRI.

    Parameters
    ----------
    shape : tuple of int
        The image or volume size. Must have either two or 3 elements.
    spacings : tuple of int
        The voxel dimensions. Should have equal length to shape. If an integer
        is provided, the same value is assumed for all axes.
    ncoil : int, optional
        Total number of coils (for all rings).
    rcoil : int or None optional
        The radius of an individual coil element. If None, it will default to
        ``shape[0] * spacings[0] / 4``.
    coil_distance : float, optional
        Distance of coil center from isocenter for central ring of coils as a
        multiple of fov_x, where ``fov_x = nx * dx``.
    orbit : float, optional
        Angular range around the cylinder covered.
    orbit_start : list of float, optional
        Can be a list of length nring. Offsets to the start of each ring in
        degrees.
    nring : int, optional
        Number of rings of coils (along a cylinder in z).
    dz_coil : float, optional
        Ring spacing in z (defaults to ``shape[2] * spacings[2] / nring``).
    scale : {'default', 'ssos'}, optional
        If ``'ssos'``, scale so that the sqrt of the sum of square of the
        center equals 1.
    dtype : {np.complex64, np.complex128}, optional
        Data type for the generated sensitivity maps.

    Returns
    -------
    smap : array
        The simulated sensitivity maps. ``smap.shape`` will be
        ``shape + (ncoils,)``.

    Notes
    -----
    based on [1]_
    Adapted from a Matlab implementation Copyright 2005-2016,
    Jeff Fessler, Amanda Funai and Mei Le, University of Michigan
    Python adaptation by Gregory Lee.

    References
    ----------
    .. [1] Grivich MI, Jackson DP.  The magnetic field of current-carrying
    polygons: An application of vector field rotations.
    Am. J. Phys. 68, 469 (2000).  doi:10.1119/1.19461
    """
    if len(shape) == 2:
        shape = tuple(shape) + (1,)
        if len(spacings) == 2:
            spacings = tuple(spacings) + (1,)

    if len(shape) != 3:
        raise ValueError("shape must have length 2 or 3")
    if np.isscalar(spacings):
        spacings = (spacings,) * 3
    nx, ny, nz = shape
    dx, dy, dz = spacings
    if rcoil is None:
        rcoil = dx * nx / 2 * 0.50
    if dz_coil is None:
        dz_coil = dz * nz / nring

    coils_per_ring = int(np.round(ncoil / nring))
    if nring * coils_per_ring != ncoil:
        raise ValueError("nring must be a divisor of ncoil")

    if dtype == np.complex128:
        real_dtype = np.float64
    elif dtype == np.complex64:
        real_dtype = np.float32
    else:
        raise ValueError("unsupported dtype")

    # coil radii
    rlist = rcoil * xp.ones((coils_per_ring, nring), dtype=real_dtype)

    # position of coil center (x, y, z)
    plist = xp.zeros((coils_per_ring, nring, 3), dtype=real_dtype)
    # normal vector (inward) from coil center
    nlist = xp.zeros((coils_per_ring, nring, 3), dtype=real_dtype)
    # unit vector orthogonal to normal vector in x-y
    olist = xp.zeros((coils_per_ring, nring, 3), dtype=real_dtype)
    # upward vector
    # ulist = xp.zeros((coils_per_ring, nring, 3), dtype=real_dtype)

    if orbit_start is None:
        orbit_start = [0] * nring
    if xp.isscalar(orbit_start):
        orbit_start = orbit_start[0] * nring
    elif len(orbit_start) == 1:
        orbit_start = [orbit_start[0]] * nring
    elif len(orbit_start) != nring:
        raise ValueError(
            "orbit_start should be a single value or a list of length nring"
        )

    # circular coil configuration, like head coils
    # list of angles in radians
    # ncoil) / coils_per_ring
    alist = xp.deg2rad(orbit) * xp.linspace(0, 1, coils_per_ring + 1)[:-1]
    z_ring = (xp.arange(1, nring + 1) - (nring + 1) / 2) * dz_coil
    for ir in range(nring):
        alist_ring = alist + xp.deg2rad(orbit_start[ir])
        for ic in range(coils_per_ring):
            phi = alist_ring[ic]
            maxval = xp.max(xp.asarray([nx / 2 * dx, ny / 2 * dy]))
            rad = maxval * coil_distance
            cp = xp.cos(phi)
            sp = xp.sin(phi)
            plist[ic, ir, :] = xp.array(
                [rad * cp, rad * sp, z_ring[ir]], dtype=real_dtype
            )
            nlist[ic, ir, :] = -xp.array(
                [cp, sp, 0], dtype=real_dtype  # cylinder
            )
            olist[ic, ir, :] = xp.array([-sp, cp, 0], dtype=real_dtype)
            # ulist[ic, ir, :] = xp.array([0, 0, 1], dtype=real_dtype)

    # object coordinates for slice z=0
    x = (xp.arange(1, nx + 1, dtype=real_dtype) - (nx + 1) / 2) * dx
    y = (xp.arange(1, ny + 1, dtype=real_dtype) - (ny + 1) / 2) * dy
    z = (xp.arange(1, nz + 1, dtype=real_dtype) - (nz + 1) / 2) * dz
    xx, yy, zz = xp.meshgrid(x, y, z, indexing="ij")
    # zz = xp.zeros_like(xx)

    smap = xp.zeros((nx, ny, nz, coils_per_ring, nring), dtype=dtype)
    for ir in range(nring):
        for ic in range(coils_per_ring):

            if nlist[ic, ir, 2] or olist[ic, ir, 2]:
                # assume z component of plist and nlist are 0
                raise Exception("Unsupported")

            # rotate coordinates to correspond to coil orientation
            zr = (
                (xx - plist[ic, ir, 0]) * nlist[ic, ir, 0]
                + (yy - plist[ic, ir, 1]) * nlist[ic, ir, 1]
                + (zz - plist[ic, ir, 2]) * nlist[ic, ir, 2]
            )
            xr = xx * nlist[ic, ir, 1] - yy * nlist[ic, ir, 0]
            yr = zz - plist[ic, ir, 2]  # translate along object z axis

            # compute sensitivity vectors in coil coordinates
            (sx, sy, sz) = _mri_smap1(xr, yr, zr, rlist[ic, ir])

            # coil response depends on tranverse magnetization only?
            # TODO: unsure if this should depend on sy and ulist in 3D
            bx = sz * nlist[ic, ir, 0] + sx * olist[ic, ir, 0]
            by = sz * nlist[ic, ir, 1] + sx * olist[ic, ir, 1]
            # bz = sz * nlist[ic, ir, 2] + sx * olist[ic, ir, 2]
            smap[:, :, :, ic, ir] = bx + 1.0j * by

    smap = smap * rlist[0] / (2 * xp.pi)  # trick: scale so near unity maximum

    if nz == 1:
        smap = smap.reshape((nx, ny, ncoil), order="F")
        scale_center = 1 / xp.sqrt(
            xp.sum(xp.abs(smap[nx // 2, ny // 2, :] ** 2))
        )
    else:
        smap = smap.reshape((nx, ny, nz, ncoil), order="F")
        scale_center = 1 / xp.sqrt(
            xp.sum(xp.abs(smap[nx // 2, ny // 2, nz // 2, :] ** 2))
        )

    if scale.lower() == "ssos":
        smap *= scale_center
    elif scale != "default":
        raise ValueError("unrecognized scale: {}".format(scale))

    return smap
