"""Non-Cartesian MRI Operator.

Non-Cartesian MRI Operator
==========================
Class implementing forward and adjoint operators for Non-Cartesian MRI. This
includes both coil maps and field maps for correction of off-resonance (and
potentially relaxation).

TODO: expand documentation here


"""

from __future__ import division, print_function, absolute_import
from math import ceil
import warnings
import numpy as np
from mrrt.mri import mri_exp_approx
from mrrt.mri.operators import NUFFT_Operator
from mrrt.utils import (
    complexify,
    get_array_module,
    have_cupy,
    masker,
    next_fast_len_multiple,
    power_method,
    prod,
    profile,
)

from mrrt.operators import BlockDiagLinOp, DiagonalOperator, LinearOperatorMulti
from mrrt.mri.operators._pixel_basis import PixelBasis

# from mrrt.nufft import compute_Q

if have_cupy:
    import cupy


# TODO: implement Toeplitz approach?

# TODO: allow fmap_basis to be only a single shot for multi-shot trajectories
# where each shot has the same temporal readout points.  This can be used to
# reduce GPU memory usage of fieldmap-corrected MRI

# TODO: There is an occasional non-reproducable failure on GPU only
#       for some table-based NUFFTs:
#       OMIT_CPU=1 py.test test_MRI_reconstruction.py -k "test_mri_2d_nocoils_nofieldmap_kernels" -v


def exp_xform(x, u, v, xp=None):
    xp, on_gpu = get_array_module(x, xp)

    x = xp.asarray(x)
    u = xp.asarray(u)
    v = xp.asarray(v)

    if x.ndim == 1:
        x = x[:, np.newaxis]

    if x.shape[0] != u.shape[1]:
        print("x.shape[0]=%d, u.shape[1]=%d." % (x.shape[0], u.shape[1]))
        raise ValueError("dimensions inconsistent")

    if u.shape[0] != v.shape[0]:
        print("u.shape[0]=%d, x.shape[0]=%d." % (u.shape[0], x.shape[0]))
        raise ValueError("dimensions inconsistent")

    tmp = -xp.dot(v.T, u)
    xp.exp(tmp, out=tmp)
    return xp.dot(tmp, x)


# @profile
def determine_fieldmap_segments(
    ti,
    zmap,
    mask=None,
    approx_type="default",
    fieldmap_segments_start=5,
    err_thresh=1e-3,
    exp_approx_args={},
    xp=np,
):
    """Determine an appropriate number of fieldmap segments."""
    fieldmap_segments = fieldmap_segments_start
    fieldmap_segments_adjust = 0
    error_acceptable = False
    if mask is not None:
        zmap = masker(zmap, mask, xp=xp)
    while error_acceptable is False:
        fmap_basis, C, hk, zk = mri_exp_approx(
            ti=ti,
            zmap=zmap,
            segments=fieldmap_segments,
            approx_type=approx_type,
            ctest=True,
            **exp_approx_args,
        )
        Eh = xp.exp(-ti[:, None] * zk.T)
        Ep = xp.dot(fmap_basis, C)
        err = xp.abs(Eh - Ep)
        # nhist = len(zk)
        # mse = xp.mean(err**2, axis=0)
        # wrms = xp.sqrt((mse[None, :] * hk[:, None]) / hk.sum())
        # ik = xp.round(xp.linspace(1,nhist,8))
        err_max = xp.max(err.mean())
        print(
            "fieldmap_segments={}, err_max={}".format(
                fieldmap_segments, err_max
            )
        )

        if err_max > err_thresh:
            print("approximation is poor.  increasing fieldmap_segments by 1")
            fieldmap_segments += 1
            fieldmap_segments_adjust += 1  # try again
        else:
            # accuracy is satisfactory as is
            error_acceptable = True

        if fieldmap_segments_adjust > 9:
            raise ValueError(
                "Unable to get reasonable accuracy with small increase in fieldmap_segments"
            )
    return fieldmap_segments


class MRI_Operator(LinearOperatorMulti):
    """MRI_Operator object for non-Cartesian MR image reconstruction.

    Note: This object can also model coil sensitivity maps and relaxation or
    off-resonance effects for MRI.

    create a system matrix object by calling:
    A = MRI_Operator( ... )
    The forward operation is then:
        y = A * x;
    The adjoint operation is then:
        y = A.H * x;

    Intended for use with iterative image reconstruction in MRI.
    """

    __weights = None
    __unitary_scaling = False
    __pixel_basis = "dirac"

    @profile
    def __init__(
        self,
        kspace,
        shape,
        mask=None,
        grid_os_factor=1.5,
        pixel_basis="dirac",
        weights=None,
        spectral_offsets=None,
        on_gpu=False,
        precision="single",
        # dtype=np.complex64,  # not used. precision determines this
        nufft_kwargs={},
        exact=False,
        fov=1,
        n_shift=None,
        exp_approx_args={},
        order="F",
        coil_sensitivities=None,
        unitary_scaling=False,
        squeeze_reps=True,
        verbose=False,
        xp=None,
        **kwargs,
    ):
        """Initialize a non-Cartesian MRI_Operator.

        Parameters
        ----------
        kspace : array_like
            2D array of k-space coordinates :math:`[n_{samples}, n_{dim}]`.
            (in inverse units of fov)
        mask : array_like or None
            boolean mask for the image array
        pixel_basis : {'dirac', 'rect'}, optional
            pixel basis for the continuous-to-discrete representation
        coil_sensitivities : array_like, optional
            2D array of coil sensitivity maps :math:`[n_{voxels}, n_{coils}]`.
        zmap : array_like, optional
            relax_map + 2j*pi*field_map.  real part is relaxation, imaginary
            part is the off-resonance field map.
        ti : array_like, optional
            time vector corresponding to the field map

        Other Parameters
        ----------------
        exp_approx_args : dict, optional
            additional kwargs for `mri_exp_approx`
        nufft_kwargs : dict, optional
            additional kwargs for `NUFFT_Operator`
        exact : bool, optional
            if True use slow, exact transform.  if False (default), use NUFFT
        n_shift : array_like, optional
            image domain shift for exact case
        fov : array_like, optional
            image field of view (e.g. in mm)
        Gnufft : NUFFT_Operator
            use the provided Gnufft operator instead of generating a new one
        fieldmap_segments : int, optional
            number of fieldmap approximation terms (see `mri_exp_approx`)
        acorr_fieldmap_segments : int, optional
            number of fieldmap approximation terms for autocorrelation
            histogram in Toeplitz version
        n_shots : int, optional
            Used to specify that the trajectory is multiple shots over
            identical time intervals.  Used to speed up zmap calculation by
            passing in `ti` for a single repetition.
        dtype : {np.float32, np.float64}
            used to specify the precision

        Attributes
        ----------
        TODO

        Notes
        -----
        any additional kwargs aside from those passed in above get passed onto
        the NUFFT_Operator object

        Update or change the zmap (e.g. for dynamic cases) via:
        A.new_zmap(zmap=zmap, ti=ti)

        Extended from original Matlab implementation:
            by Jeff Fessler, The University of Michigan

        """
        if on_gpu:
            xp = cupy
            kspace = cupy.asarray(kspace)
            loc_in = loc_out = "gpu"
        else:
            xp, on_gpu = get_array_module(kspace, xp=np)
            loc_in = loc_out = "cpu"
        self._on_gpu = on_gpu
        shape = tuple(shape)
        self.ndim = len(shape)
        self.Nd = shape  # TODO: remove?

        if "nufft_args" in kwargs:
            raise ValueError(
                "use of nufft_args is outdated. use nufft_kwargs instead."
            )

        if mask is None:
            self.mask = None
            self.nmask = prod(shape)
        else:
            self.mask = xp.asarray(mask, dtype=bool)
            if self.mask.shape != shape:
                raise ValueError("mask.shape must match the specified shape")
            nmask = xp.count_nonzero(self.mask)
            if self.xp is np:
                self.nmask = nmask
            else:
                self.nmask = nmask.get()

        if verbose:
            print("kwargs keys = {}".format(kwargs.keys()))
            print(kwargs)

        self.precision = precision
        if self.precision == "single":
            self._real_dtype = np.float32
            self._cplx_dtype = np.complex64
        else:
            self._real_dtype = np.float64
            self._cplx_dtype = np.complex128
        self.dtype = self._cplx_dtype
        if self.dtype not in [xp.complex64, xp.complex128]:
            # Note: MUST KEEP OPERATOR DTYPE REAL OR CONJUGATE OF INPUTS AND
            # OUTPUTS WILL BE TAKEN CAUSING ERRONEOUS RESULTS
            # operator type must be defined as (real) float or double
            raise ValueError("unsupported dtype: must be 32 or 64 bit float")

        if kspace.ndim != 2 or kspace.shape[1] > 3:
            raise ValueError(
                "Invalid kspace shape. Must be a 2D array where the second "
                "axis corresponds to 1D-3D spatial axes."
            )
        if kspace.dtype != self._real_dtype:
            kspace = kspace.astype(self._real_dtype, copy=False)
        self.kspace = kspace
        ndim = kspace.shape[1]

        # defaults
        self.exact = exact
        self.fov = fov
        if len(self.fov) == 1 and ndim > 1:
            self.fov = np.asarray(self.fov.tolist() * ndim)

        self.omega = self._kspace_to_omega(kspace)  # set after Nd, fov
        if xp.max(xp.abs(self.omega)) > xp.pi + 1e-6:
            warnings.warn(
                "Warning in MRI_Operator: omega exceeds pi. Was "
                "this intended?"
            )

        if self.exact:
            self.n_shift = n_shift
            if self.n_shift is None:
                self.n_shift = tuple(d // 2 for d in self.shape)
            self.n_shift = tuple(self.n_shift)
        else:
            self.n_shift = None

        # arguments for mri_exp_approx()
        self.exp_approx_args = exp_approx_args

        if order != "F":
            raise ValueError("MRI_Operator only supports order='F'.")
        self.order = order

        # SENSE-stuff
        self.coil_sensitivities = coil_sensitivities
        if self.coil_sensitivities is None:
            self.Ncoils = 1
        else:
            # need Fortran contiguous sensitivity maps
            self.coil_sensitivities = xp.asfortranarray(self.coil_sensitivities)

            if self.coil_sensitivities.ndim != 2:
                raise ValueError(
                    "'coil_sensitivities' must be either be a 2d array "
                    + "[Npixels, Ncoils] or None"
                )
            self.Ncoils = self.coil_sensitivities.shape[1]

        nargin = prod(shape)
        nargout = self.kspace.shape[0] * self.Ncoils
        self.__unitary_scaling = False

        # multispectral reconstruction
        self.spectral_offsets = spectral_offsets
        if self.spectral_offsets is not None:
            self.nspectra = len(self.spectral_offsets)
            self.spectral_offsets = tuple(self.spectral_offsets)
        else:
            self.nspectra = 1
        nargin *= self.nspectra

        #
        # initialize exact transform or NUFFT approximation object
        #
        if self.exact:
            self._init_exact()
        else:

            if on_gpu:
                self.omega = cupy.asarray(self.omega)

            # Find a size amenable to fast FFT, but also a multiple of 8
            # The choice of 8 is somewhat arbitrary, but ensures that for
            # instance a 3-level periodization wavelet transform can be
            # performed in a non-redundant fashion.
            os_grid_shape = tuple(
                [
                    next_fast_len_multiple(ceil(grid_os_factor * s), 8)
                    for s in shape
                ]
            )

            # Default NUFFT parameters
            self.nufft_kwargs = dict(
                Jd=(6,) * self.ndim,
                n_shift=tuple([n / 2 for n in self.Nd]),
                Ld=1024,
                mode="table1",
                phasing="real",
                ortho=False,  # TODO: change default to True?
            )
            if not isinstance(nufft_kwargs, dict):
                raise ValueError("nufft_kwargs must be a dictionary")
            # update defaults using any user-specified NUFFT kwargs
            valid_nufft_keys = list(self.nufft_kwargs)
            keys_to_pop = [
                k for k in nufft_kwargs.keys() if k not in valid_nufft_keys
            ]
            if keys_to_pop:
                warnings.warn(
                    "The following user-provided entries in nufft_kwargs will "
                    f"be ignored: {keys_to_pop}"
                )
                for key in keys_to_pop:
                    nufft_kwargs.pop(key, None)
            self.nufft_kwargs.update(nufft_kwargs)

            if self.n_shift is not None and any(self.n_shift):
                raise ValueError("n_shift ignored in NUFFT-based case")

            if ("squeeze_reps" in self.nufft_kwargs) and self.nufft_kwargs[
                "squeeze_reps"
            ]:
                raise ValueError(
                    "squeeze_reps = True not allowed within nufft_kwargs"
                )

            self.Gnufft = NUFFT_Operator(
                mask=mask,
                omega=self.omega,
                Nd=shape,
                Kd=os_grid_shape,
                precision=self.precision,
                on_gpu=on_gpu,
                loc_in=loc_in,
                loc_out=loc_out,
                squeeze_reps=False,
                **self.nufft_kwargs,
            )
            # store the oversampled image dimensions in the base object as well
            self.Kd = self.Gnufft.Kd

        # configure spatial basis Fourier transform
        self.pixel_basis = pixel_basis

        # configure DCF and/or basis weights
        # (must come after self._init_basis())
        self._init_density_compensation(
            weights, nargout, apply_basis_transform=True
        )

        # configure fieldmap corrected recon
        # This sets self.ti and self.zmap
        self._init_fieldmap(kwargs)

        if self.spectral_offsets is not None:
            # Note: uses self.ti as set by self._init_fieldmap
            self._init_multispectral(
                n_shots=self.kspace.shape[0] // self.ti.size
            )

        """
        select forward and adjoint operators depending on whether transform is
        exact or NUFFT-based and whether computation is to be done on the CPU
        or GPU.
        """
        if self.exact:
            if self.spectral_offsets is not None:
                raise ValueError(
                    "no multispectral implementation for the exact case"
                )
            else:
                matvec = self._forw_exact
                matvec_adj = self._back_exact
        else:
            if self.spectral_offsets is not None:
                matvec = self._forw_multispectral
                matvec_adj = self._back_multispectral
            else:
                matvec = self._forw
                matvec_adj = self._back

        self.__matvec = matvec
        self.__matvec_transp = matvec_adj
        self.__matvec_adj = matvec_adj
        super(MRI_Operator, self).__init__(
            nargin,
            nargout,
            symmetric=False,
            hermetian=False,
            matvec=self.__matvec,
            matvec_adj=self.__matvec_transp,
            matvec_transp=self.__matvec_adj,
            nd_input=False,
            nd_output=False,
            shape_in=(nargin, 1),
            shape_out=(nargout, 1),
            order="F",
            matvec_allows_repetitions=True,
            squeeze_reps=squeeze_reps,
            mask_in=None,  # self.mask,
            mask_out=None,
            dtype=self.dtype,
            loc_in=loc_in,
            loc_out=loc_out,
        )

        # make sure coils sensitivity, fieldmap etc. match the expected dtype
        self._update_array_precision()

        # if requested, scaling the weights to give an operator which is
        # approximately unitary (norm preserving)
        self.unitary_scaling = unitary_scaling

    def _init_basis(self):
        """Called by __init__ to initialize the PixelBasis."""
        self.basis = PixelBasis(
            self.kspace,
            pixel_basis=self.__pixel_basis,
            fov=self.fov,
            Nd=self.Nd,
        )

    @property
    def pixel_basis(self):
        """Return the current PixelBasis object."""
        return self.__pixel_basis

    @pixel_basis.setter
    def pixel_basis(self, pixel_basis):
        """Update the pixel basis to a new pixel_basis."""
        self.__pixel_basis = pixel_basis
        self._init_basis()

    def _init_exact(self):
        """Iniatilization for exact (slow) transform."""
        N = self.Nd
        ndim = self.ndim
        xp = self.xp
        mask = self.mask
        if self.n_shift is None:
            raise ValueError("n_shift required when exact=True")
        if True:
            # TODO: test this n-dimensional refactoring
            nn = []
            for d in range(ndim):
                nn.append(xp.arange(N[d]) - self.n_shift[d])
            nn = xp.meshgrid(*nn, indexing="ij")
            for d in range(ndim):
                self.u[:, d] = self._apply_mask(nn[d], squeeze_output=True)
            self.u = xp.empty((self.nmask, ndim), dtype=self._real_dtype)
        else:
            if len(N) == 2:
                n1, n2 = xp.meshgrid(
                    xp.arange(0, N[0]) - self.n_shift[0],
                    xp.arange(0, N[1]) - self.n_shift[1],
                    indexing="ij",
                )
                self.u = xp.zeros((xp.count_nonzero(mask), 2))
                # transposes to get Fortran style ordering to match Matlab
                self.u[:, 0] = n1.T[mask.T]
                self.u[:, 1] = n2.T[mask.T]
                self.u = self.u.T  # (2, np)
            elif len(N) == 3:
                n1, n2, n3 = xp.meshgrid(
                    xp.arange(0, N[0]) - self.n_shift[0],
                    xp.arange(0, N[1]) - self.n_shift[1],
                    xp.arange(0, N[2]) - self.n_shift[2],
                    indexing="ij",
                )
                self.u = xp.zeros((xp.count_nonzero(mask), 2))
                # transposes to get Fortran style ordering to match Matlab
                self.u[:, 0] = n1.T[mask.T]
                self.u[:, 1] = n2.T[mask.T]
                self.u[:, 2] = n3.T[mask.T]
                self.u = self.u.T  # (3, np)
            else:
                raise ValueError("Only 2D and 3D cases done")
        # self.u = complexify(self.u)
        self.v = 1j * self.omega.T  # (2, np)

    #        # TODO: use basis_transform_op to automatically handle repetitions?
    #             No: for now have folded basis.transform into self.weights
    #        if self.basis.transform is not None:
    #            self.basis_transform_op = DiagonalOperator(self.basis.transform)
    #            if self.Ncoils > 1:
    #                self.basis_transform_op = BlockDiagLinOp(
    #                    [self.basis_transform_op, ] * self.Ncoils)

    # TODO: replace [] with None
    def _init_fieldmap(self, kwargs):
        """Called by __init__ to initialize the fieldmap variables."""
        # new fieldmap stuff
        # n_shots can speed up zmap calculation in case where the
        # trajectory is multiple shots over identical time interval
        xp = self.xp
        self.zmap = kwargs.pop("zmap", None)
        self.ti = kwargs.pop("ti", None)
        if self.ti is not [] and self.ti is not None:
            self.ti = xp.asarray(self.ti)
        self.fieldmap_segments = kwargs.pop("fieldmap_segments", None)
        if (
            self.fieldmap_segments is None
            and self.ti is not None
            and self.zmap is not None
        ):
            self.fieldmap_segments = determine_fieldmap_segments(
                self.ti, self.zmap, fieldmap_segments_start=3, xp=xp
            )
        # acorr_fieldmap_segments is for autocorrelation zmap. default is kwargs['L']
        self.acorr_fieldmap_segments = kwargs.pop(
            "acorr_fieldmap_segments", self.fieldmap_segments
        )
        self.n_shots = kwargs.pop("n_shots", 1)

        # add zmap if available now.
        if self.zmap is not None:
            if (not self.exact) and (
                self.fieldmap_segments is None or self.ti is None
            ):
                raise ValueError(
                    "field-corrected recon:  zmap specified without ti and fieldmap_segments"
                )
            self.new_zmap(
                ti=self.ti,
                zmap=self.zmap,
                fieldmap_segments=self.fieldmap_segments,
                n_shots=self.n_shots,
            )

    def _init_density_compensation(
        self, weights, nargout=None, apply_basis_transform=False
    ):

        # incorporate self.basis.transform into the weights array
        if apply_basis_transform and self.basis.transform is not None:
            # TODO!!!: fix so that this doesn't break _scale_weights!!
            if self.xp.iscomplexobj(self.basis.transform):
                raise ValueError("TODO: complex case not tested")
            if weights is None:
                weights = self.basis.transform.copy()
            else:
                # fold the basis.transform into the weights
                if weights.ndim != 1:
                    raise ValueError("expected a 1D array of weights")
                if weights.size == self.basis.transform.size:
                    weights *= self.basis.transform
                elif weights.size == self.basis.transform.size * self.Ncoils:
                    weights *= self.xp.concatenate(
                        [self.basis.transform] * self.Ncoils
                    )
                else:
                    raise ValueError("invalid size for weights")

        """Initialize Density Compensation weights."""
        if weights is None:
            return None
        if nargout is None:
            nargout = self.nargout
        if weights is not None:
            xp = self.xp
            if xp is np:
                loc_in = loc_out = "cpu"
            else:
                loc_in = loc_out = "gpu"
            if isinstance(weights, xp.ndarray):
                if weights.dtype != self._real_dtype:
                    weights = weights.astype(self._real_dtype)
                weights = DiagonalOperator(
                    weights, loc_in=loc_in, loc_out=loc_out
                )
            elif not isinstance(weights, DiagonalOperator):
                raise ValueError(
                    "weights must be an ndarray or DiagonalOperator"
                )
            if weights.dtype != self._real_dtype:
                # retrieve the diagonal and convert to the desired dtype
                weights = weights.diag.diagonal().astype(self._real_dtype)
                weights = DiagonalOperator(
                    weights, loc_in=loc_in, loc_out=loc_out
                )
            if weights.nargin != nargout:
                if weights.nargin * self.Ncoils == nargout:
                    weights = BlockDiagLinOp(
                        [weights] * self.Ncoils, loc_in=loc_in, loc_out=loc_out
                    )
                else:
                    raise ValueError("mismatch between kspace size and weights")

        self.__weights = weights

    def _init_multispectral(self, n_shots):
        # initialize arrays for multi-spectral reconstruction
        xp = self.xp
        if self.spectral_offsets is None:
            return
        if self.ti is None:
            raise ValueError(
                "Using spectral_offsets requires ti to be specified"
            )
        self.offset_arrays = []
        for f in self.spectral_offsets:
            if f == 0:
                self.offset_arrays.append(None)
            elif np.isscalar(f):
                v = xp.exp(1j * 2 * xp.pi * f * self.ti.ravel(order="F"))
                if n_shots > 1:
                    v = xp.concatenate((v,) * n_shots, axis=0)
                self.offset_arrays.append(v)
            else:  # tuple of 2-tuples
                if not all(len(ff) == 2 for ff in f):
                    raise ValueError(
                        "elements of spectral offsets must be scalar or a "
                        "tuple of 2-tuples where each 2-tuple is a "
                        "(weight, offset) pair"
                    )
                if abs(1.0 - sum(ff[0] for ff in f)) > 1e-2:
                    raise ValueError("sum of the weights should equal 1.0")
                offset_set = []
                for w, ff in f:
                    v = xp.exp(1j * 2 * xp.pi * ff * self.ti.ravel(order="F"))
                    if n_shots > 1:
                        v = xp.concatenate((v,) * n_shots, axis=0)
                    offset_set.append((w, v))
                self.offset_arrays.append(tuple(offset_set))

    def _update_dtype(self, arr, mode=None):
        """ Fixup an object's dtype to match the precision of the oeprator."""
        xp = self.xp
        if mode is None:
            if xp.iscomplexobj(arr):
                if arr.dtype != self._cplx_dtype:
                    arr = arr.astype(self._cplx_dtype)
            else:
                if arr.dtype != self._real_dtype:
                    arr = arr.astype(self._real_dtype)
        elif mode == "real":
            if arr.dtype != self._real_dtype:
                arr = arr.astype(self._real_dtype)
        elif mode == "complex":
            if arr.dtype != self._cplx_dtype:
                arr = arr.astype(self._cplx_dtype)
        else:
            raise ValueError("unrecognized dtype mode")
        return arr

    def _update_array_precision(self):
        # make sure all dtypes are at a consistent precision
        self.kspace = self._update_dtype(self.kspace, "real")
        self.omega = self._update_dtype(self.omega, "real")
        if hasattr(self, "u"):
            self.u = self._update_dtype(self.u)
        if hasattr(self, "v"):
            self.v = self._update_dtype(self.v)
        if self.coil_sensitivities is not None:
            self.coil_sensitivities = self._update_dtype(
                self.coil_sensitivities, "complex"
            )
        if self.zmap is not None:
            self.zmap = self._update_dtype(self.zmap, "complex")
            if hasattr(self, "ti") and self.ti is not None:
                self.ti = self._update_dtype(self.ti, "real")
            if hasattr(self, "fmap_basis") and self.fmap_basis is not None:
                self.fmap_basis = self._update_dtype(self.fmap_basis, "complex")
            if hasattr(self, "fmap_coeffs") and self.fmap_coeffs is not None:
                self.fmap_coeffs = self._update_dtype(
                    self.fmap_coeffs, "complex"
                )
        if self.basis.transform is not None:
            self.basis.transform = self._update_dtype(
                self.basis.transform, "real"
            )

    def _kspace_to_omega(self, kspace):
        xp = self.xp
        ndim = len(self.Nd)
        # configure frequencies in physical units
        omega = xp.empty_like(kspace)
        for id in range(ndim):
            omega[:, id] = (
                2 * xp.pi * kspace[:, id] * self.fov[id] / self.Nd[id]
            )
        if omega.dtype != self._real_dtype:
            omega = omega.astype(self._real_dtype)
        return omega

    @property
    def dim(self):
        """The shape corresponding to the operator."""
        return self.shape

    def _get_weights_array(self):
        xp = self.xp
        if self.weights is None:
            return xp.ones(self.nargout // self.Ncoils, dtype=self._real_dtype)
        elif isinstance(self.weights, BlockDiagLinOp):
            # TODO: manually concatenate blocks? This assumes all are the same
            w = self.weights.blocks[0].diag
        else:
            w = self.weights.diag
        return w

    def _scale_weights(self, power_iterations=15):
        # TODO: can Anorm be a single NUFFT operation instead of all coils?
        xp = self.xp
        cplx_dtype = self._cplx_dtype
        if self.norm is not None:
            lam = power_method(self, dtype=cplx_dtype, niter=power_iterations)
        else:
            Anorm = self.H * self
            lam = power_method(Anorm, dtype=cplx_dtype, niter=power_iterations)
        self._lam = lam
        w = self._get_weights_array()
        new_weights = w / xp.sqrt(lam)
        self._init_density_compensation(
            new_weights,
            # don't apply basis transform again!
            apply_basis_transform=False,
        )
        self.__unitary_scaling = True

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        self._init_density_compensation(weights)
        if self.unitary_scaling:
            self._scale_weights()

    @property
    def unitary_scaling(self):
        """Boolean indicating if the norm of the operator is (nearly) unitary.

        i.e. Whether the np.linalg.norm(A.H * A * x) is approximately equal to
        np.linalg.norm(x) for an operator, A. If this is False, and the user
        sets it to True, the appropriately scaling of the ``weights`` property
        is computed via the power method.
        """
        return self.__unitary_scaling

    @unitary_scaling.setter
    def unitary_scaling(self, unitary_scaling):
        """Boolean indicating if the norm of the operator is (nearly) unitary.

        i.e. Whether the np.linalg.norm(A.H * A * x) is approximately equal to
        np.linalg.norm(x) for an operator, A. If the user sets it to True, the
        appropriately scaling of the ``weights`` property is computed via the
        power method.
        """
        if unitary_scaling is True:
            if self.__unitary_scaling is False:
                self._scale_weights()
        elif unitary_scaling is not False:
            raise ValueError("unitary_scaling should be True or False")
        self.__unitary_scaling = unitary_scaling

    def set_unitary_scaling_weights(self, weights):
        """Utility for using user-provided unitary weights instead of computing
        from the power method."""
        self.__weights = weights
        self.__unitary_scaling = True

    def copy_zmap(self, MRIop2):
        """ Copy zmap from an already initialized MRI Operator
        """
        if not hasattr(MRIop2, "zmap") or MRIop2.zmap is None:
            raise ValueError("No zmap to copy")
        # TODO: do some basic dimension checks
        self.ti = MRIop2.ti
        self.zmap = MRIop2.zmap
        self.fmap_coeffs = MRIop2.fmap_coeffs
        self.fieldmap_segments = MRIop2.fieldmap_segments
        self.exp_approx_type = MRIop2.exp_approx_type
        self.exp_approx_args = MRIop2.exp_approx_args
        self.fmap_basis = MRIop2.fmap_basis

    # @profile
    def copy_zmap_new_ti(self, MRIop2, ti, n_shots=1):
        """Used to share zmap, fmap_coeffs across segments in multi-segment recon."""
        if not hasattr(MRIop2, "zmap") or MRIop2.zmap is None:
            raise ValueError("No zmap to copy")

        xp = self.xp
        # make sure precision of dtypes matches the NUFFT operator
        if ti.dtype != self.Gnufft._real_dtype:
            ti = ti.astype(self.Gnufft._real_dtype)

        self.ti = ti.ravel(order="F")
        self.zmap = MRIop2.zmap
        self.fmap_coeffs = MRIop2.fmap_coeffs
        self.fieldmap_segments = MRIop2.fieldmap_segments
        self.exp_approx_type = MRIop2.exp_approx_type
        self.exp_approx_args = MRIop2.exp_approx_args

        # allti_equal = xp.max(self.ti, MRIop2.ti)
        same_ti_vector = False
        if self.ti.shape == MRIop2.ti.shape:
            if xp.max(xp.abs(self.ti - MRIop2.ti)) < 1e-6:
                same_ti_vector = True

        if same_ti_vector:
            self.fmap_basis = MRIop2.fmap_basis
        else:
            self.fmap_basis, C, hk, zk = mri_exp_approx(
                ti=self.ti,
                zmap=self.zmap,
                segments=self.fieldmap_segments,
                approx_type=self.exp_approx_type,
                **self.exp_approx_args,
            )

        # make sure precision of dtypes matches the NUFFT operator
        if self.fmap_basis.dtype != self.Gnufft._cplx_dtype:
            self.fmap_basis = self.fmap_basis.astype(self.Gnufft._cplx_dtype)

        if n_shots > 1:
            """when there are multiple shots with same time vector can just
            duplicate here."""
            self.fmap_basis = xp.tile(self.fmap_basis, [n_shots, 1])

        if xp.any(xp.isnan(self.fmap_basis)):
            raise ValueError("bug: nan values in fmap_basis")

        self.fieldmap_segments = self.fmap_basis.shape[1]

    # @profile
    def new_zmap(
        self,
        ti,
        zmap,
        fieldmap_segments=None,
        n_shots=1,
        acorr_fieldmap_segments=None,
        calc_acorr_fieldmap_basis=False,
        xp=None,
    ):
        """ Initialize the fieldmap approximation basis and coefficients

        Parameters
        ----------
        see main class documentation

        Notes
        -----
        acorr_fieldmap_segments, calc_acorr_fieldmap_basis only required for Toeplitz case
        """
        if xp is None:
            xp = self.xp
        if len(ti) != self.kspace.shape[0] // n_shots:
            raise ValueError("ti size mismatch in new_zmap()")

        # make sure precision of dtypes matches the NUFFT operator
        if not self.exact:
            if ti.dtype != self.Gnufft._real_dtype:
                ti = ti.astype(self.Gnufft._real_dtype)
            if zmap.dtype != self.Gnufft._cplx_dtype:
                zmap = zmap.astype(self.Gnufft._cplx_dtype)
        else:
            pass
            # TODO:  fix dtypes in exact case?

        self.ti = ti.ravel(order="F")

        # apply mask to zmap if it wasn't already masked
        if self.mask is not None:
            # if ((zmap.ndim == self.mask.ndim + 1) and
            #         (zmap.shape[:-1] == self.mask.shape)):
            #     # TODO: trying to allow zmap to have an extra
            #             motion-segmenets dimension
            #     self.zmap = masker(zmap, self.mask).ravel(order=self.order)
            if zmap.shape == self.mask.shape:
                self.zmap = masker(zmap, self.mask, xp=self.xp)
            elif (
                self.xp.count_nonzero(self.mask) == zmap.shape[0]
                and zmap.ndim == 1
            ):
                self.zmap = zmap
            else:
                raise ValueError("zmap size mismatch in new_zmap()")
        else:
            # if zmap.ndim != 1:
            #    raise ValueError("expected a 1D zmap (masked/raveled)")
            self.zmap = zmap.ravel(order=self.order)

        if fieldmap_segments is None:
            # auto-determine an appropriate number of segments
            fieldmap_segments = determine_fieldmap_segments(
                ti, zmap, fieldmap_segments_start=3, xp=xp
            )

        # trick to handle 'exact' case (for which fieldmap_segments is irrelevant)
        if self.exact:
            ndim = len(self.Nd)
            # trick: if already a zmap in u,v, then replace it
            # TODO: check exact case
            # print("self.u.shape = {}".format(self.u.shape))
            # print("self.v.shape = {}".format(self.v.shape))
            # print("self.zmap.shape = {}".format(self.zmap.shape))
            # print("self.ti.shape = {}".format(self.ti.shape))
            self.u = xp.hstack((self.u[:, 0:ndim], self.zmap[:, xp.newaxis]))
            self.v = xp.vstack((self.v[0:ndim, :], self.ti[xp.newaxis, :]))
            return

        if not fieldmap_segments:  # 4th argument is optional, so defaults here:
            if self.fieldmap_segments:
                fieldmap_segments = self.fieldmap_segments
            else:
                raise ValueError(
                    "user must provide self.fieldmap_segments or fieldmap_segments input"
                )

        # initialize exponential approximations for field-corrected
        # reconstruction
        if xp.any(self.zmap.real):
            self.exp_approx_type = ("hist,time,unif", [40, 10])
        else:
            # zmap_range = max(imag(self.zmap(:)))-min(imag(self.zmap(:)));
            # grl_Nbins = max(round(zmap_range/20),40);  %force enough bins to
            # keep at least 10 Hz resolution.  always use at least 40 bins
            grl_Nbins = [40]  # TODO: remove hardcode
            self.exp_approx_type = ("hist,time,unif", grl_Nbins)

        self.fmap_basis, C, hk, zk = mri_exp_approx(
            ti=self.ti,
            zmap=self.zmap,
            segments=fieldmap_segments,
            approx_type=self.exp_approx_type,
            **self.exp_approx_args,
        )

        # make sure precision of dtypes matches the NUFFT operator
        if self.fmap_basis.dtype != self.Gnufft._cplx_dtype:
            self.fmap_basis = self.fmap_basis.astype(self.Gnufft._cplx_dtype)

        if C.dtype != self.Gnufft._cplx_dtype:
            C = C.astype(self.Gnufft._cplx_dtype)

        if n_shots > 1:
            """when there are multiple shots with same time vector can just
            duplicate here."""
            # Note: on CPU this is explicitly tiled
            #       on GPU to save memory only 1 repetition will be stored
            self.fmap_basis = xp.tile(self.fmap_basis, [n_shots, 1])

        if xp.any(xp.isnan(self.fmap_basis)):
            raise ValueError("bug: nan values in fmap_basis")

        self.fmap_coeffs = C.T

        if (
            not acorr_fieldmap_segments
        ):  # 5th argument is optional, so defaults here:
            if self.acorr_fieldmap_segments:
                acorr_fieldmap_segments = self.acorr_fieldmap_segments
            # trick: use "found" L because acorr_fieldmap_segments >= fieldmap_segments generally
            elif isinstance(fieldmap_segments, (list, xp.ndarray, tuple)):
                acorr_fieldmap_segments = [
                    self.fmap_basis.shape[1],
                    fieldmap_segments[1],
                ]  # [fieldmap_segments, rmsmax]
            else:
                acorr_fieldmap_segments = self.fmap_basis.shape[1]

        # store size (seems redundant, but OK)
        self.fieldmap_segments = self.fmap_basis.shape[1]
        if isinstance(fieldmap_segments, (list, xp.ndarray, tuple)):
            self.rmsmax = fieldmap_segments[1]
            print("fieldmap_segments=%d found" % self.fieldmap_segments)

        if calc_acorr_fieldmap_basis:
            # TODO: untested
            # generate one with auto-correlation histogram too.
            # only if no relaxation map!
            if not xp.any(self.zmap.real):
                self.acorr_fieldmap_basis, C, hk, zk = mri_exp_approx(
                    ti=self.ti,
                    zmap=self.zmap,
                    segments=fieldmap_segments,
                    approx_type=self.exp_approx_type,
                    acorr=True,
                    **self.exp_approx_args,
                )

                # trick: for symmetric histogram, it should be real!
                if xp.any(self.acorr_fieldmap_basis.imag > 1e-6):
                    warnings.warn(
                        "imaginary component to self.acorr_fieldmap_basis in "
                        "MRI_Operator"
                    )
                self.acorr_fieldmap_basis = self.acorr_fieldmap_basis.real

                # make sure precision of dtypes matches the NUFFT operator
                if self.acorr_fieldmap_basis.dtype != self.Gnufft._real_dtype:
                    self.acorr_fieldmap_basis = self.acorr_fieldmap_basis.astype(
                        self.Gnufft._real_dtype
                    )

                if C.dtype != self.Gnufft._cplx_dtype:
                    C = C.astype(self.Gnufft._cplx_dtype)

                self.acorr_fieldmap_coeffs = C.T

                self.acorr_fieldmap_segments = self.acorr_fieldmap_basis.shape[
                    1
                ]
                if isinstance(
                    acorr_fieldmap_segments, (list, xp.ndarray, tuple)
                ):
                    print(
                        "acorr_fieldmap_segments=%d found"
                        % self.acorr_fieldmap_segments
                    )
            else:
                raise ValueError("zmap with real component unsupported")

    # @profile
    def _forw_exact(self, x):
        """ exact forward operation on the CPU """
        # TODO: need copy here to avoid modifying original data?
        xp = self.xp
        x = xp.asarray(x).copy()
        x = x.reshape((self.nargin, -1), order=self.order)

        if x.shape[0] != self.shape[1]:
            # [(N),(nc)] to [*N,nc]  #TODO: test
            x = xp.reshape(x, (xp.prod(self.Nd), -1), order=self.order)
            if self.mask is not None:
                x = x[self.mask.ravel(order=self.order), :]  # [np,*nc]

        x = complexify(x)  # force at least 64-bit complex

        y = exp_xform(x, self.u, self.v, xp=self.xp)  # complexify(single(...))
        y = xp.asarray(y)
        if y.ndim == 1:
            y = y[:, xp.newaxis]

        if self.weights:
            # apply density compensation
            # (self.weights incorporates self.basis.transform if necessary)
            y = self.weights * y

        y = xp.asfortranarray(y)  # TODO: necessary?
        return y  # returns an 2D Nd array

    # @profile
    def _apply_mask(self, x, order="F", squeeze_output=False):
        kwargs = dict(order=order, squeeze_output=squeeze_output, xp=self.xp)
        if self.mask is None:
            return x.reshape((self.nmask, -1), order=order)
        else:
            return masker(x, self.mask, **kwargs)

    @profile
    def _forw(self, x, legacy_loop=False):
        """ NUFFT-based forward operation on the CPU """
        xp = self.xp
        # TODO: need copy to avoid modifying original data?
        # x = xp.asarray(x).copy()  # TODO: check and only do this when necessary
        x = x.reshape((self.nargin // self.nspectra, -1), order=self.order)

        nreps = x.shape[-1]  # repetitions
        nt = self.shape[0] // self.Ncoils
        x = complexify(x)  # force at least 64-bit complex

        sn_outside_loop = True
        if sn_outside_loop:
            sn_tmp = self.Gnufft.sn.copy()
            # apply scale factors outside of fieldmap loop
            x = self._apply_mask(sn_tmp, squeeze_output=False) * x
            # TODO: make sure this is valid for the Gnufft object
            self.Gnufft.sn = None
        try:
            if self.zmap is None:  # ) or (not xp.any(self.zmap)):
                loop_over_coils = False
                if loop_over_coils:
                    if (
                        self.coil_sensitivities is not None
                    ):  # and xp.any(self.coil_sensitivities):
                        y = xp.empty(
                            (self.shape[0], nreps),
                            dtype=x.dtype,
                            order=self.order,
                        )
                        for cc in range(self.Ncoils):
                            # TODO: using broadcasting across nreps now instead
                            # of explicit tile of the sensitivities
                            y[cc * nt : (cc + 1) * nt, :] = self.Gnufft * (
                                self.coil_sensitivities[:, cc][:, xp.newaxis]
                                * x
                            )
                    else:
                        y = self.Gnufft * x
                else:
                    # x = xp.empty((self.shape[1]//self.nspectra, nreps),
                    #              dtype=y.dtype, order=self.order)
                    # multi-thread only across coils and repetitions.  this way is
                    # thread safe
                    if self.Ncoils > 1:
                        # put all coils (and/or repetitions) along the 2nd
                        # dimension
                        y_shape = (nt, self.Ncoils * nreps)
                    else:
                        y_shape = (self.shape[0], nreps)
                    y = xp.empty(y_shape, dtype=x.dtype, order=self.order)

                    for repIDX in range(nreps):
                        rep_slice = slice(
                            repIDX * self.Ncoils, (repIDX + 1) * self.Ncoils
                        )
                        if self.coil_sensitivities is not None:
                            tmp = (
                                self.coil_sensitivities
                                * x[:, repIDX : repIDX + 1]
                            )
                            y[:, rep_slice] = self.Gnufft * tmp
                        else:
                            xsl = x[:, repIDX]
                            tmp = self.Gnufft * xsl
                            y[:, rep_slice] = tmp
                    y = y.reshape((self.shape[0], nreps), order=self.order)

            else:  # approximation

                if self.coil_sensitivities is None:
                    # fieldmap, but no coil sensitivities
                    for ll in range(self.fieldmap_segments):
                        # fmap_coeffs will be broadcast when x.shape[1]>1
                        # fmap_basis will be broadcast when x.shape[1]>1
                        tmp = self.fmap_coeffs[:, ll][:, xp.newaxis] * x
                        tmp = self.Gnufft * tmp
                        tmp = self.fmap_basis[:, ll][:, xp.newaxis] * tmp
                        if ll == 0:
                            y = tmp
                        else:
                            y += tmp
                # not None and xp.any(self.coil_sensitivities):
                else:
                    loop_over_coils = True
                    if loop_over_coils:
                        # multiple coils and fieldmap
                        y = xp.empty(
                            (self.shape[0], nreps),
                            dtype=x.dtype,
                            order=self.order,
                        )
                        for ll in range(self.fieldmap_segments):
                            cx = self.fmap_coeffs[:, ll : ll + 1] * x
                            for cc in range(self.Ncoils):
                                # fmap_coeffs will be broadcast when x.shape[1]>1
                                tmp = (
                                    self.coil_sensitivities[:, cc : cc + 1] * cx
                                )
                                tmp = self.Gnufft * tmp
                                # fmap_basis will be broadcast when x.shape[1]>1
                                tmp = self.fmap_basis[:, ll : ll + 1] * tmp
                                if ll == 0:
                                    y[cc * nt : (cc + 1) * nt, :] = tmp
                                else:
                                    y[cc * nt : (cc + 1) * nt, :] += tmp
                    else:
                        # loop is over repetitions instead of coils
                        if self.Ncoils > 1:
                            # put all coils (and/or repetitions) along the 2nd
                            # dimension
                            y_shape = (nt, self.Ncoils * nreps)
                        else:
                            y_shape = (self.shape[0], nreps)
                        y = xp.empty(y_shape, dtype=x.dtype, order=self.order)
                        for ll in range(self.fieldmap_segments):
                            cx = self.fmap_coeffs[:, ll : ll + 1] * x
                            for repIDX in range(nreps):
                                rep_slice = slice(
                                    repIDX * self.Ncoils,
                                    (repIDX + 1) * self.Ncoils,
                                )
                                tmp = (
                                    self.coil_sensitivities
                                    * cx[:, repIDX : repIDX + 1]
                                )
                                tmp = self.Gnufft * tmp
                                y[:, rep_slice] = (
                                    self.fmap_basis[:, ll : ll + 1] * tmp
                                )
                        y = y.reshape((self.shape[0], nreps), order=self.order)
        finally:
            # place the scale factors back into the NUFFT object
            if sn_outside_loop:
                self.Gnufft.sn = sn_tmp

        y = xp.asarray(y)

        if self.weights is not None:
            # apply density compensation
            # (self.weights incorporates self.basis.transform if necessary)
            y = self.weights * y

        if y.ndim == 1:
            y = y[:, xp.newaxis]
        y = xp.asfortranarray(y)  # TODO: necessary?

        return y  # returns an 2D Nd array

    # @profile
    def _forw_multispectral(self, x):
        xp = self.xp
        x = xp.asarray(x).copy()
        x = x.reshape(
            (self.nargin // self.nspectra, self.nspectra, -1), order=self.order
        )

        for n, (f, v) in enumerate(  # TODO: can remove f here
            zip(self.spectral_offsets, self.offset_arrays)
        ):
            if v is not None:
                use_linops = False
                if use_linops:
                    Omega_shift = DiagonalOperator(
                        v,
                        order=self.order,
                        squeeze_reps=False,
                        loc_in=self.loc_in,
                        loc_out=self.loc_out,
                    )
                    if self.Ncoils > 1:
                        Omega_shift = BlockDiagLinOp(
                            (Omega_shift,) * self.Ncoils,
                            squeeze_reps_out=False,
                            loc_in=self.loc_in,
                            loc_out=self.loc_out,
                        )
                    tmp = self._forw(x[:, n, :])
                    if n == 0:
                        y = Omega_shift * tmp
                    else:
                        y += Omega_shift * tmp
                else:
                    # previously this case was necessary because BlockDiagLinOp
                    # had poor performance. That has since been fixed, so
                    # can just switch back to the use_linops=True case.
                    tmp = self._forw(x[:, n, :])
                    tmp_shape = tmp.shape
                    tmp = tmp.reshape(
                        (tmp.shape[0] // self.Ncoils, -1), order=self.order
                    )
                    if isinstance(v, tuple):
                        #                        if n > 0:
                        #                            y = y.reshape(tmp.shape, order=self.order)

                        # weighted sum of frequencies
                        #     (e.g. multispectral fat model)
                        tmp0 = tmp.copy()
                        for noff, (weight, vf) in enumerate(v):
                            if vf.ndim == 1:
                                vf = vf[:, np.newaxis]
                            tmp = (weight * vf) * tmp0
                            tmp = tmp.reshape(tmp_shape, order=self.order)
                            if n == 0 and noff == 0:
                                y = tmp
                            else:
                                y += tmp
                    else:
                        # single frequency with weight 1.0
                        if v.ndim == 1:
                            v = v[:, np.newaxis]
                        tmp = v * tmp
                        tmp = tmp.reshape(tmp_shape, order=self.order)
                        if n == 0:
                            y = tmp
                        else:
                            y += tmp
            else:
                if n == 0:
                    y = self._forw(x[:, n, :])
                else:
                    y += self._forw(x[:, n, :])

        if y.ndim == 1:  # TODO: necessary?
            y = y[:, xp.newaxis]
        y = xp.asfortranarray(y)  # TODO: necessary?
        return y

    # @profile
    def _back_exact(self, y):
        """ exact adjoint operation on the CPU """
        # TODO: need copy here to avoid modifying original data?
        xp = self.xp
        y = xp.asarray(y).copy()
        y = y.reshape((self.nargout, -1), order=self.order)
        if self.weights is not None:
            # apply density compensation
            y = self.weights * y

        y = complexify(y)  # force at least 64-bit complex

        # trick: conj(exp(-uv)) = exp(-conj(u) conj(v))
        vc = xp.conj(self.v)
        uc = xp.conj(self.u)
        x = exp_xform(y, vc, uc, xp=self.xp)
        # x = double6(x);
        x = xp.asarray(x)
        if x.ndim == 1:
            x = x[:, xp.newaxis]
        x = xp.asfortranarray(x)  # TODO: necessary?
        return x  # returns an 2D Nd array

    #
    #    if False:
    #        nreps = 1
    #        y = DiagonalOperator(w) * dtmp.reshape((-1, nreps), order='F')

    @profile
    def _back(self, y):
        """ NUFFT-based adjoint operation on the CPU """
        #                # coils should also be stacked within first dimension
        #        # any additional dimensions should be stacked into the second
        #        if (y.ndim > 2) or (y.shape[0] != self.nargin):
        #            x = x.reshape((nargin, -1), order='F')
        # need copy here to avoid modifying original data?
        xp = self.xp
        y = xp.asarray(y).copy()
        y = y.reshape((self.nargout, -1), order=self.order)
        nreps = y.shape[-1]
        nt = self.shape[0] // self.Ncoils  # self.kspace.shape[0]
        if self.weights is not None:
            # apply density compensation
            y = self.weights * y

        y = complexify(y)  # force at least 64-bit complex

        sn_outside_loop = True
        if sn_outside_loop:
            sn_tmp = self.Gnufft.sn.copy()
            self.Gnufft.sn = None
        try:
            if self.zmap is None:  # or (not xp.any(self.zmap)):
                x = xp.empty(
                    (self.shape[1] // self.nspectra, nreps),
                    dtype=y.dtype,
                    order=self.order,
                )
                # multi-thread only across coils and repetitions.  this way is
                # thread safe
                if self.Ncoils > 1:
                    # put all coils (and/or repetitions) along the 2nd
                    # dimension
                    y = y.reshape((nt, -1), order=self.order)

                y = self.Gnufft.T * y

                if y.ndim == 1:
                    y = y[..., xp.newaxis]

                if nreps == 1:
                    if self.coil_sensitivities is not None:
                        y *= xp.conj(self.coil_sensitivities)
                        x = y.sum(axis=1, keepdims=True)
                    else:
                        x = y
                else:
                    for repIDX in range(nreps):
                        rep_slice = slice(
                            repIDX * self.Ncoils, (repIDX + 1) * self.Ncoils
                        )
                        if self.coil_sensitivities is not None:
                            y[:, rep_slice] *= xp.conj(self.coil_sensitivities)
                            x[:, repIDX] = xp.sum(y[:, rep_slice], axis=1)
                        else:
                            x[:, repIDX] = y[:, rep_slice][:, 0]
            else:  # with fieldmap approximation
                if self.coil_sensitivities is None:  # ) or
                    # (not xp.any(self.coil_sensitivities))):
                    # fieldmap, but no coil sensitivities
                    y = y.reshape((nt, nreps), order=self.order)
                    for ll in range(0, self.fieldmap_segments):
                        # tmp = self.fmap_basis[:, ll][:, xp.newaxis] * y
                        tmp = xp.conj(self.fmap_basis[:, ll : ll + 1]) * y
                        tmp = self.Gnufft.T * tmp
                        if tmp.ndim == 1:
                            tmp = tmp[:, np.newaxis]
                        tmp *= xp.conj(self.fmap_coeffs[:, ll : ll + 1])
                        if ll == 0:
                            x = tmp
                        else:
                            x += tmp
                    del tmp
                else:
                    # fieldmap and coil sensitivities case
                    x = xp.empty(
                        (self.shape[1] // self.nspectra, nreps),
                        dtype=y.dtype,
                        order=self.order,
                    )
                    # multi-thread only across coils and repetitions.  this way is
                    # thread safe
                    if self.Ncoils > 1:
                        # put all coils (and/or repetitions) along the 2nd dim
                        y = y.reshape((nt, -1), order=self.order)
                    else:
                        # even with 1 coil need 2d for proper broadcasting
                        y = y.reshape((nt, 1), order=self.order)

                    xtmp = 0
                    for ll in range(self.fieldmap_segments):
                        tmp = xp.conj(self.fmap_basis[:, ll : ll + 1]) * y
                        tmp = self.Gnufft.T * tmp
                        if tmp.ndim == 1:
                            tmp = tmp[:, np.newaxis]
                        # can broadcast across both coils & repetitions
                        tmp *= xp.conj(self.fmap_coeffs[:, ll : ll + 1])
                        xtmp += tmp
                    del tmp
                    for repIDX in range(nreps):
                        rep_slice = slice(
                            repIDX * self.Ncoils, (repIDX + 1) * self.Ncoils
                        )
                        xtmp[:, rep_slice] *= xp.conj(self.coil_sensitivities)
                        x[:, repIDX] = xp.sum(xtmp[:, rep_slice], axis=1)
                    del xtmp
            if sn_outside_loop:
                if x.ndim > 2:
                    raise ValueError("unexpected ndim for x")
                # apply rolloff correction
                x *= xp.conj(
                    self._apply_mask(sn_tmp, squeeze_output=x.ndim == 1)
                )
        finally:
            if sn_outside_loop:
                # store rolloff factors back in Gnufft object
                self.Gnufft.sn = sn_tmp

        x = xp.asarray(x)
        if x.ndim == 1:
            x = x[:, xp.newaxis]
        # x = xp.asfortranarray(x)  # TODO: necessary?
        return x  # returns an 2D Nd array

    # @profile
    def _back_multispectral(self, y):
        xp = self.xp
        nreps = y.size // self.nargout
        x = xp.empty(
            (self.shape[1] // self.nspectra, self.nspectra, nreps),
            dtype=y.dtype,
            order=self.order,
        )
        for n, (f, v) in enumerate(
            zip(self.spectral_offsets, self.offset_arrays)
        ):
            if v is not None:
                use_linops = False
                if use_linops:
                    Omega_shift = DiagonalOperator(
                        v,
                        order=self.order,
                        squeeze_reps=False,
                        loc_in=self.loc_in,
                        loc_out=self.loc_out,
                    )
                    if self.Ncoils > 1:
                        Omega_shift = BlockDiagLinOp(
                            (Omega_shift,) * self.Ncoils,
                            squeeze_reps_out=False,
                            loc_in=self.loc_in,
                            loc_out=self.loc_out,
                        )
                    tmp = Omega_shift.H * y
                else:
                    y_shape = y.shape
                    y = y.reshape(
                        (self.shape[0] // self.Ncoils, -1), order=self.order
                    )
                    if isinstance(v, tuple):
                        # weighted sum of frequencies
                        #     (e.g. multispectral fat model)
                        for noff, (weight, vf) in enumerate(v):
                            # single frequency with weight = 1.0
                            if vf.ndim == 1:
                                vf = vf[:, np.newaxis]
                            if noff == 0:
                                tmp = xp.conj(weight * vf) * y
                            else:
                                tmp += xp.conj(weight * vf) * y
                    else:
                        # single frequency with weight = 1.0
                        if v.ndim == 1:
                            v = v[:, np.newaxis]
                        tmp = xp.conj(v) * y
                    tmp = tmp.reshape(y_shape, order=self.order)
                x[:, n, :] = self._back(tmp)
            else:
                x[:, n, :] = self._back(y)
        if x.ndim == 1:  # TODO: necessary?
            x = x[:, xp.newaxis]
        x = xp.asfortranarray(x)  # TODO: necessary?
        return x

    def prep_Toeplitz(self, weights=None):
        raise NotImplementedError("TODO")
        xp = self.xp
        if weights is not None:
            if self.weights is not None:
                raise ValueError("TODO?")
            if not isinstance(weights, DiagonalOperator):
                W_op = DiagonalOperator(weights, order="F")
            else:
                W_op = weights
        else:
            if self.weights is not None:
                raise ValueError("TODO")  # copy the existing weights
            else:
                W_op = None
            # raise ValueError("weights must be a DiagonalOperator")

        # TODO: update for basis.transform built into self.weights
        if self.basis.transform is not None:
            basis_sq = xp.real(
                xp.conj(self.basis.transform) * self.basis.transform
            )
            if W_op is not None:
                # fold the basis transform into the DCF weights
                W_op.diag.setdiag(W_op.diag.diagonal() * basis_sq)
            else:
                W_op = DiagonalOperator(basis_sq, order="F")
        self.W_op = W_op

        if not hasattr(self.Gnufft, "Q"):
            self.Gnufft.prep_toeplitz()

        # if False:
        #     if self.W_op is None:
        #         T = CompositeLinOp(self.Gnufft.H, self.Gnufft)
        #     else:
        #         T = CompositeLinOp(self.Gnufft.H, W_op, self.Gnufft)
        # else:
        #     # FT_op = FFT_Operator(self.Nd)
        #     pass

    @profile
    def norm(self, x):
        """implementation of the normal operator (A.T*A)."""
        x = complexify(x, xp=self.xp)
        # if self.mask is not None:
        #     x = xp.embed(x, self.mask, xp=self.xp)
        # try:
        #     if x.size % self.nargin != 0:
        #         raise ValueError("wrong size input")
        #     nreps = x.size // self.nargin
        # except IndexError:
        #     nreps = 1

        if hasattr(self, "Q"):
            #   D = self.weights
            raise NotImplementedError("TODO")
        #            slices = [slice(None), ]*x.ndim
        #            for d in range(len(self.Nd)):
        #                slices[d] = slice(self.Nd[d])
        #            if nreps == 1:
        #                y = fftn(x, s=self.Kd)
        #                y *= self.Q
        #                y = ifftn(y)[slices]
        #            else:
        #                x = x.reshape(tuple(self.Nd) + (nreps, ),
        #                              order=self.order)
        #                fft_axes = tuple(xp.arange(len(self.Nd)))
        #                y = fftn(x, s=self.Kd, axes=fft_axes)
        #                y *= self.Q[..., xp.newaxis]  # add an axis for repetitions
        #                y = ifftn(y, axes=fft_axes)[slices]
        else:
            y = self * x
            y = self.H * y
        return y
