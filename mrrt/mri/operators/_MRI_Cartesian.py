from functools import partial
from math import sqrt
import warnings

import numpy as np

from mrrt.operators import LinearOperatorMulti
from mrrt.utils import config, embed, fftn, ifftn, masker, prod, profile
from mrrt.utils import complexify as as_complex_array
from mrrt.mri.operators._pixel_basis import PixelBasis

if config.have_pyfftw:
    from mrrt.utils import build_fftn, build_ifftn

if config.have_cupy:
    import cupy


class MRI_Cartesian(LinearOperatorMulti):
    """ n-dimensional Fourier Transform operator with optional sampling mask.
    """

    def __init__(
        self,
        arr_shape,
        order="F",
        arr_dtype=np.float32,
        use_fft_shifts=True,
        sample_mask=None,
        ortho=False,
        coil_sensitivities=None,
        force_real_image=False,
        debug=False,
        preplan_pyfftw=True,
        pyfftw_threads=None,
        fft_axes=None,
        fftshift_axes=None,
        planner_effort="FFTW_ESTIMATE",
        loop_over_coils=False,
        preserve_memory=False,
        disable_warnings=False,
        im_mask=None,
        pixel_basis="dirac",
        rel_fov=None,
        **kwargs,
    ):
        """Cartesian MRI Operator  (with partial FFT and coil maps).

        Parameters
        ----------
        arr_shape : int
            shape of the array
        order : {'C','F'}, optional
            array ordering that will be assumed if inputs/outputs need to be
            reshaped
        arr_dtype : numpy.dtype, optional
            dtype for the array
        sample_mask : array_like, optional
            boolean mask of which FFT coefficients to keep
        coil_sensitivities : array, optional
            Array of coil sensitivities.
        ortho : bool, optional
            if True, change the normalizeation to the orthogonal case
        preplan_pyfftw : bool, optional
            if True, precompute the pyFFTW plan upon object creation
        pyfftw_threads : int, optional
            number of threads to be used by pyFFTW.  defaults to
            multiprocessing.cpu_count() // 2.
        use_fft_shifts : bool, optional
            If False, do not apply any FFT shifts
        fft_axes : tuple or None, optional
            Specify a subset of the axes to transform.  The default is to
            transform all axes.
        fftshift_axes : tuple or None, optional
            Specify a subset of the axes to fftshift.  The default is to
            shift all axes.
        im_mask : ndarray or None, optional
            Image domain mask
        force_real_image : bool, optional
        loop_over_coils : bool, optional
            If True, memory required is lower, but speed will be slower.
        preserve_memory : bool, optional
            If False, conjugate copy of coils won't be precomputed
        debug : bool, optional

        Additional Parameters
        ---------------------
        nd_input : bool, optional
        nd_output : bool, optional

        """
        if isinstance(arr_shape, (np.ndarray, list)):
            # retrieve shape from array
            arr_shape = tuple(arr_shape)
        if not isinstance(arr_shape, tuple):
            raise ValueError("expected array_shape to be a tuple or list")

        self.arr_shape = arr_shape
        self.ndim = len(arr_shape)
        self.order = order
        self.use_fft_shifts = use_fft_shifts
        self.disable_warnings = disable_warnings

        if "loc_in" in kwargs and kwargs["loc_in"] == "gpu":
            xp = cupy
            on_gpu = True
        else:
            xp = np
            on_gpu = False
        self._on_gpu = on_gpu

        if sample_mask is not None:
            # masking faster if continguity of mask matches
            if self.order == "F":
                sample_mask = xp.asfortranarray(sample_mask)
            elif self.order == "C":
                sample_mask = xp.ascontiguousarray(sample_mask)
            else:
                raise ValueError("order must be C or F")
        self.sample_mask = sample_mask

        self.force_real_image = force_real_image
        self.debug = debug
        if self.sample_mask is not None:
            if sample_mask.shape != arr_shape:
                raise ValueError("sample mask shape must match arr_shape")
            # make sure it is boolean
            self.sample_mask = self.sample_mask > 0
            # # prestore raveled mask indices to save time later during masking
            # self.sample_mask_idx = xp.where(
            #     self.sample_mask.ravel(order=self.order)
            # )

        self.preserve_memory = preserve_memory
        self.loop_over_coils = loop_over_coils

        # can specify a subset of the axes to perform the FFT/FFTshifts over
        self.fft_axes = fft_axes
        if self.fft_axes is None:
            if self.order == "C":
                # last ndim axes
                self.fft_axes = tuple([-ax for ax in range(self.ndim, 0, -1)])
            else:
                # first ndim axes
                self.fft_axes = tuple(range(self.ndim))

        if fftshift_axes is None:
            self.fftshift_axes = self.fft_axes
        else:
            self.fftshift_axes = fftshift_axes

        # configure scaling  (e.g. unitary operator or not)
        self.ortho = ortho
        if self.fft_axes is None:
            Ntrans = prod(self.arr_shape)
        else:
            Ntrans = prod(np.asarray(self.arr_shape)[np.asarray(self.fft_axes)])
        if self.ortho:
            # sqrt of product of shape along axes where FFT is performed
            self.scale_ortho = sqrt(Ntrans)
            self.gpu_scale_inverse = 1  # self.scale_ortho / Ntrans
            # self.gpu_scale_forward = self.scale_ortho
        else:
            self.scale_ortho = None
            self.gpu_scale_inverse = 1 / Ntrans

        if "mask_out" in kwargs:
            raise ValueError(
                "This operator specifies `mask_out` via the "
                "parameter `sample_mask"
            )

        if ("mask_in" in kwargs) or ("mask_out" in kwargs):
            raise ValueError(
                "This operator specifies `mask_in` via the "
                "parameter `im_mask"
            )

        if coil_sensitivities is not None:
            if coil_sensitivities.ndim == self.ndim + 1:
                # self.arr_shape + (Ncoils, )
                if self.order == "C":
                    Ncoils = coil_sensitivities.shape[0]
                else:
                    Ncoils = coil_sensitivities.shape[-1]
                Nmaps = 1
            elif coil_sensitivities.ndim == self.ndim + 2:
                # case with multiple maps (e.g. ESPIRIT soft-SENSE)
                # self.arr_shape + (Ncoils, Nmaps)
                if self.order == "C":
                    Ncoils = coil_sensitivities.shape[1]
                    Nmaps = coil_sensitivities.shape[0]
                else:
                    Ncoils = coil_sensitivities.shape[-2]
                    Nmaps = coil_sensitivities.shape[-1]
            else:
                # determine based on size
                Ncoils = coil_sensitivities.size / prod(self.arr_shape)
            if (Ncoils % 1) != 0:
                raise ValueError("sensitivity map size mismatch")
            self.Ncoils = int(Ncoils)
            self.Nmaps = Nmaps
            if self.order == "C":
                cmap_shape = (self.Nmaps, self.Ncoils) + self.arr_shape
                if not coil_sensitivities.flags.c_contiguous:
                    msg = (
                        "Converting coil_sensitivities to be C contiguous"
                        " (requires a copy).  To avoid the copy, convert to "
                        "Fortran contiguous order prior to calling "
                        "MRI_Cartesian (see np.ascontiguousarray)"
                    )
                    if not self.disable_warnings:
                        warnings.warn(msg)
                    coil_sensitivities = xp.ascontiguousarray(
                        coil_sensitivities
                    )
            else:
                cmap_shape = self.arr_shape + (self.Ncoils, self.Nmaps)
                if not coil_sensitivities.flags.f_contiguous:
                    msg = (
                        "Converting coil_sensitivities to be Fortan contiguous"
                        " (requires a copy).  To avoid the copy, convert to "
                        "Fortran contiguous order prior to calling "
                        "MRI_Cartesian (see np.asfortranarray)"
                    )
                    if not self.disable_warnings:
                        warnings.warn(msg)
                    coil_sensitivities = xp.asfortranarray(coil_sensitivities)
            if tuple(coil_sensitivities.shape) != tuple(cmap_shape):
                coil_sensitivities = coil_sensitivities.reshape(
                    cmap_shape, order=self.order
                )
            if not self.preserve_memory:
                self.coil_sensitivities_conj = xp.conj(coil_sensitivities)
        else:
            self.Ncoils = 1
            self.Nmaps = 1

        if self.Ncoils == 1:
            # TODO: currently has a shape bug if Ncoils == 1
            #       and loop_over_coils = False.
            self.loop_over_coils = True

        self.coil_sensitivities = coil_sensitivities

        if im_mask is not None:
            if im_mask.shape != arr_shape:
                raise ValueError("im_mask shape mismatch")
            if order != "F":
                raise ValueError("only order='F' supported for im_mask case")
            nargin = xp.count_nonzero(im_mask)
            self.im_mask = im_mask
        else:
            nargin = prod(arr_shape)
            self.im_mask = None
        nargin *= self.Nmaps

        # nargout = # of k-space samples
        if sample_mask is not None:
            nargout = xp.count_nonzero(sample_mask) * self.Ncoils
        else:
            nargout = nargin // self.Nmaps * self.Ncoils
        nargout = int(nargout)

        self.idx_orig = None
        self.idx_conj = None
        self.sample_mask_conj = None

        # output of FFTs will be complex, regardless of input type
        self.result_dtype = np.result_type(arr_dtype, np.complex64)

        matvec_allows_repetitions = kwargs.pop(
            "matvec_allows_repetitions", True
        )
        squeeze_reps = kwargs.pop("squeeze_reps", True)
        nd_input = kwargs.pop("nd_input", False)
        nd_output = kwargs.pop("nd_output", False)

        if (self.sample_mask is not None) and nd_output:
            raise ValueError("cannot have both nd_output and sample_mask")
        if nd_output:
            if self.Ncoils == 1:
                shape_out = self.arr_shape
            else:
                if Nmaps == 1:
                    if self.order == "C":
                        shape_out = (self.Ncoils,) + self.arr_shape
                    else:
                        shape_out = self.arr_shape + (self.Ncoils,)
                else:
                    if self.order == "C":
                        shape_out = (self.Nmaps, self.Ncoils) + self.arr_shape
                    else:
                        shape_out = self.arr_shape + (self.Ncoils, self.Nmaps)
        else:
            shape_out = (nargout, 1)

        if self.Nmaps == 1:
            shape_in = self.arr_shape
        else:
            if self.order == "C":
                shape_in = (self.Nmaps,) + self.arr_shape
            else:
                shape_in = self.arr_shape + (self.Nmaps,)
        if self.order == "C":
            self.shape_inM = (self.Nmaps,) + self.arr_shape
        else:
            self.shape_inM = self.arr_shape + (self.Nmaps,)
        self.shape_in1 = self.arr_shape
        self.have_pyfftw = config.have_pyfftw

        if self.on_gpu:
            self.preplan_pyfftw = False
        else:
            self.preplan_pyfftw = preplan_pyfftw if self.have_pyfftw else False

            if self.preplan_pyfftw:
                self._preplan_fft(pyfftw_threads, planner_effort)
                # raise ValueError("Implementation Incomplete")

        if self.on_gpu:
            self.fftn = partial(fftn, xp=cupy)
            self.ifftn = partial(ifftn, xp=cupy)
        else:
            if self.preplan_pyfftw:
                self._preplan_fft(pyfftw_threads, planner_effort)
            else:
                self.fftn = partial(fftn, xp=np)
                self.ifftn = partial(ifftn, xp=np)
        self.fftshift = xp.fft.fftshift
        self.ifftshift = xp.fft.ifftshift

        self.rel_fov = rel_fov

        self.mask = None  # TODO: implement or remove (expected by CUDA code)

        matvec = self.forward
        matvec_adj = self.adjoint
        self.norm_available = False
        self.norm = self._norm

        super(MRI_Cartesian, self).__init__(
            nargin=nargin,
            nargout=nargout,
            matvec=matvec,
            matvec_transp=matvec_adj,
            matvec_adj=matvec_adj,
            nd_input=nd_input or (im_mask is not None),
            nd_output=nd_output,
            shape_in=shape_in,
            shape_out=shape_out,
            order=self.order,
            matvec_allows_repetitions=matvec_allows_repetitions,
            squeeze_reps=squeeze_reps,
            mask_in=im_mask,
            mask_out=None,  # mask_out,
            symmetric=False,  # TODO: set properly
            hermetian=False,  # TODO: set properly
            dtype=self.result_dtype,
            **kwargs,
        )

        self._init_pixel_basis(pixel_basis=pixel_basis)

    def _preplan_fft(self, pyfftw_threads=None, planner_effort="FFTW_MEASURE"):
        """ Use FFTW builders to pre-plan the FFT for faster repeated
        calls. """
        if pyfftw_threads is None:
            import multiprocessing

            pyfftw_threads = max(1, multiprocessing.cpu_count())
        self.pyfftw_threads = pyfftw_threads
        if self.loop_over_coils:
            a_b = np.empty(self.arr_shape, dtype=self.result_dtype)
        else:
            if self.order == "C":
                a_b = np.empty(
                    (self.Ncoils,) + self.arr_shape, dtype=self.result_dtype
                )
            else:
                a_b = np.empty(
                    self.arr_shape + (self.Ncoils,), dtype=self.result_dtype
                )
        self.fftn = build_fftn(
            a_b,
            axes=self.fft_axes,
            threads=pyfftw_threads,
            overwrite_input=False,
            planner_effort=planner_effort,
        )
        self.ifftn = build_ifftn(
            a_b,
            axes=self.fft_axes,
            threads=pyfftw_threads,
            overwrite_input=False,
            planner_effort=planner_effort,
        )
        del a_b

    def _rescale_basis(self):
        if self.basis.transform is not None:
            from mrrt.utils import power_method

            lam = power_method(self.H * self)
            self.basis.transform /= sqrt(lam)
        return self.basis

    def _init_pixel_basis(self, pixel_basis=None, rel_fov=None):
        xp = self.xp
        if pixel_basis is None:
            pixel_basis = self.pixel_basis
        else:
            self.pixel_basis = pixel_basis

        if rel_fov is None:
            rel_fov = self.rel_fov
        else:
            self.rel_fov = rel_fov

        if pixel_basis != "dirac":
            if rel_fov is None:
                raise ValueError(
                    "For pixel_basis = {}, rel_fov must be specfied.".format(
                        pixel_basis
                    )
                )
        else:

            class DummyBasis(object):
                transform = None

            self.basis = DummyBasis()
            return self.basis

        rel_dx = np.asarray(self.arr_shape) / np.asarray(self.rel_fov)
        rel_dx /= rel_dx.max()
        ranges = [
            (xp.arange(s) / s - 0.5) * rel_dx[n]
            for n, s in enumerate(self.arr_shape)
        ]
        ksp = xp.meshgrid(*ranges, indexing="ij", sparse=False)
        ksp = xp.stack([k.ravel(order="F") for k in ksp], axis=-1)

        self.basis = PixelBasis(
            ksp,
            pixel_basis=self.pixel_basis,
            # don't change values below here.  already normalized the
            # k-space properly above
            dx=1.0,
            fov=self.arr_shape,
            Nd=self.arr_shape,
            xp=self.xp,
        )

        self.basis.transform = xp.asarray(
            self.basis.transform, dtype=self.result_dtype
        )
        self.basis.transform = self.basis.transform.reshape(
            self.arr_shape, order="F"
        )

        # rescale basis.transform to maintain a norm-preserving operator
        self.basis = self._rescale_basis()
        return self.basis

    @profile
    def _adjoint_single_rep(self, y, i_map=0):
        use_smaps = self.coil_sensitivities is not None
        xp = self.xp
        if self.loop_over_coils:
            if use_smaps:
                x = xp.zeros(
                    self.shape_in1,
                    order=self.order,
                    dtype=xp.result_type(y, np.complex64),
                )
            for coil in range(self.Ncoils):
                if self.order == "C":
                    coil_slice = (coil, Ellipsis)
                    imap_coil_slice = (i_map, coil, Ellipsis)
                else:
                    coil_slice = (Ellipsis, coil)
                    imap_coil_slice = (Ellipsis, coil, i_map)
                y0 = y[coil_slice]
                if self.basis.transform is not None:
                    y0 *= xp.conj(self.basis.transform)
                if self.use_fft_shifts:
                    y0 = self.ifftshift(y0, axes=self.fftshift_axes)
                if self.preplan_pyfftw:
                    x0 = self.ifftn(y0)
                else:
                    x0 = self.ifftn(y0, axes=self.fft_axes)
                if self.use_fft_shifts:
                    x0 = self.fftshift(x0, axes=self.fftshift_axes)
                if use_smaps:
                    if self.preserve_memory:
                        x += x0 * xp.conj(
                            self.coil_sensitivities[imap_coil_slice]
                        )
                    else:
                        x += x0 * self.coil_sensitivities_conj[imap_coil_slice]
                else:
                    x = x0
        else:
            if self.use_fft_shifts:
                y = self.ifftshift(y, axes=self.fftshift_axes)
            if self.basis.transform is not None:
                if self.order == "C":
                    y *= xp.conj(self.basis.transform[np.newaxis, ...])
                else:
                    y *= xp.conj(self.basis.transform[..., np.newaxis])
            if self.preplan_pyfftw:
                x = self.ifftn(y)
            else:
                x = self.ifftn(y, axes=self.fft_axes)
            if self.use_fft_shifts:
                x = self.fftshift(x, axes=self.fftshift_axes)
            if use_smaps:
                if self.order == "C":
                    imap_slice = (i_map, Ellipsis)
                    coil_axis = 0
                else:
                    imap_slice = (Ellipsis, i_map)
                    coil_axis = -1
                if self.preserve_memory:
                    x *= xp.conj(self.coil_sensitivities[imap_slice])
                else:
                    x *= self.coil_sensitivities_conj[imap_slice]
                x = x.sum(coil_axis)  # sum over coils
        return x

    @profile
    def adjoint(self, y):
        # TODO: add test for this case and the coils + sample_mask case
        # if y.ndim == 1 or y.shape[-1] == 1:
        xp = self.xp
        ncoils = self.Ncoils
        nreps = int(y.size / self.shape[0])
        if self.sample_mask is not None:
            if y.ndim == 1 and self.ndim > 1:
                y = y[:, np.newaxis]
            nmask = xp.count_nonzero(self.sample_mask)
            if self.on_gpu:
                nmask = nmask.get()
            if y.shape[0] != nmask:
                if self.order == "C":
                    y = y.reshape((-1, nmask), order=self.order)
                else:
                    y = y.reshape((nmask, -1), order=self.order)
            y = embed(y, mask=self.sample_mask, order=self.order)
        if nreps == 1:
            # 1D or single repetition or single coil nD
            if self.order == "C":
                y = y.reshape((ncoils,) + self.shape_in1, order=self.order)
            else:
                y = y.reshape(self.shape_in1 + (ncoils,), order=self.order)
            if self.Nmaps == 1:
                x = self._adjoint_single_rep(y, i_map=0)
            else:
                x = xp.zeros(
                    self.shape_inM, dtype=xp.result_type(y, xp.complex64)
                )
                if self.order == "C":
                    for i_map in range(self.Nmaps):
                        x[i_map, ...] = self._adjoint_single_rep(y, i_map=i_map)
                else:
                    for i_map in range(self.Nmaps):
                        x[..., i_map] = self._adjoint_single_rep(y, i_map=i_map)
        else:
            if self.order == "C":
                y = y.reshape(
                    (nreps, ncoils) + self.shape_in1, order=self.order
                )  # or shape_out?
                x_shape = (nreps,) + self.shape_inM
            else:
                y = y.reshape(
                    self.shape_in1 + (ncoils, nreps), order=self.order
                )  # or shape_out?
                x_shape = self.shape_inM + (nreps,)

            x = xp.zeros(x_shape, dtype=xp.result_type(y, xp.complex64))
            for i_map in range(self.Nmaps):
                if self.order == "C":
                    for rep in range(nreps):
                        x[rep, i_map, ...] = self._adjoint_single_rep(
                            y[rep, ...], i_map=i_map
                        )
                else:
                    for rep in range(nreps):
                        x[..., i_map, rep] = self._adjoint_single_rep(
                            y[..., rep], i_map=i_map
                        )
        #        if self.im_mask:
        #            x = masker(x, self.im_mask, order=self.order)
        if self.ortho:
            x *= self.scale_ortho
        if x.dtype != self.result_dtype:
            x = x.astype(self.result_dtype)
        if self.force_real_image:
            # x = x.real.astype(self.result_dtype)
            if x.dtype in [np.complex64, np.complex128]:
                x.imag[:] = 0
        return x

    @profile
    def _forward_single_rep(self, x, i_map=0):
        xp = self.xp
        if self.order == "C":
            y_shape = (self.Ncoils,) + self.shape_in1
        else:
            y_shape = self.shape_in1 + (self.Ncoils,)
        y = xp.zeros(
            y_shape, dtype=xp.result_type(x, xp.complex64), order=self.order
        )
        use_smaps = self.coil_sensitivities is not None
        if self.loop_over_coils:
            for coil in range(self.Ncoils):
                if use_smaps:
                    if self.order == "C":
                        xc = x * self.coil_sensitivities[i_map, coil, ...]
                    else:
                        xc = x * self.coil_sensitivities[..., coil, i_map]
                else:
                    xc = x
                if self.order == "C":
                    coil_slice = (coil, Ellipsis)
                else:
                    coil_slice = (Ellipsis, coil)
                if self.debug:
                    print("x.shape = {}".format(x.shape))
                    print("xc.shape = {}".format(xc.shape))
                    print("y.shape = {}".format(y.shape))
                if self.use_fft_shifts:
                    y[coil_slice] = self.ifftshift(xc, axes=self.fftshift_axes)
                else:
                    y[coil_slice] = xc
                if self.preplan_pyfftw:
                    y[coil_slice] = self.fftn(y[coil_slice])
                else:
                    y[coil_slice] = self.fftn(y[coil_slice], axes=self.fft_axes)
                if self.use_fft_shifts:
                    y[coil_slice] = self.fftshift(
                        y[coil_slice], axes=self.fftshift_axes
                    )
                if self.basis.transform is not None:
                    y *= self.basis.transform
        else:
            if use_smaps:
                if self.order == "C":
                    x = as_complex_array(x)[xp.newaxis, ...]
                    x = x * self.coil_sensitivities[i_map, ...]
                else:
                    x = as_complex_array(x)[..., xp.newaxis]
                    x = x * self.coil_sensitivities[..., i_map]
            if self.use_fft_shifts:
                x = self.ifftshift(x, axes=self.fftshift_axes)
            if self.preplan_pyfftw:
                y = self.fftn(x)
            else:
                y = self.fftn(x, axes=self.fft_axes)
            if self.use_fft_shifts:
                y = self.fftshift(y, axes=self.fftshift_axes)
            if self.basis.transform is not None:
                if self.order == "C":
                    y *= self.basis.transform[xp.newaxis, ...]
                else:
                    y *= self.basis.transform[..., xp.newaxis]

        if self.sample_mask is not None:
            # y = y[self.sample_mask]
            y = masker(
                y,
                mask=self.sample_mask,
                order=self.order,
                # mask_idx_ravel=self.sample_mask_idx,
            )
        return y

    @profile
    def forward(self, x):
        xp = self.xp
        if self.force_real_image:
            if x.dtype in [np.complex64, np.complex128]:
                x.imag[:] = 0
        if self.im_mask is None:
            size_1rep = self.nargin
        else:
            size_1rep = prod(self.shape_inM)
        if x.size < size_1rep:
            raise ValueError("data, x, too small to transform.")
        elif x.size == size_1rep:
            nreps = 1
            # 1D or single repetition nD
            x = x.reshape(self.shape_inM, order=self.order)
            if self.order == "C":
                y = self._forward_single_rep(x[0, ...], i_map=0)
                for i_map in range(1, self.Nmaps):
                    y += self._forward_single_rep(x[i_map, ...], i_map=i_map)
            else:
                y = self._forward_single_rep(x[..., 0], i_map=0)
                for i_map in range(1, self.Nmaps):
                    y += self._forward_single_rep(x[..., i_map], i_map=i_map)
        else:
            if self.order == "C":
                nreps = x.shape[0]
                x_shape = (nreps,) + self.shape_inM
                if self.sample_mask is not None:
                    y_shape = (nreps, self.Ncoils, self.nargout)
                else:
                    y_shape = (nreps, self.Ncoils) + self.shape_in1
            else:
                nreps = x.shape[-1]
                x_shape = self.shape_inM + (nreps,)
                if self.sample_mask is not None:
                    y_shape = (self.nargout, self.Ncoils, nreps)
                else:
                    y_shape = self.shape_in1 + (self.Ncoils, nreps)
            x = x.reshape(x_shape, order=self.order)
            if self.sample_mask is not None:
                # number of samples by number of repetitions
                y = xp.zeros(
                    y_shape,
                    dtype=xp.result_type(x, xp.complex64),
                    order=self.order,
                )
            else:
                y = xp.zeros(
                    y_shape,
                    dtype=xp.result_type(x, xp.complex64),
                    order=self.order,
                )
            if self.order == "C":
                for rep in range(nreps):
                    y[rep, ...] = self._forward_single_rep(
                        x[rep, 0, ...], i_map=0
                    ).reshape(y.shape[1:], order=self.order)
                    for i_map in range(1, self.Nmaps):
                        y[rep, ...] += self._forward_single_rep(
                            x[rep, i_map, ...], i_map=i_map
                        ).reshape(y.shape[1:], order=self.order)
            else:
                for rep in range(nreps):
                    y[..., rep] = self._forward_single_rep(
                        x[..., 0, rep], i_map=0
                    ).reshape(y.shape[:-1], order=self.order)
                    for i_map in range(1, self.Nmaps):
                        y[..., rep] += self._forward_single_rep(
                            x[..., i_map, rep], i_map=i_map
                        ).reshape(y.shape[:-1], order=self.order)
        if self.ortho:
            y /= self.scale_ortho

        if y.dtype != self.result_dtype:
            y = y.astype(self.result_dtype)

        return y

    def _norm_single_rep(self, x):
        # chain forward and adjoint together for a single repetition
        y = self._forward_single_rep(x)
        if self.sample_mask is not None:
            y = embed(y, mask=self.sample_mask, order=self.order)
        if self.order == "C":
            y_shape = (self.Ncoils,) + self.shape_in1
        else:
            y_shape = self.shape_in1 + (self.Ncoils,)
        y = y.reshape(y_shape, order=self.order)
        x = self._adjoint_single_rep(y)
        return x

    @profile
    def _norm(self, x):
        # forward transform, immediately followed by inverse transform
        # slightly faster than calling self.adjoint(self.forward(x))
        xp = self.xp
        if self.force_real_image:
            if x.dtype in [np.complex64, np.complex128]:
                x.imag[:] = 0
        if self.im_mask is None:
            size_1rep = self.nargin
        else:
            size_1rep = prod(self.shape_inM)
        if x.size < size_1rep:
            raise ValueError("data, x, too small to transform.")
        elif x.size == size_1rep:
            nreps = 1
            # 1D or single repetition nD
            x = x.reshape(self.shape_inM, order=self.order)
            y = self._norm_single_rep(x)
        else:
            nreps = x.size // size_1rep
            if self.order == "C":
                x = x.reshape((nreps,) + self.shape_inM, order=self.order)
                y = xp.zeros_like(x)
                for rep in range(nreps):
                    y[rep, ...] = self._norm_single_rep(x[rep, ...]).reshape(
                        y.shape[1:], order=self.order
                    )
            else:
                x = x.reshape(self.shape_inM + (nreps,), order=self.order)
                y = xp.zeros_like(x)
                for rep in range(nreps):
                    y[..., rep] = self._norm_single_rep(x[..., rep]).reshape(
                        y.shape[:-1], order=self.order
                    )
        if y.dtype != self.result_dtype:
            y = y.astype(self.result_dtype)

        if not self.nd_input:
            if self.squeeze_reps_in and (nreps == 1):
                y = xp.ravel(y, order=self.order)
            else:
                if self.order == "C":
                    y = y.reshape((nreps, -1), order=self.order)
                else:
                    y = y.reshape((-1, nreps), order=self.order)
        return y
