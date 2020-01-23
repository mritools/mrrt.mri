from itertools import product

import numpy as np
import pytest

from mrrt.mri.operators import MRI_Cartesian
from mrrt.utils import config, masker, embed
from mrrt.utils import fftn, fftnc, ifftn, ifftnc

all_xp = [np]
if config.have_cupy:
    import cupy

    if cupy.cuda.runtime.getDeviceCount() > 0:
        all_xp += [cupy]


def get_loc(xp):
    """Location arguments corresponding to numpy or CuPy case."""
    if xp is np:
        return dict(loc_in="cpu", loc_out="cpu")
    else:
        return dict(loc_in="gpu", loc_out="gpu")


def get_data(xp, shape=(128, 127)):
    rstate = xp.random.RandomState(1234)
    # keep one dimension odd to make sure shifts work correctly
    c = rstate.randn(*shape)
    return c


if config.have_pyfftw:
    preplan_pyfftw_vals = [True, False]
else:
    preplan_pyfftw_vals = [False]


# @dec.slow
@pytest.mark.parametrize(
    "xp, shift, nd_in, nd_out, order, real_dtype",
    product(
        all_xp,
        [True, False],
        [True, False],
        [True, False],
        ["C", "F"],
        [np.float32, np.float64, np.complex64, np.complex128],
    ),
)
def test_FFT(xp, shift, nd_in, nd_out, order, real_dtype):
    rtol = 1e-4
    atol = 1e-4
    cimg = get_data(xp).astype(real_dtype)
    cplx_type = xp.result_type(cimg.dtype, xp.complex64)

    FTop = MRI_Cartesian(
        cimg.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )

    """
    test forward transform
    """
    tmp = FTop * cimg
    if nd_out:
        assert tmp.shape == cimg.shape
    else:
        assert tmp.ndim == 1
    if xp.isrealobj(cimg):
        assert tmp.real.dtype == cimg.dtype
    else:
        assert tmp.dtype == cimg.dtype

    tmp = tmp.reshape(cimg.shape, order=order)
    if shift:
        numpy_tmp = xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(cimg)))
    else:
        numpy_tmp = xp.fft.fftn(cimg)

    if numpy_tmp.dtype != cplx_type:
        numpy_tmp = numpy_tmp.astype(cplx_type)

    xp.testing.assert_allclose(tmp, numpy_tmp, rtol=rtol, atol=atol)

    """
    test adjoint transform
    """
    tmp2 = FTop.H * tmp
    if nd_in:
        assert tmp2.shape == cimg.shape
    else:
        assert tmp2.ndim == 1

    if xp.isrealobj(cimg):
        assert tmp2.real.dtype == cimg.dtype
    else:
        assert tmp2.dtype == cimg.dtype

    tmp2 = tmp2.reshape(cimg.shape, order=order)
    if shift:
        numpy_tmp2 = xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(numpy_tmp)))
    else:
        numpy_tmp2 = xp.fft.ifftn(numpy_tmp)
    xp.testing.assert_allclose(tmp2, numpy_tmp2, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "xp, shift, nd_in, nd_out, order",
    product(all_xp, [True, False], [True, False], [True, False], ["C", "F"]),
)
def test_FFT_roundtrips(xp, shift, nd_in, nd_out, order):
    rtol = 1e-3
    atol = 1e-3
    c = get_data(xp)

    FTop = MRI_Cartesian(
        c.shape,
        order=order,
        nd_input=nd_in,
        nd_output=nd_out,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=False,
        **get_loc(xp),
    )

    """
    test round trip transform with paranthesis grouping
    """
    tmp = FTop.H * (FTop * c)
    xp.testing.assert_allclose(
        tmp.reshape(c.shape, order=order).real, c, rtol=rtol, atol=atol
    )

    """
    test round trip without paranthesis
    """
    tmp = FTop.H * FTop * c
    xp.testing.assert_allclose(
        tmp.reshape(c.shape, order=order).real, c, rtol=rtol, atol=atol
    )

    """
    test round trip with composite operator
    """
    tmp = (FTop.H * FTop) * c
    xp.testing.assert_allclose(
        tmp.reshape(c.shape, order=order).real, c, rtol=rtol, atol=atol
    )


# @dec.slow
@pytest.mark.parametrize(
    "xp, shift, nd_in, nd_out, order, ortho, fft_axes",
    product(
        all_xp,
        [True, False],
        [True, False],
        [True, False],
        ["C", "F"],
        [True, False],
        [(0,), (1,), (0, 1), None],
    ),
)
def test_FFT_axes_subsets_and_ortho(
    xp, shift, nd_in, nd_out, order, ortho, fft_axes
):
    rtol = 1e-3
    atol = 1e-3
    c = get_data(xp)
    """ Test applying FFT only along particular axes. """
    FTop = MRI_Cartesian(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        fft_axes=fft_axes,
        ortho=ortho,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )

    """
    test forward transform
    """
    tmp = FTop * c
    if nd_out:
        assert tmp.shape == c.shape
    else:
        assert tmp.ndim == 1
    assert tmp.real.dtype == c.dtype

    tmp = tmp.reshape(c.shape, order=order)
    if ortho:
        fftargs = dict(norm="ortho")
    else:
        fftargs = {}
    if shift:
        numpy_tmp = xp.fft.fftshift(
            xp.fft.fftn(
                xp.fft.ifftshift(c, axes=fft_axes), axes=fft_axes, **fftargs
            ),
            axes=fft_axes,
        )
    else:
        numpy_tmp = xp.fft.fftn(c, axes=fft_axes, **fftargs)
    xp.testing.assert_allclose(tmp, numpy_tmp, rtol=rtol, atol=atol)

    """
    test adjoint transform
    """
    tmp2 = FTop.H * numpy_tmp
    if nd_in:
        assert tmp2.shape == c.shape
    else:
        assert tmp2.ndim == 1
    assert tmp2.real.dtype == c.dtype

    tmp2 = tmp2.reshape(c.shape, order=order)
    if shift:
        numpy_tmp2 = xp.fft.fftshift(
            xp.fft.ifftn(
                xp.fft.ifftshift(numpy_tmp, axes=fft_axes),
                axes=fft_axes,
                **fftargs,
            ),
            axes=fft_axes,
        )
    else:
        numpy_tmp2 = xp.fft.ifftn(numpy_tmp, axes=fft_axes, **fftargs)
    xp.testing.assert_allclose(tmp2, numpy_tmp2, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "xp, shift, nd_in, nd_out, order",
    product(all_xp, [True, False], [True, False], [True, False], ["C", "F"]),
)
def test_FFT_2reps(xp, shift, nd_in, nd_out, order):
    """ multiple input case. """
    c = get_data(xp)
    FTop = MRI_Cartesian(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )

    """
    test forward transform with 2 repetitions
    """
    nreps = 2
    if order == "F":
        c2 = np.stack((c,) * nreps, axis=-1)
    else:
        c2 = np.stack((c,) * nreps, axis=0)
    tmp = FTop * c2
    if nd_out:
        if order == "F":
            assert tmp.shape == c.shape + (nreps,)
        else:
            assert tmp.shape == (nreps,) + c.shape
    else:
        assert tmp.ndim == 2
        if order == "F":
            assert tmp.shape[-1] == nreps
        else:
            assert tmp.shape[0] == nreps
    """
    test adjoint transform with 2 repetitions
    """
    tmp2 = FTop.H * tmp
    if nd_in:
        if order == "F":
            assert tmp2.shape == c.shape + (nreps,)
        else:
            assert tmp2.shape == (nreps,) + c.shape
    else:
        assert tmp2.ndim == 2

    # scipy compatibility
    xp.testing.assert_array_equal(tmp, FTop.matvec(c2))
    xp.testing.assert_array_equal(tmp2, FTop.rmatvec(tmp))


# @dec.slow
@pytest.mark.parametrize(
    "xp, shift, nd_in, nd_out, loop, preplan_pyfftw, order",
    product(
        all_xp,
        [True, False],
        [True, False],
        [True, False],
        [True, False],
        preplan_pyfftw_vals,
        ["C", "F"],
    ),
)
def test_FFT_coilmap(xp, shift, nd_in, nd_out, loop, preplan_pyfftw, order):
    """ multiple coil  case. """
    ncoils = 4
    c = get_data(xp)
    if order == "F":
        cmap = 1 / ncoils * xp.ones((c.shape + (ncoils,)), dtype=c.dtype)
    else:
        cmap = 1 / ncoils * xp.ones(((ncoils,) + c.shape), dtype=c.dtype)

    if xp != np and preplan_pyfftw:
        # pyFFTW preplan case doesn't apply to the GPU
        return

    if order == "C":
        cmap = xp.ascontiguousarray(cmap)
    else:
        cmap = xp.asfortranarray(cmap)
    FTop = MRI_Cartesian(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        loop_over_coils=loop,
        preplan_pyfftw=preplan_pyfftw,
        coil_sensitivities=cmap,
        gpu_force_reinit=False,
        disable_warnings=True,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )
    """
    test forward transform with 4 coils
    """
    tmp = FTop * c
    if nd_out:
        if order == "F":
            assert tmp.shape == c.shape + (ncoils,)
        else:
            assert tmp.shape == (ncoils,) + c.shape
    else:
        assert tmp.ndim == 1

    """
    test adjoint transform
    """
    tmp2 = FTop.H * tmp
    if nd_in:
        assert tmp2.shape == c.shape
    else:
        assert tmp2.ndim == 1
    assert tmp2.real.dtype == c.dtype


# @dec.slow
@pytest.mark.parametrize(
    "xp, shift, nd_in, nd_out, loop, preplan_pyfftw, order",
    product(
        all_xp,
        [True, False],
        [True, False],
        [True, False],
        [True, False],
        preplan_pyfftw_vals,
        ["C", "F"],
    ),
)
def test_FFT_coilmap_2reps(
    xp, shift, nd_in, nd_out, loop, preplan_pyfftw, order
):
    """ multiple coil, multiple input case. """
    ncoils = 4
    c = get_data(xp)
    if order == "F":
        cmap = 1 / ncoils * xp.ones((c.shape + (ncoils,)), dtype=c.dtype)
    else:
        cmap = 1 / ncoils * xp.ones(((ncoils,) + c.shape), dtype=c.dtype)

    if xp != np and preplan_pyfftw:
        # pyFFTW preplan case doesn't apply to the GPU
        return

    if order == "C":
        cmap = xp.ascontiguousarray(cmap)
    else:
        cmap = xp.asfortranarray(cmap)
    FTop = MRI_Cartesian(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        loop_over_coils=loop,
        preplan_pyfftw=preplan_pyfftw,
        coil_sensitivities=cmap,
        gpu_force_reinit=False,
        disable_warnings=True,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )
    """
    test forward transform with 4 coils and 2 repetitions
    """
    nreps = 2

    if order == "F":
        c2 = np.stack((c,) * nreps, axis=-1)
    else:
        c2 = np.stack((c,) * nreps, axis=0)
    tmp = FTop * c2
    if order == "F":
        if nd_out:
            assert tmp.shape == c.shape + (ncoils, nreps)
        else:
            assert tmp.ndim == 2
            assert tmp.shape[-1] == nreps  # ncoils is folded into first dim
    else:
        if nd_out:
            assert tmp.shape == (nreps, ncoils) + c.shape
        else:
            assert tmp.ndim == 2
            assert tmp.shape[0] == nreps  # ncoils is folded into first dim
    """
    test adjoint transform with 2 repetitions
    """
    tmp2 = FTop.H * tmp
    if nd_in:
        if order == "F":
            assert tmp2.shape == c.shape + (nreps,)
        else:
            assert tmp2.shape == (nreps,) + c.shape
    else:
        assert tmp2.ndim == 2


@pytest.mark.parametrize(
    "xp, shift, nd_in, nd_out, order",
    product(all_xp, [True, False], [True, False], [True, False], ["C", "F"]),
)
def test_FFT_composite(xp, shift, nd_in, nd_out, order):
    """Testing composite forward and adjoint operator."""
    rtol = 1e-4
    atol = 1e-4
    c = get_data(xp)
    FTop = MRI_Cartesian(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )

    # create new linear operator for forward followed by inverse transform
    FtF = FTop.H * FTop
    """
    test forward transform
    """
    tmp = FtF * c
    if nd_in:
        assert tmp.shape == c.shape
    else:
        assert tmp.ndim == 1
    assert tmp.real.dtype == c.dtype

    tmp = tmp.reshape(c.shape, order=order)
    xp.testing.assert_allclose(tmp, c, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "xp, shift, nd_in, order",
    product(all_xp, [True, False], [True, False], ["C", "F"]),
)
def test_partial_FFT_allsamples(xp, shift, nd_in, order):
    rtol = 1e-4
    atol = 1e-4
    c = get_data(xp)
    sample_mask = xp.ones(c.shape)
    """ masked FFT without missing samples """
    # TODO: when all samples are present, can test against normal FFT
    nd_out = False
    FTop = MRI_Cartesian(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        sample_mask=sample_mask,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )  # no missing samples

    # create new linear operator for forward followed by inverse transform
    FtF = FTop.H * FTop

    """
    test forward transform
    """
    tmp = FtF * c
    if nd_in:
        assert tmp.shape == c.shape
    else:
        assert tmp.ndim == 1
    assert tmp.real.dtype == c.dtype

    tmp = tmp.reshape(c.shape, order=order)
    xp.testing.assert_allclose(tmp, c, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "xp, shift, nd_in, order",
    product(all_xp, [True, False], [True, False], ["C", "F"]),
)
def test_partial_FFT(xp, shift, nd_in, order):
    """ masked FFT with missing samples """
    # TODO: check accuracy against brute force DFT
    c = get_data(xp)
    rstate = xp.random.RandomState(1234)
    sample_mask = rstate.rand(*(128, 127)) > 0.5

    nd_out = False
    FTop = MRI_Cartesian(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        sample_mask=sample_mask,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )

    # create new linear operator for forward followed by inverse transform
    FtF = FTop.H * FTop

    """
    test round trip transform
    """
    tmp = FtF * c
    if nd_in:
        assert tmp.shape == c.shape
    else:
        assert tmp.ndim == 1
    assert tmp.real.dtype == c.dtype

    tmp = tmp.reshape(c.shape, order=order)


@pytest.mark.parametrize(
    "xp, shift, nd_in, order",
    product(
        all_xp,
        [True, False],
        [True, False],
        ["F"],  # TODO: also implement order = 'C' case?
    ),
)
def test_partial_FFT_with_im_mask(xp, shift, nd_in, order):
    """ masked FFT with missing samples and masked image domain """
    # TODO: check accuracy against brute force DFT
    c = get_data(xp)
    rstate = xp.random.RandomState(1234)
    sample_mask = rstate.rand(*(128, 127)) > 0.5

    x, y = xp.meshgrid(
        xp.arange(-c.shape[0] // 2, c.shape[0] // 2),
        xp.arange(-c.shape[1] // 2, c.shape[1] // 2),
        indexing="ij",
        sparse=True,
    )

    # make a circular mask
    im_mask = xp.sqrt(x * x + y * y) < c.shape[0] // 2

    nd_out = False
    FTop = MRI_Cartesian(
        c.shape,
        order=order,
        im_mask=im_mask,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        sample_mask=sample_mask,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )

    # test forward only
    forw = embed(
        FTop * masker(c, im_mask, order=order), sample_mask, order=order
    )
    if shift:
        expected_forw = sample_mask * fftnc(c * im_mask)
    else:
        expected_forw = sample_mask * fftn(c * im_mask)
    xp.testing.assert_allclose(forw, expected_forw, rtol=1e-7, atol=1e-4)

    # test roundtrip
    roundtrip = FTop.H * (FTop * masker(c, im_mask, order=order))
    if shift:
        expected_roundtrip = masker(
            ifftnc(sample_mask * fftnc(c * im_mask)), im_mask, order=order
        )
    else:
        expected_roundtrip = masker(
            ifftn(sample_mask * fftn(c * im_mask)), im_mask, order=order
        )
    xp.testing.assert_allclose(
        roundtrip, expected_roundtrip, rtol=1e-7, atol=1e-4
    )

    # TODO: grlee77: fix FtF operation on masked data and uncomment case below
    # # create new linear operator for forward followed by inverse transform
    # FtF = FTop.H * FTop
    # roundtrip2 = FtF * masker(c, im_mask, order=order)
    # xp.testing.assert_allclose(
    #     roundtrip2, expected_roundtrip, rtol=1e-7, atol=1e-4
    # )

    # test roundtrip with 2 reps
    c2 = xp.stack([c] * 2, axis=-1)
    roundtrip = FTop.H * (FTop * masker(c2, im_mask, order=order))
    if shift:
        expected_roundtrip = masker(
            ifftnc(
                sample_mask[..., xp.newaxis]
                * fftnc(c2 * im_mask[..., xp.newaxis], axes=(0, 1)),
                axes=(0, 1),
            ),
            im_mask,
            order=order,
        )
    else:
        expected_roundtrip = masker(
            ifftn(
                sample_mask[..., xp.newaxis]
                * fftn(c2 * im_mask[..., xp.newaxis], axes=(0, 1)),
                axes=(0, 1),
            ),
            im_mask,
            order=order,
        )
    xp.testing.assert_allclose(
        roundtrip, expected_roundtrip, rtol=1e-7, atol=1e-4
    )


@pytest.mark.parametrize(
    "xp, shift, nd_in, order",
    product(all_xp, [True, False], [True, False], ["C", "F"]),
)
def test_partial_FFT_coils(xp, shift, nd_in, order):
    """ masked FFT with missing samples and multiple coils """
    # TODO: check accuracy against brute force DFT
    ncoils = 4
    c = get_data(xp)
    if order == "F":
        cmap = 1 / ncoils * xp.ones((c.shape + (ncoils,)), dtype=c.dtype)
    else:
        cmap = 1 / ncoils * xp.ones(((ncoils,) + c.shape), dtype=c.dtype)
    rstate = xp.random.RandomState(1234)
    sample_mask = rstate.rand(*(128, 127)) > 0.5

    nd_out = False
    if order == "C":
        cmap = xp.ascontiguousarray(cmap)
    else:
        cmap = xp.asfortranarray(cmap)
    FTop = MRI_Cartesian(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        sample_mask=sample_mask,
        coil_sensitivities=cmap,
        gpu_force_reinit=False,
        disable_warnings=True,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )

    # create new linear operator for forward followed by inverse transform
    FtF = FTop.H * FTop

    """
    test round-trip transform
    """
    tmp = FtF * c
    if nd_in:
        assert tmp.shape == c.shape
    else:
        assert tmp.ndim == 1

    y = FTop * c

    assert tmp.real.dtype == c.dtype

    # now test with two sets of coil maps (e.g. ESPIRIT Soft-SENSE)
    for M in [2, 3]:
        if order == "F":
            cmap_multi = xp.asfortranarray(xp.stack((cmap,) * M, axis=-1))
            c_multi = xp.asfortranarray(xp.stack((c,) * M, axis=-1))
        else:
            cmap_multi = xp.asfortranarray(xp.stack((cmap,) * M, axis=0))
            c_multi = xp.asfortranarray(xp.stack((c,) * M, axis=0))

        FTop_multi = MRI_Cartesian(
            c.shape,
            order=order,
            use_fft_shifts=shift,
            nd_input=nd_in,
            nd_output=nd_out,
            sample_mask=sample_mask,
            coil_sensitivities=cmap_multi,
            gpu_force_reinit=False,
            disable_warnings=True,
            mask_kspace_on_gpu=(not shift),
            **get_loc(xp),
        )
        y_multi = FTop_multi * c_multi

        # identical data and cmap for each set, so y_multi should just be
        # M * y
        assert xp.max(xp.abs(y_multi - M * y)) < 1e-4

        FtF = FTop_multi.H * FTop_multi
        tmp = FtF * c_multi
        if nd_in:
            assert tmp.shape == c_multi.shape
        else:
            assert tmp.ndim == 1
