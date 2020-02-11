import numpy as np
from mrrt.utils import embed, fftnc, get_array_module, ifftnc


def mri_partial_fourier_nd(
    partial_kspace,
    pf_mask,
    niter=5,
    tw_inner=8,
    tw_outer=3,
    fill_conj=False,
    init=None,
    verbose=False,
    show=False,
    return_all_estimates=False,
    xp=None,
):
    """Partial Fourier reconstruction.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    The implementation is based on a multi-dimensional iterative reconstruction
    technique as described in [1]_.  This is an extension of the 1D iterative
    methods described in [2]_ and [3]_.  The concept of partial Fourier imaging
    was first proposed in [4]_, [5]_.

    References
    ----------
    .. [1] Xu, Y. and Haacke, E. M.  Partial Fourier imaging in
        multi-dimensions: A means to save a full factor of two in time.
        J. Magn. Reson. Imaging, 2001; 14:628–635.
        doi:10.1002/jmri.1228

    .. [2] Haacke, E.; Lindskogj, E. & Lin, W. A fast, iterative,
        partial-Fourier technique capable of local phase recovery.
        Journal of Magnetic Resonance, 1991; 92:126-145.

    .. [3] Liang, Z.-P.; Boada, F.; Constable, R. T.; Haacke, E.; Lauterbur,
        P. & Smith, M. Constrained reconstruction methods in MR imaging.
        Rev Magn Reson Med, 1992; 4: 67-185

    .. [4] Margosian, P.; Schmitt, F. & Purdy, D. Faster MR imaging: imaging
        with half the data Health Care Instrum, 1986; 1:195.

    .. [5] Feinberg, D. A.; Hale, J. D.; Watts, J. C.; Kaufman, L. & Mark, A.
        Halving MR imaging time by conjugation: demonstration at 3.5 kG.
        Radiology, 1986, 161, 527-531

    """
    xp, on_gpu = get_array_module(partial_kspace, xp)
    partial_kspace = xp.asarray(partial_kspace)
    # dtype = partial_kspace.dtype

    pf_mask = xp.asarray(pf_mask)
    if pf_mask.dtype != xp.bool:
        pf_mask = pf_mask.astype(xp.bool)
    im_shape = pf_mask.shape
    ndim = pf_mask.ndim

    if not xp.all(xp.asarray(im_shape) % 2 == 0):
        raise ValueError(
            "This function assumes all k-space dimensions have even length."
        )

    if partial_kspace.size != xp.count_nonzero(pf_mask):
        raise ValueError(
            "partial kspace should have total size equal to the number of "
            "non-zeros in pf_mask"
        )

    kspace_init = embed(partial_kspace, pf_mask)
    img_est = ifftnc(kspace_init)

    lr_kspace = xp.zeros_like(kspace_init)

    nz = xp.where(pf_mask)
    lr_slices = [slice(None)] * ndim
    pf_slices = [slice(None)] * ndim
    win2_slices = [slice(None)] * ndim
    lr_shape = [0] * ndim
    # pf_shape = [0, ]*ndim
    win2_shape = [0] * ndim
    for d in range(ndim):
        nz_min = xp.min(nz[d])
        nz_max = xp.max(nz[d])
        if hasattr(nz_min, "get"):
            # 0-dim GPU array to scalar
            nz_min, nz_max = nz_min.get(), nz_max.get()
        i_mid = im_shape[d] // 2
        if nz_min == 0:
            i_end = nz_max
            width = i_end - i_mid
        else:
            i_start = nz_min
            width = i_mid - i_start
        lr_slices[d] = slice(i_mid - width, i_mid + width + 1)
        lr_shape[d] = 2 * width + 1

        # pf_slices[d] = slice(nz_min, nz_max + 1)
        pf_shape = nz_max - nz_min + 1
        pf_slices[d] = slice(nz_min, nz_max + 1)
        win2_shape[d] = pf_shape + tw_outer
        if nz_min == 0:
            # pf_slices[d] = slice(nz_min, nz_max + 1 + tw_outer)
            win2_slices[d] = slice(tw_outer, tw_outer + pf_shape)
        else:
            # pf_slices[d] = slice(nz_min - tw_outer, nz_max + 1)
            win2_slices[d] = slice(pf_shape)

    lr_slices = tuple(lr_slices)
    win2_slices = tuple(win2_slices)
    pf_slices = tuple(pf_slices)
    # lr_mask = xp.zeros(pf_mask, dtype=xp.zeros)
    lr_win = hanning_apodization_window(lr_shape, tw_inner, xp)
    lr_kspace[lr_slices] = kspace_init[lr_slices] * lr_win

    img_lr = ifftnc(lr_kspace)
    phi = xp.angle(img_lr)

    pf_win = hanning_apodization_window(win2_shape, tw_outer, xp)[win2_slices]

    lr_mask = xp.zeros(pf_mask.shape, dtype=xp.float32)
    lr_mask[lr_slices] = lr_win

    win2_mask = xp.zeros(pf_mask.shape, dtype=xp.float32)
    win2_mask[pf_slices] = pf_win

    if show and ndim == 2:
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(2, 2)
        axes = axes.ravel()
        axes[0].imshow(pf_mask)
        axes[0].set_title("PF mask")
        axes[1].imshow(lr_mask)
        axes[1].set_title("LR Filter")
        axes[2].imshow(win2_mask)
        axes[2].set_title("Filter")
        axes[3].imshow(xp.abs(img_est).T)
        axes[3].set_title("Initial Estimate")
        for ax in axes:
            ax.set_xticklabels("")
            ax.set_yticklabels("")

    if verbose:
        norm0 = xp.linalg.norm(img_est)
        max0 = xp.max(xp.abs(img_est))
    if return_all_estimates:
        all_img_est = [img_est]

    # POCS iterations
    for i in range(niter):
        # step 5
        rho1 = xp.abs(img_est) * xp.exp(1j * phi)

        if verbose:
            change2 = xp.linalg.norm(rho1 - img_est) / norm0
            change1 = xp.max(xp.abs(rho1 - img_est)) / max0
            print("change = {}%% {}%%".format(100 * change2, 100 * change1))

        # step 6
        s1 = fftnc(rho1)

        # steps 7 & 8
        full_kspace = win2_mask * kspace_init + (1 - win2_mask) * s1

        # step 9
        img_est = ifftnc(full_kspace)
        if return_all_estimates:
            all_img_est.append(img_est)

    if return_all_estimates:
        return all_img_est
    return img_est


def hanning_apodization_window(shape, transition_length, xp=np):
    # n-dimensional equivalent of ir_mri_pf_gen_window
    shape = tuple(shape)
    ndim = len(shape)

    if xp.isscalar(transition_length):
        transition_lengths = (transition_length,) * ndim
    else:
        transition_lengths = tuple(transition_length)

    if len(transition_lengths) != ndim:
        raise ValueError(
            "shape and transition_lengths must have the same length"
        )

    awin = 1.0
    for i, tl in enumerate(transition_lengths):
        # (tl + 1) because end points are zero
        hann_len = (tl + 1) * 2 + 1  # always odd so central value will be 1
        hann_1d = xp.hanning(hann_len)[1:-1]  # crop first and last zero
        if hann_1d.max() != 1:
            hann_1d /= hann_1d.max()
        hann_1d_stretch = xp.ones((shape[i],))
        hann_1d_stretch[:tl] = hann_1d[:tl]
        hann_1d_stretch[-tl:] = hann_1d[-tl:]
        s = [1] * ndim
        s[i] = hann_1d_stretch.size
        awin = awin * xp.reshape(hann_1d_stretch, s)  # broadcasting
    awin /= awin.max()
    return awin
