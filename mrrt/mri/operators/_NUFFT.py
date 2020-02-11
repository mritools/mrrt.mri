import warnings
import numpy as np

from mrrt.nufft import NufftBase
from mrrt.operators import LinearOperatorMulti
from mrrt.utils import fftn, ifftn, next_fast_len
from mrrt.utils import complexify, embed, prod, profile


def compute_Q(G, wi=None, Nd_os=2, Kd_os=1.35, J=5, **extra_nufft_kwargs):
    """Compute Q such that IFFT(Q*FFT(x)) = (G.H * G * x).

    Notes
    -----
    requires that G.Kd ~= 2*G.Nd for good accuracy.
    can get away with Kd_os < substantially less than 2

    References
    ----------
    ..[1] Wajer FTAW, Pruessmann KP. Major Speedup of Reconstruction for
    Sensitivity Encoding with Arbitrary Trajectories.
    Proc. Intl. Soc. Mag. Reson. Med. 9 (2001), p.767.

    ..[2] Eggers H, Boernert P, Boesiger P.  Comparison of Gridding- and
    Convolution-Based Iterative Reconstruction Algorithms For
    Sensitivity-Encoded Non-Cartesian Acquisitions.
    Proc. Intl. Soc. Mag. Reson. Med. 10 (2002)

    ..[3] Liu C, Moseley ME, Bammer R.  Fast SENSE Reconstruction Using Linear
    System Transfer Function.
    Proc. Intl. Soc. Mag. Reson. Med. 13 (2005), p.689.
    """
    from mrrt.mri.operators import MRI_Operator, NUFFT_Operator

    if isinstance(G, NUFFT_Operator):
        Gnufft_op = G
    elif isinstance(G, MRI_Operator):
        Gnufft_op = G.Gnufft
    else:
        raise ValueError("G must be an NUFFT_Operator or MRI_Operator")

    # need reasonably accurate gridding onto a 2x oversampled grid
    Nd = (Nd_os * Gnufft_op.Nd).astype(np.intp)
    Kd = next_fast_len((Kd_os * Nd).astype(np.intp))
    if any(k < 2 * n for (k, n) in zip(G.Kd, G.Nd)):
        warnings.warn(
            "Q operator unlikely to be accurate.  Recommend using G "
            "with a grid oversampling factor of 2"
        )
    if Nd_os != 2:
        warnings.warn("recommend keeping Nd_os=2")

    G2 = NUFFT_Operator(
        omega=Gnufft_op.om,
        Nd=Nd,
        Kd=Kd,
        Jd=(J,) * len(Nd),
        Ld=Gnufft_op.Ld,
        n_shift=Nd / 2,
        mode=Gnufft_op.mode,
        phasing="real",  # ONLY WORKS IF THIS IS REAL!
        **extra_nufft_kwargs,
    )

    if wi is None:
        wi = np.ones(Gnufft_op.om.shape[0], dtype=Gnufft_op._cplx_dtype)

    psft = G2.H * wi
    # TODO: allow DiagonalOperator too for weights
    psft = np.fft.fftshift(psft.reshape(G2.Nd, order=G2.order))
    return fftn(psft)


def compute_Q_v2(G, copy_X=True):
    """Alternative version of compute_Q.

    experimental:  not recommended over compute_Q()
    """
    from mrrt.nufft._nufft import nufft_adj

    ones = np.ones(G.kspace.shape[0], G.Gnufft._cplx_dtype)
    sf = np.sqrt(prod(G.Gnufft.Kd))
    return sf * fftn(nufft_adj(G.Gnufft, ones, copy_X=True, return_psf=True))


class NUFFT_Operator(NufftBase, LinearOperatorMulti):

    """ Linear Operator wrapper around NufftBase.

    Parameters
    ----------
    Nd : array_like
        image domain dimensions
    omega : array_like
        k-space coordinates normalized to the range [-pi, pi]
        e.g. if kspace is in mm^-1 and dx in mm:
        ``omega[:, d] = 2 * np.pi * kspace[:, d] * dx[d]``
    mask : array_like, optional
        logical support mask over the image
    kwargs : dict, optional
        any additional keyword arguments to pass onto `NufftBase`

    Attributes
    ----------
    mask
    dim

    """

    @profile
    def __init__(
        self,
        Nd,
        omega,
        mask=None,
        squeeze_reps=True,
        loc_in="cpu",
        loc_out="cpu",
        order="F",
        **kwargs,
    ):

        if loc_in == "cpu":
            xp = np
        else:
            import cupy

            xp = cupy

        # masked boolean if mask is True everywhere or no mask is provided
        if mask is not None:
            mask = xp.asarray(mask, dtype=bool)

        self.masked = (
            (mask is not None)
            and (not isinstance(mask, tuple))
            and (mask.size != xp.count_nonzero(mask))
        )
        if isinstance(mask, tuple):
            self.__mask = None
        else:
            self.__mask = mask
        self.__Nd = Nd

        nargout = omega.shape[0]
        if self.masked:
            nargin = xp.count_nonzero(mask)
        else:
            nargin = prod(self.__Nd)

        if order != "F":
            raise ValueError("NUFFT_Operator only supports order='F'.")

        # initialize NUFFT
        NufftBase.__init__(self, Nd=Nd, omega=omega, order=order, **kwargs)

        # set appropriate operations as initialized by NufftBase
        self.__matvec = self.fft
        self.__matvec_transp = self.adj
        self.__matvec_adj = self.adj

        # construct the linear operator
        LinearOperatorMulti.__init__(
            self,
            nargin,
            nargout,
            symmetric=False,
            hermetian=False,
            matvec=self.__matvec,
            matvec_adj=self.__matvec_adj,
            matvec_transp=self.__matvec_adj,
            nd_input=False,  # True,
            nd_output=False,
            loc_in=loc_in,
            loc_out=loc_out,
            shape_in=self.Nd,
            shape_out=(nargout, 1),
            order=order,
            matvec_allows_repetitions=True,
            # MRI_Operator code assumes squeeze_reps_* are False
            squeeze_reps=squeeze_reps,
            mask_in=self.mask,
            mask_out=None,
            dtype=self._cplx_dtype,
        )

    @property
    def mask(self, create=False):
        "The image domain mask corresponding to the operator."
        if self.__mask is None and create:
            return self.xp.ones(self.Nd, dtype=np.bool)
        else:
            return self.__mask

    @mask.setter
    def mask(self, mask):
        "The image domain mask corresponding to the operator."
        if mask is None:
            self.masked = False
        elif self.__mask is not None and mask.shape != self.__mask.shape:
            raise ValueError("Shape of mask inconsistent with existing object")
        elif mask.sum() == mask.size:
            # True everywhere, so don't need the mask
            self.masked = False
            mask = None
        else:
            self.masked = True
        self.__mask = mask

    @property
    def dim(self):
        "The shape corresponding to the operator."
        return self.shape

    def prep_toeplitz(self, weights=None, Kd_os=1.35, J=5):
        if weights is not None:
            weights = complexify(weights)
        self.Q = compute_Q(self, weights=weights, Kd_os=Kd_os, J=J)

    @profile
    def norm(self, x):
        # if not hasattr(self, 'Q') or self.Q is None:
        #     warnings.warn("Toeplitz Q did not exist, creating it...")
        #     self.prep_toeplitz()

        x = complexify(x)
        if self.masked:
            x = embed(x, self.mask, order=self.order)
        try:
            if x.size % self.nargin != 0:
                raise ValueError("wrong size input")
            Nrepetitions = x.size // self.nargin
        except IndexError:
            Nrepetitions = 1

        if hasattr(self, "Q"):
            slices = [slice(None)] * x.ndim
            for d in range(len(self.Nd)):
                slices[d] = slice(self.Nd[d])
            if Nrepetitions == 1:
                y = fftn(x, s=self.Q.shape)
                y *= self.Q
                y = ifftn(y)[slices]
            else:
                if self.order == "C":
                    x_shape = (Nrepetitions,) + tuple(self.Nd)
                else:
                    x_shape = tuple(self.Nd) + (Nrepetitions,)
                x = x.reshape(x_shape, order=self.order)
                fft_axes = tuple(np.arange(len(self.Nd)))
                y = fftn(x, s=self.Q.shape, axes=fft_axes)
                y *= self.Q[..., np.newaxis]  # add an axis for repetitions
                y = ifftn(y, axes=fft_axes)[slices]
        else:
            y = self.H * (self * x)
        return y
