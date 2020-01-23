import numpy as np
from mrrt.utils import prod

# TODO: remove unused sinc case


class PixelBasis:
    r""" Pixel basis for representing a continuous-to-discrete mapping between
    a continuous image and its sampled (discrete) values.
    """

    def __init__(self, kspace, pixel_basis="dirac", dx=None, fov=None, Nd=None):
        """ Initialize a pixel-basis and its transform.

        Parameters
        ----------
        kspace : array
            2D array of k-space coordinates :math:`[n_{samples}, n_{dim}]`.
        pixel_basis : {'dirac', 'rect'}
            basis type
        dx : array_like, optional
            image pixel spacings
        fov : array_like, optional
            image field of view (i.e. `Nd` * `dx`)
        Nd : array_like, optional
            image matrix size along each dimension

        Attributes
        ----------
        type : str
            basis type
        dx : array or None
            pixel spacing in image domain
        fov : array or None
            image field of view (i.e. `Nd` * `dx`)
        Nd : array or None
            image matrix size along each dimension
        transform : array
            Fourier transform of the basis evaluated at the k-space sample
            locations.

        Notes
        -----
        A continuous function, :math:`f(\vec{x})`, is represented by a
        discrete set of pixel values :math:`x_{i}` in basis,
        :math:`b(\vec{x})`

        .. math::

            f\left(\vec{x}\right)=\sum_{i=1}^{N}\,x_{i}b\left(\vec{x}-x_{i}\right)

        units for `dx`, `fov` must match (e.g. mm).
        `kspace` units (e.g. mm^-1) should be the inverse of those for `dx`,
        `fov`.

        References
        ----------
        .. [1] Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P.  SENSE:
           Sensitivity Encoding for Fast MRI.  Magn. Reson. Med. 1999;
           42:952-962.
        .. [2] Sutton, BP.  Physics Based Iterative Reconstruction for MRI:
           Compensating and Estimating Field Inhomogeneity and T2\* Relaxation.
           Doctoral Dissertation.  University of Michigan. 2003.
        .. [3] Barrett HH, Myers KJ.  "7.1: Objects and Images" In *Foundations
           of Image Science*.  2003. Wiley-Interscience.

        """
        self.type = pixel_basis
        self.dx = dx
        self.fov = fov
        self.Nd = Nd
        if self.dx is not None:
            self.dx = np.atleast_1d(self.dx)
        if self.fov is not None:
            self.fov = np.atleast_1d(self.fov)
        if self.Nd is not None:
            self.Nd = np.atleast_1d(self.Nd)
        else:
            self.Nd = np.round(self.fov / self.dx).astype(int)
        self.kspace = kspace
        if kspace.ndim != 2:
            raise ValueError("kspace must be 2D: [nsamples x ndim]")

        nk, ndim = kspace.shape
        if self.type == "dirac":
            self.transform = None
        elif self.type == "rect":
            if self.dx is None:
                if (self.fov is None) or (self.Nd is None):
                    raise ValueError("must specify dx or both fov and Nd")
                self.dx = self.fov / self.Nd
            if len(self.dx) != ndim:
                if len(self.dx) == 1:
                    self.dx = np.asarray(self.dx.tolist() * ndim)
                else:
                    raise ValueError("len(dx) != ndim")
            self.transform = np.ones((nk,))

            for idx in range(ndim):
                # allow for zero-sized pixels (Dirac impulses)
                if self.dx[idx] > 0:
                    # dx* sinc(dx*kx) term in Eq. 6.15 of Brad Sutton's thesis
                    self.transform = self.transform * (
                        self.dx[idx] * np.sinc(self.dx[idx] * kspace[:, idx])
                    )

            # make the average scaling close to 1
            self.transform /= self.transform.mean()
        elif self.type in ["sinc", "dirac*dx"]:
            # simply provide an appropriate "scale factor" dx to relate Fourier
            # integral and summation
            self.transform = np.ones((nk,)) * prod(
                f / n for f, n in zip(self.fov, self.Nd)
            )

            # make the average scaling close to 1
            self.transform /= self.transform.mean()
        else:
            raise ValueError("unknown PixelBasis type: {}".format(pixel_basis))

    def __str__(self):
        repstr = "Pixel Basis:\ntype={}, ndim={}, dx={}".format(
            self.type, len(self.Nd), self.dx
        )
        return repstr

    def __repr__(self):
        # repstr = ("PixelBasis(kspace, pixel_basis={}, ".format(self.type) +
        #           "dx={}, fov={}, Nd={})".format(self.fov, self.dx, self.Nd))
        repstr = (
            "PixelBasis(kspace, pixel_basis='{}', dx={}, fov={}, Nd={})"
        ).format(self.type, self.dx, self.fov, self.Nd)

        return repstr
