"""MRI test objects created from basic shapes with analytical expression.

These are used in testing and benchmarking scripts as well as some demos.

The following shapes are implemented based on analytical expressions of their
Fourier transform:

    Gaussians
    Rectangles
    Cylinders
    Diracs

The class AnalyticalComposite is used to generate a composite phantom built up
of analytical objects of the above type.

The following functions generate specific test phantoms as used in various
testing and benchmarking scripts:

    mri_object_1d
    mri_object_2d
    mri_object_2d_multispectral
    mri_object_3d

Much of the code in this module is ported from equivalent Matlab code by
Jeff Fessler and his students (Michigan Image Reconstruction Toolbox).
"""
from abc import ABC, abstractmethod
import warnings

import numpy as np

from mrrt.utils import rect, jinc


__all__ = [
    "AnalyticalComposite",
    "Cylinders",
    "Diracs",
    "Gaussians",
    "Rectangles",
    "mri_object_1d",
    "mri_object_2d",
    "mri_object_2d_multispectral",
    "mri_object_3d",
]


class Fourier_Analytical(ABC):

    fixup = False

    def __init__(self, ndim):
        self.ndim = ndim

    def _fixup_shape(self, x, d, fixup=None):
        if fixup is None:
            fixup = self.fixup
        if fixup and x.ndim != self.ndim:
            if x.ndim == 1:
                new_shape = [1] * self.ndim
                new_shape[d] = x.size
                x = x.reshape(new_shape)
            else:
                raise ValueError(
                    "Each coordinate array must be 1d or have full" "shape."
                )
        return x

    @abstractmethod
    def image(self, *coords):
        """Image domain representation."""

    @abstractmethod
    def kspace(self, *coords):
        """Fourier domain representation not implemented."""


class Gaussians(Fourier_Analytical):
    def __init__(
        self,
        amplitudes,
        centers,
        widths,
        dtype=np.float64,
        fixup=False,
        offsets=None,
    ):
        centers = np.asarray(centers)
        amplitudes = np.asarray(amplitudes)
        widths = np.asarray(widths)
        if amplitudes.ndim != 1:
            raise ValueError("amplitudes should be a shape (N, ) array")
        if centers.ndim != 2:
            raise ValueError("centers should be a shape (N, ndim) array")
        if widths.ndim != 2:
            raise ValueError("centers should be a shape (N, ndim) array")
        if (centers.shape[0] != amplitudes.shape[0]) or (
            widths.shape[0] != amplitudes.shape[0]
        ):
            raise ValueError("shape mismatch")
        self.n = amplitudes.size
        self.ndim = centers.shape[1]
        self.amplitudes = amplitudes
        self.centers = centers
        self.widths = widths
        self.dtype = dtype
        self.fixup = fixup
        self.offsets = offsets
        if self.offsets is None:
            # set of tuples for each component where the first element of the
            # tumple is the relative amplitude of the component and the second
            # is it's frequency offset.
            self.offsets = [0] * self.n
        self.unique_offsets = np.unique(self.offsets)
        self.n_offsets = self.unique_offsets.size

    def image(self, *coords, squeeze=True):
        # Evaluate values in the image domain.
        if len(coords) != self.ndim:
            raise ValueError(
                "The number of coordinate arrays must equal the number of "
                "dimensions"
            )
        denom = np.sqrt(np.log(256)) / np.sqrt(2 * np.pi)
        widths = self.widths / denom
        if self.n_offsets == 1 and squeeze:
            out = 0.0
            for n in range(self.n):
                g = 1.0
                for d in range(self.ndim):
                    x = self._fixup_shape(coords[d], d)
                    c = self.centers[n, d]
                    w = widths[n, d]
                    g = g * np.exp(-np.pi * ((x - c) / w) ** 2)
                out += self.amplitudes[n] * g
        else:
            all_out = [0] * self.n_offsets
            for m, off in enumerate(self.unique_offsets):
                out = 0
                for n in np.where(self.offsets == off)[0]:
                    g = 1.0
                    for d in range(self.ndim):
                        x = self._fixup_shape(coords[d], d)
                        c = self.centers[n, d]
                        w = widths[n, d]
                        g = g * np.exp(-np.pi * ((x - c) / w) ** 2)
                    out += self.amplitudes[n] * g
                all_out[m] = out
            out = np.stack(all_out, axis=-1)

        return np.asarray(out, dtype=self.dtype)

    def kspace(self, *coords, t=None):
        # Evaluate values in the Fourier domain.
        if len(coords) != self.ndim:
            raise ValueError(
                "The number of coordinate arrays must equal the number of "
                "dimensions"
            )
        denom = np.sqrt(np.log(256)) / np.sqrt(2 * np.pi)
        widths = self.widths / denom
        out = 0.0
        for n in range(self.n):
            g = 1.0
            omega = 2 * np.pi * self.offsets[n]
            for d in range(self.ndim):
                x = self._fixup_shape(coords[d], d)
                c = self.centers[n, d]
                w = widths[n, d]
                g = g * w
                g = g * np.exp(-np.pi * (x * w) ** 2)
                g = g * np.exp(-2j * np.pi * (x * c))
            if omega == 0:
                out += self.amplitudes[n] * g
            else:
                if t is None:
                    raise ValueError(
                        "for non-zero offsets, a time vector "
                        "corresponding to the trajectory must be "
                        "provided"
                    )
                out += self.amplitudes[n] * g * np.exp(1j * omega * t)
        return np.asarray(out, dtype=np.result_type(self.dtype, np.complex64))


class Rectangles(Fourier_Analytical):
    def __init__(
        self, amplitudes, centers, widths, dtype=np.float64, offsets=None
    ):
        centers = np.asarray(centers)
        amplitudes = np.asarray(amplitudes)
        widths = np.asarray(widths)
        if amplitudes.ndim != 1:
            raise ValueError("amplitudes should be a shape (N, ) array")
        if centers.ndim != 2:
            raise ValueError("centers should be a shape (N, ndim) array")
        if widths.ndim != 2:
            raise ValueError("centers should be a shape (N, ndim) array")
        if (centers.shape[0] != amplitudes.shape[0]) or (
            widths.shape[0] != amplitudes.shape[0]
        ):
            raise ValueError("shape mismatch")
        self.n = amplitudes.size
        self.ndim = centers.shape[1]
        self.amplitudes = amplitudes
        self.centers = centers
        self.widths = widths
        self.dtype = dtype
        self.offsets = offsets
        if self.offsets is None:
            # set of tuples for each component where the first element of the
            # tumple is the relative amplitude of the component and the second
            # is it's frequency offset.
            self.offsets = [0] * self.n
        self.unique_offsets = np.unique(self.offsets)
        self.n_offsets = self.unique_offsets.size

    def image(self, *coords, squeeze=True):
        # Evaluate values in the image domain.
        if len(coords) != self.ndim:
            raise ValueError(
                "The number of coordinate arrays must equal the number of "
                "dimensions"
            )
        if self.n_offsets == 1 and squeeze:
            out = 0.0
            for n in range(self.n):
                g = 1.0
                for d in range(self.ndim):
                    x = self._fixup_shape(coords[d], d)
                    c = self.centers[n, d]
                    w = self.widths[n, d]
                    g = g * rect((x - c) / w)
                out += self.amplitudes[n] * g
        else:
            out = [0] * self.n_offsets
            for m, off in enumerate(self.unique_offsets):
                o = 0
                for n in np.where(self.offsets == off)[0]:
                    g = 1.0
                    for d in range(self.ndim):
                        x = self._fixup_shape(coords[d], d)
                        c = self.centers[n, d]
                        w = self.widths[n, d]
                        g = g * rect((x - c) / w)
                    o += self.amplitudes[n] * g
                out[m] = o
            out = np.stack(out, axis=-1)
        return np.asarray(out, dtype=self.dtype)

    def kspace(self, *coords, t=None):
        # Evaluate values in the Fourier domain.
        if len(coords) != self.ndim:
            raise ValueError(
                "The number of coordinate arrays must equal the number of "
                "dimensions"
            )
        out = 0.0
        for n in range(self.n):
            omega = 2 * np.pi * self.offsets[n]
            g = 1.0
            ph = 0.0
            for d in range(self.ndim):
                f = self._fixup_shape(coords[d], d)
                c = self.centers[n, d]
                w = self.widths[n, d]
                sinc_term = w * np.sinc(f * w)
                g = g * sinc_term
                ph = ph + f * c
            g = g * np.exp(-2j * np.pi * ph)
            if omega == 0:
                out += self.amplitudes[n] * g
            else:
                if t is None:
                    raise ValueError(
                        "for non-zero offsets, a time vector "
                        "corresponding to the trajectory must be "
                        "provided"
                    )
                out += self.amplitudes[n] * g * np.exp(1j * omega * t)
        return np.asarray(out, dtype=np.result_type(self.dtype, np.complex64))


class Cylinders(Fourier_Analytical):
    def __init__(
        self,
        amplitudes,
        centers,
        radii,
        heights=None,
        dtype=np.float64,
        offsets=None,
    ):
        centers = np.asarray(centers)
        amplitudes = np.asarray(amplitudes)
        radii = np.asarray(radii)
        if amplitudes.ndim != 1:
            raise ValueError("amplitudes should be a shape (N, ) array")
        if radii.ndim != 1:
            raise ValueError("radii should be a shape (N, ) array")
        if centers.ndim != 2:
            raise ValueError("centers should be a shape (N, ndim) array")
        if (centers.shape[0] != amplitudes.shape[0]) or (
            radii.shape[0] != amplitudes.shape[0]
        ):
            raise ValueError("shape mismatch")
        if heights is not None:
            heights = np.asarray(heights)
            if heights.ndim != 1:
                raise ValueError("heights should be a shape (N, ) array")
            if heights.shape[0] != amplitudes.shape[0]:
                raise ValueError("shape mismatch")
            if centers.shape[1] != 3:
                raise ValueError("heights requires 3d centers")
        self.n = amplitudes.size
        self.ndim = centers.shape[1]
        self.amplitudes = amplitudes
        self.centers = centers
        self.radii = radii
        self.heights = heights
        self.dtype = dtype
        self.offsets = offsets
        if self.offsets is None:
            # set of tuples for each component where the first element of the
            # tumple is the relative amplitude of the component and the second
            # is it's frequency offset.
            self.offsets = [0] * self.n
        self.unique_offsets = np.unique(self.offsets)
        self.n_offsets = self.unique_offsets.size

    def image(self, *coords, squeeze=True):
        # Evaluate values in the image domain.
        if len(coords) != self.ndim:
            raise ValueError(
                "The number of coordinate arrays must equal the number of "
                "dimensions"
            )

        if self.heights is not None:
            ndim_circ = self.ndim - 1
        else:
            ndim_circ = self.ndim

        if self.n_offsets == 1 and squeeze:
            out = 0.0
            for n in range(self.n):
                rr = 0.0
                for d in range(ndim_circ):
                    x = self._fixup_shape(coords[d], d)
                    c = self.centers[n, d]
                    rr = rr + (x - c) ** 2
                circ = rr < self.radii[n] ** 2
                if self.heights is None:
                    out += self.amplitudes[n] * circ
                else:
                    z = self._fixup_shape(coords[-1], self.ndim - 1)
                    zc = self.centers[n, -1]
                    zh = self.heights[n]
                    out += self.amplitudes[n] * circ * rect((z - zc) / zh)
        else:
            out = [0] * self.n_offsets
            for m, off in enumerate(self.unique_offsets):
                o = 0
                for n in np.where(self.offsets == off)[0]:
                    rr = 0.0
                    for d in range(ndim_circ):
                        x = self._fixup_shape(coords[d], d)
                        c = self.centers[n, d]
                        rr = rr + (x - c) ** 2
                    circ = rr < self.radii[n] ** 2
                    if self.heights is None:
                        out += self.amplitudes[n] * circ
                    else:
                        z = self._fixup_shape(coords[-1], self.ndim - 1)
                        zc = self.centers[n, -1]
                        zh = self.heights[n]
                        o += self.amplitudes[n] * circ * rect((z - zc) / zh)
                out[m] = o
            out = np.stack(out, axis=-1)
        return np.asarray(out, dtype=self.dtype)

    def kspace(self, *coords, t=None):
        # Evaluate values in the Fourier domain.
        # Evaluate values in the image domain.
        if len(coords) != self.ndim:
            raise ValueError(
                "The number of coordinate arrays must equal the number of "
                "dimensions"
            )
        if self.ndim == 2:
            # disc
            u, v = coords[0], coords[1]
        if self.ndim == 3:
            # cylinder
            u, v, w = coords[0], coords[1], coords[2]
            w = self._fixup_shape(w, 2)
        u = self._fixup_shape(u, 0)
        v = self._fixup_shape(v, 1)

        out = 0.0
        for n in range(self.n):
            omega = 2 * np.pi * self.offsets[n]
            r = self.radii[n]
            jinc_term = r ** 2 * 4 * jinc(2 * np.sqrt(u ** 2 + v ** 2) * r)
            if self.heights is not None:
                zh = self.heights[n]
                sinc_term = zh * np.sinc(w * zh)
                xc, yc, zc = self.centers[n]
                ph = xc * u + yc * v + zc * w
                o = (
                    self.amplitudes[n]
                    * jinc_term
                    * sinc_term
                    * np.exp(-2j * np.pi * ph)
                )
            else:
                xc, yc = self.centers[n]
                ph = xc * u + yc * v
                o = self.amplitudes[n] * jinc_term * np.exp(-2j * np.pi * ph)
            if omega == 0:
                out = out + o
            else:
                if t is None:
                    raise ValueError(
                        "for non-zero offsets, a time vector "
                        "corresponding to the trajectory must be "
                        "provided"
                    )
                out = out + o * np.exp(1j * omega * t)
        return np.asarray(out, dtype=np.result_type(self.dtype, np.complex64))


class Diracs(Fourier_Analytical):
    def __init__(self, amplitudes, centers, dtype=np.float64):
        centers = np.asarray(centers)
        amplitudes = np.asarray(amplitudes)
        if amplitudes.ndim != 1:
            raise ValueError("amplitudes should be a shape (N, ) array")
        if centers.ndim != 2:
            raise ValueError("centers should be a shape (N, ndim) array")
        if centers.shape[0] != amplitudes.shape[0]:
            raise ValueError("shape mismatch")
        self.n = amplitudes.size
        self.ndim = centers.shape[1]
        self.amplitudes = amplitudes
        self.centers = centers
        self.dtype = dtype

    def image(self, *coords):
        # Evaluate values in the image domain.
        if len(coords) != self.ndim:
            raise ValueError(
                "The number of coordinate arrays must equal the number of "
                "dimensions"
            )
        out = 0.0
        for n in range(self.n):
            g = 1.0
            for d in range(self.ndim):
                x = self._fixup_shape(coords[d], d)
                c = self.centers[n, d]
                g = g * (x == c)
            out += self.amplitudes[n] * g
        out[out != 0] = np.inf
        warnings.warn("Image of dirac is invalid")
        return np.asarray(out, dtype=self.dtype)

    def kspace(self, *coords):
        # Evaluate values in the Fourier domain.
        if len(coords) != self.ndim:
            raise ValueError(
                "The number of coordinate arrays must equal the number of "
                "dimensions"
            )
        out = 0.0
        for n in range(self.n):
            g = 0.0
            for d in range(self.ndim):
                x = self._fixup_shape(coords[d], d)
                c = self.centers[n, d]
                g = g + x * c
            out += self.amplitudes[n] * np.exp(-2j * np.pi * g)
        return np.asarray(out, dtype=np.result_type(self.dtype, np.complex64))


class AnalyticalComposite(object):
    """
    Object that describes image-domain and Fourier-domain spectra of simple
    structures such as rectangles, disks and Gaussian bumps.

    These functions are useful for simple "idealized" MRI simulations
    where the data is modeled as analytical Fourier samples,
    i.e., no field inhomogeneity and no relaxation effects.
    """

    def __init__(self, objects):

        if isinstance(objects, Fourier_Analytical):
            objects = [objects]

        for obj in objects:
            if not isinstance(obj, Fourier_Analytical):
                raise ValueError(
                    "objects should be a list of analytical Fourier objects"
                )
        self.objects = objects

        # bookkeeping stuff for cases where different object sets may have
        # different numbers of spectral offsets
        tmp = np.concatenate([o.unique_offsets for o in self.objects])
        self.unique_offsets = np.unique(tmp)
        # reorder so offset=0 is always the first component
        self.unique_offsets = np.concatenate(
            (
                self.unique_offsets[self.unique_offsets == 0],
                sorted(self.unique_offsets[self.unique_offsets != 0]),
            )
        )
        self.n_offsets_total = self.unique_offsets.size
        self.img_idx = [
            [
                np.where(self.unique_offsets == o)[0][0]
                for o in obj.unique_offsets
            ]
            for obj in self.objects
        ]

    def image(self, *coords):
        for n, object_set in enumerate(self.objects):
            o = object_set.image(*coords, squeeze=False)
            if n == 0:
                out = np.zeros(
                    o.shape[:-1] + (self.n_offsets_total,), dtype=o.dtype
                )
            out[..., self.img_idx[n]] = out[..., self.img_idx[n]] + o.real
        return np.squeeze(out)

    def kspace(self, *coords, t=None):
        out = 0
        for object_set in self.objects:
            out = out + object_set.kspace(*coords, t=t)
        return out


def mri_object_1d(fov, units="mm", dtype=np.float64):
    if len(fov) > 1:
        fov = fov[0]

    if units == "cm":
        ref_fov = 25.6
    else:
        ref_fov = 256.0  # mm

    rectangles = Rectangles(
        amplitudes=[1, 1, 1],
        centers=((0,), (-50,), (0,)),
        widths=((200,), (40,), (50,)),
        dtype=dtype,
    )
    rectangles.centers = rectangles.centers / ref_fov * np.asarray(fov)
    rectangles.widths = rectangles.widths / ref_fov * np.asarray(fov)

    return AnalyticalComposite([rectangles])


def mri_object_2d(fov, units="mm", dtype=np.float64):

    if units == "cm":
        ref_fov = 25.6
    else:
        ref_fov = 256.0  # mm

    rectangles = Rectangles(
        amplitudes=[1, 1, 1, 1],
        centers=((0, 0), (-50, -50), (50, -50), (0, 50)),
        widths=((200, 200), (40, 40), (20, 20), (50, 50)),
        dtype=dtype,
    )
    rectangles.centers = rectangles.centers / ref_fov * np.asarray(fov)
    rectangles.widths = rectangles.widths / ref_fov * np.asarray(fov)

    gaussians = Gaussians(
        amplitudes=[1, 1, 1, 1, 1, 1, 1, 1],
        centers=(
            (-70, 0),
            (-60, 0),
            (-50, 0),
            (-40, 0),
            (-20, 0),
            (0, 0),
            (20, 0),
            (50, 0),
        ),
        widths=((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)),
        dtype=dtype,
    )
    gaussians.centers = gaussians.centers / ref_fov * np.asarray(fov)
    gaussians.widths = gaussians.widths / ref_fov * np.asarray(fov)

    return AnalyticalComposite([rectangles, gaussians])


def mri_object_2d_multispectral(
    fov, spectral_offsets=None, units="mm", dtype=np.float64
):
    """Returns a pair of analytical objects.

    The first object returned is the object magnitude and the second is the
    spectral offsets.

    Examples
    --------
    from mrrt.utils import ImageGeometry
    import matplotlib.pyplot as plt
    fov = 256
    ig = ImageGeometry(shape=(128, 128), fov=fov)
    o1, o2 = mri_object_2d_multispectral(fov, n_fat_peaks=6)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(o1.image(*ig.grid()))
    axes[1].imshow(o2.image(*ig.grid()))
    """

    if units == "cm":
        ref_fov = 25.6
    else:
        ref_fov = 256.0  # mm

    # Generate objects for the magnitude component
    if len(spectral_offsets) > 7:
        raise ValueError("maximum of 7 spectral offsets supported")
    offsets = np.zeros(7)
    offsets[-len(spectral_offsets) :] = spectral_offsets
    rectangles = Rectangles(
        amplitudes=[1, 1, 1, 1, 1, 1, 1],
        offsets=offsets,
        centers=(
            (0, 0),
            (-50, -50),
            (50, -50),
            (0, -50),
            (-50, 50),
            (50, 50),
            (0, 50),
        ),
        widths=(
            (200, 200),
            (30, 30),
            (30, 30),
            (30, 30),
            (30, 30),
            (30, 30),
            (30, 30),
        ),
        dtype=dtype,
    )
    rectangles.centers = rectangles.centers / ref_fov * np.asarray(fov)
    rectangles.widths = rectangles.widths / ref_fov * np.asarray(fov)

    gaussians = Gaussians(
        amplitudes=[1, 1, 1, 1, 1, 1, 1, 1],
        centers=(
            (-70, 0),
            (-60, 0),
            (-50, 0),
            (-40, 0),
            (-20, 0),
            (0, 0),
            (20, 0),
            (50, 0),
        ),
        widths=((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)),
        dtype=dtype,
    )
    gaussians.centers = gaussians.centers / ref_fov * np.asarray(fov)
    gaussians.widths = gaussians.widths / ref_fov * np.asarray(fov)

    return AnalyticalComposite([rectangles, gaussians])


def mri_object_3d(fov, units="mm", dtype=np.float64):

    # cylinders
    cylinders = Cylinders(
        amplitudes=[1],
        centers=[(0, 0, 0)],
        radii=[0.4 * fov[0]],
        heights=[0.9 * fov[2]],
    )

    if units == "cm":
        ref_fov = 25.6
    else:
        ref_fov = 256.0  # mm

    # Rectangles
    rectangles = Rectangles(
        amplitudes=[1, 1, 1],
        centers=((-50, -50, 0), (50, -50, 40), (0, 50, -40)),
        widths=((40, 40, 40), (20, 20, 50), (30, 30, 60)),
        dtype=dtype,
    )
    rectangles.centers = rectangles.centers / ref_fov * np.asarray(fov)
    rectangles.widths = rectangles.widths / ref_fov * np.asarray(fov)

    # Gaussians
    gaussians = Gaussians(
        amplitudes=[1, 1, 1, 1, 1, 1, 1, 1],
        centers=(
            (-70, 0, 0),
            (-60, 0, 0),
            (-50, 0, 0),
            (-40, 0, 0),
            (-20, 0, 0),
            (0, 0, 0),
            (20, 0, 0),
            (50, 0, 0),
        ),
        widths=(
            (1, 1, 1),
            (2, 2, 2),
            (3, 3, 3),
            (4, 4, 4),
            (5, 5, 5),
            (6, 6, 6),
            (7, 7, 7),
            (8, 8, 8),
        ),
        dtype=dtype,
    )
    gaussians.centers = gaussians.centers / ref_fov * np.asarray(fov)
    gaussians.widths = gaussians.widths / ref_fov * np.asarray(fov)
    return AnalyticalComposite([cylinders, rectangles, gaussians])
