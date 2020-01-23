"""Phase unwrapping (1D, 2D, 3D) routines from scikit-image."""
try:
    from skimage.restoration import unwrap_phase  # noqa

    __all__ = ["unwrap_phase"]
except ImportError:
    __all__ = []
