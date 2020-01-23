import numpy as np
import pytest

from mrrt.mri.sim import mri_object_1d, mri_object_2d, mri_object_3d

from mrrt.utils import ImageGeometry, ifftnc


# TODO: add test for mri_object_2d_multispectral


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mri_object_1d(dtype):
    obj = mri_object_1d(fov=(240,), units="mm", dtype=dtype)
    ig = ImageGeometry((240,), distances=(1.0,))

    img = obj.image(*ig.grid())
    ksp = obj.kspace(*ig.fgrid()) / ig.distances[0]

    norm_err = np.linalg.norm(img - ifftnc(ksp)) / np.linalg.norm(img)
    assert norm_err < 0.1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mri_object_2d(dtype):
    obj = mri_object_2d(fov=(240, 240), units="mm", dtype=dtype)
    ig = ImageGeometry((240, 240), distances=(1.0, 1.0))

    img = obj.image(*ig.grid())
    ksp = obj.kspace(*ig.fgrid()) / np.prod(ig.distances)

    norm_err = np.linalg.norm(img - ifftnc(ksp)) / np.linalg.norm(img)
    assert norm_err < 0.1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mri_object_3d(dtype):
    obj = mri_object_3d(fov=(120, 120, 40), units="mm", dtype=dtype)
    ig = ImageGeometry((240, 240, 80), distances=(0.5, 0.5, 0.5))

    img = obj.image(*ig.grid())
    ksp = obj.kspace(*ig.fgrid()) / np.prod(ig.distances)

    norm_err = np.linalg.norm(img - ifftnc(ksp)) / np.linalg.norm(img)
    assert norm_err < 0.2
