"""Utilities for coil intensity non-unformity correction (aka bias correction).
"""

import numpy as np
from mrrt.utils import get_array_module


__all__ = ["compute_bias_field"]


def compute_bias_field(
    recon, affine=None, down=None, basename="xtrue", xp=None
):
    # TODO: call N4BiasFieldCorrection via SimpleITK or the ITK Python wrappers
    #       instead of via shell
    import nibabel as nib
    import subprocess

    xp, on_gpu = get_array_module(recon, xp)
    if affine is None:
        affine = np.eye(4)
    ndim = recon.ndim
    recon_abs = np.abs(recon)
    recon_abs = (32767 * recon_abs / recon_abs.max()).astype(np.uint16)
    if xp != np:
        recon_abs = recon_abs.get()
    recon_nii = nib.Nifti1Image(recon_abs, affine=affine)
    recon_nii.to_filename("{0}.nii".format(basename))

    if down is None:
        if ndim == 3:
            down = 3
            cstr = "[200x200x200x200]"
        else:
            down = 1
            cstr = "[400x200x200]"
    cmd = (
        "N4BiasFieldCorrection"
        " -d {0}"
        " -s {1} -c " + cstr + " -i {2}.nii"
        " -o [{2}_biascor.nii, {2}_biasfield.nii]"
    )
    cmd = cmd.format(ndim, down, basename)
    try:
        subprocess.check_output(cmd, shell=True)
    except RuntimeError:
        print(
            "Subprocess failed: is ANTS N4BiasFieldCorrection "
            "on the system path?"
        )
        raise
    bias = nib.load("{0}_biasfield.nii".format(basename)).get_data()
    np.save("{}_biasfield".format(basename), np.asarray(bias, dtype=np.float32))
    return xp.asarray(bias)
