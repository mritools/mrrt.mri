"""Coil compression, pre-whitening and sensitivy map estimation utilities.

Much of this code is modified from idmrmrd-python-tools
https://github.com/ismrmrd/ismrmrd-python-tools

Much code that was 2D only has been updated to work with 3D (or nD) data.
The user can specify which axis of the array corresponds to the coils.

The function calculate_csm_inati_iter is an adaptation of a coil sensitivity
map estimation function by Souheil Inati that is included in Gadgetron.

"""

from ._coil_maps import *  # noqa
from ._coil_pca import *  # noqa
from ._prewhiten import *  # noqa
from ._bias import *  # noqa
