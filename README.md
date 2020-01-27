
mrrt.mri
========

This package contains a range of Python functions related to magnetic resonance
imaging (MRI) reconstruction.

- Operators for Cartesian and Non-Cartesian MRI
- Coil Sensitivity Map estimation
- Coil Compression
- k-space data Prewhitening
- determination of basis functions for off-resonance and/or relaxation correction
- basic analytical MRI phantoms
- partial Fourier recontruction

Installation
------------

pip install mri

**Requires:**

- [NumPy]  (>=1.14)
- [mrrt.nufft]
- [mrrt.operators]
- [mrrt.utils]

**Recommended:**

- [CuPy]  (>=6.1)

**To run the test suite, users will also need:**

- [pytest]

Basic Usage
------------
TODO

Documentation
------------
TODO


[CuPy]: https://cupy.chainer.org
[mrrt.operators]: https://github.com/mritools/mrrt.operators
[mrrt.nufft]: https://github.com/mritools/mrrt.nufft
[mrrt.utils]: https://github.com/mritools/mrrt.utils
[NumPy]: https://numpy.org
[pytest]: https://docs.pytest.org/en/latest/
[SciPy]: https://scipy.org

