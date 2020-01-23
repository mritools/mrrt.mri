"""
example script to use in testing for CUDA memory leaks via:

cuda-memcheck python -m pycuda.debug example_debug.py

To check thread race conditions:
cuda-memcheck --tool racecheck python example_debug.py

To check for uninitialized global memory accesses
cuda-memcheck --tool initcheck python example_debug.py

To check for incorrect synchronization
cuda-memcheck --tool synccheck python example_debug.py


Can also profile using nvidia's nvprof

nvprof --print-gpu-trace python example_debug.py

"""

from mrrt.operators.bench.bench_mri import (
    bench_mri_2d_nocoils_nofieldmap,
    bench_mri_2d_16coils_fieldmap,
)

bench_mri_2d_nocoils_nofieldmap(
    navg_time=1, include_GPU=True, include_CPU=False, print_timings=True
)

bench_mri_2d_16coils_fieldmap(
    navg_time=1, include_GPU=True, include_CPU=False, print_timings=True
)
