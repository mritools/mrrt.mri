# import cupy
from mrrt.mri.operators.bench.bench_mri import bench_mri_3d_16coils_fieldmap

# if hasattr(cupy.fft, "cache") and hasattr(cupy.fft.cache, "enable"):
#     cupy.fft.cache.enable()

bench_mri_3d_16coils_fieldmap(
    plot_timings=True, include_CPU=False, include_GPU=True
)
