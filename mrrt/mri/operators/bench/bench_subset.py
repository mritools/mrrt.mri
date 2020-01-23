import mrrt.mri.operators
from mrrt.mri.operators.bench.bench_MRI import (
    bench_MRI_2d_nocoils_nofieldmap,
    bench_MRI_2d_16coils_nofieldmap,
    bench_MRI_2d_16coils_fieldmap,
    bench_MRI_2d_16coils_fieldmap_multispectral,
    bench_MRI_3d_nocoils_nofieldmap,
    bench_MRI_3d_nocoils_fieldmap,
    bench_MRI_3d_16coils_nofieldmap,
    bench_MRI_3d_16coils_fieldmap,
)


def run_all_bench(
    plot_timings=True,
    include_GPU=True,
    include_CPU=True,
    print_timings=True,
    phasings=["real", "complex"],
    save_dir="/tmp",
):
    common_kwargs = dict(
        plot_timings=plot_timings,
        include_GPU=include_GPU,
        include_CPU=include_CPU,
        print_timings=print_timings,
        phasings=phasings,
        save_dir=save_dir,
    )
    # 2d cases
    bench_MRI_2d_nocoils_nofieldmap(Navg_time=10, **common_kwargs)
    bench_MRI_2d_16coils_nofieldmap(Navg_time=4, **common_kwargs)
    bench_MRI_2d_16coils_fieldmap(Navg_time=4, **common_kwargs)
    bench_MRI_2d_16coils_fieldmap_multispectral(
        Navg_time=2, nspectra=2, **common_kwargs
    )

    # 3d cases
    bench_MRI_3d_nocoils_nofieldmap(Navg_time=4, **common_kwargs)
    bench_MRI_3d_nocoils_fieldmap(Navg_time=4, **common_kwargs)
    bench_MRI_3d_16coils_nofieldmap(Navg_time=2, **common_kwargs)
    bench_MRI_3d_16coils_fieldmap(Navg_time=2, **common_kwargs)


common_kwargs = dict(
    plot_timings=True,
    include_GPU=True,
    include_CPU=False,
    print_timings=True,
    phasings=["complex"],
    save_dir="/tmp",
)

from cupy.cuda import device

device.get_cusparse_handle()
import cupy

cupy.fft.cache.enable()

# bench_MRI_2d_16coils_nofieldmap(Navg_time=64, **common_kwargs)
# bench_MRI_2d_16coils_fieldmap(Navg_time=32, **common_kwargs)
bench_MRI_2d_16coils_fieldmap_multispectral(Navg_time=64, nspectra=2, **common_kwargs)
