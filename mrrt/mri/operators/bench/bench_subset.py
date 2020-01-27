from cupy.cuda import device
import matplotlib.pyplot as plt

from mrrt.mri.operators.bench.bench_mri import (
    bench_mri_2d_nocoils_nofieldmap,
    bench_mri_2d_16coils_nofieldmap,
    bench_mri_2d_16coils_fieldmap,
    bench_mri_2d_16coils_fieldmap_multispectral,
    bench_mri_3d_nocoils_nofieldmap,
    bench_mri_3d_nocoils_fieldmap,
    bench_mri_3d_16coils_nofieldmap,
    bench_mri_3d_16coils_fieldmap,
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
    bench_mri_2d_nocoils_nofieldmap(navg_time=10, **common_kwargs)
    bench_mri_2d_16coils_nofieldmap(navg_time=4, **common_kwargs)
    bench_mri_2d_16coils_fieldmap(navg_time=4, **common_kwargs)
    bench_mri_2d_16coils_fieldmap_multispectral(
        navg_time=2, nspectra=2, **common_kwargs
    )

    # 3d cases
    bench_mri_3d_nocoils_nofieldmap(navg_time=4, **common_kwargs)
    bench_mri_3d_nocoils_fieldmap(navg_time=4, **common_kwargs)
    bench_mri_3d_16coils_nofieldmap(navg_time=2, **common_kwargs)
    bench_mri_3d_16coils_fieldmap(navg_time=2, **common_kwargs)


common_kwargs = dict(
    plot_timings=True,
    include_GPU=True,
    include_CPU=True,
    print_timings=True,
    phasings=["complex"],
    save_dir="/tmp",
)


device.get_cusparse_handle()

# cupy.fft.cache.enable()

# bench_mri_3d_16coils_fieldmap(navg_time=2, **common_kwargs)
# bench_mri_2d_16coils_nofieldmap(navg_time=64, **common_kwargs)
bench_mri_2d_16coils_fieldmap(navg_time=32, **common_kwargs)
# bench_mri_2d_16coils_fieldmap_multispectral(
#     navg_time=64, nspectra=2, **common_kwargs
# )
plt.show()
