""" Basic benchmarking of some Non-Cartesian MRI reconstruction cases. """
from __future__ import print_function, division

import numpy as np

from matplotlib import pyplot as plt

from mrrt.mri.operators.tests.test_mri_noncartesian import _test_mri_multi


def _split_timings(bench_tuple, print_timings=False):
    t_adj = [(k, v["MRI: adjoint"]) for k, v in bench_tuple.items()]
    adjoint_times = sorted(t_adj, key=lambda x: x[1], reverse=True)

    t_for = [(k, v["MRI: forward"]) for k, v in bench_tuple.items()]
    forward_times = sorted(t_for, key=lambda x: x[1], reverse=True)

    t_create = [(k, v["MRI: object creation"]) for k, v in bench_tuple.items()]
    creation_times = sorted(t_create, key=lambda x: x[1], reverse=True)

    t_norm = [(k, v["MRI: norm"]) for k, v in bench_tuple.items()]
    norm_times = sorted(t_norm, key=lambda x: x[1], reverse=True)

    if print_timings:
        print("creation times")
        for k, v in creation_times[::-1]:
            print("\t{}: {}".format(k, v))
        print("forward times")
        for k, v in forward_times[::-1]:
            print("\t{}: {}".format(k, v))
        print("adjoint times")
        for k, v in adjoint_times[::-1]:
            print("\t{}: {}".format(k, v))
        print("norm times")
        for k, v in norm_times[::-1]:
            print("\t{}: {}".format(k, v))
    return creation_times, forward_times, adjoint_times, norm_times


def _autolabel(ax, rects, fmt="%0.3g"):
    """
    Attach a text label to the right of each bar displaying its width
    """
    # use mean_width to determine text spacing so that it will be the same
    # for all bars
    mean_width = np.mean([rect.get_width() for rect in rects])
    max_width = 0
    for rect in rects:
        width = rect.get_width()
        t = ax.text(
            width + 0.02 * mean_width,
            rect.get_y() + rect.get_height() / 2.0,
            fmt % width,
            va="center",
            ha="left",
        )
        if width > max_width:
            t_max = t
            max_width = width

    # determine size of the text as in:
    #   http://stackoverflow.com/questions/5320205/matplotlib-text-dimensions
    bb = t_max.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    bb_ax = ax.get_window_extent()
    stretch_factor = (bb_ax.width + bb.width) / bb_ax.width
    # stretch xlim a bit to accomodate the labels
    ax.set_xlim((ax.get_xlim()[0], stretch_factor * ax.get_xlim()[1]))


def _plot_timings(
    labels_times_tuple, title=None, xlabel=None, xlim=None, ax=None, fmt="%0.3g"
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[12, 6])
    nlabel = len(labels_times_tuple)
    labels = [a[0] for a in labels_times_tuple]
    labels = [l.replace("single", "s") for l in labels]
    labels = [l.replace("double", "d") for l in labels]
    labels = [l.replace("complex", "C") for l in labels]
    labels = [l.replace("real", "R") for l in labels]
    rects = ax.bar(
        np.zeros(nlabel),
        height=0.8,
        bottom=np.arange(nlabel),
        width=[a[1] for a in labels_times_tuple],
        tick_label=labels,
        orientation="horizontal",
        align="center",
    )
    _autolabel(ax, rects, fmt=fmt)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    # ax.axis('tight')
    if xlim is not None:
        ax.set_xlim(xlim)
    #        mean_width = np.mean([r.get_width() for r in rects])
    #        for rect in rects:
    #            width = rect.get_width()
    #            if width < xlim[-1]:
    #                t = ax.text(xlim[-1] + 0.02*mean_width,
    #                            rect.get_y() + rect.get_height()/2.,
    #                            '%0.3g' % width,
    #                            va='center', ha='left')

    # center the text of any multi-line labels
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), multialignment="center")
    return ax


if False:
    # BART CPU without Toeplitz 2873 ms
    # BART CPU with Toeplitz 2154 ms
    # BART GPU with Toeplitz 2027 ms
    # BART GPU without Toeplitz 2027 msG
    # ('BART (CPU)', 2154),

    from matplotlib import rcParams

    rcParams["font.size"] = 14

    labels_times_tuple = (
        ("Michigan\nIRT", 5861),
        # fftn/ifftn with FFTW_MEAURE planner effort
        ("PythonCPU\n(16 cores)", 2670),
        ("BART (GPU)", 2027),
        ("gpuNUFFT", 353),
        ("PythonGPU\n(No Precomp)", 58.5),
        ("PythonGPU\n(Full_Precomp)", 56.2),
    )
    fig1, ax1 = plt.subplots(1, 1)
    _plot_timings(
        labels_times_tuple[::-1],
        title="96x96x96 with 16 coils",
        xlabel="$A^{H}A$ execution time (ms)",
        ax=ax1,
        fmt="%d",
    )
    plt.tight_layout()

    labels_times_tuple = (
        ("Single Coil", 122),
        ("Single Coil\n(no Precomp)", 138),
        ("16 Coils", 588),
        ("16 Coils\n(no Precomp)", 727),
        ("16 Coils +\nFieldmap\n(6 time segments)", 3090),
        ("16 Coils +\nFieldmap\n(6 time segments)\n(No Precomp)", 4080),
    )
    fig2, ax2 = plt.subplots(1, 1)
    _plot_timings(
        labels_times_tuple[::-1],
        title="PythonGPU\n(256x256x256 voxels)",
        xlabel="$A^{H}A$ execution time (ms)",
        ax=ax2,
        fmt="%d",
    )
    plt.tight_layout()

    labels_times_tuple = (
        ("SimpleITK\nCPU\n(16 cores))", 706),
        ("PythonGPU", 57.7),
        ("PythonGPU\n(warp field\npreloaded)", 37.8),
    )
    fig3, ax3 = plt.subplots(1, 1)
    _plot_timings(
        labels_times_tuple[::-1],
        title=("Warp with cubic interpolation" "\n(200x148x96 complex array)"),
        xlabel="execution time (ms)",
        ax=ax3,
        fmt="%d",
    )
    plt.tight_layout()

    out_dir = "/home/lee8rx/Dropbox/Grants/SSE/LaTex_NIH/Figures/"
    from os.path import join as pjoin

    fig1.savefig(pjoin(out_dir, "comparison2.png"))
    fig1.savefig(pjoin(out_dir, "comparison2.eps"))
    fig1.savefig(pjoin(out_dir, "comparison2.pdf"))
    fig2.savefig(pjoin(out_dir, "gpu_cases.png"))
    fig2.savefig(pjoin(out_dir, "gpu_cases.eps"))
    fig2.savefig(pjoin(out_dir, "gpu_cases.pdf"))
    fig3.savefig(pjoin(out_dir, "warp.png"))
    fig3.savefig(pjoin(out_dir, "warp.eps"))
    fig3.savefig(pjoin(out_dir, "warp.pdf"))
    # 96x96x96x16coils with 400 shots instead of 150
    # PythonGPU: 90.1 ms
    # BART CPU with Toeplitz: 2205 ms
    # BART GPU with Toeplitz: 2223 ms

    # for data of size 128x128x128
    ("ShearLab3D (dec)", 14.626),
    # ('ShearLab3D (thresh)', 5.264),
    ("ShearLab3D (rec)", 16.303),
    ("UDCT (dec)", 3.37),
    ("UDCT (rec)", 1.71),
    ("DAST (Matlab) (dec)", 1.298),  # (double precision)
    # ('DAST (Matlab) (thresh)', 8.145),
    ("DAST (Matlab) (rec)", 1.577),
    ("DAST (Python) (dec)", 0.845),  # (single precision)
    # ('DAST (Python) (dec)', 0.786),  # w/ batchFFT
    # ('DAST (Python) (thresh)', 1.668),
    ("DAST (Python) (rec)", 0.865),
    ("Surfacelet (dec, R=6.4, mex)", 1.538)
    ("Surfacelet (dec, R=6.4, pure Matlab)", 2.137)
    ("Surfacelet (rec, R=6.4, mex)", 1.334)
    ("Surfacelet (rec, R=6.4, pure Matlab)", 2.282)
    # ('DAST (Python) (rec)', 0.770),   # w/ batchFFT


def _plot_all_timings(
    creation_times,
    forward_times,
    adjoint_times,
    norm_times=None,
    title_substring="",
    uniform_xscale=False,
    plot_creation_times=True,
):
    nsub = 2
    if plot_creation_times:
        nsub += 1
    if norm_times is not None:
        nsub += 1
    if nsub == 4:
        fig, axes = plt.subplots(2, 2, figsize=[16, 16])
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(nsub, 1, figsize=[16, 12])
    max_time_create = np.max([t[1] for t in creation_times])
    max_time_forward = np.max([t[1] for t in forward_times])
    max_time_adjoint = np.max([t[1] for t in adjoint_times])
    max_time = np.maximum(max_time_forward, max_time_adjoint)
    if norm_times is not None:
        max_time_norm = np.max([t[1] for t in norm_times])
        max_time = np.maximum(max_time, max_time_norm)
    if uniform_xscale:
        max_time = np.maximum(max_time, max_time_create)
        xlim1 = [0, max_time]
        xlim = xlim1
    else:
        xlim1 = None
        xlim = None  # [0, max_time]

    print([0, max_time])
    n = 0
    if plot_creation_times:
        _plot_timings(
            creation_times,
            title="creation: " + title_substring,
            xlabel="time (s)",
            xlim=xlim1,
            ax=axes[n],
        )
        n += 1
    _plot_timings(
        forward_times,
        title="forward: " + title_substring,
        xlabel="time (s)",
        xlim=xlim,
        ax=axes[n],
    )
    n += 1
    _plot_timings(
        adjoint_times,
        title="adjoint: " + title_substring,
        xlabel="time (s)",
        xlim=xlim,
        ax=axes[n],
    )
    n += 1
    if norm_times is not None:
        _plot_timings(
            norm_times,
            title="norm: " + title_substring,
            xlabel="time (s)",
            xlim=xlim,
            ax=axes[n],
        )
        n += 1
    if uniform_xscale and plot_creation_times:
        axes[0].set_xticks([])
        axes[1].set_xticks([])
        axes[0].set_xlabel("")
        axes[1].set_xlabel("")
    if not plot_creation_times:
        axes[0].set_xticks([])
        axes[0].set_xlabel("")

    return axes


def _save_bench(save_dir, fname_base, timing_tuples, axes=None):
    import pickle
    from os.path import join as pjoin

    if axes is not None:
        axes[0].get_figure().savefig(pjoin(save_dir, fname_base + ".png"))
    with open(pjoin(save_dir, fname_base + ".p"), "wb") as f:
        pickle.dump(timing_tuples, f)


def _old_bar_graph(bench_tuple):
    t = bench_tuple
    fields = ["MRI: adjoint", "MRI: forward", "MRI: object creation"]
    for field in fields:
        sorted_keys = sorted(t.keys())
        plt.figure()
        y = [t[key][field] for key in sorted_keys]
        x = np.arange(len(y))
        width = 0.8
        plt.bar(
            x,
            y,
            tick_label=[s.replace(",", "\n") for s in sorted_keys],
            width=width,
        )
        plt.xticks(plt.xticks()[0] + width / 2)
        plt.title(field, fontdict=dict(fontsize=24))


# navg_time=10; plot_timings=True; include_GPU=True; include_CPU=True; print_timings=True; phasings=['real', 'complex']


def bench_mri_2d_nocoils_nofieldmap(
    navg_time=(4, 16),
    plot_timings=False,
    include_GPU=False,
    include_CPU=True,
    print_timings=True,
    phasings=["real", "complex"],
    save_dir=None,
):

    recon_cases = []
    if include_GPU:
        recon_cases += ["GPU,Sp", "GPU,Tab"]
        precisions = ["single"]
    else:
        precisions = ["single", "double"]
    if include_CPU:
        recon_cases += ["CPU,Tab", "CPU,Sp"]

    t = _test_mri_multi(
        ndim=2,
        N0=64,
        grid_os_factor=1.5,
        J0=4,
        Ld=4096,
        n_coils=1,
        fieldmap_segments=None,
        precisions=precisions,
        phasings=phasings,
        recon_cases=recon_cases,
        compare_to_exact=False,
        navg_time=navg_time,
    )

    t_create, t_forward, t_adj, t_norm = _split_timings(t, print_timings)
    if plot_timings:
        axes = _plot_all_timings(
            t_create,
            t_forward,
            t_adj,
            t_norm,
            title_substring="2D (no coils, nofieldmap)",
        )
    else:
        axes = None

    if save_dir is not None:
        fname_base = "bench_2d_001coil_001fmapseg"
        _save_bench(
            save_dir, fname_base, [t_create, t_forward, t_adj, t_norm], axes
        )


def bench_mri_2d_16coils_nofieldmap(
    navg_time=(4, 16),
    plot_timings=False,
    include_GPU=False,
    include_CPU=True,
    print_timings=True,
    phasings=["real", "complex"],
    save_dir=None,
):

    recon_cases = []
    if include_GPU:
        recon_cases += ["GPU,Sp", "GPU,Tab"]
        precisions = ["single"]
    else:
        precisions = ["single", "double"]
    if include_CPU:
        recon_cases += ["CPU,Tab", "CPU,Sp"]

    t = _test_mri_multi(
        ndim=2,
        N0=128,
        grid_os_factor=1.5,
        J0=4,
        Ld=4096,
        n_coils=16,
        fieldmap_segments=None,
        precisions=precisions,
        phasings=phasings,
        recon_cases=recon_cases,
        compare_to_exact=False,
        navg_time=navg_time,
    )

    t_create, t_forward, t_adj, t_norm = _split_timings(t, print_timings)
    if plot_timings:
        axes = _plot_all_timings(
            t_create,
            t_forward,
            t_adj,
            t_norm,
            title_substring="2D (16 coils, nofieldmap)",
        )
    else:
        axes = None

    if save_dir is not None:
        fname_base = "bench_2d_016coil_001fmapseg"
        _save_bench(
            save_dir, fname_base, [t_create, t_forward, t_adj, t_norm], axes
        )


def bench_mri_2d_16coils_fieldmap(
    navg_time=(2, 8),
    plot_timings=False,
    include_GPU=False,
    include_CPU=True,
    print_timings=True,
    phasings=["real", "complex"],
    save_dir=None,
):

    recon_cases = []
    if include_GPU:
        recon_cases += ["GPU,Sp", "GPU,Tab"]
        precisions = ["single"]
    else:
        precisions = ["single", "double"]
    if include_CPU:
        recon_cases += ["CPU,Tab", "CPU,Sp"]

    t = _test_mri_multi(
        ndim=2,
        N0=64,
        grid_os_factor=1.5,
        J0=4,
        Ld=4096,
        n_coils=16,
        fieldmap_segments=6,
        precisions=precisions,
        phasings=phasings,
        recon_cases=recon_cases,
        compare_to_exact=False,
        navg_time=4,
    )

    t_create, t_forward, t_adj, t_norm = _split_timings(t, print_timings)
    if plot_timings:
        axes = _plot_all_timings(
            t_create,
            t_forward,
            t_adj,
            t_norm,
            title_substring="2D (16 coils, 6 fieldmap seg)",
        )
    else:
        axes = None

    if save_dir is not None:
        fname_base = "bench_2d_016coil_006fmapseg"
        _save_bench(
            save_dir, fname_base, [t_create, t_forward, t_adj, t_norm], axes
        )


def bench_mri_2d_16coils_fieldmap_multispectral(
    navg_time=(1, 4),
    plot_timings=False,
    include_GPU=False,
    include_CPU=True,
    print_timings=True,
    phasings=["real", "complex"],
    save_dir=None,
    nspectra=2,
):

    recon_cases = []
    if include_GPU:
        recon_cases += ["GPU,Sp", "GPU,Tab"]
        precisions = ["single"]
    else:
        precisions = ["single", "double"]
    if include_CPU:
        recon_cases += ["CPU,Tab", "CPU,Sp"]
    # spectral_offsets = [0, 434.316]
    spectral_offsets = np.asarray(
        [0, 434.316, 332.124, 485.412, -76.644, 49.8186, 247.8156]
    )
    spectral_offsets = spectral_offsets[:nspectra]

    t = _test_mri_multi(
        ndim=2,
        N0=64,
        grid_os_factor=1.5,
        J0=4,
        Ld=4096,
        n_coils=16,
        fieldmap_segments=6,
        precisions=precisions,
        phasings=phasings,
        recon_cases=recon_cases,
        compare_to_exact=False,
        navg_time=4,
        spectral_offsets=spectral_offsets,
    )

    t_create, t_forward, t_adj, t_norm = _split_timings(t, print_timings)
    if plot_timings:
        title_string = "2D (16 coils, 6 fieldmap seg, %d spectral)" % nspectra
        axes = _plot_all_timings(
            t_create, t_forward, t_adj, t_norm, title_substring=title_string
        )
    else:
        axes = None

    if save_dir is not None:
        fname_base = "bench_2d_016coil_006fmapseg_%dspectra" % nspectra
        _save_bench(
            save_dir, fname_base, [t_create, t_forward, t_adj, t_norm], axes
        )


def bench_mri_3d_nocoils_nofieldmap(
    navg_time=(2, 8),
    plot_timings=False,
    include_GPU=False,
    include_CPU=True,
    print_timings=True,
    phasings=["real", "complex"],
    save_dir=None,
):

    recon_cases = []
    if include_GPU:
        recon_cases += ["GPU,Sp", "GPU,Tab"]
        precisions = ["single"]
    else:
        precisions = ["single", "double"]
    if include_CPU:
        recon_cases += ["CPU,Tab", "CPU,Sp"]

    t = _test_mri_multi(
        ndim=3,
        N0=64,
        grid_os_factor=1.5,
        J0=4,
        Ld=4096,
        n_coils=1,
        fieldmap_segments=None,
        precisions=precisions,
        phasings=phasings,
        recon_cases=recon_cases,
        compare_to_exact=False,
        navg_time=navg_time,
    )

    t_create, t_forward, t_adj, t_norm = _split_timings(t, print_timings)
    if plot_timings:
        axes = _plot_all_timings(
            t_create,
            t_forward,
            t_adj,
            t_norm,
            title_substring="3D (no coils, nofieldmap)",
        )
    else:
        axes = None

    if save_dir is not None:
        fname_base = "bench_3d_001coil_001fmapseg"
        _save_bench(
            save_dir, fname_base, [t_create, t_forward, t_adj, t_norm], axes
        )


def bench_mri_3d_nocoils_fieldmap(
    navg_time=(2, 8),
    plot_timings=False,
    include_GPU=False,
    include_CPU=True,
    print_timings=True,
    phasings=["real", "complex"],
    save_dir=None,
):

    recon_cases = []
    if include_GPU:
        recon_cases += ["GPU,Sp", "GPU,Tab"]
        precisions = ["single"]
    else:
        precisions = ["single", "double"]
    if include_CPU:
        recon_cases += ["CPU,Tab", "CPU,Sp"]

    t = _test_mri_multi(
        ndim=3,
        N0=64,
        grid_os_factor=1.5,
        J0=4,
        Ld=4096,
        n_coils=1,
        fieldmap_segments=6,
        precisions=precisions,
        phasings=phasings,
        recon_cases=recon_cases,
        compare_to_exact=False,
        navg_time=navg_time,
    )

    t_create, t_forward, t_adj, t_norm = _split_timings(t, print_timings)
    if plot_timings:
        axes = _plot_all_timings(
            t_create,
            t_forward,
            t_adj,
            t_norm,
            title_substring="3D (no coils, 6 fieldmap seg)",
        )
    else:
        axes = None

    if save_dir is not None:
        fname_base = "bench_3d_001coil_006fmapseg"
        _save_bench(
            save_dir, fname_base, [t_create, t_forward, t_adj, t_norm], axes
        )


def bench_mri_3d_16coils_nofieldmap(
    navg_time=(1, 4),
    plot_timings=False,
    include_GPU=False,
    include_CPU=True,
    print_timings=True,
    phasings=["real", "complex"],
    save_dir=None,
):

    recon_cases = []
    if include_GPU:
        recon_cases += ["GPU,Sp", "GPU,Tab"]
        precisions = ["single"]
    else:
        precisions = ["single", "double"]
    if include_CPU:
        recon_cases += ["CPU,Tab", "CPU,Sp"]

    t = _test_mri_multi(
        ndim=3,
        N0=64,
        grid_os_factor=1.5,
        J0=4,
        Ld=4096,
        n_coils=16,
        fieldmap_segments=None,
        precisions=precisions,
        phasings=phasings,
        recon_cases=recon_cases,
        compare_to_exact=False,
        navg_time=navg_time,
    )

    t_create, t_forward, t_adj, t_norm = _split_timings(t, print_timings)
    if plot_timings:
        axes = _plot_all_timings(
            t_create,
            t_forward,
            t_adj,
            t_norm,
            title_substring="3D (no coils, nofieldmap)",
        )
    else:
        axes = None

    if save_dir is not None:
        fname_base = "bench_3d_016coil_001fmapseg"
        _save_bench(
            save_dir, fname_base, [t_create, t_forward, t_adj, t_norm], axes
        )


def bench_mri_3d_16coils_fieldmap(
    navg_time=(1, 4),
    plot_timings=False,
    include_GPU=False,
    include_CPU=True,
    print_timings=True,
    phasings=["real", "complex"],
    save_dir=None,
):

    recon_cases = []
    if include_GPU:
        recon_cases += ["GPU,Sp", "GPU,Tab"]
        precisions = ["single"]
    else:
        precisions = ["single", "double"]
    if include_CPU:
        recon_cases += ["CPU,Tab", "CPU,Sp"]

    t = _test_mri_multi(
        ndim=3,
        N0=64,
        grid_os_factor=1.5,
        J0=4,
        Ld=4096,
        n_coils=16,
        fieldmap_segments=6,
        precisions=precisions,
        phasings=phasings,
        recon_cases=recon_cases,
        compare_to_exact=False,
        navg_time=navg_time,
    )

    t_create, t_forward, t_adj, t_norm = _split_timings(t, print_timings)
    if plot_timings:
        axes = _plot_all_timings(
            t_create,
            t_forward,
            t_adj,
            t_norm,
            title_substring="3D (16 coils, 6 fieldmap seg)",
        )
    else:
        axes = None

    if save_dir is not None:
        fname_base = "bench_3d_016coil_006fmapseg"
        _save_bench(
            save_dir, fname_base, [t_create, t_forward, t_adj, t_norm], axes
        )


# plot_timings=True; include_GPU=True; include_CPU=True; print_timings=True; phasings=['real', 'complex']; save_dir='/tmp'


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
    if include_GPU:
        import cupy
        from cupy.cuda import device

        device.get_cusparse_handle()
        cupy.fft.cache.enable()

    # 2d cases
    bench_mri_2d_nocoils_nofieldmap(navg_time=(4, 20), **common_kwargs)
    bench_mri_2d_16coils_nofieldmap(navg_time=(2, 8), **common_kwargs)
    bench_mri_2d_16coils_fieldmap(navg_time=(2, 8), **common_kwargs)
    bench_mri_2d_16coils_fieldmap_multispectral(
        navg_time=4, nspectra=2, **common_kwargs
    )

    # 3d cases
    bench_mri_3d_nocoils_nofieldmap(navg_time=(2, 4), **common_kwargs)
    bench_mri_3d_nocoils_fieldmap(navg_time=(2, 4), **common_kwargs)
    bench_mri_3d_16coils_nofieldmap(navg_time=(1, 4), **common_kwargs)
    bench_mri_3d_16coils_fieldmap(navg_time=(1, 4), **common_kwargs)
