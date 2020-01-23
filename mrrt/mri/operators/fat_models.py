"""
Multi-peak fat/water models for Magnetic Resonance Imaging.
"""

import numpy as np


__all__ = ["fat_model"]


def six_peak_fat_model(gamma=42.58, field_strength=3, components=6):
    """Six peak fat model.

    Parameters
    ----------
    gamma : float, optional
        The gyromagnetic ratio in MHz/Tesla (default value is for is for 1H).
    field_strengh : float, optional
        MRI field strength (in Tesla).

    Returns
    -------
    relative_frequencies_Hz : array
        A numpy array containing the relative frequencies in Hz for the fat
        peaks.
    fat_rel_amps : array
        The relative amplitude of the fat peaks (the sum of all amplitudes
        is normalized to 1).

    Notes
    -----
    The output arrays are sorted in order of descending relative amplitude.

    These values are taken from the model used by the FWQPBO toolbox and for
    the 2012 ISMRM fat/water workshop challenge.

    The six components in this model are (offsets listed at 3 Tesla):
        Component 1:  Ampltidue 0.693: ,  Offset (Hz): 434.3
        Component 2:  Ampltidue 0.128: ,  Offset (Hz): 332.1
        Component 3:  Ampltidue 0.087: ,  Offset (Hz): 485.4
        Component 4:  Ampltidue 0.048: ,  Offset (Hz): -76.6
        Component 5:  Ampltidue 0.039: ,  Offset (Hz): 49.8
        Component 6:  Ampltidue 0.004: ,  Offset (Hz): 247.8

    These are similar to the six peak models from [1]_, [2]_ and references
    therein.

    References
    ----------
    .. [1] H. Yu, A. Shimakawa, C.A. McKenzie, E. Brodsky, J.H. Brittain,
           S.B. Reeder. Multi-Echo Water-Fat Separation and Simultaneous R2*
           Estimation with Multi-Frequency Fat Spectrum Modeling.
           Magn. Reson. Med. 2008; 60(5):1122-1134.
           DOI:10.1002/mrm.21737

    .. [2] D. Hernando, Z.-P. Liang, P. Kellman.  Chemical Shift-Based
           Water/Fat Separation: A Comparison of Signal Models.
           Magn. Reson. Med. 2010; 64(3):811-822.
           DOI:10.1002/mrm.22455
    """
    water_cs = 4.7  # water chemical shift in ppm

    # fat peak chemical shifts in ppm
    fat_cs = np.asarray([5.3, 4.31, 2.76, 2.1, 1.3, 0.9])

    # relative amplitudes of the peaks
    fat_rel_amps = np.asarray([0.048, 0.039, 0.004, 0.128, 0.693, 0.087])

    relative_shifts = water_cs - fat_cs

    relative_frequencies_Hz = gamma * field_strength * relative_shifts

    # indices to sort in descending order
    si = np.argsort(fat_rel_amps)[::-1]
    fat_rel_amps = fat_rel_amps[si]
    relative_frequencies_Hz = relative_frequencies_Hz[si]

    if components < 1 or components > 6:
        raise ValueError("components must be between 1 and 6")
    if components < 6:
        # truncate and renormalize
        relative_frequencies_Hz = relative_frequencies_Hz[:components]
        fat_rel_amps = fat_rel_amps[:components]

    # make sure sum of amplitudes is 1.0
    fat_rel_amps /= np.sum(fat_rel_amps)

    return -relative_frequencies_Hz, fat_rel_amps


def three_peak_fat_model(gamma=42.58, field_strength=3):
    """Three peak fat model from the liver as given in [1]_.

    Parameters
    ----------
    gamma : float, optional
        The gyromagnetic ratio in MHz/Tesla (default value is for is for 1H).
    field_strengh : float, optional
        MRI field strength (in Tesla).

    Returns
    -------
    relative_frequencies_Hz : array
        A numpy array containing the relative frequencies in Hz for the fat
        peaks.
    fat_rel_amps : array
        The relative amplitude of the fat peaks (the sum of all amplitudes
        is normalized to 1).

    Notes
    -----
    The values used are an average of patient 1 and 2 of Table 1 in [1]_.

    References
    ----------
    .. [1] H. Yu, A. Shimakawa, C.A. McKenzie, E. Brodsky, J.H. Brittain,
           S.B. Reeder. Multi-Echo Water-Fat Separation and Simultaneous R2*
           Estimation with Multi-Frequency Fat Spectrum Modeling.
           Magn. Reson. Med. 2008; 60(5):1122-1134.
           DOI:10.1002/mrm.21737
    """

    # relative amplitudes of the peaks
    fat_rel_amps = np.asarray([0.74, 0.18, 0.08])
    fat_rel_amps /= fat_rel_amps.sum()

    freqs_3T_1H = np.asarray([420, 318, -94])
    relative_frequencies_Hz = gamma / 42.58 * field_strength / 3 * freqs_3T_1H

    return -relative_frequencies_Hz, fat_rel_amps


def single_peak_fat_model(gamma=42.58, field_strength=3):
    """Simple, single peak fat model.

    Parameters
    ----------
    gamma : float, optional
        The gyromagnetic ratio in MHz/Tesla (default value is for is for 1H).
    field_strengh : float, optional
        MRI field strength (in Tesla).

    Returns
    -------
    relative_frequencies_Hz : array
        A numpy array containing the relative frequencies in Hz for the fat
        peaks.
    fat_rel_amps : array
        The relative amplitude of the fat peaks (the sum of all amplitudes
        is normalized to 1).

    """

    # relative amplitudes of the peaks
    fat_rel_amps = np.asarray([1.0])
    fat_rel_amps /= fat_rel_amps.sum()

    freqs_3T_1H = np.asarray([420])
    relative_frequencies_Hz = gamma / 42.58 * field_strength / 3 * freqs_3T_1H

    return -relative_frequencies_Hz, fat_rel_amps


def fat_model(field_strength, num_peaks=6, gamma=42.58):
    """Simple, single peak fat model.

    Parameters
    ----------
    field_strengh : float, optional
        MRI field strength (in Tesla).
    num_peaks : int
        The number of fat peaks included in the model. Must be in the range
        ``[1, 6]``.
    gamma : float, optional
        The gyromagnetic ratio in MHz/Tesla (default value is for is for 1H).

    Returns
    -------
    relative_frequencies_Hz : array
        A numpy array containing the relative frequencies in Hz for the fat
        peaks.
    fat_rel_amps : array
        The relative amplitude of the fat peaks (the sum of all amplitudes
        is normalized to 1).

    """
    if num_peaks == 1:
        return single_peak_fat_model(gamma=gamma, field_strength=field_strength)
    elif num_peaks == 3:
        return three_peak_fat_model(gamma=gamma, field_strength=field_strength)
    elif num_peaks <= 6:
        return six_peak_fat_model(
            gamma=gamma, field_strength=field_strength, components=num_peaks
        )
    else:
        raise ValueError("unsupported number of peaks")
