from __future__ import annotations

from typing import Tuple

import numpy as np

from ft4ftirs.data.interferogram import Interferogram
from ft4ftirs.data.spectrum import Spectrum


def spectral_resolution(
    interferogram: Interferogram, apodization_factor: float = 1.0
) -> float:
    """
    Estimate the effective spectral resolution in cm⁻¹.

    The unapodized (Rayleigh) resolution is ``1 / max_opd``.  Apodization
    broadens the instrument line shape (ILS) by a window-dependent factor.

    Parameters
    ----------
    interferogram : Interferogram
        The interferogram whose acquisition parameters define the resolution.
    apodization_factor : float, default 1.0
        ILS broadening factor for the applied window function.
        Common values: Boxcar=1.0, Triangular=1.33, Hann=1.21,
        BH3=1.61, BH4=1.81, NB-Weak=1.10, NB-Medium=1.30, NB-Strong=1.55.

    Returns
    -------
    float
        Effective spectral resolution in cm⁻¹ (FWHM of the ILS).
    """
    return interferogram.unapodized_resolution * apodization_factor


def peak_to_peak_noise(spectrum: Spectrum, wn_min: float, wn_max: float) -> float:
    """
    Estimate spectral noise as the peak-to-peak amplitude in a baseline region.

    Parameters
    ----------
    spectrum : Spectrum
        Input spectrum (typically single-beam or absorbance).
    wn_min, wn_max : float
        Wavenumber window (cm⁻¹) assumed to contain no analyte bands —
        a flat spectral region used purely for noise characterisation.

    Returns
    -------
    float
        Peak-to-peak noise amplitude (max − min) in the selected region.

    Raises
    ------
    ValueError
        If the selected window contains fewer than 2 points.
    """
    region = spectrum.trim(wn_min, wn_max)
    if region.n_points < 2:
        raise ValueError(
            f"Noise region [{wn_min}, {wn_max}] cm⁻¹ contains fewer than 2 points."
        )
    return float(np.ptp(region.intensities))


def rms_noise(spectrum: Spectrum, wn_min: float, wn_max: float) -> float:
    """
    Estimate spectral noise as the RMS deviation in a baseline region.

    A linear baseline is removed from the selected region before computing
    the RMS to avoid sensitivity to tilt.

    Parameters
    ----------
    spectrum : Spectrum
        Input spectrum.
    wn_min, wn_max : float
        Flat (analyte-free) spectral region in cm⁻¹.

    Returns
    -------
    float
        RMS noise in the selected region (same units as ``spectrum.intensities``).
    """
    region = spectrum.trim(wn_min, wn_max)
    if region.n_points < 4:
        raise ValueError("Need at least 4 points for RMS noise estimation.")
    # Remove linear baseline to eliminate tilt contribution
    coeffs = np.polyfit(region.wavenumbers, region.intensities, 1)
    baseline = np.polyval(coeffs, region.wavenumbers)
    residuals = region.intensities - baseline
    return float(np.sqrt(np.mean(residuals**2)))


def snr(
    spectrum: Spectrum,
    signal_range: Tuple[float, float],
    noise_range: Tuple[float, float],
) -> float:
    """
    Compute signal-to-noise ratio.

    Parameters
    ----------
    spectrum : Spectrum
        Input spectrum.
    signal_range : (float, float)
        Wavenumber window (cm⁻¹) containing the analyte peak of interest.
    noise_range : (float, float)
        Flat, analyte-free wavenumber window (cm⁻¹) for noise estimation.

    Returns
    -------
    float
        SNR = peak_signal / rms_noise (dimensionless).
    """
    signal_region = spectrum.trim(*signal_range)
    noise_val = rms_noise(spectrum, *noise_range)
    peak = float(np.max(np.abs(signal_region.intensities)))
    if noise_val == 0.0:
        return np.inf
    return peak / noise_val


def centerburst_quality(interferogram: Interferogram) -> float:
    """
    Return the dynamic range of the centerburst relative to the noise floor.

    Computed as the ratio of the centerburst amplitude to the RMS of the
    interferogram tails (the final 10 % of each wing), which approximate
    detector noise when the optical signal has decayed.

    Parameters
    ----------
    interferogram : Interferogram

    Returns
    -------
    float
        Centerburst amplitude / tail RMS.  Values < 100 often indicate
        saturation, contamination, or detector issues.
    """
    n = interferogram.n_points
    tail_len = max(2, n // 10)
    tail = np.concatenate(
        [
            interferogram.signal[:tail_len],
            interferogram.signal[-tail_len:],
        ]
    )
    tail_rms = float(np.sqrt(np.mean(tail**2)))
    centerburst_amp = float(np.max(np.abs(interferogram.signal)))
    if tail_rms == 0.0:
        return np.inf
    return centerburst_amp / tail_rms
