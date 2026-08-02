from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

# ===========================================================================
# Abstract base
# ===========================================================================


class PhaseCorrector(ABC):
    """
    Abstract base class for FTIR phase correction methods.

    Phase errors in FTIR arise from electronic delays, optical misalignment,
    and asymmetric detector/amplifier responses.  Left uncorrected they cause
    dispersive line shapes and baseline distortion.
    """

    @abstractmethod
    def __call__(
        self,
        complex_spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        zpd_index: int,
        apodized_signal: np.ndarray,
    ) -> np.ndarray:
        """
        Return the real-valued, phase-corrected spectrum.

        Parameters
        ----------
        complex_spectrum : np.ndarray, shape (M,)
            Full complex FFT output (one-sided, positive frequencies).
        wavenumbers : np.ndarray, shape (M,)
            Wavenumber axis corresponding to ``complex_spectrum``.
        zpd_index : int
            Index of the centerburst in the *original* (pre-zero-fill) signal.
        apodized_signal : np.ndarray, shape (N,)
            The apodized interferogram before zero-filling (needed by Mertz).

        Returns
        -------
        np.ndarray, shape (M,)
            Real-valued phase-corrected intensities.
        """


# ===========================================================================
# Mertz method (recommended)
# ===========================================================================


class MertzPhaseCorrector(PhaseCorrector):
    def __init__(
        self,
        phase_resolution_fraction: float = 0.125,
        phase_window: str = "triangular",
        phase_resolution_cm: Optional[float] = None,
        laser_wavenumber: Optional[float] = None,
    ) -> None:
        self.phase_resolution_fraction = phase_resolution_fraction
        self._phase_window_name = phase_window
        self.phase_resolution_cm = phase_resolution_cm
        self.laser_wavenumber = laser_wavenumber

    def _triangular_window(self, size: int) -> np.ndarray:
        n = np.arange(size)
        w = 1.0 - np.abs(2.0 * n / (size - 1) - 1.0)
        return np.clip(w, 0.0, None)

    def estimate_phase(
        self,
        complex_spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        zpd_index: int,
        apodized_signal: np.ndarray,  # the UNPADDED apodized signal (length n)
    ) -> np.ndarray:
        """Low-resolution Mertz phase, on the same grid as ``complex_spectrum``.

        Exposed separately from :meth:`__call__` so the phase estimated from ONE
        (sign-definite) channel — e.g. the hot-minus-cold blackbody difference —
        can be applied to OTHER channels, as in the finite-interferogram total-power
        calibration of Heizmann et al. (2025) / Revercomb (1988) complex calibration.
        """
        # Two DISTINCT lengths must not be conflated:
        #   n        — length of the (unpadded) apodized signal; the modulus for
        #              extracting the centerburst segment around the ZPD.
        #   fft_size — length of the FFT that produced `complex_spectrum`; the
        #              low-resolution phase spectrum MUST be reconstructed on this
        #              same grid or it lands on the wrong wavenumber axis. The
        #              pipeline hands us the one-sided spectrum truncated to
        #              fft_size // 2, so fft_size = 2 * len(complex_spectrum).
        n = len(apodized_signal)
        fft_size = 2 * len(complex_spectrum)

        if self.phase_resolution_cm is not None and self.laser_wavenumber is not None:
            n_phase = max(
                4, int(round(2.0 * self.laser_wavenumber / self.phase_resolution_cm))
            )
        else:
            n_phase = max(4, int(fft_size * self.phase_resolution_fraction))

        n_phase = min(n_phase, n)  # cannot exceed the available signal samples
        if n_phase % 2 != 0:
            n_phase -= 1
        half = n_phase // 2

        # Extract the short symmetric window directly around the ZPD index
        left = (zpd_index - half) % n
        right = (zpd_index + half) % n

        if left < right:
            segment = apodized_signal[left:right]
        else:  # Handle wrap-around edges safely
            segment = np.concatenate([apodized_signal[left:], apodized_signal[:right]])

        # Apply window
        segment = segment * self._triangular_window(n_phase)

        # Re-envelope the low-res segment into a FULL-LENGTH (padded) array with the
        # ZPD at index 0, so its FFT grid matches `complex_spectrum` exactly.
        low_res_padded = np.zeros(fft_size, dtype=complex_spectrum.dtype)
        low_res_padded[:half] = segment[half:]
        low_res_padded[-half:] = segment[:half]

        low_res_complex = np.fft.fft(low_res_padded)
        return np.angle(low_res_complex[: len(complex_spectrum)])

    def __call__(
        self,
        complex_spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        zpd_index: int,
        apodized_signal: np.ndarray,
    ) -> np.ndarray:
        phase = self.estimate_phase(
            complex_spectrum, wavenumbers, zpd_index, apodized_signal
        )
        return np.real(complex_spectrum * np.exp(-1j * phase))


# ===========================================================================
# Savitzky-Golay smoothed phase correction
# ===========================================================================


class SavitzkyGolayPhaseCorrector(PhaseCorrector):
    """
    Phase correction via unwrapping and Savitzky-Golay smoothing.

    The raw phase of the full complex spectrum is unwrapped to remove ±π
    discontinuities, then smoothed with a Savitzky-Golay filter to suppress
    noise-induced fluctuations.  The smoothed phase is subtracted from the
    spectrum via complex rotation.

    This approach is computationally cheap and works well for small,
    smoothly-varying instrumental phases.  Use :class:`MertzPhaseCorrector`
    when the phase varies rapidly across the spectral range.

    Parameters
    ----------
    smoothing_bandwidth_cm : float, default 32.0
        Half-width (in cm⁻¹) of the Savitzky-Golay smoothing window.
        Larger values → smoother phase estimate.
    polyorder : int, default 3
        Polynomial order for the Savitzky-Golay filter.
    """

    def __init__(
        self,
        smoothing_bandwidth_cm: float = 32.0,
        polyorder: int = 3,
    ) -> None:
        self.smoothing_bandwidth_cm = smoothing_bandwidth_cm
        self.polyorder = polyorder

    def __call__(
        self,
        complex_spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        zpd_index: int,
        apodized_signal: np.ndarray,
    ) -> np.ndarray:
        raw_phase = np.unwrap(np.angle(complex_spectrum))

        step = wavenumbers[1] - wavenumbers[0]
        window_pts = max(self.polyorder + 2, int(self.smoothing_bandwidth_cm / step))
        if window_pts % 2 == 0:
            window_pts += 1
        window_pts = min(
            window_pts,
            len(raw_phase) if len(raw_phase) % 2 != 0 else len(raw_phase) - 1,
        )

        smoothed_phase = savgol_filter(raw_phase, window_pts, polyorder=self.polyorder)
        return np.real(complex_spectrum * np.exp(-1j * smoothed_phase))
