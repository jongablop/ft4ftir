from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import hilbert


def _envelope(signal: np.ndarray) -> np.ndarray:
    """Amplitude envelope via Hilbert transform.

    For carrier-modulated signals (e.g. FTIR interferograms) the instantaneous
    absolute value oscillates at the carrier frequency.  Taking the modulus of
    the analytic signal removes the carrier and yields the slowly-varying
    envelope, which is what sub-sample ZPD finders should locate.
    """
    return np.abs(hilbert(signal))


class ZpdFinder(ABC):
    """
    Abstract strategy for locating the ZPD (centerburst) of an interferogram.

    Sub-classes implement :meth:`__call__` and return the ZPD position as a
    **float** so that sub-sample finders can preserve fractional accuracy.
    Integer callers (array indexing) should round the result themselves.
    """

    @abstractmethod
    def __call__(self, signal: np.ndarray) -> float:
        """
        Return the ZPD position in sample units.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Raw or apodized interferogram intensities.

        Returns
        -------
        float
            ZPD position.  May be fractional for sub-sample finders.
        """


# ===========================================================================
# Min-max — midpoint of positive and negative centerburst peaks
# ===========================================================================


class MinMaxZpdFinder(ZpdFinder):
    """
    ZPD as the midpoint between the signal maximum and minimum.

    The centerburst produces the largest positive and negative excursions in
    the interferogram.  Their average index estimates the ZPD and is more
    robust than a pure argmax because it uses information from both polarity
    peaks.  For a symmetric centerburst the result coincides with the true
    ZPD; slight asymmetry averages out.

    No Hilbert transform is required — the method operates directly on the
    raw signal.
    """

    def __call__(self, signal: np.ndarray) -> float:
        i_max = float(np.argmax(signal))
        i_min = float(np.argmin(signal))
        return (i_max + i_min) / 2.0


# ===========================================================================
# Argmax — baseline, integer precision
# ===========================================================================


class ArgmaxZpdFinder(ZpdFinder):
    """
    Locate the ZPD as the sample with maximum absolute amplitude.

    This is the standard method used by most instrument software.  It is
    noise-sensitive and has integer (one-sample) precision.
    """

    def __call__(self, signal: np.ndarray) -> float:
        return float(np.argmax(signal))


# ===========================================================================
# Parabolic interpolation — sub-sample, fast
# ===========================================================================


class ParabolicZpdFinder(ZpdFinder):
    """
    Sub-sample ZPD refinement via parabolic interpolation of the envelope.

    The Hilbert-transform envelope is computed first to remove the carrier
    oscillation; a parabola is then fitted through the three samples around
    the envelope peak.  The analytical maximum of the parabola gives a
    fractional ZPD position accurate to ≪ 1 sample for smooth, high-SNR
    centrebursts.

    This is computationally cheap (envelope + O(1) fit) and is the most
    common sub-sample refinement used in real-time instrument firmware.

    References
    ----------
    Quinn, *IEEE Trans. Signal Process.* 42(5), 1321–1323 (1994).
    """

    def __call__(self, signal: np.ndarray) -> float:
        env = _envelope(signal)
        k = int(np.argmax(env))
        n = len(env)
        if k == 0 or k == n - 1:
            return float(k)
        denom = env[k - 1] - 2.0 * env[k] + env[k + 1]
        if denom == 0.0:
            return float(k)
        delta = 0.5 * (env[k - 1] - env[k + 1]) / denom
        return float(k) + delta


# ===========================================================================
# Centroid — sub-sample, robust to noise
# ===========================================================================


class CentroidZpdFinder(ZpdFinder):
    """
    ZPD as the envelope-weighted centroid around the argmax.

    The Hilbert-transform envelope is computed first; then
    ``Σ i · env[i]^p / Σ env[i]^p`` is evaluated over a symmetric window
    around the envelope peak.  More robust to noise and asymmetry than
    parabolic interpolation.  The power ``p`` sharpens the weight
    distribution; ``p = 2`` (energy centroid) is the default.

    Parameters
    ----------
    half_window : int, default 20
        Half-width of the window (samples) around the envelope peak.
    power : float, default 2.0
        Exponent applied to the envelope to compute weights.
        ``p = 1`` → amplitude centroid; ``p = 2`` → energy centroid.
    """

    def __init__(self, half_window: int = 20, power: float = 2.0) -> None:
        if half_window < 1:
            raise ValueError("half_window must be >= 1.")
        if power <= 0:
            raise ValueError("power must be positive.")
        self.half_window = half_window
        self.power = power

    def __call__(self, signal: np.ndarray) -> float:
        env = _envelope(signal)
        k = int(np.argmax(env))
        n = len(env)
        lo = max(0, k - self.half_window)
        hi = min(n, k + self.half_window + 1)
        indices = np.arange(lo, hi, dtype=float)
        weights = env[lo:hi] ** self.power
        total = float(weights.sum())
        if total == 0.0:
            return float(k)
        return float(np.dot(indices, weights) / total)


# ===========================================================================
# Gaussian (log-parabola) fit — sub-sample, most accurate for smooth bursts
# ===========================================================================


class GaussianZpdFinder(ZpdFinder):
    """
    ZPD from a Gaussian (log-parabola) fit to the Hilbert-transform envelope.

    Computes the amplitude envelope via the Hilbert transform, then fits
    ``log(env[i]) = c₀ + c₁·i + c₂·i²`` over a window around the envelope
    peak.  The analytical maximum ``-c₁ / (2·c₂)`` is the centerburst
    position.

    For a true Gaussian envelope this fit is exact; for real interferograms
    it provides the best least-squares Gaussian centre.  The fit is linear
    (no iterations) and more accurate than parabolic interpolation when the
    envelope is asymmetric about the peak sample.

    Parameters
    ----------
    half_window : int, default 50
        Half-width of the fitting window (samples) around the envelope peak.
        Should span at least one full centerburst width.
    """

    def __init__(self, half_window: int = 50) -> None:
        if half_window < 2:
            raise ValueError("half_window must be >= 2.")
        self.half_window = half_window

    def __call__(self, signal: np.ndarray) -> float:
        env = _envelope(signal)
        k = int(np.argmax(env))
        n = len(env)
        lo = max(0, k - self.half_window)
        hi = min(n, k + self.half_window + 1)
        amp = env[lo:hi]
        # Only fit samples with positive envelope (log undefined at zero)
        mask = amp > 0.0
        if mask.sum() < 3:
            return float(k)
        x = np.arange(lo, hi, dtype=float)[mask]
        y = np.log(amp[mask])
        # Quadratic fit: y = c2*x^2 + c1*x + c0  (polyfit returns [c2, c1, c0])
        c2, c1, _ = np.polyfit(x, y, 2)
        if c2 >= 0.0:
            # Not a valid downward-opening parabola — fall back to envelope argmax
            return float(k)
        return -c1 / (2.0 * c2)
