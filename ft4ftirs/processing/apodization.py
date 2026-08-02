from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import numpy as np

# ===========================================================================
# Abstract base
# ===========================================================================


class ApodizationWindow(ABC):
    """
    Abstract base class for FTIR apodization window functions.

    Subclasses implement ``__call__`` to return a normalised window array
    of arbitrary length.  The window is defined over the full interferogram
    (no truncation); ZPD-centred alignment is handled by :class:`Apodizer`.

    References
    ----------
    Griffiths & de Haseth, *Fourier Transform Infrared Spectrometry*, 2nd ed.,
    Wiley, 2007, Chapter 4.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical name of this window function."""

    @abstractmethod
    def __call__(self, size: int) -> np.ndarray:
        """
        Generate the window array.

        Parameters
        ----------
        size : int
            Number of points.

        Returns
        -------
        np.ndarray, shape (size,)
            Window values normalised to peak = 1, all values in [0, 1].
        """

    def _normalise(self, w: np.ndarray) -> np.ndarray:
        w = np.clip(w, 0.0, None)
        return w / w.max()


# ===========================================================================
# Concrete window implementations
# ===========================================================================


class BoxcarWindow(ApodizationWindow):
    """
    Rectangular (no apodization) window.

    Maximum spectral resolution, worst sidelobe suppression (–13 dB).
    Use only when the interferogram is genuinely double-sided and symmetric.
    """

    @property
    def name(self) -> str:
        return "Boxcar"

    def __call__(self, size: int) -> np.ndarray:
        return np.ones(size)


class TriangularWindow(ApodizationWindow):
    """
    Triangular (Bartlett) window.

    Resolution loss factor ≈ 1.33. Simple, moderate sidelobe suppression (–26 dB).
    """

    @property
    def name(self) -> str:
        return "Triangular"

    def __call__(self, size: int) -> np.ndarray:
        n = np.arange(size)
        w = 1.0 - np.abs(2.0 * n / (size - 1) - 1.0)
        return self._normalise(w)


class HannWindow(ApodizationWindow):
    """
    Hann (raised-cosine) window.

    Resolution loss factor ≈ 1.21. Good sidelobe suppression (–31.5 dB).
    Common default for mid-IR spectrometers.
    """

    @property
    def name(self) -> str:
        return "Hann"

    def __call__(self, size: int) -> np.ndarray:
        n = np.arange(size)
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (size - 1)))
        return self._normalise(w)


class BlackmanHarris3TermWindow(ApodizationWindow):
    """
    Blackman-Harris 3-term window.

    Resolution loss factor ≈ 1.61. Very low sidelobes (–67 dB).

    References
    ----------
    Harris, *Proc. IEEE* 66, 51–83 (1978).
    """

    @property
    def name(self) -> str:
        return "BlackmanHarris3Term"

    def __call__(self, size: int) -> np.ndarray:
        n = np.arange(size)
        a0, a1, a2 = 0.42323, 0.49755, 0.07922
        w = (
            a0
            - a1 * np.cos(2.0 * np.pi * n / (size - 1))
            + a2 * np.cos(4.0 * np.pi * n / (size - 1))
        )
        return self._normalise(w)


class BlackmanHarris4TermWindow(ApodizationWindow):
    """
    Blackman-Harris 4-term window.

    Resolution loss factor ≈ 1.81. Extremely low sidelobes (–92 dB).
    Preferred for quantitative band-area measurements in dense spectra.

    References
    ----------
    Harris, *Proc. IEEE* 66, 51–83 (1978).
    """

    @property
    def name(self) -> str:
        return "BlackmanHarris4Term"

    def __call__(self, size: int) -> np.ndarray:
        n = np.arange(size)
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        w = (
            a0
            - a1 * np.cos(2.0 * np.pi * n / (size - 1))
            + a2 * np.cos(4.0 * np.pi * n / (size - 1))
            - a3 * np.cos(6.0 * np.pi * n / (size - 1))
        )
        return self._normalise(w)


class NortonBeerWeakWindow(ApodizationWindow):
    """
    Norton-Beer "weak" function.

    Resolution loss factor ≈ 1.10.  Minimal broadening, moderate sidelobe
    suppression.  Defined as w(x) = Σ c_j (1 − x²)^j, x ∈ [−1, 1].

    References
    ----------
    Norton & Beer, *J. Opt. Soc. Am.* 66, 259 (1976), Table I.
    """

    _COEFFS = (0.384093, -0.087577, 0.703484)

    @property
    def name(self) -> str:
        return "NortonBeerWeak"

    def __call__(self, size: int) -> np.ndarray:
        x = np.linspace(-1.0, 1.0, size)
        c0, c1, c2 = self._COEFFS
        w = c0 + c1 * (1 - x**2) + c2 * (1 - x**2) ** 2
        return self._normalise(w)


class NortonBeerMediumWindow(ApodizationWindow):
    """
    Norton-Beer "medium" function.

    Resolution loss factor ≈ 1.30.  Balanced resolution / sidelobe trade-off.
    IUPAC recommendation for routine mid-IR quantitative analysis.

    References
    ----------
    Norton & Beer, *J. Opt. Soc. Am.* 66, 259 (1976), Table I.
    """

    _COEFFS = (0.152442, -0.136176, 0.983734)

    @property
    def name(self) -> str:
        return "NortonBeerMedium"

    def __call__(self, size: int) -> np.ndarray:
        x = np.linspace(-1.0, 1.0, size)
        c0, c1, c2 = self._COEFFS
        w = c0 + c1 * (1 - x**2) + c2 * (1 - x**2) ** 2
        return self._normalise(w)


class NortonBeerStrongWindow(ApodizationWindow):
    """
    Norton-Beer "strong" function.

    Resolution loss factor ≈ 1.55.  Strong sidelobe suppression with lower
    resolution penalty than Blackman-Harris windows.

    References
    ----------
    Norton & Beer, *J. Opt. Soc. Am.* 66, 259 (1976), Table I.
    """

    _COEFFS = (0.045335, 0.554883, 0.399782)

    @property
    def name(self) -> str:
        return "NortonBeerStrong"

    def __call__(self, size: int) -> np.ndarray:
        x = np.linspace(-1.0, 1.0, size)
        c0, c1, c2 = self._COEFFS
        w = c0 + c1 * (1 - x**2) + c2 * (1 - x**2) ** 2
        return self._normalise(w)


# ===========================================================================
# Registry
# ===========================================================================

WINDOW_REGISTRY: Dict[str, Type[ApodizationWindow]] = {
    cls().name: cls  # type: ignore[abstract]
    for cls in [
        BoxcarWindow,
        TriangularWindow,
        HannWindow,
        BlackmanHarris3TermWindow,
        BlackmanHarris4TermWindow,
        NortonBeerWeakWindow,
        NortonBeerMediumWindow,
        NortonBeerStrongWindow,
    ]
}


def get_window(name: str) -> ApodizationWindow:
    """
    Retrieve a window instance by canonical name.

    Parameters
    ----------
    name : str
        One of the keys in :data:`WINDOW_REGISTRY`.

    Returns
    -------
    ApodizationWindow

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    if name not in WINDOW_REGISTRY:
        raise ValueError(
            f"Unknown apodization window '{name}'. "
            f"Available: {sorted(WINDOW_REGISTRY.keys())}"
        )
    return WINDOW_REGISTRY[name]()


# ===========================================================================
# Apodizer — applies a window to an interferogram, ZPD-centred
# ===========================================================================


class Apodizer:
    """
    Applies an :class:`ApodizationWindow` to a raw interferogram.

    The window is aligned to the centerburst (ZPD) of the interferogram,
    not to the array boundaries, ensuring physical consistency for both
    symmetric and asymmetric (one-sided dominant) scans.

    Parameters
    ----------
    window : ApodizationWindow
        Window function to use.
    laser_wavenumber : float
        HeNe laser wavenumber in cm⁻¹ (used only for bookkeeping / OPD axis).
    zpd_finder : ZpdFinder, optional
        Strategy for locating the ZPD when no ``zpd_index`` is supplied to
        :meth:`__call__`.  Defaults to
        :class:`~ftirpy.processing.zpd.ArgmaxZpdFinder`.
    """

    def __init__(
        self,
        window: ApodizationWindow,
        laser_wavenumber: float,
        zpd_finder=None,
    ) -> None:
        self.window = window
        self.laser_wavenumber = laser_wavenumber
        if zpd_finder is None:
            from ft4ftirs.processing.zpd import ArgmaxZpdFinder

            self.zpd_finder = ArgmaxZpdFinder()
        else:
            self.zpd_finder = zpd_finder

    def __call__(
        self, signal: np.ndarray, zpd_index: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply the window to ``signal``.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Raw interferogram intensities.
        zpd_index : int, optional
            Pre-computed centerburst index.  When supplied, skips the
            internal :attr:`zpd_finder` call — useful when the caller
            already has the ZPD (e.g. :class:`~ftirpy.processing.pipeline.SpectralPipeline`)
            to avoid computing it twice.

        Returns
        -------
        np.ndarray, shape (N,)
            Apodized interferogram.
        """
        signal = np.asarray(signal, dtype=float)
        size = len(signal)
        if zpd_index is None:
            zpd_index = int(round(self.zpd_finder(signal)))

        w = self.window(size)
        shift = zpd_index - size // 2
        aligned = np.roll(w, shift)

        return signal * aligned

    @classmethod
    def from_name(
        cls, name: str, laser_wavenumber: float, zpd_finder=None
    ) -> "Apodizer":
        """Convenience constructor using a window name string."""
        return cls(
            window=get_window(name),
            laser_wavenumber=laser_wavenumber,
            zpd_finder=zpd_finder,
        )
