from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ft4ftirs.processing.zpd import ZpdFinder


class ScanDirection(enum.Enum):
    """Mirror scan direction recorded during acquisition."""

    FORWARD = "forward"
    BACKWARD = "backward"
    AVERAGED = "averaged"
    UNKNOWN = "unknown"


@dataclass
class Interferogram:
    """
    Container for a raw FTIR interferogram and its acquisition metadata.

    Parameters
    ----------
    signal : np.ndarray
        Raw interferogram intensities (detector counts, scaled by CSF).
    laser_wavenumber : float
        HeNe reference laser wavenumber in cm⁻¹ (typically ~15798 cm⁻¹).
        Determines the OPD sampling step: dx = 1 / laser_wavenumber.
    x_index : np.ndarray, optional
        Digitizer sample indices from the instrument. Inferred as
        ``np.arange(len(signal))`` when not provided.
    scan_direction : ScanDirection, optional
        Mirror scan direction.  Set by the file reader when the source file
        records it; defaults to ``ScanDirection.UNKNOWN``.
    zpd_finder : ZpdFinder, optional
        Strategy used to locate the centerburst.  Defaults to
        :class:`~ft4ftirs.processing.zpd.ArgmaxZpdFinder`.  Pass a different
        finder (e.g. :class:`~ft4ftirs.processing.zpd.ParabolicZpdFinder`) for
        sub-sample ZPD accuracy.
    metadata : dict, optional
        Instrument parameters (scanner velocity, resolution setting, gain, etc.)
        preserved from the source file for traceability.

    Notes
    -----
    The ZPD (centerburst) position is determined by :attr:`zpd_finder` each
    time :attr:`zpd_index` or :attr:`zpd_position` is accessed.
    """

    signal: np.ndarray
    laser_wavenumber: float
    x_index: Optional[np.ndarray] = None
    scan_direction: ScanDirection = field(default_factory=lambda: ScanDirection.UNKNOWN)
    zpd_finder: Optional[ZpdFinder] = field(default=None, compare=False, repr=False)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=float)
        if self.signal.ndim != 1:
            raise ValueError("signal must be 1-D.")
        if self.laser_wavenumber <= 0:
            raise ValueError("laser_wavenumber must be positive.")
        if self.x_index is None:
            self.x_index = np.arange(len(self.signal), dtype=float)
        else:
            self.x_index = np.asarray(self.x_index, dtype=float)
        if len(self.x_index) != len(self.signal):
            raise ValueError("x_index and signal must have the same length.")
        if self.zpd_finder is None:
            from ft4ftirs.processing.zpd import MinMaxZpdFinder, ArgmaxZpdFinder

            self.zpd_finder = ArgmaxZpdFinder()

        self.signal = self.signal - np.mean(self.signal)

    # ------------------------------------------------------------------
    # Derived physical quantities
    # ------------------------------------------------------------------

    @property
    def n_points(self) -> int:
        """Number of digitizer samples."""
        return len(self.signal)

    @property
    def dx(self) -> float:
        """OPD sampling step in cm (= 1 / laser_wavenumber)."""
        return 1.0 / self.laser_wavenumber

    @property
    def zpd_position(self) -> float:
        """
        Centerburst position in fractional sample units.

        Computed by :attr:`zpd_finder`; may be non-integer for sub-sample
        finders (:class:`~ft4ftirs.processing.zpd.ParabolicZpdFinder`,
        :class:`~ft4ftirs.processing.zpd.CentroidZpdFinder`,
        :class:`~ft4ftirs.processing.zpd.GaussianZpdFinder`).
        """
        return self.zpd_finder(self.signal)

    @property
    def zpd_index(self) -> int:
        """Integer sample index of the centerburst, rounded from :attr:`zpd_position`."""
        return int(round(self.zpd_position))

    @property
    def opd(self) -> np.ndarray:
        """
        Optical path difference axis in cm, centred at the ZPD sample.

        Returns
        -------
        np.ndarray
            OPD values, zero at the centerburst.
        """
        return (self.x_index - self.zpd_index) * self.dx

    @property
    def max_opd(self) -> float:
        """
        Maximum retardation in cm, defined as ``n_points * dx / 2``.

        This is the Fourier-limit of spectral resolution before apodization.
        """
        return self.n_points * self.dx / 2.0

    @property
    def unapodized_resolution(self) -> float:
        """
        Theoretical spectral resolution in cm⁻¹ (= 1 / max_opd).

        Apodization degrades this by a factor that depends on the window
        (e.g., ×1.21 for Hann, ×1.81 for Blackman-Harris 4-term).
        """
        return 1.0 / self.max_opd

    def center_interferogram(self, target_index: Optional[int] = None) -> None:
        """
        Centers the ZPD centerburst to a target index in-place by physically
        shifting the array elements and zeroing out wrapped edges.

        Parameters
        ----------
        target_index : int, optional
            The desired destination index for the ZPD peak.
            Defaults to the exact middle: ``len(signal) // 2``.
        """
        if target_index is None:
            target_index = len(self.signal) // 2

        # 1. Calculate how many indices we need to shift
        current_zpd = self.zpd_index
        shift_amount = target_index - current_zpd

        if shift_amount == 0:
            return

        # 2. Physically shift the array elements in-place
        self.signal = np.roll(self.signal, shift_amount)

        # 3. Clean up wrapped edges to keep the baseline quiet at the borders
        if shift_amount > 0:
            # Shifted right: data at the beginning wrapped around from the tail.
            self.signal[:shift_amount] = 0.0
        else:
            # Shifted left: data at the end wrapped around from the head.
            self.signal[shift_amount:] = 0.0
