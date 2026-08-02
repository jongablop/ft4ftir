from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple

import numpy as np


class SpectralQuantity(Enum):
    """Physical quantity represented by the spectrum intensities."""

    SINGLE_BEAM = auto()
    TRANSMITTANCE = auto()  # T = I_sample / I_reference,  dimensionless [0, 1]
    ABSORBANCE = auto()  # A = -log10(T),                dimensionless [0, ∞)
    REFLECTANCE = auto()  # R = I_sample / I_reference,  dimensionless [0, 1]
    ABSORBANCE_REFLECTANCE = auto()  # -log10(R)


@dataclass
class Spectrum:
    """
    A processed single-sided FTIR spectrum.

    Parameters
    ----------
    wavenumbers : np.ndarray
        Wavenumber axis in cm⁻¹, monotonically increasing.
    intensities : np.ndarray
        Spectral intensities corresponding to each wavenumber point.
    quantity : SpectralQuantity
        Physical interpretation of the intensity values.
    metadata : dict, optional
        Processing parameters and instrument information for reproducibility.

    Notes
    -----
    Wavenumbers are in the positive half-spectrum convention: index 0
    corresponds to 0 cm⁻¹ (DC) and the last index to the Nyquist limit
    (= ``laser_wavenumber / 2``).
    """

    wavenumbers: np.ndarray
    intensities: np.ndarray
    quantity: SpectralQuantity = SpectralQuantity.SINGLE_BEAM
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavenumbers = np.asarray(self.wavenumbers, dtype=float)
        self.intensities = np.asarray(self.intensities, dtype=float)
        if self.wavenumbers.shape != self.intensities.shape:
            raise ValueError("wavenumbers and intensities must have the same shape.")
        if self.wavenumbers.ndim != 1:
            raise ValueError("wavenumbers must be 1-D.")

    # ------------------------------------------------------------------
    # Descriptive properties
    # ------------------------------------------------------------------

    @property
    def n_points(self) -> int:
        """Number of spectral points."""
        return len(self.wavenumbers)

    @property
    def point_spacing(self) -> float:
        """Mean spacing between adjacent wavenumber points in cm⁻¹."""
        return float(np.mean(np.diff(self.wavenumbers)))

    @property
    def wavenumber_range(self) -> Tuple[float, float]:
        """(min, max) of the wavenumber axis in cm⁻¹."""
        return float(self.wavenumbers[0]), float(self.wavenumbers[-1])

    # ------------------------------------------------------------------
    # Slicing / trimming
    # ------------------------------------------------------------------

    def trim(self, wn_min: float, wn_max: float) -> Spectrum:
        """
        Return a copy restricted to ``[wn_min, wn_max]`` cm⁻¹.

        Parameters
        ----------
        wn_min, wn_max : float
            Wavenumber bounds in cm⁻¹ (inclusive).

        Returns
        -------
        Spectrum
            A new Spectrum instance with the wavenumber axis trimmed.
        """
        mask = (self.wavenumbers >= wn_min) & (self.wavenumbers <= wn_max)
        return Spectrum(
            wavenumbers=self.wavenumbers[mask].copy(),
            intensities=self.intensities[mask].copy(),
            quantity=self.quantity,
            metadata=self.metadata.copy(),
        )
