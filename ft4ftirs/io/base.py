from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from ft4ftirs.data.interferogram import Interferogram
from ft4ftirs.processing.apodization import Apodizer


class SpectrometerReader(ABC):
    """
    Abstract base class for instrument file readers.

    Subclasses parse a vendor-specific binary or text format and return
    a standardised :class:`~ft4ftirs.data.interferogram.Interferogram`
    together with the instrument-recommended :class:`~ft4ftirs.processing.apodization.Apodizer`.

    Implement :meth:`load` to support a new instrument format.
    """

    @abstractmethod
    def load(self, path: Union[str, Path]) -> dict:
        """
        Parse the file at ``path`` and return a normalised data dictionary.

        Parameters
        ----------
        path : str or Path
            Path to the instrument data file.

        Returns
        -------
        dict with keys:
            ``"interferogram"`` : :class:`~ft4ftirs.data.interferogram.Interferogram`
                Raw interferogram data and laser metadata.
            ``"apodizer"`` : :class:`~ft4ftirs.processing.apodization.Apodizer`
                Apodizer configured with the instrument-recommended window.
            ``"signal_gain"`` : float
                Detector gain reported by the instrument.
            ``"wn_min"`` : float or None
                Low-frequency cut-off in cm⁻¹ (OPUS ``HFQ`` parameter).
                Stored in ``interferogram.metadata`` and used by
                :class:`~ft4ftirs.processing.pipeline.SpectralPipeline` to
                automatically trim the output spectrum.
            ``"wn_max"`` : float or None
                High-frequency cut-off in cm⁻¹ (OPUS ``LFQ`` parameter).
        """
