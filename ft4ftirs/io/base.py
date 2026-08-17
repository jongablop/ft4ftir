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
            ``"interferograms"`` : list of :class:`~ft4ftirs.data.interferogram.Interferogram`
                Every scan direction the file contains — two entries for a
                bidirectional acquisition, one otherwise.  Transform each
                separately and combine the resulting spectra with
                :func:`~ft4ftirs.processing.scan_averaging.average_spectra`;
                the scan directions carry different phase errors, so their
                interferograms must not be averaged.
            ``"interferogram"`` : :class:`~ft4ftirs.data.interferogram.Interferogram`
                ``interferograms[0]``, kept as a single-scan convenience.
            ``"apodizer"`` : :class:`~ft4ftirs.processing.apodization.Apodizer`
                Apodizer configured with the instrument-recommended window.
            ``"phase_corrector"`` : :class:`~ft4ftirs.processing.phase_correction.PhaseCorrector` or None
                Phase corrector configured from the instrument's recorded
                phase-correction settings.  ``None`` when the file records no
                phase correction, in which case
                :class:`~ft4ftirs.processing.pipeline.SpectralPipeline` returns
                the magnitude spectrum.
            ``"signal_gain"`` : float
                Detector gain reported by the instrument.
            ``"zero_filling_factor"`` : int
                Zero-filling factor recorded by the instrument (OPUS ``ZFF``).
                Pass to :class:`~ft4ftirs.processing.pipeline.SpectralPipeline`
                to reproduce the instrument's spectral point spacing.
            ``"wn_min"`` : float or None
                Low-frequency cut-off in cm⁻¹ (OPUS ``HFQ`` parameter — despite
                the name, ``HFQ`` holds the *lower* wavenumber bound).
                Stored in ``interferogram.metadata`` and used by
                :class:`~ft4ftirs.processing.pipeline.SpectralPipeline` to
                automatically trim the output spectrum.
            ``"wn_max"`` : float or None
                High-frequency cut-off in cm⁻¹ (OPUS ``LFQ`` parameter).
        """
