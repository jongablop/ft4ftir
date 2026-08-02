from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from brukeropusreader import read_file

from ft4ftirs.data.interferogram import Interferogram, ScanDirection
from ft4ftirs.io.base import SpectrometerReader
from ft4ftirs.processing.apodization import Apodizer, get_window
from ft4ftirs.processing.phase_correction import MertzPhaseCorrector
from ft4ftirs.processing.scan_averaging import average_forward_backward

# Mapping from Bruker APF codes to ft4ftirs window names
_BRUKER_APF_MAP: dict[str, str] = {
    "B3": "BlackmanHarris3Term",
    "B4": "BlackmanHarris4Term",
    "BX": "Boxcar",
    "TR": "Triangular",
    "HN": "Hann",
    "NB": "NortonBeerWeak",  # Bruker "NB" → closest to Norton-Beer Weak
    "NM": "NortonBeerMedium",
    "NS": "NortonBeerStrong",
}

_DEFAULT_WINDOW = "BlackmanHarris3Term"

# Bruker AQM (Acquisition Mode) codes → scan direction.
# "DD" (Double Direction) means the IgSm block contains forward + backward
# scans concatenated; the reader splits and averages them automatically.
# Integer codes 0/1 are placeholders until confirmed from real files.
_BRUKER_AQM_MAP: dict = {
    0: ScanDirection.FORWARD,  # TODO: confirm integer code for forward-only
    1: ScanDirection.BACKWARD,  # TODO: confirm integer code for backward-only
    "DD": ScanDirection.AVERAGED,
}


class BrukerOpusReader(SpectrometerReader):
    """
    Reader for Bruker OPUS binary interferogram files.

    Extracts the single-channel interferogram (``IgSm`` block), applies the
    instrument scaling factor (``CSF``), and reads the acquisition parameters
    needed to construct an :class:`~ft4ftirs.data.interferogram.Interferogram`
    and the matching :class:`~ft4ftirs.processing.apodization.Apodizer`.

    Parameters
    ----------
    block : str, default "IgSm"
        OPUS data block to read.  ``"IgSm"`` is the sample single-channel
        interferogram; use ``"IgRf"`` for the reference/background scan.
    """

    def __init__(self, block: str = "IgSm") -> None:
        self.block = block

    def load(self, path: Union[str, Path]) -> dict:
        """
        Load a Bruker OPUS file.

        Parameters
        ----------
        path : str or Path
            Path to the ``.0``, ``.1``, etc. OPUS file.

        Returns
        -------
        dict
            Keys: ``"interferogram"``, ``"apodizer"``, ``"signal_gain"``,
            ``"phase_corrector"``.
            ``"phase_corrector"`` is a :class:`~ft4ftirs.processing.phase_correction.MertzPhaseCorrector`
            configured from the file's ``PHR`` parameter when available, otherwise
            using the fractional fallback.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"OPUS file not found: {path}")

        opus = read_file(str(path))

        # --- Interferogram block ---
        raw = np.asarray(opus[self.block], dtype=float)
        csf: float = float(opus.get(f"{self.block} Data Parameter", {}).get("CSF", 1.0))
        scaled = raw * csf * 0.125

        x_index = np.asarray(opus.get_range(self.block), dtype=float)

        # --- Instrument parameters ---
        instrument = opus.get("Instrument", {})
        hfl: float = float(instrument.get("HFL", 0))  # true HeNe laser wn
        signal_gain: float = float(instrument.get("ASG", 1.0))

        # --- Acquisition mode / scan direction (needed before sampling detection) ---
        acq_params = opus.get("Acquisition", {})
        raw_aqm = (
            instrument.get("AQM")
            or acq_params.get("AQM")
            or opus.get("Measurement", {}).get("AQM")
        )
        scan_direction = (
            _BRUKER_AQM_MAP.get(raw_aqm, ScanDirection.UNKNOWN)
            if raw_aqm is not None
            else ScanDirection.UNKNOWN
        )

        # --- Fourier transformation parameters ---
        ft_params = opus.get("Fourier Transformation", {})
        apf_code: str = ft_params.get("APF", "B3")
        window_name = _BRUKER_APF_MAP.get(apf_code, _DEFAULT_WINDOW)
        phr_raw = ft_params.get("PHR")
        phase_resolution_cm: Optional[float] = (
            float(phr_raw) if phr_raw is not None else None
        )
        # HFQ is the high-pass cutoff (= wn_min); LFQ is the low-pass cutoff (= wn_max).
        hfq_raw = ft_params.get("HFQ")
        lfq_raw = ft_params.get("LFQ")
        wn_min: Optional[float] = float(hfq_raw) if hfq_raw is not None else None
        wn_max: Optional[float] = float(lfq_raw) if lfq_raw is not None else None

        laser_wn = hfl
        base_metadata = {
            "source_file": str(path),
            "opus_block": self.block,
            "apf_code": apf_code,
            "csf": csf,
            "signal_gain": signal_gain,
            "phase_resolution_cm": phase_resolution_cm,
            "wn_min": wn_min,
            "wn_max": wn_max,
        }

        if raw_aqm == "DD":
            # Bidirectional scan: array is [forward | backward] concatenated.
            # Split in half, wrap each in an Interferogram, then average.
            n_half = len(scaled) // 2
            fwd_igram = Interferogram(
                signal=scaled[:n_half],
                laser_wavenumber=laser_wn,
                x_index=x_index[:n_half],
                scan_direction=ScanDirection.FORWARD,
                metadata=base_metadata,
            )
            bwd_igram = Interferogram(
                signal=scaled[n_half:],
                laser_wavenumber=laser_wn,
                x_index=x_index[n_half:],
                scan_direction=ScanDirection.BACKWARD,
                metadata=base_metadata,
            )
            interferogram = average_forward_backward(fwd_igram, bwd_igram)
            # interferogram = fwd_igram

        else:
            interferogram = Interferogram(
                signal=scaled,
                laser_wavenumber=laser_wn,
                x_index=x_index,
                scan_direction=scan_direction,
                metadata=base_metadata,
            )

        apodizer = Apodizer.from_name(window_name, laser_wavenumber=laser_wn)

        phase_corrector = MertzPhaseCorrector(
            phase_resolution_cm=phase_resolution_cm,
            laser_wavenumber=laser_wn,
        )

        return {
            "interferogram": interferogram,
            "apodizer": apodizer,
            "signal_gain": signal_gain,
            "phase_corrector": phase_corrector,
            "wn_min": wn_min,
            "wn_max": wn_max,
        }
