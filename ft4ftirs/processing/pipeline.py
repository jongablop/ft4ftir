from __future__ import annotations

from typing import Optional

import numpy as np

from ft4ftirs.data.interferogram import Interferogram
from ft4ftirs.data.spectrum import Spectrum, SpectralQuantity
from ft4ftirs.processing.apodization import Apodizer
from ft4ftirs.processing.phase_correction import PhaseCorrector
from ft4ftirs.processing.zero_filling import next_power_of_two


class SpectralPipeline:
    """
    FTIR interferogram → single-beam spectrum pipeline.

    Steps
    -----
    1. Apodization — ZPD-centred window to suppress Gibbs oscillations.
    2. Zero-filling — roll ZPD to index 0, pad to next power-of-two.
    3. FFT — numpy.fft.fft on the padded signal.
    4. Wavenumber axis — ``k · 2 · laser_wavenumber / N_fft``.
       ``laser_wavenumber`` is the HeNe reference frequency.  The interferogram
       is sampled at every HeNe zero crossing, i.e. every ``λ_HeNe / 2`` of
       optical path difference, so ``dx = 1 / (2 · laser_wavenumber)`` and the
       folding (Nyquist) wavenumber is ``laser_wavenumber`` itself.
    5. Spectrum recovery — magnitude ``|FFT[k]|`` by default (robust,
       no phase estimation required).  Pass a
       :class:`~ft4ftirs.processing.phase_correction.PhaseCorrector` for
       Mertz or Savitzky-Golay phase correction.

    Parameters
    ----------
    apodizer : Apodizer
    phase_corrector : PhaseCorrector, optional
        When ``None`` (default) the spectrum is taken as ``|FFT[k]|``.
    zero_filling_factor : int, default 1
        Number of spectral points per resolution element, following Bruker's
        ``ZFF`` convention: the resolution element is set by the maximum
        retardation — the longer *wing* of the interferogram — so the factor
        multiplies that wing, not the full double-sided record, before rounding
        up to the next power of two.  The transform is never made smaller than
        ``next_power_of_two(len(signal))``, so no acquired sample is ever
        discarded; for a double-sided interferogram this means factors 1 and 2
        coincide (the full record already carries two points per resolution
        element).
    """

    def __init__(
        self,
        apodizer: Apodizer,
        phase_corrector: Optional[PhaseCorrector] = None,
        zero_filling_factor: int = 1,
    ) -> None:
        if zero_filling_factor < 1:
            raise ValueError("zero_filling_factor must be >= 1.")
        self.apodizer = apodizer
        self.phase_corrector = phase_corrector
        self.zero_filling_factor = zero_filling_factor

    def __call__(self, interferogram: Interferogram) -> Spectrum:
        # 1. Dynamically locate the true ZPD centerburst peak
        # zpd = int(np.argmax(np.abs(interferogram.signal)))

        interferogram.center_interferogram()
        zpd = interferogram.zpd_index

        # 2. Apodization
        apodized = self.apodizer(interferogram.signal, zpd_index=zpd)

        # 3. The Standard FTIR Center-Padding Workflow:
        # `n` is the number of acquired samples and drives the padding below; the
        # zero-filling factor is counted from a different length. Bruker's ZFF is
        # spectral points per resolution element, and the resolution element is set
        # by the maximum retardation -- the longer wing of the interferogram -- so
        # the factor multiplies the wing, not the whole double-sided record.
        n = len(apodized)
        wing = max(zpd, n - 1 - zpd)
        fft_size = max(
            next_power_of_two(n),  # never discard acquired samples
            next_power_of_two(wing * self.zero_filling_factor),
        )

        # Step A: Pad the centered, apodized data with zeros evenly on both sides
        # This keeps the ZPD dead-center and allows the wings to decay smoothly.
        pad_width = (fft_size - n) // 2
        # Handle odd/even padding imbalances cleanly
        remainder = (fft_size - n) % 2
        padded_centered = np.pad(
            apodized, (pad_width, pad_width + remainder), mode="constant"
        )

        # Step B: Shift the ZPD from the center to index 0 using the standard library
        # This replaces your manual slicing and handles odd/even shapes perfectly.
        padded_fft_ready = np.fft.ifftshift(padded_centered)

        # 4. FFT
        complex_spectrum = np.fft.rfft(padded_fft_ready)
        n_positive = fft_size // 2
        complex_spectrum_pos = complex_spectrum[:n_positive]

        # 5. Wavenumber axis
        wavenumbers = (
            np.arange(n_positive) * (2.0 * interferogram.laser_wavenumber) / fft_size
        )

        # 6. Linear Phase Correction
        # Because the ZPD is located at index `zpd` instead of index 0,
        # the FFT introduces a predictable linear phase ramp. We subtract it out directly:
        # frequencies = np.arange(n_positive) / fft_size
        # linear_phase_correction = np.exp(2j * np.pi * zpd * frequencies)
        # complex_spectrum_pos = complex_spectrum_pos * linear_phase_correction

        # 7. Spectrum recovery
        if self.phase_corrector is None:
            corrected = np.abs(complex_spectrum_pos)
        else:
            # Pass the phase-aligned spectrum to the corrector
            corrected = self.phase_corrector(
                complex_spectrum=complex_spectrum_pos,
                wavenumbers=wavenumbers,
                zpd_index=zpd,
                apodized_signal=apodized,
            )

        # ... metadata and Spectrum return block ...

        metadata = {
            "laser_wavenumber_cm": interferogram.laser_wavenumber,
            "apodization_window": getattr(
                self.apodizer.window,
                "name",
                type(self.apodizer.window).__name__,
            ),
            "zero_filling_factor": self.zero_filling_factor,
            "phase_corrector": (
                "magnitude"
                if self.phase_corrector is None
                else type(self.phase_corrector).__name__
            ),
            "fft_size": fft_size,
            **interferogram.metadata,
        }

        spectrum = Spectrum(
            wavenumbers=wavenumbers,
            intensities=corrected,
            quantity=SpectralQuantity.SINGLE_BEAM,
            metadata=metadata,
        )

        wn_min = interferogram.metadata.get("wn_min")
        wn_max = interferogram.metadata.get("wn_max")
        if wn_min is not None or wn_max is not None:
            lo = wn_min if wn_min is not None else spectrum.wavenumbers[0]
            hi = wn_max if wn_max is not None else spectrum.wavenumbers[-1]
            spectrum = spectrum.trim(lo, hi)

        return spectrum
