from pathlib import Path

import pandas as pd

from ft4ftirs.io.bruker_opus import BrukerOpusReader
from ft4ftirs.processing.pipeline import SpectralPipeline
from ft4ftirs.processing.scan_averaging import average_spectra

# This example shows how to replicate the preprocessing and the Fourier
# transform of an interferogram stored in an OPUS file.

opus_file_path = Path("./example_opus.0")

# Read the interferogram, the instrument-recommended apodizer/phase
# corrector, and the detector gain from the OPUS file.
reader = BrukerOpusReader()
data = reader.load(opus_file_path)

interferograms = data["interferograms"]
apodizer = data["apodizer"]
phase_corrector = data["phase_corrector"]
signal_gain = data["signal_gain"]
zero_filling_factor = data["zero_filling_factor"]

# Run the interferogram -> single-beam spectrum pipeline (apodization,
# zero-filling, FFT and phase correction).  The zero-filling factor comes
# from the file's ZFF parameter, so the spectral point spacing matches the
# one OPUS produced.
pipeline = SpectralPipeline(apodizer, phase_corrector, zero_filling_factor)

# A bidirectional (AQM = DD) file holds a forward and a backward scan.  They
# carry different phase errors, so each is transformed and phase-corrected on
# its own and only the resulting spectra are averaged.  Averaging the
# interferograms instead would imprint an artefact no alignment can remove.
spectrum = average_spectra([pipeline(ig) for ig in interferograms])

# Scale the flux by the signal gain
scaled_flux = spectrum.intensities / signal_gain

# Save to CSV
csv_filename = opus_file_path.with_name(opus_file_path.stem + "_forward.csv")
df = pd.DataFrame({
    "wavenumber / cm^-1": spectrum.wavenumbers,
    "intensity / a.u.": scaled_flux,
})
df.to_csv(csv_filename, index=False)
