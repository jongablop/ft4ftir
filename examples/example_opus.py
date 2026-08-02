from pathlib import Path

import pandas as pd

from ft4ftirs.io.bruker_opus import BrukerOpusReader
from ft4ftirs.processing.pipeline import SpectralPipeline

# This example shows how to replicate the preprocessing and the Fourier
# transform of an interferogram stored in an OPUS file.

opus_file_path = Path("./example_opus.0")

# Read the interferogram, the instrument-recommended apodizer/phase
# corrector, and the detector gain from the OPUS file.
reader = BrukerOpusReader()
data = reader.load(opus_file_path)

interferogram = data["interferogram"]
apodizer = data["apodizer"]
phase_corrector = data["phase_corrector"]
signal_gain = data["signal_gain"]

# Run the interferogram -> single-beam spectrum pipeline (apodization,
# zero-filling, FFT and phase correction).
pipeline = SpectralPipeline(apodizer, phase_corrector)
spectrum = pipeline(interferogram)

# Scale the flux by the signal gain
scaled_flux = spectrum.intensities / signal_gain

# Save to CSV
csv_filename = opus_file_path.with_name(opus_file_path.stem + "_forward.csv")
df = pd.DataFrame({
    "wavenumber / cm^-1": spectrum.wavenumbers,
    "intensity / a.u.": scaled_flux,
})
df.to_csv(csv_filename, index=False)
