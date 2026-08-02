# FT4FTIRS

FT4FTIRS (Fourier Tranform for Fourier Transform Infrared Spectroscopy) is a Python package for processing interferograms from FTIR (Fourier Transform Infrared Spectroscopy) data. 
It supports reading Bruker OPUS files, processing interferograms (apodization, zero-filling, Fourier transform), 
and extracting spectra for further analysis.

## Installation

You can install the package using `pip`:

```bash
pip install ft4ftirs
```

## Quick start

```python
from ft4ftirs.io.bruker_opus import BrukerOpusReader
from ft4ftirs.processing.pipeline import SpectralPipeline

# Load an interferogram plus the instrument-recommended apodizer and
# phase corrector from a Bruker OPUS file.
reader = BrukerOpusReader()
data = reader.load("sample.0")

# Run the interferogram -> single-beam spectrum pipeline (apodization,
# zero-filling, FFT and phase correction).
pipeline = SpectralPipeline(data["apodizer"], data["phase_corrector"])
spectrum = pipeline(data["interferogram"])

print(spectrum.wavenumbers, spectrum.intensities)
```

See `examples/example_opus.py` for a complete, runnable script.

## Package layout

- `ft4ftirs.data` — `Interferogram` and `Spectrum` containers.
- `ft4ftirs.io` — instrument file readers (currently Bruker OPUS).
- `ft4ftirs.processing` — apodization, ZPD finding, phase correction, and the `SpectralPipeline`.
- `ft4ftirs.analysis` — transmittance/absorbance/reflectance conversion and spectral quality metrics.
