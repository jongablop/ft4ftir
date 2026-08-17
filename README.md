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
from ft4ftirs.processing.scan_averaging import average_spectra

# Load the interferogram(s) plus the instrument-recommended apodizer, phase
# corrector and zero-filling factor from a Bruker OPUS file.
reader = BrukerOpusReader()
data = reader.load("sample.0")

# Run the interferogram -> single-beam spectrum pipeline (apodization,
# zero-filling, FFT and phase correction).  Passing the file's zero-filling
# factor reproduces the spectral point spacing OPUS produced.
pipeline = SpectralPipeline(
    data["apodizer"], data["phase_corrector"], data["zero_filling_factor"]
)

# A bidirectional file holds a forward and a backward scan.  They carry
# different phase errors, so transform each one separately and average the
# resulting *spectra* -- never the interferograms.
spectrum = average_spectra([pipeline(ig) for ig in data["interferograms"]])

print(spectrum.wavenumbers, spectrum.intensities)
```

See `examples/example_opus.py` for a complete, runnable script.

## Package layout

- `ft4ftirs.data` — `Interferogram` and `Spectrum` containers.
- `ft4ftirs.io` — instrument file readers (currently Bruker OPUS).
- `ft4ftirs.processing` — apodization, ZPD finding, phase correction, and the `SpectralPipeline`.
- `ft4ftirs.analysis` — transmittance/absorbance/reflectance conversion and spectral quality metrics.
