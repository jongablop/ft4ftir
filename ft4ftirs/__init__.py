"""
ft4ftirs
===========
FTIR interferogram-to-spectrum processing pipeline.

Quick start
-----------
>>> from ft4ftirs.io.bruker_opus import BrukerOpusReader
>>> from ft4ftirs.processing.pipeline import SpectralPipeline
>>>
>>> reader = BrukerOpusReader()
>>> data = reader.load("sample.0")
>>> pipeline = SpectralPipeline(
...     data["apodizer"], data["phase_corrector"], data["zero_filling_factor"]
... )
>>> # Transform each scan direction separately, then average the spectra.
>>> spectrum = average_spectra([pipeline(ig) for ig in data["interferograms"]])
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # The version is derived from the git tag at build time by
    # setuptools-scm and read back here from the installed metadata,
    # so there is no version literal anywhere in the source tree.
    __version__ = version("ft4ftirs")
except PackageNotFoundError:  # not installed (e.g. run from a source tree)
    __version__ = "0.0.0+unknown"

from ft4ftirs.data.interferogram import Interferogram, ScanDirection
from ft4ftirs.data.spectrum import Spectrum, SpectralQuantity
from ft4ftirs.processing.apodization import (
    Apodizer,
    ApodizationWindow,
    BoxcarWindow,
    TriangularWindow,
    HannWindow,
    BlackmanHarris3TermWindow,
    BlackmanHarris4TermWindow,
    NortonBeerWeakWindow,
    NortonBeerMediumWindow,
    NortonBeerStrongWindow,
    get_window,
    WINDOW_REGISTRY,
)
from ft4ftirs.processing.phase_correction import (
    MertzPhaseCorrector,
    SavitzkyGolayPhaseCorrector,
)
from ft4ftirs.processing.pipeline import SpectralPipeline
from ft4ftirs.processing.scan_averaging import (
    average_spectra,
    average_forward_backward,
)
from ft4ftirs.processing.zpd import (
    ZpdFinder,
    MinMaxZpdFinder,
    ArgmaxZpdFinder,
    ParabolicZpdFinder,
    CentroidZpdFinder,
    GaussianZpdFinder,
)
from ft4ftirs.analysis.conversion import (
    to_transmittance,
    to_absorbance,
    to_reflectance,
    to_absorbance_reflectance,
)
from ft4ftirs.analysis.metrics import (
    spectral_resolution,
    snr,
    rms_noise,
    peak_to_peak_noise,
    centerburst_quality,
)

__all__ = [
    "__version__",
    # data
    "Interferogram",
    "ScanDirection",
    "Spectrum",
    "SpectralQuantity",
    # processing
    "Apodizer",
    "ApodizationWindow",
    "BoxcarWindow",
    "TriangularWindow",
    "HannWindow",
    "BlackmanHarris3TermWindow",
    "BlackmanHarris4TermWindow",
    "NortonBeerWeakWindow",
    "NortonBeerMediumWindow",
    "NortonBeerStrongWindow",
    "get_window",
    "WINDOW_REGISTRY",
    "MertzPhaseCorrector",
    "SavitzkyGolayPhaseCorrector",
    "SpectralPipeline",
    "average_spectra",
    "average_forward_backward",
    # ZPD finders
    "ZpdFinder",
    "MinMaxZpdFinder",
    "ArgmaxZpdFinder",
    "ParabolicZpdFinder",
    "CentroidZpdFinder",
    "GaussianZpdFinder",
    # analysis
    "to_transmittance",
    "to_absorbance",
    "to_reflectance",
    "to_absorbance_reflectance",
    "spectral_resolution",
    "snr",
    "rms_noise",
    "peak_to_peak_noise",
    "centerburst_quality",
]
