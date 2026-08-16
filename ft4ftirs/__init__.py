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
>>> pipeline = SpectralPipeline(data["apodizer"])
>>> spectrum = pipeline(data["interferogram"])
"""

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
from ft4ftirs.processing.scan_averaging import average_forward_backward
from ft4ftirs.processing.zpd import (
    ZpdFinder,
    MinMaxZpdFinder,
    ArgmaxZpdFinder,
    ParabolicZpdFinder,
    CentroidZpdFinder,
    GaussianZpdFinder,
)
from ft4ftirs.analysis.conversion import to_transmittance, to_absorbance, to_reflectance
from ft4ftirs.analysis.metrics import (
    spectral_resolution,
    snr,
    rms_noise,
    peak_to_peak_noise,
)

__version__ = "1.0.0"
__all__ = [
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
    "spectral_resolution",
    "snr",
    "rms_noise",
    "peak_to_peak_noise",
]
