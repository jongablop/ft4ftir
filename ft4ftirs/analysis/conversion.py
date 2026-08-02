from __future__ import annotations

import numpy as np

from ft4ftirs.data.spectrum import Spectrum, SpectralQuantity


def to_transmittance(sample: Spectrum, reference: Spectrum) -> Spectrum:
    """
    Compute transmittance from a sample and reference single-beam spectrum.

    .. math::

        T(\\tilde{\\nu}) = \\frac{I_{\\text{sample}}(\\tilde{\\nu})}
                                  {I_{\\text{reference}}(\\tilde{\\nu})}

    Parameters
    ----------
    sample : Spectrum
        Sample single-beam spectrum.
    reference : Spectrum
        Background/reference single-beam spectrum measured under the same
        instrument conditions.

    Returns
    -------
    Spectrum
        Transmittance spectrum (dimensionless, values in [0, 1]).

    Raises
    ------
    ValueError
        If the wavenumber axes of ``sample`` and ``reference`` do not match.
    """
    _check_axes_match(sample, reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(
            reference.intensities != 0,
            sample.intensities / reference.intensities,
            np.nan,
        )
    return Spectrum(
        wavenumbers=sample.wavenumbers.copy(),
        intensities=T,
        quantity=SpectralQuantity.TRANSMITTANCE,
        metadata={**sample.metadata, "reference_metadata": reference.metadata},
    )


def to_absorbance(sample: Spectrum, reference: Spectrum) -> Spectrum:
    """
    Compute absorbance (Beer-Lambert) from sample and reference single-beam spectra.

    .. math::

        A(\\tilde{\\nu}) = -\\log_{10}\\!\\left(
            \\frac{I_{\\text{sample}}(\\tilde{\\nu})}
                  {I_{\\text{reference}}(\\tilde{\\nu})}
        \\right)

    Parameters
    ----------
    sample : Spectrum
        Sample single-beam spectrum.
    reference : Spectrum
        Background single-beam spectrum.

    Returns
    -------
    Spectrum
        Absorbance spectrum (dimensionless, Beer-Lambert units).

    Notes
    -----
    Negative absorbance values (emission or artefact bands) are preserved.
    Zero or negative transmittance values produce ``nan`` and a
    ``RuntimeWarning``.
    """
    T = to_transmittance(sample, reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = -np.log10(np.where(T.intensities > 0, T.intensities, np.nan))
    return Spectrum(
        wavenumbers=T.wavenumbers.copy(),
        intensities=A,
        quantity=SpectralQuantity.ABSORBANCE,
        metadata=T.metadata,
    )


def to_reflectance(sample: Spectrum, reference: Spectrum) -> Spectrum:
    """
    Compute reflectance from a sample and reference single-beam spectrum.

    Physically identical to transmittance in terms of the calculation; the
    distinction lies in the measurement geometry (ATR, DRIFTS, specular
    reflection).

    Parameters
    ----------
    sample : Spectrum
        Sample single-beam spectrum.
    reference : Spectrum
        Reference single-beam spectrum (mirror, background, or gold standard).

    Returns
    -------
    Spectrum
        Reflectance spectrum with ``SpectralQuantity.REFLECTANCE``.
    """
    R = to_transmittance(sample, reference)
    return Spectrum(
        wavenumbers=R.wavenumbers.copy(),
        intensities=R.intensities.copy(),
        quantity=SpectralQuantity.REFLECTANCE,
        metadata=R.metadata,
    )


def to_absorbance_reflectance(sample: Spectrum, reference: Spectrum) -> Spectrum:
    """
    Compute absorbance-reflectance (−log₁₀ R), used for DRIFTS and ATR spectra.

    Parameters
    ----------
    sample, reference : Spectrum
        Sample and reference single-beam spectra.

    Returns
    -------
    Spectrum
        −log₁₀(R) spectrum.
    """
    R = to_reflectance(sample, reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        AR = -np.log10(np.where(R.intensities > 0, R.intensities, np.nan))
    return Spectrum(
        wavenumbers=R.wavenumbers.copy(),
        intensities=AR,
        quantity=SpectralQuantity.ABSORBANCE_REFLECTANCE,
        metadata=R.metadata,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_axes_match(a: Spectrum, b: Spectrum) -> None:
    if a.n_points != b.n_points or not np.allclose(
        a.wavenumbers, b.wavenumbers, rtol=1e-6
    ):
        raise ValueError(
            "sample and reference spectra have different wavenumber axes. "
            "Interpolate or resample to a common grid before conversion."
        )
