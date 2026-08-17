from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from ft4ftirs.data.interferogram import Interferogram, ScanDirection
from ft4ftirs.data.spectrum import Spectrum


def average_spectra(spectra: Sequence[Spectrum], rtol: float = 1e-6) -> Spectrum:
    """
    Average several spectra onto their common wavenumber axis.

    This is the correct way to combine the forward and backward halves of a
    bidirectional (OPUS ``AQM = DD``) acquisition: transform and phase-correct
    each scan direction *independently*, then average the resulting spectra.

    Do **not** average the interferograms instead.  The two scan directions
    carry different phase errors (direction-dependent electronic and optical
    delays), so summing the raw interferograms imprints a coherent artefact
    that ZPD alignment cannot remove — see :func:`average_forward_backward`
    for the measured comparison.

    Parameters
    ----------
    spectra : sequence of Spectrum
        Spectra to average.  All must share the same wavenumber axis and the
        same :class:`~ft4ftirs.data.spectrum.SpectralQuantity`.
    rtol : float, default 1e-6
        Relative tolerance for the wavenumber-axis equality check.

    Returns
    -------
    Spectrum
        Point-wise mean of the input intensities, carrying the first
        spectrum's wavenumber axis and metadata plus ``n_spectra_averaged``.

    Raises
    ------
    ValueError
        If ``spectra`` is empty, or the axes or quantities do not match.
    """
    spectra = list(spectra)
    if not spectra:
        raise ValueError("Need at least one spectrum to average.")

    first = spectra[0]
    for i, s in enumerate(spectra[1:], start=1):
        if s.n_points != first.n_points:
            raise ValueError(
                f"Spectrum {i} has {s.n_points} points, spectrum 0 has "
                f"{first.n_points}.  Interpolate onto a common axis first."
            )
        if not np.allclose(s.wavenumbers, first.wavenumbers, rtol=rtol):
            raise ValueError(
                f"Spectrum {i} has a different wavenumber axis from spectrum 0.  "
                "Interpolate onto a common axis first."
            )
        if s.quantity != first.quantity:
            raise ValueError(
                f"Spectrum {i} is {s.quantity.name}, spectrum 0 is "
                f"{first.quantity.name}.  Cannot average different quantities."
            )

    return Spectrum(
        wavenumbers=first.wavenumbers.copy(),
        intensities=np.mean([s.intensities for s in spectra], axis=0),
        quantity=first.quantity,
        metadata={**first.metadata, "n_spectra_averaged": len(spectra)},
    )


def average_forward_backward(
    forward: Interferogram,
    backward: Interferogram,
    zpd_tolerance: int = 5,
) -> Interferogram:
    """
    Average a forward and backward scan into a single interferogram.

    .. deprecated::
        Use :func:`average_spectra` instead — transform and phase-correct each
        scan direction independently, then average the resulting spectra.

        Averaging *interferograms* is not sound.  The two scan directions carry
        different phase errors, so the sum contains a coherent artefact that ZPD
        alignment cannot remove.  Measured against the spectrum OPUS stored in
        ``examples/example_opus.0`` (RMS residual as a fraction of that
        spectrum's peak-to-peak amplitude):

        ======================================================  =========
        forward scan alone                                        0.030 %
        backward scan alone                                       0.032 %
        this function (interferogram average, ZPD-aligned)        0.247 %
        this function, best achievable rigid sub-sample shift     0.195 %
        :func:`average_spectra` (phase-corrected independently)    0.012 %
        ======================================================  =========

        Averaging the interferograms is thus ~8x *worse* than using a single
        scan direction, while averaging the spectra is better than either —
        which is what √2 noise reduction should look like.

    The backward scan is time-reversed (``np.flip``) before averaging so both
    scans share the same OPD direction.

    Parameters
    ----------
    forward : Interferogram
        Forward-direction scan.
    backward : Interferogram
        Backward-direction scan (time-reversed relative to the forward scan).
    zpd_tolerance : int, default 5
        Maximum allowed index offset between the forward ZPD and the expected
        position of the backward ZPD (``N - 1 - forward.zpd_index``).  Raise
        ``ValueError`` if the mismatch exceeds this threshold.

    Returns
    -------
    Interferogram
        Averaged interferogram with ``scan_direction=ScanDirection.AVERAGED``
        and the forward scan's ``x_index`` and ``laser_wavenumber``.

    Raises
    ------
    ValueError
        If the scans differ in length, laser wavenumber, or ZPD alignment.

    Notes
    -----
    For a symmetric bidirectional scan the ZPD of the backward scan satisfies
    ``zpd_bwd ≈ N - 1 - zpd_fwd``.  Verify this before averaging; if the
    instrument uses different retardation ranges for each direction, resample
    to a common grid first.
    """
    warnings.warn(
        "average_forward_backward() averages interferograms, which imprints a "
        "phase artefact that ZPD alignment cannot remove (measured ~8x worse "
        "than using a single scan direction). Transform each scan direction "
        "separately and combine with average_spectra() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if forward.n_points != backward.n_points:
        raise ValueError(
            f"Forward ({forward.n_points} pts) and backward ({backward.n_points} pts) "
            "scans must have the same length."
        )
    if not np.isclose(forward.laser_wavenumber, backward.laser_wavenumber, rtol=1e-4):
        raise ValueError(
            f"Laser wavenumbers differ: {forward.laser_wavenumber:.4f} vs "
            f"{backward.laser_wavenumber:.4f} cm⁻¹."
        )

    n = forward.n_points
    expected_bwd_zpd = n - 1 - forward.zpd_index
    shift = expected_bwd_zpd - backward.zpd_index  # signed mismatch
    if abs(shift) > zpd_tolerance:
        raise ValueError(
            f"ZPD mismatch: backward ZPD at index {backward.zpd_index}, "
            f"expected {expected_bwd_zpd} (= N-1 - forward ZPD {forward.zpd_index}), "
            f"tolerance ±{zpd_tolerance} samples. "
            "Verify that both scans cover the same OPD range."
        )

    # Flip backward scan to match the forward OPD direction, then shift by
    # any residual misalignment so the two centrbursts are co-phased before
    # averaging.  Without this alignment a k-sample offset introduces a
    # cos(π·ν·k/laser_wn) modulation that distorts the spectrum.
    flipped = np.flip(backward.signal).copy()
    if shift != 0:
        flipped = np.roll(flipped, -shift)

    averaged = (forward.signal + flipped) / 2.0

    return Interferogram(
        signal=averaged,
        laser_wavenumber=forward.laser_wavenumber,
        x_index=forward.x_index.copy(),
        scan_direction=ScanDirection.AVERAGED,
        metadata={
            **forward.metadata,
            "averaged_forward_backward": True,
            "backward_metadata": backward.metadata,
        },
    )
