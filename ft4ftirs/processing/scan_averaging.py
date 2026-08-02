from __future__ import annotations

import numpy as np

from ft4ftirs.data.interferogram import Interferogram, ScanDirection


def average_forward_backward(
    forward: Interferogram,
    backward: Interferogram,
    zpd_tolerance: int = 5,
) -> Interferogram:
    """
    Average a forward and backward scan into a single interferogram.

    The backward scan is time-reversed (``np.flip``) before averaging so both
    scans share the same OPD direction.  Averaging two co-phased scans reduces
    random noise by √2 relative to a single scan.

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
