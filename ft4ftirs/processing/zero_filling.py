from __future__ import annotations

import numpy as np


def next_power_of_two(n: int) -> int:
    """
    Return the smallest power of two that is ≥ ``n``.

    Parameters
    ----------
    n : int
        Input length.

    Returns
    -------
    int
        Smallest 2^k such that 2^k ≥ n.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    return int(2 ** np.ceil(np.log2(n)))


def zero_fill(
    signal: np.ndarray, zpd_index: int, zero_filling_factor: int = 1
) -> np.ndarray:
    """
    Symmetrically zero-pads an interferogram by inserting zeros in the middle,
    leaving the ZPD at its original index position.

    ``zero_filling_factor`` follows Bruker's ``ZFF`` convention: it is the
    number of spectral points per resolution element, and the resolution
    element is set by the maximum retardation — the longer wing of the
    interferogram — so the factor multiplies that wing rather than the full
    record.  The result is never shorter than ``next_power_of_two(len(signal))``,
    so no acquired sample is discarded.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    wing = max(zpd_index, n - 1 - zpd_index)
    target = max(
        next_power_of_two(n),
        next_power_of_two(wing * zero_filling_factor),
    )

    if target == n:
        return signal

    padded = np.zeros(target, dtype=float)

    # Place the left wing at the start, and push the right wing to the very end
    padded[:zpd_index] = signal[:zpd_index]
    padded[-(n - zpd_index) :] = signal[zpd_index:]
    return padded
