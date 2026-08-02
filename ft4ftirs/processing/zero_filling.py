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


def get_next_power_of_two(self, n):
    return int(2 ** np.ceil(np.log2(n)))


def zero_fill(
    signal: np.ndarray, zpd_index: int, zero_filling_factor: int = 1
) -> np.ndarray:
    """
    Symmetrically zero-pads an interferogram by inserting zeros in the middle,
    leaving the ZPD at its original index position.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    target = int(2 ** np.ceil(np.log2(n * zero_filling_factor)))

    if target == n:
        return signal

    padded = np.zeros(target, dtype=float)

    # Place the left wing at the start, and push the right wing to the very end
    padded[:zpd_index] = signal[:zpd_index]
    padded[-(n - zpd_index) :] = signal[zpd_index:]
    return padded
