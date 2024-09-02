from typing import Any
import numpy as np
import cmath

from ft4ftir.interferogram import Interferogram
from ft4ftir.apodization import Apodization

class FourierTransform:

    def __init__(
        self,
        interferogram: Interferogram,
        apodization: Apodization
    ) -> None:
        
        self.interferogram = interferogram
        self.apodization = apodization

    def get_next_power_of_two(self, n):
        """
        Calculate the next power of two greater than or equal to n.

        Parameters:
        - n (int): Input integer.

        Returns:
        - int: The next power of two.
        """
        return int(2 ** np.ceil(np.log2(n)))
    
    def zero_fill(self, interferogram, target_size):
        """
        Zero-fill the interferogram to a target size.

        Parameters:
        - interferogram (numpy.ndarray): Input interferogram data.
        - target_size (int): Target size after zero-filling.

        Returns:
        - numpy.ndarray: Zero-filled interferogram.
        """
        filled_interferogram = np.zeros(target_size)
        half_size = int(interferogram.size / 2)
        
        filled_interferogram[:half_size] = interferogram[half_size:]
        filled_interferogram[-half_size:] = interferogram[:half_size][::-1]
        
        return filled_interferogram

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        # Preprocess the interferogram
        apodized_interferogram = self.apodization(self.interferogram.interferogram)
        next_power_of_two = self.get_next_power_of_two(apodized_interferogram.size)
        reversed_zerofilled_interferogram = self.zero_fill(apodized_interferogram, target_size=next_power_of_two) 

        # FFT
        uncorrected_spectrum = np.fft.fft(reversed_zerofilled_interferogram)

        # Phase correction
        corrected_spectrum_size = int(next_power_of_two / 2)
        corrected_spectrum = np.zeros(corrected_spectrum_size)

        for i in range(int(corrected_spectrum_size / 2)):
            # Calculate phase
            phase = cmath.phase(uncorrected_spectrum[i])
            
            # Apply phase correction
            corrected_spectrum[i] = np.abs(np.real(uncorrected_spectrum[i] * np.exp(-1j * phase)))
            
        # Generate wavenumbers
        wavenumbers = np.zeros(corrected_spectrum_size)
        wavenumbers = np.arange(corrected_spectrum_size) * self.interferogram.laser_wavenumber / corrected_spectrum_size
        
        return corrected_spectrum, wavenumbers
