import numpy as np
from typing import Union

class Interferogram:

    def __init__(
        self,
        x_index: np.array,
        interferogram: np.array,
        laser_wavenumber: Union[int, float]    
    ) -> None:
        
        self.laser_wavenumber = laser_wavenumber
        self.x_index = x_index
        self.interferogram = interferogram