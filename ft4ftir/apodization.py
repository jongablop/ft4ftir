from typing import Any, Optional
import numpy as np

class Apodization:

    def __init__(
        self, 
        name: str, 
        size: int,
        window: Optional[np.array] = np.array([])
    ) -> None:
        
        self.apodization_functions = {
            "BlackmanHarris3Term": self.blackman_harris_3_term,
        }

        self.name = name 
        self.size = size 
        self.window = window 
        self.function = None

        if self.window.size == 0:

            self.function = self.apodization_functions.get(self.name)
            self.window = self.function(size = self.size)

    def __call__(self, interferogram) -> Any:
        return interferogram*self.window
    
    def blackman_harris_3_term(self, size: Optional[int] = None):
        if size is None:
            size = self.size
        return np.blackman(size)
