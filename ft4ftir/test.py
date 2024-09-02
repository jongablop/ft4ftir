



from typing import Any


class FourierTransform:

    def __init__(self, param1:int, param2: float, **kwargs) -> None:
        pass

    def getter(self):
        pass

    def __call__(self, interferogram, *args: Any, **kwds: Any) -> Any:
        a = self.apodization * interferogram

        return a
    

if __name__ == "__main__":

    ft = FourierTransform()
    array1 = []
    array2 = []

    result = ft(array1, array2)
    print(result)