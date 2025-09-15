import warnings


class Backend:
    @staticmethod
    def initiate():
        import numpy as np
        lib = np

        try: 
            import cupynumeric as cp
        
        except ImportError:
            warnings.warn(
                """
WARNING: The backend engine was unable to find the CUDA required for GPU acceleration. 
As a result, it will default to CPU execution using NumPy. This may lead to significantly slower computations, especially for large-scale models or datasets.
"""
            )
        
        else:
            lib = cp

        return lib

class device:

    def __repr__(self) -> str:
        from arrx import lib
        import numpy as np
        kind = 'cpu' if lib.ndarray == np.ndarray else 'cuda'
        return f"arrx.device('{kind}')"

    @staticmethod
    def put(data):
        import arrx
        rawdata = data._rawbuffer
        rawdata = arrx.lib.array(rawdata)
        return arrx.array(rawdata)
    
