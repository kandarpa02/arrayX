import warnings


class Backend:
    @staticmethod
    def initate():
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
