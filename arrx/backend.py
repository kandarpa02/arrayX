import warnings


class Backend:
    @staticmethod
    def initiate():
        import jax
        jax.config.update("jax_enable_x64", True)

# class device:

#     def __repr__(self) -> str:
#         from arrx import lib
#         import numpy as np
#         kind = 'cpu' if lib.ndarray == np.ndarray else 'cuda'
#         return f"arrx.device('{kind}')"

#     @staticmethod
#     def put(data):
#         import arrx
#         rawdata = data._rawbuffer
#         rawdata = arrx.lib.array(rawdata)
#         return arrx.array(rawdata)
    
