import torch

def _device(arg):
    """Ensure argument is a torch.device"""
    if arg is None: return None
    if isinstance(arg, torch.device): return arg
    if isinstance(arg, str): return torch.device(arg)
    raise TypeError(f"Invalid device: {arg}")

def _dtype(arg):
    """Ensure argument is torch.dtype"""
    if arg is None: return None
    if isinstance(arg, torch.dtype): return arg
    if isinstance(arg, str):
        try: return getattr(torch, arg)
        except AttributeError: raise TypeError(f"Invalid dtype string: '{arg}'")
    raise TypeError(f"Invalid dtype: {arg}")

def _neo_dtype(arg):
    """Convert torch dtype to string representation without torch. prefix"""
    arg = str(arg)
    if 'torch.' in arg:
        return arg.removeprefix('torch.')
