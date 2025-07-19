from neo.backend import get_xp
import warnings

DTYPE_MAP = {
    "float16":  {"numpy": "float16",  "cupy": "float16"},
    "float32":  {"numpy": "float32",  "cupy": "float32"},
    "float64":  {"numpy": "float64",  "cupy": "float64"},
    
    "bfloat16": {"numpy": "float32",  "cupy": "bfloat16"},  # numpy fallback to float32
    
    "int16":    {"numpy": "int16",    "cupy": "int16"},
    "int32":    {"numpy": "int32",    "cupy": "int32"},
    "int64":    {"numpy": "int64",    "cupy": "int64"},

    "bool":     {"numpy": "bool_",    "cupy": "bool_"},
}


def get_dtype(name: str, device: str = "cpu"):
    backend = "cupy" if device == "cuda" else "numpy"
    xp = get_xp(device)
    
    try:
        dtype_str = DTYPE_MAP[name][backend]
        return getattr(xp, dtype_str)
    except (KeyError, AttributeError):
        fallback = "float32" if "float" in name or "bfloat" in name else "int32"
        warnings.warn(
            f"[Neo] Unsupported dtype '{name}' for backend '{backend}'. Falling back to '{fallback}'.",
            category=UserWarning
        )
        return getattr(xp, fallback)
