def _lib():
    """
    Import the underlying library (numpy or cupynumeric) to resolve recursive import issues.
    """
    from arrx import lib
    return lib


class Dtype:
    """
    Base class for all dtypes in arrx.
    
    Attributes:
        name (str): Name of the dtype.
    """

    def __init__(self, name: str = None) -> None: # type:ignore
        self.name = name or type(self).__name__

    def __repr__(self):
        return f"arrx.{self.name}"

    def __str__(self):
        return f"{self.name}"

    def set(self):
        """
        Returns the actual dtype object from the underlying library.
        
        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement set()")
    
    def __call__(self):
        return self.set()


# FLOATING TYPES 

class floating(Dtype):
    """
    Base class for all floating point dtypes.
    """

    def set(self):
        return _lib().floating


class float16(floating):
    """
    16-bit floating point dtype.
    """

    def __init__(self):
        super().__init__('float16')

    def set(self):
        return _lib().float16()


class float32(floating):
    """
    32-bit floating point dtype.
    """

    def __init__(self):
        super().__init__('float32')

    def set(self):
        return _lib().float32()


class float64(floating):
    """
    64-bit floating point dtype.
    """

    def __init__(self):
        super().__init__('float64')

    def set(self):
        return _lib().float64()


class float128(floating):
    """
    128-bit floating point dtype (if supported by the library).
    """

    def __init__(self):
        super().__init__('float128')

    def set(self):
        return _lib().float128()


# INTEGER TYPES 

class integer(Dtype):
    """
    Base class for all integer dtypes.
    """

    def set(self):
        return _lib().integer


class int8(integer):
    """
    8-bit signed integer dtype.
    """

    def __init__(self):
        super().__init__('int8')

    def set(self):
        return _lib().int8()


class int16(integer):
    """
    16-bit signed integer dtype.
    """

    def __init__(self):
        super().__init__('int16')

    def set(self):
        return _lib().int16()


class int32(integer):
    """
    32-bit signed integer dtype.
    """

    def __init__(self):
        super().__init__('int32')

    def set(self):
        return _lib().int32()


class int64(integer):
    """
    64-bit signed integer dtype.
    """

    def __init__(self):
        super().__init__('int64')

    def set(self):
        return _lib().int64()


class int128(integer):
    """
    128-bit signed integer dtype (if supported by the library).
    """

    def __init__(self):
        super().__init__('int128')

    def set(self):
        raise NotImplementedError(f"Dtype ({self.name}) is not implemented")


# UNSIGNED INTEGER TYPES 

class unsignedinteger(Dtype):
    """
    Base class for all unsigned integer dtypes.
    """

    def set(self):
        return _lib().unsignedinteger()


class uint8(unsignedinteger):
    """
    8-bit unsigned integer dtype.
    """

    def __init__(self):
        super().__init__('uint8')

    def set(self):
        return _lib().uint8()


class uint16(unsignedinteger):
    """
    16-bit unsigned integer dtype.
    """

    def __init__(self):
        super().__init__('uint16')

    def set(self):
        return _lib().uint16()


class uint32(unsignedinteger):
    """
    32-bit unsigned integer dtype.
    """

    def __init__(self):
        super().__init__('uint32')

    def set(self):
        return _lib().uint32()


class uint64(unsignedinteger):
    """
    64-bit unsigned integer dtype.
    """

    def __init__(self):
        super().__init__('uint64')

    def set(self):
        return _lib().uint64()


class uint128(unsignedinteger):
    """
    128-bit unsigned integer dtype (if supported by the library).
    """

    def __init__(self):
        super().__init__('uint128')

    def set(self):
        raise NotImplementedError(f"Dtype ({self.name}) is not implemented")


# BOOLEAN TYPE 

class boolean(Dtype):
    """
    Boolean dtype representing True/False values.
    """

    def __init__(self):
        super().__init__('bool')

    def set(self):
        return bool


def dmap(data_type):
    import numpy as np
    lib = _lib()
    Dtype_MAP = {
        # signed ints
        lib.int8  : int8(),  lib.int16 : int16(),  lib.int32 : int32(),  lib.int64 : int64(),
        np.int8   : int8(),  np.int16  : int16(),  np.int32  : int32(),  np.int64  : int64(),

        # floats
        lib.float16 : float16(), lib.float32 : float32(), lib.float64 : float64(),
        np.float16  : float16(), np.float32  : float32(), np.float64  : float64(),

        # unsigned
        lib.uint16 : uint16(), lib.uint32 : uint32(), lib.uint64 : uint64(),
        np.uint16  : uint16(), np.uint32  : uint32(), np.uint64  : uint64(),

        # booleans
        np.bool_   : boolean(),
        getattr(lib, "bool_", None): boolean() if hasattr(lib, "bool_") else None,
    }

    return Dtype_MAP.get(data_type, None)
