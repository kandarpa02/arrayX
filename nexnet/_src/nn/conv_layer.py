from ..nn.conv_functional import *
from ..nn.layers.base_module import Module
from ..nn.initializers import *
from nexnet import RNGKey
from typing import Any, Callable
import torch

def to_tup(shape):
    if isinstance(shape, int):
        return (shape, shape)
    elif isinstance(shape, tuple) and len(shape) > 1:
        return shape
    else:
        raise ValueError(f'Invalid shape is passed {shape}')


class Conv1D(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel: int | tuple,
            stride: int | tuple = 1,
            padding: int | tuple = 0,
            dilation: int = 1,
            groups: int = 1,
            initializer: Callable | Any = None,
            bias: bool = True,
            name: str = ''
            ):

        super().__init__(name)
        self.in_chan, self.out_chan = in_channels, out_channels
        self.kernel = to_tup(kernel)
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.nonlin = None
        self.init_fn = xavier_uniform if initializer is None else initializer

    def __call__(self, x: torch.Tensor, rng: RNGKey) -> torch.Tensor:
        kernel_shape = (self.out_chan, self.in_chan) + self.kernel  # (out, in, kW)
        with self.name_context():
            weight = self.param(
                name="weight",
                shape=kernel_shape,
                dtype=x.dtype,
                init_fn=self.init_fn,
                rng=rng
            )
            bias = self.param("bias", (self.out_chan,), x.dtype, zero_init, rng) if self.bias else None

        return conv1d(
            x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


class Conv2D(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel: int | tuple,
            stride: int | tuple = 1,
            padding: int | tuple = 0,
            dilation: int = 1,
            groups: int = 1,
            initializer: Callable | Any = None,
            bias: bool = True,
            name: str = ''
            ):

        super().__init__(name)
        self.in_chan, self.out_chan = in_channels, out_channels
        self.kernel = to_tup(kernel)
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.nonlin = None
        self.init_fn = xavier_uniform if initializer is None else initializer

    def __call__(self, x: torch.Tensor, rng: RNGKey) -> torch.Tensor:
        kernel_shape = (self.out_chan, self.in_chan) + self.kernel
        with self.name_context():
            weight = self.param(
                name="weight",
                shape=kernel_shape,
                dtype=x.dtype,
                init_fn=self.init_fn,
                rng=rng
            )
            bias = self.param("bias", (self.out_chan,), x.dtype, zero_init, rng) if self.bias else None

        return conv2d(
            x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


class Conv3D(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel: int | tuple,
            stride: int | tuple = 1,
            padding: int | tuple = 0,
            dilation: int = 1,
            groups: int = 1,
            initializer: Callable | Any = None,
            bias: bool = True,
            name: str = ''
            ):

        super().__init__(name)
        self.in_chan, self.out_chan = in_channels, out_channels
        self.kernel = to_tup(kernel)
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.nonlin = None
        self.init_fn = xavier_uniform if initializer is None else initializer

    def __call__(self, x: torch.Tensor, rng: RNGKey) -> torch.Tensor:
        kernel_shape = (self.out_chan, self.in_chan) + self.kernel
        with self.name_context():
            weight = self.param(
                name="weight",
                shape=kernel_shape,
                dtype=x.dtype,
                init_fn=self.init_fn,
                rng=rng
            )
            bias = self.param("bias", (self.out_chan,), x.dtype, zero_init, rng) if self.bias else None

        return conv3d(
            x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
