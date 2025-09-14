# ArrayX

Neural Networks are fun right? Like you just write some fancy stuff and it can tell what digit is written on the image, not only that, it can classify audio, video, text and many more!

And all these are possible after you put the correct parameters into a function, and in order to get the correct parameters we compute partial derivatives of the function w.r.t. the parameters to see which parametere contributes how much to the function; Here chain rule of derivative comes into the picture.

I studied the core mechanism behind the **automatic differentiation**, and coded a minimal autograd engine which supports higher order gradients too, here it goes:


```python
>>> import arrx as rx
>>> x = rx.array(5.)
>>> f = lambda x: x ** 3
>>> dx = rx.grad(f)
>>> d2x = rx.grad(dx)
>>> d3x = rx.grad(d2x)
>>> print('dx: ', dx(x))
dx:  75.0
>>> print('d2x: ', d2x(x))
d2x:  30.0
>>> print('d3x: ', d3x(x))
d3x:  6.0

```
