# ArrayX

for lightning fast numerical computing in python.

---

Neural networks are fascinating - with just a few lines of code, you can build models that recognize handwritten digits, classify audio, interpret text, and much more!

At the heart of these models lies **automatic differentiation**; a powerful technique that allows you to compute gradients efficiently. By calculating partial derivatives of functions with respect to parameters, we can understand how each parameter influences the outcome and optimize it accordingly using methods like gradient descent.

Inspired by this, I explored the underlying principles of autograd systems and built this lightweight yet functional automatic differentiation engine. It supports **higher-order gradients**, enabling you to compute derivatives of derivatives seamlessly.

---

## Example usage

```python
>>> import arrx as rx
>>> x = rx.array(5.0)
>>> f = lambda x: x ** 3
>>> dx = rx.grad(f)     # 1st derivative
>>> d2x = rx.grad(dx)   # 2nd derivative
>>> d3x = rx.grad(d2x)  # 3rd derivative
>>> print('dx:', dx(x))
dx: 75.0
>>> print('d2x:', d2x(x))
d2x: 30.0
>>> print('d3x:', d3x(x))
d3x: 6.0
