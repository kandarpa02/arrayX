## neo.LiteTensor

**neo.LiteTensor** is an n-dimensional data structure containing values of several data types (e.g., float32, int32, bool), metadata of the data such as `dtype`, `device` and `shape`.

`neo.LiteTensor(data:Any, d_type:str = 'float32', device:str = 'cpu')`

#### Simple exampple of LiteTensor creation and computing grads

```python
>>> import neo
>>> from neo import autograd
>>> x = neo.Lite([[5, -1]], dtype='float32')
>>> y = neo.Lite([[4], [0.5]], dtype='float32')
>>> value, grad = autograd.session.value_and_grad(neo.matmul)(x, y)
>>> print(value)
lite_tensor([[19.5]])
>>> print(grad.values())
dict_values([tensor([[4.0000, 0.5000]]), tensor([[ 5.], [-1.]])])
>>> 
```