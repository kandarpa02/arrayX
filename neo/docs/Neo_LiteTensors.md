## neo.LiteTensor

**neo.LiteTensor** is an n-dimensional data structure containing values of several data types (e.g., float32, int32, bool), metadata of the data such as `dtype`, `device` and `shape`.

`neo.LiteTensor(data:Any, d_type:str = 'float32', device:str = 'cpu')`


#### arguments:

- data: torch.Tensor, integer, float.
- d_type: float32, int32 etc.
- device: 'cpu' (default) or 'cuda'.



