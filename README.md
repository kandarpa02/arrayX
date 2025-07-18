# *#neo*
### *More functional than PyTorch less complex than JAX*

#### Design Principle:
**neonet** is a minmal, lightweight, efficient yet a very powerful Machine Leanring Library. It follows the functional and stateless structure of **JAX**, defining custom `backward` rule via `autograd.Policy` module, just like `torch.autograd.Function` of **PyTorch**. 


#### The Backend Math:
Leading ML frameworks use C/C++ and CUDA as backend but, which is great but while developing **neonet** I thought learning C/C++ then working on the framework will take so long and there is already **NumPy** which is backed by `BLAS` library and **CuPy** which uses `cuBLAS` under the hood. So I integrated them for **CPU** and **GPU** acceleration.
In future I will integrate **Triton** to enable equation fusing and `Jit` compilation (*GPU only*)


#### Minimal Example:
Here is a minimal example how we can define new backward logic and compute grads with **neonet**

```python
import neo
import neo.numpy as nep
from neo import autograd
from neo.functions import fn_forward


# You can define any funcion and its backward rule
# with autograd.Policy module, its inner working a bit
# verbose, I wil make everythng clear once it is complete

class IF_IT_WORKS_DONT_TOUCH_IT(autograd.Policy):
    def forward(self, X, Y, b):
        self.ctx.save(X, Y, b)
        return (X @ Y) + b
    
    def backward(self, grad):
        X, Y, b = self.ctx.release
        x_grad = grad @ Y.T
        y_grad = X.T @ grad
        b_grad = grad.sum(axis=0) if b.size > 1 else grad.sum()
        return x_grad, y_grad, b_grad


X = neo.randn((3,4), device='cuda')
Y = neo.randn((4,2), device='cuda')
b = neo.randn((2,), device='cuda')


forward = fn_forward(IF_IT_WORKS_DONT_TOUCH_IT) # Returns a function & record nodes 

out, grads = autograd.value_and_grad(forward)(X, Y, b)
print("Output :\n", out, "\n")

matrices = list(grads.values())
names = ["X_grad", "Y_grad", "b_grad"]

for name, mat in zip(names, matrices):
    print(f"Matrix {name}:\n{mat}\n")

"""
Output :
 [[ 0.97534184  0.05345273]
 [ 0.63562825  0.53376321]
 [ 0.07361545 -0.63269553]] 

Matrix X_grad:
[[-0.12554877  0.90374159 -0.40608373  0.5634923 ]
 [-0.12554877  0.90374159 -0.40608373  0.5634923 ]
 [-0.12554877  0.90374159 -0.40608373  0.5634923 ]]

Matrix Y_grad:
[[ 4.3551331   4.3551331 ]
 [ 0.11424104  0.11424104]
 [ 0.07168733  0.07168733]
 [-1.56919827 -1.56919827]]

Matrix b_grad:
[3. 3.]"""

```

Now if we do the same thing with **JAX**:

```python
import jax.numpy as jnp
from jax import grad as gfn

# .numpy() method is used to get NumPy arrays from neo.Array object 
# JAX wants NumPy arrays so first convert it to 'cpu' then expose NumPy arrays

X_, Y_, b_ = X.to('cpu').numpy(), Y.to('cpu').numpy(), b.to('cpu').numpy()

grads_jax = gfn(lambda x, y, b: (x@y + b).sum(), argnums=[0,1,2])(X_, Y_, b_)

matrices = list(grads_jax)
names = ["X_JAX_grad", "Y_JAX_grad", "b_JAX_grad"]

for name, mat in zip(names, matrices):
    print(f"Matrix {name}:\n{mat}\n")

"""
Matrix X_JAX_grad:
[[-0.12554878  0.90374154 -0.4060837   0.5634923 ]
 [-0.12554878  0.90374154 -0.4060837   0.5634923 ]
 [-0.12554878  0.90374154 -0.4060837   0.5634923 ]]

Matrix Y_JAX_grad:
[[ 4.355133    4.355133  ]
 [ 0.114241    0.114241  ]
 [ 0.07168728  0.07168728]
 [-1.5691983  -1.5691983 ]]

Matrix b_JAX_grad:
[3. 3.]"""

```
**I am building this for my personal use so I picked my favourite features only! For that this library might feel verbose, still I encourage y'll to try it out, its fun ⌁**