<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="neo/media/neo_d1.png">
    <source media="(prefers-color-scheme: light)" srcset="neo/media/neo_l1.png">
    <img alt="Neo Logo" src="neo_logo_light.png" width="800">
  </picture>
</p>


---


***# neo: more functional than PyTorch less complex than JAX***

### Design Principle:
**neonet** is a minmal, lightweight, efficient yet a very powerful Machine Leanring Library. It follows the functional and stateless structure of **JAX**, defining custom `backward` rule via `autograd.Policy` module, just like `torch.autograd.Function` of **PyTorch**. 


### The Backend Math:
Leading ML frameworks use C/C++ and CUDA as backend but, which is great but while developing **neonet** I thought learning C/C++ then working on the framework will take so long and there is already **NumPy** which is backed by `BLAS` library and **CuPy** which uses `cuBLAS` under the hood. So I integrated them for **CPU** and **GPU** acceleration.
In future I will integrate **Triton** to enable equation fusing and `Jit` compilation (*GPU only*)


### Benchmark: Backward Pass (100 runs)
| Device | Neo | PyTorch | Verdict |
|--------|-----|---------|---------|
| CUDA   | 0.8731s | 0.8330s | Torch |
| CPU    | 25.76s  | 29.30s  | Neo |


#### Minimal Example:
Here is a minimal example how we can define new backward logic and compute grads with **neonet**

```python
import neo
import neo.numpy as nep
from neo import autograd
from neo.functions import fn_forward


# You can define any funcion and its backward rule
# with autograd.Policy module, its inner working a bit
# verbose, I will make everythng clear once it is complete

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

out, grads = autograd.session.value_and_grad(forward)(X, Y, b)
print("Output :\n", out, "\n")

matrices = list(grads.values())
names = ["X_grad", "Y_grad", "b_grad"]

for name, mat in zip(names, matrices):
    print(f"Matrix {name}:\n{mat}\n")

"""
Output :
 [[-0.65263305  1.20131121]
 [-1.33377853 -0.93973197]
 [-0.24288831 -1.21855391]] 

Matrix X_grad:
[[0.18543391 0.23330489 1.71311085 0.72454788]
 [0.18543391 0.23330489 1.71311085 0.72454788]
 [0.18543391 0.23330489 1.71311085 0.72454788]]

Matrix Y_grad:
[[ 1.4496104   1.4496104 ]
 [ 2.33568928  2.33568928]
 [-2.0695625  -2.0695625 ]
 [-0.51349937 -0.51349937]]

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
[[0.18543386 0.23330486 1.7131109  0.72454786]
 [0.18543386 0.23330486 1.7131109  0.72454786]
 [0.18543386 0.23330486 1.7131109  0.72454786]]

Matrix Y_JAX_grad:
[[ 1.4496104  1.4496104]
 [ 2.3356893  2.3356893]
 [-2.0695624 -2.0695624]
 [-0.5134994 -0.5134994]]

Matrix b_JAX_grad:
[3. 3.]"""

```
**I am building this for my personal use so I picked my favourite features only! For that this library might feel verbose, still I encourage y'll to try it out, its fun ⌁**