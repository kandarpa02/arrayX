
# *#neo*

## *This readme is outdated, backend is being changed, will update this soon!*

**neo** is an experimental machine learning system designed with an emphasis on **minimalism**, **functional design**, and **performance clarity**. At its core, it implements a fully custom **reverse-mode autodiff engine** called:

### Eager Functional Reverse-Mode Autodiff (EFRMA)

This system:

- Tracks computations via a stateless **eager trace** (like a tape),
- Uses explicit `value_and_grad` evaluations (no global flags or magic),
- Lets you define **custom gradients** via `autograd.Policy`,
- Optimizes execution overhead by avoiding dynamic metadata or nested Python object trees.

Unlike PyTorch, **neo Arrays do not store**:
- `.grad`
- `.grad_fn`
- Backward hooks
- Recursive metadata

This leads to:
- Lower Python overhead
- Simpler tracing
- Explicit, functional-style programs

neo is not a full ML framework, but a **research-grade functional autodiff system**, targeting researchers and autodiff hobbyists who want *clarity over abstraction*.


### Design Principle:
**neo** is a minmal, lightweight, efficient yet a very powerful Machine Leanring Library. It follows the functional and stateless structure of **JAX**, defining custom `backward` rule via `autograd.Policy` module, just like `torch.autograd.Function` of **PyTorch**. 


### The Backend Math:
Leading ML frameworks use C/C++ and CUDA as backend, which is great but while developing **neonet**, I realized learning low-level C/C++ just to get started would take too long. Instead, I leveraged **NumPy** (which uses `BLAS`) for CPU and **CuPy** (which wraps `cuBLAS`) for GPU acceleration, making prototyping fast and efficient.

**Note:** For now, Neo uses NumPy/CuPy as the backend array libraries. In the near future, these will be replaced by `torch.Tensor.detach()` using **only the raw Torch tensor object**, without its autograd, module system, or Python overhead to achieve high-performance array math on both CPU and GPU.

Eventually, I also plan to integrate **Triton** for equation fusion and `jit`-compiled GPU kernels.


### Benchmark: Backward Pass for 100 runs (smaller is better)
| Device | Neo | PyTorch | Verdict |
|--------|-----|---------|---------|
| CUDA   | 0.8731s | 0.8330s | PyTorch |
| CPU    | 25.76s  | 29.30s  | Neo |


### Experiment: 3-Layer MLP on MNIST

Here is the colab link of the test, you can inspect that too [`neo-mnist`](https://drive.google.com/file/d/1mp5-0ZaFidrBdPWbzSt391DQaoAVujHp/view?usp=sharing)

- Architecture: `784 → 256 → 64 → 10`
- Optimizer: Vanilla Gradient Descent (`Neo`), SGD (`Torch`)
- LR: `0.5`
- Batch size: `64 (train)`, `1000 (val)`

> **Remark:** Despite Neo being built entirely in Python (including its autodiff system and just a standard gradient descent optimizer), it still achieves **~95–98% of PyTorch's performance** on a 3-layer MLP for MNIST, thanks to its clean reversed-mode design and raw array backends (NumPy/CuPy). The performance gap mainly comes from Python-side overhead and lack of fused ops, which can be further optimized in future versions.

#### Neo
```
Epoch: 1  Train Acc: 0.8641  Val Acc: 0.8076  (17.14s)
Epoch: 2  Train Acc: 0.9282  Val Acc: 0.8728  (17.72s)
Epoch: 3  Train Acc: 0.9422  Val Acc: 0.8845  (16.86s)
Epoch: 4  Train Acc: 0.9491  Val Acc: 0.8883  (17.51s)
Epoch: 5  Train Acc: 0.9536  Val Acc: 0.9036  (16.84s)
Epoch: 6  Train Acc: 0.9577  Val Acc: 0.9030  (17.38s)
Epoch: 7  Train Acc: 0.9592  Val Acc: 0.9106  (16.95s)
Epoch: 8  Train Acc: 0.9615  Val Acc: 0.9135  (17.02s)
Epoch: 9  Train Acc: 0.9657  Val Acc: 0.9117  (17.30s)
Epoch: 10 Train Acc: 0.9660  Val Acc: 0.9211  (16.86s)
```

#### PyTorch
```
Epoch: 1  Train Acc: 0.9277  Val Acc: 0.9607  (14.19s)
Epoch: 2  Train Acc: 0.9677  Val Acc: 0.9469  (13.42s)
Epoch: 3  Train Acc: 0.9784  Val Acc: 0.9705  (13.43s)
Epoch: 4  Train Acc: 0.9833  Val Acc: 0.9737  (13.56s)
Epoch: 5  Train Acc: 0.9851  Val Acc: 0.9671  (13.50s)
Epoch: 6  Train Acc: 0.9879  Val Acc: 0.9756  (13.51s)
Epoch: 7  Train Acc: 0.9915  Val Acc: 0.9766  (13.72s)
Epoch: 8  Train Acc: 0.9946  Val Acc: 0.9752  (14.41s)
Epoch: 9  Train Acc: 0.9940  Val Acc: 0.9792  (13.81s)
Epoch: 10 Train Acc: 0.9949  Val Acc: 0.9755  (13.89s)
```


#### Minimal Example:
Here is a minimal example how we can define new backward logic and compute grads with **neo**

```python
import neo
import neo.numpy as nep
from neo import autograd
from neo.functions import neo_function


# You can define any funcion and its backward rule
# with autograd.Policy module, its inner working is a bit
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


forward = neo_function(IF_IT_WORKS_DONT_TOUCH_IT) # Returns a function & records nodes 

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

### Why neo?

neo is intentionally *not* built to be production-ready but to:
- Study the anatomy of modern autodiff
- Create a functional playground to tinker with gradients
- Allow you to define your own rules, math, and optimization stack

---

**I am building this for personal research, so neo reflects my favorite abstractions. It may feel verbose, but every layer is transparentand fun ⌁**
