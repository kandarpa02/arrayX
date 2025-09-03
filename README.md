# * NEXnet*


**nexnet was built out of frustration with PyTorch’s `backward()` limitations and JAX’s compiler obsession.**
**I just wanted to define the gradient myself 👾 — not fight with a compiler to get it. 😵‍💫**

---

## Eager Functional Reverse-Mode Autodiff (EFRMA)
This system:

- Tracks computations via a stateless **eager trace** (like a tape),
- Uses explicit `value_and_grad` evaluations,
- Lets you define **custom gradients** via `autograd.Policy`
- Best for **Model agnosting Meta Learning** where we compute higher order derivatives.

### Workflow
```text
┌───────────────────────┐      ┌───────────────────────┐      ┌──────────────────────-─┐
│    Forward Pass       │      │    Tape Recording     │      │    Backward Pass       │
├───────────────────────┤      ├───────────────────────┤      ├───────────────────────-┤
│                       │      │                       │      │                        │
│  Input Tensors        │      │  TapeContext.push()   │      │  Initialize:           │
│  (LiteTensor)         ├──────>                       ├──────>  grad_dict = {out:1}   │
│                       │      │  Create Node:         │      │                        │
│  ┌───────────────┐    │      │  - output = result    │      │  Reverse traverse tape │
│  │ @function     │    │      │  - parents = inputs   │      │                        │
│  │  Policy       │    │      │  - bwd_fn = op.bwd    │      │  for node in reversed: │
│  │  .forward()   ├───┐│      │                       │      │    grad = grad_dict.pop│
│  └───────────────┘   ││      │  Add to Tape          │      │    grads =             │
│                       │      │                       │      │      node.bwd_fn(grad) │
│  Output Tensor ◄──────┘      │  TapeContext.pop()    │      │                        │
│                       │      │                       │      │    Accumulate gradients│
└───────────────────────┘      └───────────────────────┘      │    for parents:        │
          ▲                                                   │      grad_dict[parent] │
          │                                                   │          += grad       │
          │                                                   │                        │
          └───────────────────────────────────────────────────┘  Return input_grads    │
                                                                                       │
                                                               └───────────────────────┘
```
#### A simple example of the workflow

```text
┌───────────────────────────┐          ┌─────────────────────────────┐
│      User Code            │          │  @function Transformation   │
│   x = LiteTensor(5.5)     │          │ (Policy → Function)         │
│   y = LiteTensor(6.2)     │          ├─────────────────────────────┤
│   z = multiplication(x,y) ├─────────>│ 1. Instantiates:            │
└───────────────────────────┘          │    op = multiplication()    │ 
                                       │ 2. Processes args:          │
                                       │    valargs = [x.data, y.data] 
                                       │    valargs_strict = [x,y]   │
                                       │ 3. Calls:                   │
                                       │    op.forward(5.5, 6.2)     │
                                       └─────────────┬───────────────┘
                                                     ▼
┌─────────────────────────────┐         ┌────────────────────────────────┐
│  multiplication.forward     │         │    Tape Recording              │
│  • self.ctx.save(5.5,6.2)   │         ├────────────────────────────────┤
│  • Returns: 5.5 * 6.2 = 34.1|         │ Creates Node:                  │
│                             |         │   output = z (LiteTensor(34.1))│
└─────────────┬───────────────┘         │   parents = (x,y)              │
              │                         │   bwd_fn = op.backward         │
              ▼                         └─────────────┬──────────────────┘
        z = LiteTensor(34.1)                          ▼
                                       ┌─────────────────────────────┐
                                       │   Tape State:               │
                                       │   [Node(z, (x,y), backward)]│
                                       └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          Backward Pass                              │
├─────────────────────────────────────────────────────────────────────┤
│  • Initialize: grad_dict = {id(z): 1.0}                             │
│  • Process last node (multiplication):                              │
│      - grad_z = 1.0                                                 │
│      - Call: op.backward(grad_z) → (6.2*1.0, 5.5*1.0) = (6.2, 5.5)  │
│      - Update grad_dict:                                            │
│          grad_dict[id(x)] = 6.2                                     │
│          grad_dict[id(y)] = 5.5                                     │
│  • Return: (z, {x:6.2, y:5.5})                                      │
└─────────────────────────────────────────────────────────────────────┘
```



Unlike PyTorch, **nexnet's LiteTensor does not store**:
- `.grad`
- `.grad_fn`
- Backward hooks
- Recursive metadata

This leads to:
- Lower Python overhead
- Simpler tracing
- Explicit, functional-style programs

nexnet is not a full ML framework, but a **research-grade functional autodiff system**, targeting researchers who want *clarity over abstraction*. Best for people who want to conduct fast paced experiments on new theories


## Why nexnet?

Modern deep learning frameworks like PyTorch and TensorFlow are engineering marvels; fast, feature-rich, and battle-tested at scale. However, their internal complexity often makes them difficult to inspect, modify, or understand at a fundamental level. For students, researchers, and curious developers interested in the *"how"* behind autodiff, optimization, and training loops, these frameworks can feel like opaque black boxes.

**nexnet** was created to fill this gap: a minimalist, modular reverse-mode autodiff system designed for clarity, extensibility, and hands-on learning. It provides a clean, functional interface to core deep learning mechanics, computation graphs, gradient propagation, and optimization, without hiding them behind abstractions. The entire system is written in Python (with selective Cython acceleration), making it easy to read, modify, and extend.

What sets nexnet apart is that it doesn’t just prioritize transparency, it also delivers surprising performance. Thanks to a carefully designed trace-based execution model, nexnet achieves **~97-99% of PyTorch’s training performance** on real workloads like MNIST MLPs, despite lacking kernel fusion, mixed precision, or GPU acceleration. This balance of **simplicity and speed** makes nexnet ideal for:
- Learning the internals of deep learning frameworks,
- Prototyping new autodiff rules or optimizers,
- Experimenting with gradient manipulation,
- Building research tools without heavyweight dependencies.

In short, **nexnet is not a replacement for PyTorch — it is a companion for those who want to understand what’s under the hood, and who believe that clarity is power.**


## Design Principle:
**nexnet** is a minmal, lightweight, efficient yet a very powerful Machine Leanring Library. It follows the functional and stateless structure of **JAX**, defining custom `backward` rule via `autograd.Policy` module, just like `torch.autograd.Function` of **PyTorch** but easier to use!


## The Backend Math:
Leading ML frameworks use C/C++ and CUDA as backend, which is great but while developing **nexnetnet**, I realized learning low-level C/C++ just to get started would take too long. Instead, I went with **PyTorch's Tensors**(`torch.Tensor.detach()`) without its autograd and other functionalities, as the **torch.Tensor** is already very mature and a battle tested backend. In future I will manually define the compute heavy function like `softmax` and `matmul` in **Triron**.

## Experiment: 3-Layer MLP on MNIST

Here is the kaggle link of the test, you can inspect that too [`nexnet-mnist`](https://www.kaggle.com/code/kandarpasarkar/mini-mlpe644d0e26d)

- **Architecture:** `784 → 256 → 64 → 10`  
- **Optimizer:** SGD (default settings for both)  
- **Learning Rate:** default  
- **Batch Size:** `64 (train)`, `1000 (val)`  
- **Initialization:** Xavier, `seed = 0`  

---

## Performance Comparison: nexnet vs. PyTorch

**Remark:** Despite being implemented primarily in Python (with some Cython acceleration), **nexnet** achieves ~99% of PyTorch’s speed on a 3-layer MLP for MNIST. This is thanks to its clean reverse-mode autodiff design and minimal graph overhead. The remaining performance gap is mainly due to Python function dispatch and lack of fused ops, which are optimizable in future versions.

> **Why does nexnet converge slightly faster?**
While both nexnet and PyTorch use the same initialization, architecture, and training setup, tiny implementation-level differences can affect convergence. nexnet’s custom autograd system may apply ops with less internal overhead, fewer dispatch layers, and slightly more deterministic gradient flow. Meanwhile, PyTorch; being a production-scale framework, performs additional runtime checks, optimizations, and backend dispatching, which can subtly affect training dynamics. These minor factors accumulate and may explain the small differences in early convergence and final accuracy.

### nexnet Results

#### nexnet
```
Epoch: 1  Train Acc: 0.5860  Val Acc: 0.7896  (12.65s)
Epoch: 2  Train Acc: 0.8256  Val Acc: 0.8590  (12.42s)
Epoch: 3  Train Acc: 0.8687  Val Acc: 0.8864  (12.58s)
Epoch: 4  Train Acc: 0.8889  Val Acc: 0.8991  (12.52s)
Epoch: 5  Train Acc: 0.9001  Val Acc: 0.9099  (12.52s)
Epoch: 6  Train Acc: 0.9087  Val Acc: 0.9155  (12.32s)
Epoch: 7  Train Acc: 0.9145  Val Acc: 0.9211  (12.39s)
Epoch: 8  Train Acc: 0.9196  Val Acc: 0.9232  (12.96s)
Epoch: 9  Train Acc: 0.9242  Val Acc: 0.9259  (12.46s)
Epoch: 10  Train Acc: 0.9272  Val Acc: 0.9280  (12.37s)
```

#### PyTorch
```
Epoch: 1  Train Acc: 0.5821  Val Acc: 0.7896  (12.67s)
Epoch: 2  Train Acc: 0.8234  Val Acc: 0.8590  (12.43s)
Epoch: 3  Train Acc: 0.8667  Val Acc: 0.8864  (12.56s)
Epoch: 4  Train Acc: 0.8871  Val Acc: 0.8991  (12.56s)
Epoch: 5  Train Acc: 0.8983  Val Acc: 0.9099  (12.42s)
Epoch: 6  Train Acc: 0.9071  Val Acc: 0.9155  (12.63s)
Epoch: 7  Train Acc: 0.9133  Val Acc: 0.9211  (12.44s)
Epoch: 8  Train Acc: 0.9181  Val Acc: 0.9232  (12.42s)
Epoch: 9  Train Acc: 0.9226  Val Acc: 0.9260  (12.53s)
Epoch: 10  Train Acc: 0.9258  Val Acc: 0.9279  (12.59s)
```

## Minimal Example:
Here is a minimal example how we can define new backward logic and compute grads with **nexnet**

```python
import nexnet
from nexnet.functions import function


# You can define any function and its backward rule
# with nexnet.Policy module, its inner working is a bit
# verbose, I will make everythng clear once it is complete

@function # Returns a function & records nodes 
class IF_IT_WORKS_DONT_TOUCH_IT(nexnet.Policy):
    def forward(self, X, Y, b):
        self.ctx.save(X, Y, b)
        return (X @ Y) + b
    
    def backward(self, grad):
        X, Y, b = self.ctx.release
        x_grad = grad @ Y.T
        y_grad = X.T @ grad
        b_grad = grad.sum(axis=0) if b.numel() > 1 else grad.sum()
        return x_grad, y_grad, b_grad


X = nexnet.randn((3,4), device='cuda')
Y = nexnet.randn((4,2), device='cuda')
b = nexnet.randn((2,), device='cuda')

output = None

@nexnet.build_computation_graph(inputs=[X, Y, b])
def forward(X, y, b):
    out = IF_IT_WORKS_DONT_TOUCH_IT(X, Y, b)
    global output
    output = out
    return out.sum()

print("Output :\n", output.numpy(), "\n")

matrices = forward.grad
names = ["X_grad", "Y_grad", "b_grad"]

for name, mat in zip(names, matrices):
    print(f"Matrix {name}:\n{mat.numpy()}\n") 
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

# .numpy() method is used to get NumPy arrays from nexnet.LiteTensor object 
# JAX wants NumPy arrays so first convert it to 'cpu' then expose NumPy arrays

X_, Y_, b_ = X.numpy(), Y.numpy(), b.numpy()

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

### Should you use it?

nexnet is intentionally *not* built to be production-ready but to:
- Study the anatomy of modern autodiff
- Create a functional playground to tinker with gradients
- Allow you to define your own rules, math, and optimization stack

---

**I am building this for personal research, so nexnet reflects my favorite abstractions. It may feel verbose, but every layer is transparentand fun ⌁**
