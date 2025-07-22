from neo._src.autograd import Policy
import torch
from ..math import neolib

class log_e(Policy):
    def forward(self, x):
        self.ctx.save(x)
        return neolib.log(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        return grad / x

class log_10(Policy):
    def forward(self, x):
        self.ctx.save(x)
        return neolib.log10(x)

    def backward(self, grad):
        x, = self.ctx.release
        return grad / (x * torch.log(torch.tensor(10)))
