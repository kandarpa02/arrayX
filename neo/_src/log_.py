from neo._src.autograd import Policy
import torch

class log(Policy):
    def forward(self, x):
        self.ctx.save(x)
        return torch.log(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        return grad / x

class log10(Policy):
    def forward(self, x):
        self.ctx.save(x)
        return torch.log10(x)

    def backward(self, grad):
        x, = self.ctx.release
        return grad / (x * torch.log(torch.tensor(10)))
