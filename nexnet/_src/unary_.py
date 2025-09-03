from nexnet._src.autograd import Policy
import torch

class abs(Policy):
    def forward(self, x):
        self.ctx.save(x)
        return torch.abs(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        return grad * torch.sign(x)

class sign(Policy):
    def forward(self, x):
        self.ctx.save()
        return torch.sign(x)
    
    def backward(self, grad):
        return torch.zeros_like(grad)

class exp(Policy):
    def forward(self, x):
        out = torch.exp(x)
        self.ctx.save(out)
        return out
    
    def backward(self, grad):
        out, = self.ctx.release
        return out * grad


class sqrt(Policy):
    def forward(self, x):
        self.ctx.save(x)
        return torch.sqrt(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        return (0.5 / torch.sqrt(x)) * grad
