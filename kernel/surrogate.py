import torch
import torch.nn as nn

class SurrogateFunctionBase(nn.Module):
    def __init__(self, alpha, spiking=True, n=1, threshold=1.0):
        super().__init__()
        self.spiking = spiking
        self.alpha = alpha
        self.n = n
        self.threshold = threshold

    def set_spiking_mode(self, spiking: bool):
        self.spiking = spiking

    def extra_repr(self):
        return f'alpha={self.alpha}, spiking={self.spiking}'

    @staticmethod
    def spiking_function(x, alpha, n, threshold):
        raise NotImplementedError

    @staticmethod
    def primitive_function(x, alpha, n, threshold):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return self.spiking_function(x, self.alpha, self.n, self.threshold)
        else:
            return self.primitive_function(x, self.alpha, self.n, self.threshold)

@torch.jit.script
def multi_level(x: torch.Tensor, n: int, threshold: float):
    l = int(2**n)-1
    r = (x >= 0).float()
    for i in range(1, l):
        r += ((x >= -float(i)/l * threshold) ^ (x >= -float(i-1)/l * threshold)) * float(l-i)/l
    return r.to(x)

@torch.jit.script
def multi_level_parallelized(x: torch.Tensor, n: int, threshold: float):
    l = int(2**n) - 1
    # Create indices from 1 to l-1 as a tensor
    i = torch.arange(1, l, dtype=torch.float32, device=x.device)
    
    # Compute the thresholds for each step
    lower_bounds = -i / l * threshold
    upper_bounds = -torch.cat((torch.tensor([0.0], device=x.device), lower_bounds[:-1]))

    # Expand x for broadcasting and calculate contributions
    x_expanded = x.unsqueeze(0)  # Shape (1, x.size())
    contributions = ((x_expanded >= lower_bounds.view(-1, 1)) ^ (x_expanded >= upper_bounds.view(-1, 1))) * ((l - i).view(-1, 1) / l)

    # Sum contributions and add the initial term
    r = (x >= 0).float() + contributions.sum(dim=0)

    return r.to(x)

@torch.jit.script
def sigmoid_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float, n: int, threshold: float):
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1. - sgax) * sgax * alpha, None, None, None

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, n, threshold):
        shift = (2**(n-1) -1) / (2**n-1) * threshold
        if x.requires_grad:
            ctx.save_for_backward(x+shift)
            ctx.alpha = alpha
            ctx.n = n
            ctx.threshold = threshold
        return multi_level(x, n, threshold)

    @staticmethod
    def backward(ctx, grad_output):
        return sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.alpha, ctx.n, ctx.threshold)

class Sigmoid(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True, n=1, threshold=1.0):
        super().__init__(alpha, spiking, n, threshold)

    @staticmethod
    def spiking_function(x, alpha, n, threshold):
        return sigmoid.apply(x, alpha, n, threshold)

    # @staticmethod
    # @torch.jit.script
    # def primitive_function(x: torch.Tensor, alpha: float, n: int, threshold: float):
    #     return (x * alpha).sigmoid()

    @staticmethod
    def backward(grad_output, x, alpha, n, threshold):
        shift = (2**(n-1) -1) / (2**n-1) * threshold
        return sigmoid_backward(grad_output, x+shift, alpha, n, threshold)[0]
