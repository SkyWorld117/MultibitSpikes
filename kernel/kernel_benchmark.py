# def spikingjelly():
#     from spikingjelly.activation_based import neuron, surrogate, functional, layer

#     benchmark_title = f"SpikingJelly PyTorch<br>v0.0.0.0.15"

#     def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
#         class Model(nn.Module):
#             def __init__(self, tau=5.0):
#                 super().__init__()
#                 self.model = nn.Sequential(
#                     layer.Linear(n_neurons, n_neurons),
#                     neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m'),
#                 )

#             def forward(self, x):
#                 functional.reset_net(self.model)
#                 return self.model(x)

#         model = Model().to(device)
#         input_static = torch.randn(n_steps, batch_size, n_neurons).to(device)
#         with torch.no_grad():
#             model(input_static)
#         return dict(model=model, input=input_static, n_neurons=n_neurons)

#     def forward_fn(bench_dict):
#         model, input_static = bench_dict["model"], bench_dict["input"]
#         bench_dict["output"] = model(input_static)
#         return bench_dict

#     def backward_fn(bench_dict):
#         output = bench_dict["output"]
#         loss = output.sum()
#         loss.backward(retain_graph=True)

#     return prepare_fn, forward_fn, backward_fn, benchmark_title

import torch
import torch.nn as nn
import neurons, surrogate
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import surrogate as org_surrogate

class Kernel(nn.Module):
    def __init__(self, n):
        super(Kernel, self).__init__()
        if n == 1:
            self.lif = neurons.LIFNode(surrogate_function=org_surrogate.Sigmoid())
        else:
            self.lif = neurons.LIFNode(surrogate_function=surrogate.Sigmoid(n=n))
    
    def forward(self, x):
        functional.reset_net(self.lif)
        return self.lif(x)
    
def forward_path(n):
    model = Kernel(n).to('cuda')
    input = torch.randn(16384, requires_grad=True).to('cuda')
    output = model(input)
    return output

def backward_path(n):
    output = forward_path(n)
    loss = output.sum()
    loss.backward(retain_graph=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--backward', action='store_true')

args = parser.parse_args()

forward_path(args.n)
if args.backward:
    backward_path(args.n)
