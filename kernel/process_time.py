from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import neurons, surrogate
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import surrogate as org_surrogate

import time

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
    input = torch.randn(2**25, requires_grad=True).to('cuda')
    start = time.process_time()
    output = model(input)
    end = time.process_time()
    return output, end - start

def backward_path(n):
    output, forward_time = forward_path(n)
    loss = output.sum()
    start = time.process_time()
    loss.backward(retain_graph=True)
    end = time.process_time()
    return forward_time, end - start

def repeat_experiment(n, num_repeats):
    forward_times = []
    backward_times = []
    for _ in range(num_repeats):
        forward_time, backward_time = backward_path(n)
        forward_times.append(forward_time)
        backward_times.append(backward_time)
    return forward_times, backward_times

def run(N, R):
    forwards = []
    backwards = []
    for n in range(N):
        forward_times, backward_times = repeat_experiment(n+1, R)
        forwards.append(forward_times)
        backwards.append(backward_times)

    forwards = np.array(forwards)
    backwards = np.array(backwards)
    return forwards, backwards

def plot(forwards, backwards):
    N = forwards.shape[0]

    mean_forward = np.mean(forwards, axis=1)
    mean_backward = np.mean(backwards, axis=1)
    std_forward = np.std(forwards, axis=1)
    std_backward = np.std(backwards, axis=1)

    x = np.arange(1, N+1)

    plt.plot(x, mean_forward, label='forward')
    # plt.fill_between(x, np.max(mean_forward - std_forward, 0), mean_forward + std_forward, alpha=0.3)
    plt.fill_between(x, mean_forward - std_forward, mean_forward + std_forward, alpha=0.3)
    plt.plot(x, mean_backward, label='backward')
    # plt.fill_between(x, np.max(mean_backward - std_backward, 0), mean_backward + std_backward, alpha=0.3)
    plt.fill_between(x, mean_backward - std_backward, mean_backward + std_backward, alpha=0.3)

    plt.legend()
    plt.xlabel('n')
    plt.ylabel('time (s)')
    plt.xticks(np.arange(1, N+1))
    plt.ylim(0,)
    plt.show()

forwards, backwards = run(8, 10)
plot(forwards, backwards)
plt.savefig('process_time_parallel_sparse_A100.pdf')
