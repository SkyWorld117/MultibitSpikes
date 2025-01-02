from spikingjelly.activation_based.model import spiking_resnet, sew_resnet, parametric_lif_net
from spikingjelly.activation_based import layer
from spikingjelly.activation_based.encoding import PoissonEncoder

import torch
import torch.ao.quantization
import torch.nn as nn

from copy import deepcopy

__all__ = ['CIFAR10Net', 'CIFAR10DVSNet', 'FashionMNISTNet', 'MNISTNet', 'NMNISTNet', 'DVSGestureNet']

class CIFAR10Net(nn.Module):
    def __init__(self, channels=256, spiking_neuron: callable = None, quantize: bool = False, **kwargs):
        super().__init__()

        self.encoder = PoissonEncoder()

        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels

                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))

            conv.append(layer.MaxPool2d(2, 2))


        self.conv = nn.Sequential(*conv)

        self.flatten = layer.Flatten()
        self.dropout1 = layer.Dropout(0.5)
        self.fc1 = layer.Linear(channels * 8 * 8, 2048)
        self.spiking1 = spiking_neuron(**deepcopy(kwargs))

        self.dropout2 = layer.Dropout(0.5)
        self.fc2 = layer.Linear(2048, 100)
        self.spiking2 = spiking_neuron(**deepcopy(kwargs))

        self.voting = layer.VotingLayer(10)

        self.num_spiking = 3
        self.nnz = [ [] for _ in range(self.num_spiking + 1) ]

    def forward(self, x):
        # x = self.encoder(x)
        self.nnz[0].append(x.nonzero().size(0) / x.numel())

        x = self.conv(x)
        self.nnz[1].append(x.nonzero().size(0) / x.numel())

        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.spiking1(x)
        self.nnz[2].append(x.nonzero().size(0) / x.numel())

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.spiking2(x)
        self.nnz[3].append(x.nonzero().size(0) / x.numel())

        x = self.voting(x)
        
        return x

class CIFAR10DVSNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, quantize: bool = False, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(nn.Conv2d(in_channels, channels, 3, padding=1, bias=False))
            conv.append(nn.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(nn.MaxPool2d(2))

        self.conv = nn.Sequential(*conv)

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(channels * 8 * 8, 512)
        self.spiking1 = spiking_neuron(**deepcopy(kwargs))

        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 100)
        self.spiking2 = spiking_neuron(**deepcopy(kwargs))

        self.voting = layer.VotingLayer(10)

        self.num_spiking = 3
        self.nnz = [ [] for _ in range(self.num_spiking + 1) ]

    def forward(self, x):
        self.nnz[0].append(x.nonzero().size(0) / x.numel())

        x = self.conv(x)
        self.nnz[1].append(x.nonzero().size(0) / x.numel())

        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.spiking1(x)
        self.nnz[2].append(x.nonzero().size(0) / x.numel())

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.spiking2(x)
        self.nnz[3].append(x.nonzero().size(0) / x.numel())

        x = self.voting(x)

        return x

class FashionMNISTNet(nn.Module):
    def __init__(self, spiking_neuron: callable = None, quantize: bool = False, **kwargs):
        super().__init__()

        self.encoder = PoissonEncoder()

        if quantize:
            self.quant1 = torch.ao.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.mp1 = nn.MaxPool2d(2)
        if quantize:
            self.dequant1 = torch.ao.quantization.DeQuantStub()
        self.spiking1 = spiking_neuron(**deepcopy(kwargs))

        if quantize:
            self.quant2 = torch.ao.quantization.QuantStub()
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.mp2 = nn.MaxPool2d(2)
        if quantize:
            self.dequant2 = torch.ao.quantization.DeQuantStub()
        self.spiking2 = spiking_neuron(**deepcopy(kwargs))

        if quantize:
            self.quant3 = torch.ao.quantization.QuantStub()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4 * 4, 10)
        if quantize:
            self.dequant3 = torch.ao.quantization.DeQuantStub()
        self.spiking3 = spiking_neuron(**deepcopy(kwargs))

        self.quantize = quantize

        self.num_spiking = 3
        self.nnz = [ [] for _ in range(self.num_spiking + 1) ]

    def forward(self, x):
        x = self.encoder(x)

        self.nnz[0].append(x.nonzero().size(0) / x.numel())

        if self.quantize:
            x = self.quant1(x)
        x = self.conv1(x)
        x = self.mp1(x)
        if self.quantize:
            x = self.dequant1(x)
        x = self.spiking1(x)
        self.nnz[1].append(x.nonzero().size(0) / x.numel())

        if self.quantize:
            x = self.quant2(x)
        x = self.conv2(x)
        x = self.mp2(x)
        if self.quantize:
            x = self.dequant2(x)
        x = self.spiking2(x)
        self.nnz[2].append(x.nonzero().size(0) / x.numel())

        if self.quantize:
            x = self.quant3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        if self.quantize:
            x = self.dequant3(x)
        x = self.spiking3(x)
        self.nnz[3].append(x.nonzero().size(0) / x.numel())
        
        return x

class MNISTNet(FashionMNISTNet):
    pass

class NMNISTNet(nn.Module):
    def __init__(self, spiking_neuron: callable = None, quantize: bool = False, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, 5)
        self.mp1 = nn.MaxPool2d(2)
        self.spiking1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.mp2 = nn.MaxPool2d(2)
        self.spiking2 = spiking_neuron(**deepcopy(kwargs))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*5*5, 10)
        self.spiking3 = spiking_neuron(**deepcopy(kwargs))

        self.num_spiking = 3

        self.nnz = [ [] for _ in range(self.num_spiking+1) ]

    def forward(self, x: torch.Tensor):
        self.nnz[0].append(x.nonzero().size(0) / x.numel())

        x = self.conv1(x)
        x = self.mp1(x)
        x = self.spiking1(x)
        self.nnz[1].append(x.nonzero().size(0) / x.numel())

        x = self.conv2(x)
        x = self.mp2(x)
        x = self.spiking2(x)
        self.nnz[2].append(x.nonzero().size(0) / x.numel())

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.spiking3(x)
        self.nnz[3].append(x.nonzero().size(0) / x.numel())

        return x

class DVSGestureNet(nn.Module):

    def __init__(self, spiking_neuron: callable = None, quantize: bool = False, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, 5)
        self.mp1 = nn.MaxPool2d(2)
        self.spiking1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.mp2 = nn.MaxPool2d(2)
        self.spiking2 = spiking_neuron(**deepcopy(kwargs))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*29*29, 11)
        self.spiking3 = spiking_neuron(**deepcopy(kwargs))

        self.num_spiking = 3

        self.nnz = [ [] for _ in range(self.num_spiking+1) ]

    def forward(self, x: torch.Tensor):
        self.nnz[0].append(x.nonzero().size(0) / x.numel())

        x = self.conv1(x)
        x = self.mp1(x)
        x = self.spiking1(x)
        self.nnz[1].append(x.nonzero().size(0) / x.numel())

        x = self.conv2(x)
        x = self.mp2(x)
        x = self.spiking2(x)
        self.nnz[2].append(x.nonzero().size(0) / x.numel())

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.spiking3(x)
        self.nnz[3].append(x.nonzero().size(0) / x.numel())

        return x
    
# class DVSGestureNet(nn.Module):
#     def __init__(self, channels=16, spiking_neuron: callable = None, **kwargs):
#         super().__init__()

#         conv = []
#         for i in range(2):
#             if conv.__len__() == 0:
#                 in_channels = 2
#             else:
#                 in_channels = channels

#             conv.append(nn.Conv2d(in_channels, channels, 3, padding=1))
#             # conv.append(nn.BatchNorm2d(channels))
#             conv.append(spiking_neuron(**deepcopy(kwargs)))
#             conv.append(nn.MaxPool2d(2))

#         self.conv = nn.Sequential(*conv)

#         self.flatten = nn.Flatten()
#         # self.dropout1 = nn.Dropout(0.5)
#         # self.fc1 = nn.Linear(channels * 4 * 4, 256)
#         # self.fc1 = nn.Linear(channels * 4 * 4, 110)
#         self.fc1 = nn.Linear(channels * 32 * 32, 11)
#         self.spiking1 = spiking_neuron(**deepcopy(kwargs))
        
#         # self.dropout2 = nn.Dropout(0.5)
#         # self.fc2 = nn.Linear(256, 110)
#         # self.spiking2 = spiking_neuron(**deepcopy(kwargs))

#         # self.voting = layer.VotingLayer(10)

#         self.num_spiking = 3

#         self.train_nnz = [ [] for _ in range(self.num_spiking+1) ]
#         self.test_nnz = [ [] for _ in range(self.num_spiking+1) ]

#         self.is_train = True
    
#     def forward(self, x: torch.Tensor):
#         if self.is_train:
#             self.train_nnz[0].append(x.nonzero().size(0) / x.numel())
#         else:
#             self.test_nnz[0].append(x.nonzero().size(0) / x.numel())

#         x = self.conv(x)
#         if self.is_train:
#             self.train_nnz[1].append(x.nonzero().size(0) / x.numel())
#         else:
#             self.test_nnz[1].append(x.nonzero().size(0) / x.numel())

#         x = self.flatten(x)
#         # x = self.dropout1(x)
#         x = self.fc1(x)
#         x = self.spiking1(x)
#         if self.is_train:
#             self.train_nnz[2].append(x.nonzero().size(0) / x.numel())
#         else:
#             self.test_nnz[2].append(x.nonzero().size(0) / x.numel())

#         # x = self.dropout2(x)
#         # x = self.fc2(x)
#         # x = self.spiking2(x)
#         if self.is_train:
#             self.train_nnz[3].append(x.nonzero().size(0) / x.numel())
#         else:
#             self.test_nnz[3].append(x.nonzero().size(0) / x.numel())

#         # x = self.voting(x)
        
#         return x