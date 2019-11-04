import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--method', type=str, default='dopri5')

parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class HarmonicOscillator(nn.Module):
    def __init__(self, m, k):
        super().__init__()
        self.register_buffer('m', m)
        self.register_buffer('k', k)
        self.register_buffer('M', torch.zeros(2,2))
        self.M[0,1] = 1
        self.M[1,0] = -k / m

        self.net = nn.Sequential(nn.Linear(2,10), 
                                 nn.Softplus(),
                                 nn.Linear(10,2))
        
    def forward(self, t, x):
        delta = 0
        return (self.M @ x) + delta

    def exact(self, x0, t):
        omega = torch.sqrt(self.k / self.m)
        c1 = x0[0]
        c2 = x0[1] / omega
        A = torch.sqrt(c1**2 + c2**2)
        phi = torch.atan2(c2, c1)

        x = A * torch.cos(omega * t - phi)
        v = -A * omega * torch.sin(omega * t - phi)
        return torch.stack([x, v], 1)


def torch2np(x):
    return x.cpu().detach().numpy()

class Solver(nn.Module):
    def __init__(self, r=1):
        super().__init__()
        self.r = 1
        self.finegrid = torch.linspace(0,1,num=r)

    def forward(self, func, x0, times):
        for t in times:
            Yr t = []
            Fr = []
            for r in range(self.r):
                func(t, x)



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Constants
    m = 0.01 + torch.abs(torch.randn(1))
    k = 0.01 + torch.abs(torch.randn(1))
    # Initial conditions
    x0 = torch.randn(2).to(device)

    methods = ['euler', 'midpoint', 'rk4', 'dopri5', 'adams']
    errors = []
    for method in methods:
        model = HarmonicOscillator(m, k).to(device)
        t = torch.linspace(0.,10.,100).to(device)

        y = odeint(model, x0, t, method=method)
        ytrue = model.exact(x0, t)

        t = torch2np(t)
        y = torch2np(y)
        ytrue = torch2np(ytrue)

        diff = L2diff(y[:,0], ytrue[:,0])
        errors.append(diff)
        print("Method: {:s}, L2 error: {:f}".format(method, L2error(y[:,0], ytrue[:,0])))

    plot_diff(t, np.stack(errors, 1), methods)

def plot(t, y, ytrue):
    plt.figure(figsize=(12,8))
    plt.plot(t, y[:,0], 'b.')
    plt.plot(t, y[:,1], 'r.')
    plt.plot(t, ytrue[:,0], 'b')
    plt.plot(t, ytrue[:,1], 'r')
    plt.show()
    
def plot_diff(t, diff, methods):
    plt.figure(figsize=(12,8))
    for i, method in enumerate(methods):
        plt.semilogy(t, diff[:,i], label=method)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Log error", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(t[0], t[-1])
    plt.legend()
    plt.show()

def L2diff(y, ytrue):
    return np.sqrt((y - ytrue)**2)

def L2error(y, ytrue):
    return np.sqrt(np.sum((y - ytrue)**2))

if __name__ == '__main__':
    main()
