import torch
import torch.nn as nn
import numpy as np
from conf import device


class CompletelyLearned(nn.Module):
    """Just a linear neural network layer, initialized with random Gaussian entries"""

    def __init__(self, m, n):
        super(CompletelyLearned, self).__init__()
        self.m = m
        self.n = n
        self.param = nn.Parameter(torch.normal(0, 1/ np.sqrt(n), size=(1, m, n)).to(device))
 
    def forward(self, b, test=False):
        return self.param.repeat(b, 1, 1)

def to_superpixel(phi):
  if len(phi.shape) == 2:
    m, n = phi.shape
    b = 1
    twod=True
  if len(phi.shape) == 3:
    b, m, n = phi.shape
    twod=False

  weight = torch.ones((3, 3)).to(device)
  weight = weight.unsqueeze(0).unsqueeze(0)
  phi = phi.reshape(b*m , 1, int(np.sqrt(n)), int(np.sqrt(n))).contiguous()
  phi = torch.nn.functional.conv2d(phi, weight, bias=None, stride=1, padding=1, dilation=1, groups=1).clamp(0, 1)
  if twod:
      return phi.reshape(m, n)
  return phi.reshape(b, m, n)

class Pooling(nn.Module):
    """Class for learning pooling masks (d ones per row)"""

    def __init__(self, m, n, d, initial_scalar, random_seed):
        super(Pooling, self).__init__()
        self.m = m
        self.n = n
        self.d = d
        torch.manual_seed(random_seed)
        noise = - torch.log(- torch.log(torch.rand((1, m, n), device=device))) / 1000
        self.temperature = 1
        self.param = nn.Parameter(noise)
        self.scalar = nn.Parameter(torch.tensor( initial_scalar,device=device))


    def forward(self, b, test=False):

        if not test:
            logits = self.param.repeat(b, 1, 1)
            noise = - torch.log(- torch.log(torch.rand((b, self.m, self.n), device=device)))

            probs = torch.softmax((logits + noise / 1000) / self.temperature, dim=2)
            values, index = torch.topk(probs, self.d, dim=2)

            y_hard = torch.zeros_like(probs, memory_format=torch.legacy_contiguous_format).scatter_(2, index, 1.0)
            phi = y_hard - probs.detach() + probs

        if test:
            phi = torch.zeros(1, self.m, self.n, device=device)
            noise = 1 * - torch.log(- torch.log(torch.rand((1, self.m, self.n), device=device))) / 1000
            val, index = torch.topk(self.param + noise, self.d, dim=2)
            phi.scatter_(2, index, 1.0)
            phi = phi.reshape(1, self.m, self.n)
            phi = phi.repeat(b, 1, 1)

        norm = (phi.norm(dim=2).unsqueeze(2) + 10e-3)
        return phi / norm * torch.maximum(torch.tensor(0.01).to(device),self.scalar)

class Pixel(nn.Module):
    """Class for learning pixel masks (d ones per row)"""

    def __init__(self, m, n, d, initial_scalar, random_seed, use_superpixel):
        super(Pixel, self).__init__()
        self.m = m
        self.n = n
        self.d = d
        self.use_superpixel = use_superpixel
        torch.manual_seed(random_seed)
        noise = - torch.log(- torch.log(torch.rand((1, m, n), device=device))) / 1000
        self.temperature = 1
        self.param = nn.Parameter(noise)
        self.scalar = nn.Parameter(torch.ones(1) * initial_scalar)


    def forward(self, b, test=False):

        if not test:
            logits = self.param.repeat(b, 1, 1)
            noise = - torch.log(- torch.log(torch.rand((b, self.m, self.n), device=device)))

            probs = torch.softmax((logits + noise / 1000) / self.temperature, dim=2)
            values, index = torch.topk(probs, self.d, dim=2)

            y_hard = torch.zeros_like(probs, memory_format=torch.legacy_contiguous_format).scatter_(2, index, 1.0)
            phi = y_hard - probs.detach() + probs
            

        if test:
            phi = torch.zeros(1, self.m, self.n, device=device)
            noise = 1 * - torch.log(- torch.log(torch.rand((1, self.m, self.n), device=device))) / 1000
            val, index = torch.topk(self.param + noise, self.d, dim=2)
            phi.scatter_(2, index, 1.0)
            phi = phi.reshape(1, self.m, self.n)
            phi = phi.repeat(b, 1, 1)

        if self.use_superpixel:
          phi = to_superpixel(phi)
        norm = (phi.norm(dim=2).unsqueeze(2) + 10e-3)
        return phi / norm * torch.maximum(torch.tensor(0.01).to(device),self.scalar)


class LeftDRegularGraph(nn.Module):

    def __init__(self, m, n, d, initial_scalar, random_seed):
        super(LeftDRegularGraph, self).__init__()
        self.m = m
        self.n = n
        self.d = d
        torch.manual_seed(random_seed)
        noise = - torch.log(- torch.log(torch.rand((1, m, n), device=device))) / 1000

        self.param = nn.Parameter(noise)
        self.temperature = 1
        self.scalar =nn.Parameter(torch.ones(1) * initial_scalar)

    def forward(self, b, test=False):

        if not test:
            logits = self.param.repeat(b, 1, 1)
            noise = - torch.log(- torch.log(torch.rand((b, self.m, self.n), device=device)))

            probs = torch.softmax((logits + noise / 1000) / 1, dim=1)
            values, index = torch.topk(probs, self.d, dim=1)

            y_hard = torch.zeros_like(probs, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
            phi = y_hard - probs.detach() + probs

        if test:
            phi = torch.zeros(self.m, self.n, device=device)
            val, index = torch.topk(self.param[0], self.d, dim=0)
            phi.scatter_(0, index, 1.0)
            phi = phi.reshape(1, self.m, self.n)
            phi = phi.repeat(b, 1, 1)


        return phi / np.sqrt(self.d) * torch.maximum(torch.tensor(0.01).to(device),self.scalar)


class LoadedFromNumpy(nn.Module):
  """Loads a Numpy Matrix"""

  def __init__(self, constant, m, n, path, d):
    super(LoadedFromNumpy, self).__init__()
    self.phi = torch.tensor(np.load(path)).to(device)
    self.param = torch.zeros(m, n)
    self.d = d
    self.constant = constant
  
  def __call__(self, b, test=False):
    return self.phi.unsqueeze(0).repeat(b, 1, 1) * self.constant

class ConstructedPooling:
  """Constructs the pooling matrix from: 
  Petersen, Hendrik Bernd, Bubacarr Bah, and Peter Jung. 
  'Practical high-throughput, non-adaptive and noise-robust SARS-CoV-2 testing' ."""
  def __init__(self, scalar):
    self.scalar = scalar
    Q = 31 # pool size
    N = Q**2 # population size
    prevalence = 0.73 # infection rate [%]
    S1 = np.int(np.round((prevalence*N/100))) # between 1 to N
    #print('Infected/Population: {}/{}\n'.format(S1,N))
    # Pooling strategy to mix viral loads
    # Permutation matrix
    P = np.zeros((Q,Q))
    P[0,Q-1] = 1
    for q in range(0,Q-1):
        P[q+1,q] = 1
    #print('Permutation matrix P.shape:', P.shape)
    #print(P)
    # Pooling matrix (A) to mix viral loads
    M = (S1+1)*Q # number of qPCR tests
    A = np.ones((M,N))
    for s in range(1,S1+1 +1):
        for q in range(1,Q +1):
            A_sq = 1/(S1+1) * np.linalg.matrix_power(P,(s-1)*(q-1))
            A[(s-1)*Q:s*Q,(q-1)*Q:q*Q] = A_sq
    self.A = torch.tensor(A).float().to(device)
    self.param = self.A
  
  def __call__(self, b, test=False):
    return self.A.unsqueeze(0).repeat(b, 1, 1) * self.scalar

  def to(self, device):
    return self




def circulant(tensor, dim):
    """From: https://stackoverflow.com/questions/69820726/is-there-a-way-to-compute-a-circulant-matrix-in-pytorch
    get a circulant version of the tensor along the {dim} dimension.
    The additional axis is appended as the last dimension.
    E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,))

class CircularConv(nn.Module):
  
  def __init__(self, m, n, kernel_width, d, initial_scalar, random_seed):
        super(CircularConv, self).__init__()
        self.m = m 
        self.n = n
        self.d = d
        self.kernel_width = kernel_width
        self.scalar = nn.Parameter(torch.tensor([initial_scalar], device=device).float())

        torch.manual_seed(random_seed)
        noise = - torch.log(- torch.log(torch.rand((1, kernel_width), device=device))) / 1000
        self.kernel_param = nn.Parameter(noise)
        noise = - torch.log(- torch.log(torch.rand((1, n), device=device))) / 1000
        self.mask_param = nn.Parameter(noise)
        self.temperature = 1
        
  def forward(self, b, test=False):

        if not test:
            logits = self.kernel_param.repeat(b, 1)
            noise = - torch.log(- torch.log(torch.rand((b, self.kernel_width), device=device)))

            probs = torch.softmax((logits + noise / 1000) / self.temperature, dim=1)
            values, index = torch.topk(probs, self.d, dim=1)

            y_hard = (0 * torch.ones_like(probs, memory_format=torch.legacy_contiguous_format)).scatter_(1, index, 1.0)
            kernel = y_hard - probs.detach() + probs



            logits = self.mask_param.repeat(b, 1)
            noise = - torch.log(- torch.log(torch.rand((b, self.n), device=device)))
            probs = torch.softmax((logits + noise / 1000) / self.temperature, dim=1)
            values, index = torch.topk(probs, self.m, dim=1)
            y_hard = (0 * torch.ones_like(probs, memory_format=torch.legacy_contiguous_format)).scatter_(1, index, 1.0)
            mask = y_hard - probs.detach() + probs
            mask = mask.reshape(b, self.n, 1)


        if test:
            kernel = ( 0 *  torch.ones(1, self.kernel_width, device=device))
            noise = 0.2 * - torch.log(- torch.log(torch.rand((1, self.kernel_width), device=device))) / 1000
            val, index = torch.topk(self.kernel_param + noise, self.d, dim=1)
            kernel.scatter_(1, index, 1.0)
            kernel = kernel.reshape(1, self.kernel_width)
            kernel = kernel.repeat(b, 1)

            mask = ( 0 *  torch.ones(1, self.n, device=device))
            noise = 0.2 * - torch.log(- torch.log(torch.rand((1, self.n), device=device))) / 1000
            val, index = torch.topk(self.mask_param + noise, self.m, dim=1)
            mask.scatter_(1, index, 1.0)
            mask = mask.reshape(1, self.n, 1)
            mask = mask.repeat(b, 1, 1)
        
        kernel = kernel / np.sqrt(self.n) * torch.maximum(torch.tensor(0.01).to(device),self.scalar)

        phi = circulant(kernel, -1)
        return phi[mask.bool().squeeze(2)].reshape(-1, self.m, 784)
