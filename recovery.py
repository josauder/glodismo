import torch
import torch.nn as nn
import torch.nn.functional as F
from conf import device 
import numpy as np

def get_median_backward_op(phi, n, d, test=False, train_matrix=True):
    if test:
        def backward_op(y):
                  xh = torch.zeros(y.shape[0], n, device=device)
                  c = y[:, :, None].repeat(1, 1, n)
                  i, ii, iii = torch.where(
                      torch.abs(phi.unsqueeze(0).repeat(y.shape[0], 1, 1).transpose(2, 1)) > 0.0001)
                  l = c[i, iii, ii] 
                  l = l.reshape(y.shape[0], n, d)
                  l1 = torch.median(l, dim=-1)[0]  #
                  return l1
        return backward_op
    else:
        
        if train_matrix:
            def backward_op(y):
                xh = torch.zeros(y.shape[0], n, device=device)
                c = y[:, :, None].repeat(1, 1, n)  # .reshape(b,m,n)
                i, ii, iii = torch.where(torch.abs(phi.transpose(2, 1)) > 0.0001) 
                l = c[i, iii, ii]  # .reshape(b,d,n)#
                l = l.reshape(y.shape[0], n, d)
                l1 = torch.median(l, dim=-1)[0]  #
                return l1
            return backward_op

        else:
            def backward_op(y):
                xh = torch.zeros(y.shape[0], n, device=device)
                c = y[:, :, None].repeat(1, 1, n)  # .reshape(b,m,n)
                i, ii, iii = torch.where(torch.abs(phi.repeat(y.shape[0], 1, 1).transpose(2, 1)) > 0.0001)  
                l = c[i, iii, ii]  # .reshape(b,d,n)#
                l = l.reshape(y.shape[0], n, d)
                l1 = torch.median(l, dim=-1)[0]  #
                return l1  
            return backward_op


def hard_threshold(x, s):
    x = x.reshape(x.shape[0], -1)

    abs_ = torch.abs(x)
    topk, _ = torch.topk(abs_, s, dim=1)
    topk, _ = topk.min(dim=1)
    index = (abs_ >= topk.unsqueeze(1)).float()
    return (index * x)

class IHT(nn.Module):
    def __init__(self, k, s):
        super(IHT, self).__init__()
        self.k = k
        self.s = s

    def forward(self, y, forward_op, backward_op, psi, psistar):
        x = torch.zeros_like(backward_op(y))
        for i in range(self.k):
            a = forward_op(psistar(x))
            b = y - a
            c = psi(backward_op(b))
            d = x + c
            x = hard_threshold(d, self.s)
        return x

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        self.load_state_dict(torch.load(name, map_location=device))

def soft_thresh_plain(a, theta):
    arg = torch.atan2(a[:, :, 1], a[:, :, 0])
    abs = torch.norm(a, dim=-1)
    r = torch.max(torch.zeros_like(abs), abs - theta * torch.ones_like(abs))
    real = r * torch.cos(arg)
    imag = r * torch.sin(arg)
    ff = torch.stack([real, imag])
    ff = ff.permute(1, 2, 0)
    return ff.contiguous()

def soft_threshold(x, theta, p):
    if torch.is_complex(x):
      x = torch.view_as_real(x).contiguous()

      if p == 0:
          return torch.view_as_complex(soft_thresh_plain(x, theta))
      abs_ = torch.norm(x, dim=-1)
      topk, _ = torch.topk(abs_, int(p), dim=1)
      topk, _ = topk.min(dim=1)
      index = (abs_ > topk.unsqueeze(1)).float().unsqueeze(2)
      real1 = index * x
      real2 = (1 - index) * soft_thresh_plain(x, theta)
      return torch.view_as_complex(real1 + real2)
    return soft_threshold_real(x, theta, p)

def soft_threshold_real(x, theta, p):
    shape = x.shape
    x = x.reshape(x.shape[0], -1)
    if p == 0:
        return (torch.sign(x) * torch.relu(torch.abs(x) - theta)).reshape(shape)

    abs_ = torch.abs(x)
    topk, _ = torch.topk(abs_, int(p), dim=1)
    topk, _ = topk.min(dim=1)
    index = (abs_ > topk.unsqueeze(1)).float()
    return (index * x + (1 - index) * torch.sign(x) * torch.relu(torch.abs(x) - theta)).reshape(shape)


        
class NormLSTM(nn.Module):
    def __init__(self, n, s, k, in_dim, out_dim, dim=128):
        super(NormLSTM, self).__init__()
        self.n = n
        self.k = k
        self.s = s
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.lstm = nn.LSTMCell(in_dim, dim)
        self.lll = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, out_dim)
        self.softplus = nn.Softplus()

        self.hidden = nn.Parameter(torch.Tensor(np.random.normal(size=self.dim)))
        self.cellstate = nn.Parameter(torch.Tensor(np.random.normal(size=self.dim)))

        self.bl1_mean = 0
        self.cl1_mean = 0
        self.bl1_std = 0
        self.cl1_std = 0
        self.initialized_normalizers = False

    def get_initial(self, batch_size):
        return (
            self.cellstate.unsqueeze(0).repeat(batch_size, 1),
            self.hidden.unsqueeze(0).repeat(batch_size, 1),
        )

    def forward(self, running_vars, hidden, cellstate):
        #TODO: optimize!
        l1s = [torch.norm(var, dim=0, p=1) for var in running_vars]

        if not self.initialized_normalizers:
            self.means = [l1.mean().item() for l1 in l1s]
            self.stds = [l1.std().item() for l1 in l1s]
            self.initialized_normalizers = True

        stack = torch.stack(
            [(l1s[i] - self.means[i]) / self.stds[i] for i in range(len(l1s))]
        ).T

        hidden, cellstate = self.lstm(stack, (hidden, cellstate))
        out = self.softplus(self.linear(torch.relu(self.lll(cellstate))))
        return out, hidden, cellstate

class NA_ALISTA(nn.Module):
    def __init__(self, k, s,  m, n, lstm_hidden=128):
        super(NA_ALISTA, self).__init__()
        self.m = m
        self.n = n
        self.k = k
        self.s = s
        self.initial = nn.Parameter(torch.view_as_complex(torch.randn((1, self.n, 2))))

        self.regressor = NormLSTM(n, s, k, in_dim=2, out_dim=2, dim=lstm_hidden)

        self.alpha = 0.99

    def forward(self, y, forward_op, backward_op, psi, psistar):
        x = torch.zeros_like(backward_op(y), device=device)
        cellstate, hidden = self.regressor.get_initial(y.shape[0])

        for i in range(self.k):
            a = forward_op(psistar(x))
            b = y - a
            c = psi(backward_op(b))
            pred, hidden, cellstate = self.regressor(
                (b.reshape(x.shape[0], -1).T, c.reshape(x.shape[0], -1).T),
                 hidden,
                cellstate
            )
            gamma = pred[:, :1]
            theta = pred[:, 1:]

            d = x + (gamma * c)
            s = np.clip(np.linspace(self.s / self.k, self.s * 1.4, self.k), 0, self.s*1.2).astype(np.int32)
            x = soft_threshold(d, theta, s[i])
        return x

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        self.load_state_dict(torch.load(name, map_location=device))

class NA_NNLAD(nn.Module):
    def __init__(self, k, s,  m, n, lstm_hidden=128):
        super(NA_NNLAD, self).__init__()
        self.m = m
        self.n = n
        self.k = k
        self.s = s
                
        self.x0 = nn.Parameter(0.005 + 0.02 * torch.rand(n))
        self.v0 = nn.Parameter(0.005 + 0.02 * torch.rand(n))
        self.w0 = nn.Parameter(0.005 + 0.02 * torch.rand(m))

        self.initial = nn.Parameter(torch.view_as_complex(torch.randn((1, self.n, 2))))

        self.regressor = NormLSTM(n, s, k, in_dim=5, out_dim=2, dim=lstm_hidden)

        self.alpha = 0.99

    def forward(self, y, forward_op, backward_op, psi, psistar):
        
        x = self.x0.unsqueeze(0).repeat(y.shape[0], 1)
        v = self.v0.unsqueeze(0).repeat(y.shape[0], 1)
        w = self.w0.unsqueeze(0).repeat(y.shape[0], 1)

        w_tilde = backward_op(w)
        x_tilde = forward_op(x)
        v_tilde = forward_op(v)
  
        
        cellstate, hidden = self.regressor.get_initial(y.shape[0])


        for i in range(self.k):

            b = (v_tilde - y)
            c = x
            pred, hidden, cellstate = self.regressor(
                (b.reshape(x.shape[0], -1).T, 
                c.reshape(x.shape[0], -1).T, 
                w.reshape(x.shape[0], -1).T, 
                w_tilde.reshape(x.shape[0], -1).T, 
                i / self.k), 
                hidden,
                cellstate
            )
            sigma = (pred[:, :1])
            tau = (pred[:, 1:])
            w = w + sigma * (v_tilde - y)
            w = torch.minimum(torch.ones_like(w), torch.abs(w)) * torch.sign(w)
            w_tilde = backward_op(w)
            v = -x

            #print(x[0][0], w_tilde[0][0],x[0][0] - self.tau * w_tilde[0][0] )
            x = torch.maximum(torch.zeros_like(x), x - tau * w_tilde)
            v = v + 2 * x
            v_tilde = forward_op(v)
            x_tilde = 0.5 * (v_tilde + x_tilde)
            

        return x, None

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        self.load_state_dict(torch.load(name, map_location=device))


class NNLAD(nn.Module):
    def __init__(self, k, sigma, tau):
        super(NNLAD, self).__init__()
        self.k = k
        self.sigma = sigma
        self.tau = tau


    def forward(self, y, forward_op, backward_op, psi, psistar):
        
        x = torch.zeros_like(backward_op(y))
       
        v = torch.zeros_like(x)
        w = torch.zeros_like(y)

        w_tilde = backward_op(w)
        x_tilde = forward_op(x)
        v_tilde = forward_op(v)
        for i in range(self.k):
            w = w + self.sigma * (v_tilde - y)
            w = torch.minimum(torch.ones_like(w), torch.abs(w)) * torch.sign(w)
            w_tilde = backward_op(w)
            v = -x

            x = torch.maximum(torch.zeros_like(x), x - self.tau * w_tilde)
            v = v + 2 * x
            v_tilde = forward_op(v)
            x_tilde = 0.5 * (v_tilde + x_tilde)

        return x

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        self.load_state_dict(torch.load(name, map_location=device))