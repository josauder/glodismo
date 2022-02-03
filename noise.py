import torch
import numpy as np
import torch.nn as nn
from conf import device 


class GaussianNoise(nn.Module):
    """
    Create Gaussian noise on the input with specified signal to noise ration snr.
    """
    def __init__(self, snr):
        super(GaussianNoise, self).__init__()
        self.snr = snr

    def forward(self, y):
        std = torch.std(y, dim=1) * np.power(10.0, -self.snr / 20)
        noise = torch.normal(torch.zeros_like(y, device=device),
                            std=(torch.zeros_like(y, device=device) + std.reshape(-1, 1)))
        return y + noise


class Noiseless(nn.Module):
    """
    Create Gaussian noise on the input with specified signal to noise ration snr.
    """
    def __init__(self):
        super(Noiseless, self).__init__()
        self.snr = 'inf'


    def forward(self, y):
        return y

class StudentNoise(nn.Module):
    """
    Create Gaussian noise on the input with specified signal to noise ration snr.
    """
    def __init__(self, snr):
        super(StudentNoise, self).__init__()
        self.snr = snr


    def forward(self, y):
        std = torch.std(y, dim=1) * np.power(10.0, -self.snr / 20)
        student = torch.distributions.studentT.StudentT(1, loc=0.0, scale=std.mean(), validate_args=None)
        #noise = torch.normal(torch.zeros_like(y, device=device),
        #                     std=(torch.zeros_like(y, device=device) + std.reshape(-1, 1)))
        noise = student.sample(y.shape).to(device)
        return y + noise
