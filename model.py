# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:05:46 2022

@author: phy71
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.fft import dct, idct

def discrete_cosine_transform(u, axis=-1):
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    D = torch.tensor(dct(np.eye(n), axis=-1), dtype=torch.float, device=u.device)
    y = u@D
    
    if axis != -1:
        y = torch.transpose(y, -1, axis)
        
    return y

def inverse_discrete_cosine_transform(u, axis=-1):
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    D = torch.tensor(idct(np.eye(n), axis=-1), dtype=torch.float, device=u.device)
    y = u@D
    
    if axis != -1:
        y = torch.transpose(y, -1, axis)
        
    return y

class SoftThresholding(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.T = torch.nn.Parameter(torch.rand(self.num_features)/10)
              
    def forward(self, x):
        return torch.mul(torch.sign(x), torch.nn.functional.relu(torch.abs(x)-torch.abs(self.T)))

class DCT1D(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.ST = SoftThresholding(self.num_features)    
        self.v = torch.nn.Parameter(torch.rand(self.num_features))
         
    def forward(self, x):
        f0 = x
        f1 = discrete_cosine_transform(f0)
        f2 = self.v*f1
        f3 = self.ST(f2)
        f4 = inverse_discrete_cosine_transform(f3)
        y = f4
        return y

class dct_net(nn.Module):
    def __init__(self):
        super(dct_net, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
			torch.nn.ReLU(),
			torch.nn.Linear(16, 8),
			torch.nn.ReLU(),
			torch.nn.Linear(8, 4),
			torch.nn.ReLU(),
            torch.nn.Linear(4, 8),
			torch.nn.ReLU(),
			torch.nn.Linear(8, 16),
			torch.nn.ReLU(),
			torch.nn.Linear(16, 32),
			torch.nn.Sigmoid(),
            
            DCT1D(32),
            )
    
    def forward(self, x):
        return self.layer1(x)