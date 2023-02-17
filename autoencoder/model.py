# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:05:46 2022

@author: phy71
"""


import torch
import torch.nn as nn

class autoencorder(nn.Module):
    def __init__(self):
        super(autoencorder, self).__init__()
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
			torch.nn.Sigmoid()
            )
    
    def forward(self, x):
        return self.layer1(x)
