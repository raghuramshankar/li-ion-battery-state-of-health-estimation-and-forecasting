# %%
import math
import torch
import gpytorch
from matplotlib import pyplot as plt

if '__ipython__':
    %matplotlib widget
    %load_ext autoreload
    %autoreload 2

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 100)

# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

